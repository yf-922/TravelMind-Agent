import hashlib
import json
import pytest
from app.evaluation import rag_retrieval as r
from app.evaluation.rag_chunking import split_text
from app.evaluation.rag_protocol import validate_corpus, validate_annotations, freeze_payload
from scripts.evaluate_rag_v2 import score_case, summarize_rows, paired_bootstrap


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=True):
        return list(text)+[0,0]


@pytest.mark.parametrize('strategy',['semantic','token'])
def test_chunks_preserve_content_and_never_truncate(strategy):
    text='长'*1000+'。下一句。'
    chunks=split_text(text,CharacterTokenizer(),strategy,max_tokens=50)
    assert ''.join(chunks)==text
    assert all(len(c)+2<=50 for c in chunks)


def test_empty_chunking():
    assert split_text('',CharacterTokenizer())==[]


def test_duplicate_retrieval_cannot_inflate_recall():
    grade=score_case({'relevant_chunks':[{'chunk_id':'a','relevance':2}]},[{'chunk_id':'a'},{'chunk_id':'a'}])
    assert grade['recall']==1 and grade['precision']==1


def test_duplicate_channel_cannot_double_rrf():
    out=r.rrf([{'chunk_id':'a'},{'chunk_id':'a'}],[],2)
    assert out[0]['rrf_score']==pytest.approx(1/61)


def test_model_revision_invalidates_index():
    rows=[{'text':'same'}]
    assert r.index_fingerprint(rows,{'revision':'one'})!=r.index_fingerprint(rows,{'revision':'two'})


def test_empty_rows_do_not_load_default():
    with pytest.raises(ValueError):r.search('q','keyword',3,rows=[])


@pytest.mark.parametrize('query',['',None,'   '])
def test_empty_query_rejected(query):
    with pytest.raises(ValueError):r.search(query,'keyword',3,rows=[{'chunk_id':'a','text':'text'}])


def test_filters_apply_before_retrieval(monkeypatch):
    rows=[{'chunk_id':'a','text':'one','user_id':'one'},{'chunk_id':'b','text':'two','user_id':'two'}]
    monkeypatch.setattr(r,'_keyword',lambda query,rows,limit:rows)
    assert [x['chunk_id'] for x in r.search('q','keyword',3,rows=rows,filters={'user_id':'one'})]==['a']
    assert r.search('q','keyword',3,rows=rows,filters={'user_id':'unknown'})==[]


def test_bm25_index_cached():
    r._bm25_index.cache_clear()
    rows=[{'chunk_id':'a','text':'上海博物馆预约'},{'chunk_id':'b','text':'南京交通'}]
    r._bm25('预约',rows)
    r._bm25('交通',rows)
    assert r._bm25_index.cache_info().misses==1


def test_errors_are_zero_not_excluded():
    grade={k:1 for k in ('hit','recall','precision','rr','ndcg','fact_coverage')}
    rows=[{'answerable':True,'grade':grade,'error':None,'context_chars':1},
          {'answerable':True,'grade':None,'error':'timeout','context_chars':0}]
    summary=summarize_rows(rows)
    assert summary['metrics']['recall']==.5
    assert summary['completed_only_metrics']['recall']==1
    assert summary['availability']==.5


def test_bootstrap_accounts_for_failed_pairs():
    base={'id':'x','answerable':True,'intent_group':'g','error':None,'grade':{'recall':1}}
    bad={**base,'error':'timeout','grade':None}
    assert paired_bootstrap({'cases':[base]},{'cases':[bad]},iterations=100)['mean_delta']==-1


def test_corpus_quota_reports_missing_cities():
    assert any('丽江' in x for x in validate_corpus([],True))


def test_machine_labels_cannot_freeze():
    with pytest.raises(ValueError):freeze_payload([],[],{})


def test_human_status_without_signature_is_rejected():
    case={'id':'a','intent_group':'g','split':'test','type':'answerable',
          'relevant_chunks':[{'chunk_id':'x','relevance':2}],'annotation_status':'human_reviewed'}
    errors=validate_annotations([case],[{'chunk_id':'x'}],True,False)
    assert any('signed human review' in x for x in errors)


def test_same_intent_cannot_leak():
    cases=[{'id':str(i),'intent_group':'g','split':split} for i,split in enumerate(['dev','test'])]
    assert 'intent group leaks across splits' in validate_annotations(cases,[],False,False)


def test_chunking_preserves_newlines_and_punctuation():
    text='。\n\n第一句！！下一句？\n'
    assert ''.join(split_text(text,CharacterTokenizer(),max_tokens=10))==text


def test_generation_preflight_and_neighbors():
    from scripts.evaluate_rag_generation import preflight,neighboring_ks
    assert neighboring_ks(1)==[1,2,3]
    assert neighboring_ks(10)==[5,8,10]
    budget=preflight([{'query':'q'}]*60,[2,3,5],3,1,2,100,200)
    assert budget['calls_upper_bound']==1080
    assert budget['paid_execution'] is False
    assert budget['estimated_cost_ceiling']>0


def test_judge_invalid_output_rejected():
    from scripts.evaluate_rag_generation import validate_judge
    with pytest.raises(ValueError):validate_judge({'correctness':2})


def test_v2_quota_and_honest_pending_status():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    rows=json.loads((root/'knowledge/benchmark_v2_curated/chunks.json').read_text(encoding='utf-8'))
    cases=json.loads((root/'evaluation/rag_benchmark_v2_draft.json').read_text(encoding='utf-8'))
    assert validate_corpus(rows,True)==[]
    assert validate_annotations(cases,rows,False,True)==[]
    assert all(c['annotation_status']=='pending_review' for c in cases)
    assert any(r['source_type']=='project_rule' for r in rows)
    assert all(r['fact_validity']=='planning_advice_not_official_fact' for r in rows if r['source_type']=='project_rule')


def test_source_review_required_even_if_query_labels_are_signed():
    row={'chunk_id':'a','text':'fact','content_hash':hashlib.sha256(b'fact').hexdigest(),
         'url':'https://example.org','city':'南京','topic':'预约','collected_at':'2026-10-08','source_type':'official'}
    assert validate_corpus([row])==[]
    assert any('signed source review' in e for e in validate_corpus([row],require_review=True))


def test_source_reconstruction_does_not_use_sorted_chunk_ids():
    from scripts.build_rag_variants import source_paragraphs
    first={'chunk_id':'z','url':'page','evidence_group':'p','text':'第一句。'}
    second={'chunk_id':'a','url':'page','evidence_group':'p','text':'第二句。'}
    unrelated={'chunk_id':'b','url':'other','evidence_group':'p','text':'其他。'}
    restored=source_paragraphs([second,unrelated,first],[first,second,unrelated])
    assert restored[0]['text']=='第一句。第二句。'
    assert len(restored)==2
    with pytest.raises(ValueError):source_paragraphs([first],[])


def test_generation_budget_counts_failed_and_unknown_usage_calls():
    from scripts.evaluate_rag_generation import CallBudget
    budget=CallBudget(1,1,1,2,100)
    reserved=budget.reserve([('human','南京')])
    budget.reconcile(reserved,None)
    assert budget.snapshot()['unknown_usage_calls']==1
    assert budget.committed_cost==reserved
    with pytest.raises(RuntimeError,match='budget exhausted'):budget.reserve([('human','q')])


def test_generation_budget_reconciles_actual_provider_usage():
    from scripts.evaluate_rag_generation import CallBudget
    budget=CallBudget(2,1,1,2,100)
    reserved=budget.reserve([('human','q')])
    budget.reconcile(reserved,{'input_tokens':10,'output_tokens':20})
    assert budget.committed_cost==pytest.approx(.00005)


def test_generation_summary_keeps_failures_in_denominator():
    from scripts.evaluate_rag_generation import summarize_generation
    good={'judge':{s:1 for s in ('correctness','evidence_support','preference_match','refusal_correctness')},
          'generator_usage':{'input_tokens':10,'output_tokens':5},'human_review_status':'pending'}
    report=summarize_generation([good,{'error':'timeout'}],2)
    assert report['availability']==.5
    assert report['judge_scores_including_failures']['correctness']==.5
    assert report['judge_scores_completed_only']['correctness']==1
    assert report['actual_provider_usage']['generator']['input_tokens']==10
    assert report['human_review_complete'] is False


def test_pinned_tokenizer_never_downloads_in_retrieval(monkeypatch):
    import transformers
    seen=[]
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *args,**kwargs:seen.append(kwargs) or CharacterTokenizer())
    r.embedding_tokenizer.cache_clear()
    try:
        assert r.token_count('hello')==7
        assert seen[0]['revision']==r.REVISION and seen[0]['local_files_only'] is True
    finally:r.embedding_tokenizer.cache_clear()


def test_freeze_roundtrip_rejects_changed_labels_and_config(tmp_path,monkeypatch):
    from app.evaluation.rag_protocol import verify_freeze
    monkeypatch.setattr(r,'token_count',lambda text:100)
    rows=json.loads((r.ROOT/'knowledge/benchmark_v2_curated/chunks.json').read_text(encoding='utf-8'))
    cases=json.loads((r.ROOT/'evaluation/rag_benchmark_v2_draft.json').read_text(encoding='utf-8'))
    # Synthetic test signatures never exported to benchmark assets.
    for row in rows:
        row.update(review_status='human_reviewed',annotator='unit-test-only',reviewed_at='2026-10-08',change_log=['test fixture'])
    for case in cases:
        case.update(annotation_status='human_reviewed',annotator='unit-test-only',reviewed_at='2026-10-08',change_log=['test fixture'])
    config={'mode':'bm25','k':3,'rrf_c':60}
    path=tmp_path/'freeze.json'
    path.write_text(json.dumps(freeze_payload(cases,rows,config)),encoding='utf-8')
    assert verify_freeze(path,cases,rows,config)['status']=='frozen'
    with pytest.raises(ValueError,match='changed'):verify_freeze(path,cases,rows,{**config,'k':5})
    cases[0]['query']+='changed'
    with pytest.raises(ValueError,match='changed'):verify_freeze(path,cases,rows,config)


def test_unpinned_diagnostic_model_rejected_without_changing_default():
    original=r.embedding_config()
    with pytest.raises(ValueError):r.configure_embedding({**original,'revision':'main'})
    assert r.embedding_config()==original


def test_not_run_generation_reports_null_quality():
    from scripts.evaluate_rag_generation import summarize_generation
    report=summarize_generation([],540)
    assert report['execution_status']=='not_run'
    assert report['judge_scores_including_failures']['correctness'] is None


def test_freeze_checks_actual_tokenizer_hard_budget(monkeypatch):
    from app.evaluation import rag_protocol as protocol
    monkeypatch.setattr(protocol,'validate_corpus',lambda *args:[])
    monkeypatch.setattr(protocol,'validate_annotations',lambda *args:[])
    monkeypatch.setattr(r,'token_count',lambda text:513)
    with pytest.raises(ValueError,match='exceeds embedding token budget'):
        freeze_payload([],[{'chunk_id':'long','text':'long'}],{'mode':'bm25','k':3,'rrf_c':60})


def test_vector_index_repairs_stale_documents_before_query(monkeypatch):
    import sys
    from types import SimpleNamespace
    class Collection:
        documents={'a':'stale'}
        writes=0
        def get(self,include):return {'ids':list(self.documents),'documents':list(self.documents.values())}
        def delete(self,ids):self.documents={}
        def upsert(self,ids,documents,**kwargs):
            self.documents=dict(zip(ids,documents));self.writes+=1
    collection=Collection()
    monkeypatch.setitem(sys.modules,'chromadb',SimpleNamespace(PersistentClient=lambda **kwargs:
        SimpleNamespace(get_or_create_collection=lambda **kwargs:collection)))
    monkeypatch.setattr(r,'token_count',lambda text:5)
    monkeypatch.setattr(r,'embedding_model',lambda:SimpleNamespace(encode=lambda *a,**k:SimpleNamespace(tolist=lambda:[[1.,0.]])))
    rows=[{'chunk_id':'a','text':'current'}]
    assert r._collection(rows).documents=={'a':'current'}
    r._collection(rows)
    assert collection.writes==1


def test_negative_case_with_zero_relevance_is_not_counted_answerable(monkeypatch):
    from scripts.evaluate_rag_v2 import evaluate
    monkeypatch.setattr('scripts.evaluate_rag_v2.search',lambda *args,**kwargs:[])
    case={'id':'n','type':'out_of_corpus','query':'outside','relevant_chunks':[{'chunk_id':'x','relevance':0}]}
    report=evaluate([case],'bm25',3)
    assert report['cases'][0]['answerable'] is False
    assert report['no_answer_by_type']['out_of_corpus']['empty_rate']==1


def test_top_k_metric_budget_and_context_separator_budget():
    from scripts.evaluate_rag_generation import context_for
    case={'relevant_chunks':[{'chunk_id':i,'relevance':2} for i in ('a','b')]}
    grade=score_case(case,[{'chunk_id':'a'},{'chunk_id':'b'}],1)
    assert grade['recall']==.5 and grade['precision']==1
    rows=[{'chunk_id':'a','text':'one'},{'chunk_id':'b','text':'two'}]
    assert len(context_for(rows,14))<=14
