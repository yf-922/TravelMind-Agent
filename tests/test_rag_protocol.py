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
