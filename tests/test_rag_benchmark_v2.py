import math
import pytest
from app.evaluation import rag_retrieval as r
from scripts.evaluate_rag_v2 import score_case


def test_chunk_recall_is_not_source_hit():
    case = {"relevant_chunks": [{"chunk_id": "a", "relevance": 2}, {"chunk_id": "b", "relevance": 1}]}
    grade = score_case(case, [{"chunk_id": "a"}, {"chunk_id": "wrong"}])
    assert grade["recall"] == 0.5
    assert grade["hit"] is True
    assert grade["precision"] == 0.5
    assert grade["rr"] == 1
    assert grade["ndcg"] == pytest.approx(2 / (2 + 1/math.log2(3)))


def test_standard_rrf_does_not_round_before_sorting():
    rows = [{"chunk_id": str(i), "text": str(i)} for i in range(20)]
    result = r.rrf(rows, [], 10, 100)
    assert [x["chunk_id"] for x in result] == [str(i) for i in range(10)]
    assert result[0]["rrf_score"] == 1/101


def test_hybrid_cannot_silently_degrade(monkeypatch):
    monkeypatch.setattr(r, "_vector", lambda *a: (_ for _ in ()).throw(RuntimeError("embedding unavailable")))
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        r.search("query", "hybrid", 10, rows=[{"chunk_id": "a", "text": "a"}])


def test_top_k_ten_not_truncated_to_five(monkeypatch):
    rows = [{"chunk_id": str(i), "text": "text"} for i in range(12)]
    monkeypatch.setattr(r, "_vector", lambda query, source, limit: source[:limit])
    assert len(r.search("q", "vector", 10, rows=rows)) == 10


def test_fingerprint_changes_on_content_update():
    assert r.corpus_fingerprint([{"text": "old"}]) != r.corpus_fingerprint([{"text": "new"}])


def test_k_selection_uses_both_recall_and_ndcg():
    from scripts.evaluate_rag_v2 import select_k
    reports=[{'k':k,'errors':0,'metrics':{'recall':recall,'ndcg':ndcg}} for k,recall,ndcg in [(1,.9,.9),(3,.99,.91),(5,1,.92)]]
    assert select_k(reports)==3


def test_duplicate_chunks_do_not_inflate_fact_coverage():
    case={'relevant_chunks':[{'chunk_id':'a','relevance':2,'evidence_group':'same'}, {'chunk_id':'b','relevance':2,'evidence_group':'same'}]}
    result=score_case(case,[{'chunk_id':'a','evidence_group':'same'}])
    assert result['recall']==.5
    assert result['fact_coverage']==1


def test_same_source_different_chunk_is_not_relevant():
    assert score_case({'relevant_chunks':[{'chunk_id':'a','relevance':2}]},[{'chunk_id':'b','source':'same'}])['recall']==0


def test_versioned_corpus_has_unique_ids_hashes_and_unknown_indoor_tags():
    rows=r.load_chunks()
    assert len({x['chunk_id'] for x in rows})==len(rows)
    assert len({x['content_hash'] for x in rows})==len(rows)
    assert all(x['indoor']=='unknown' and x['url'].startswith('https://') for x in rows)


def test_draft_intent_groups_cannot_leak_across_splits():
    import json
    cases=json.loads((r.ROOT/'evaluation/rag_benchmark_v1_draft.json').read_text(encoding='utf-8'))
    groups={}
    for case in cases:
        assert case['annotation_status']=='pending_review'
        groups.setdefault(case['intent_group'],set()).add(case['split'])
    assert all(len(splits)==1 for splits in groups.values())
