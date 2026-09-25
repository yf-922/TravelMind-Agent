import json
from pathlib import Path

from scripts.evaluate_rag import evaluate, validate_citations


ROOT = Path(__file__).resolve().parents[1]


def test_rag_golden_set_meets_offline_recall_gate(monkeypatch):
    monkeypatch.setenv("TRAVEL_KNOWLEDGE_ENABLED", "0")
    cases = json.loads((ROOT / "evaluation" / "rag_golden_set.json").read_text(encoding="utf-8"))

    report = evaluate(cases, top_k=3)

    assert report["case_count"] == 30
    assert report["answerable_cases"] == 24
    assert report["no_answer_cases"] == 6
    assert report["metrics"]["recall_at_k"] >= 0.90
    assert report["metrics"]["no_answer_accuracy"] >= 0.90


def test_citation_validator_rejects_sources_not_returned_by_retrieval():
    retrieved = [{"source": "trusted", "chunk_id": "trusted-0"}]
    result = validate_citations(
        "[source: trusted#trusted-0] [source: forged#forged-9]", retrieved
    )

    assert result["citations"] == 2
    assert result["valid_citations"] == 1
    assert result["validity_rate"] == 0.5


def test_rag_evaluation_scores_no_answer_cases_as_correct_rejections(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluate_rag.travel_knowledge.search_travel_knowledge",
        lambda query, limit, mode: [] if "unrelated" in query else [{
            "source": "guide", "chunk_id": "guide-0", "text": "grounded",
            "retrieval_mode": "keyword",
        }],
    )
    report = evaluate([
        {"id": "positive", "difficulty": "normal", "query": "travel", "expected_sources": ["guide"]},
        {"id": "negative", "difficulty": "no_answer", "query": "unrelated", "expected_sources": []},
    ], top_k=3)

    assert report["metrics"]["recall_at_k"] == 1.0
    assert report["metrics"]["no_answer_accuracy"] == 1.0
    assert report["metrics"]["case_accuracy"] == 1.0


def test_in_domain_unknowns_are_not_misgraded_as_retrieval_failures(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluate_rag.travel_knowledge.search_travel_knowledge",
        lambda query, limit, mode: [{
            "source": "responsible_trip_checklist", "chunk_id": "responsible_trip_checklist-0",
            "text": "不得编造实时资料", "retrieval_mode": "keyword",
        }],
    )
    cases = json.loads((ROOT / "evaluation" / "rag_hard_negative_set.json").read_text(encoding="utf-8"))
    report = evaluate(cases)

    assert report["case_count"] == 8
    assert report["generation_only_cases"] == 8
    assert all(row["case_passed"] is None for row in report["cases"])
    assert report["metrics"]["case_accuracy"] is None
    assert report["metrics"]["generation_only_context_rate"] == 1.0
