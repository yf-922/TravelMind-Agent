import json
from pathlib import Path

from scripts.evaluate_memory import evaluate_agent_memory, evaluate_semantic_memory


ROOT = Path(__file__).resolve().parents[1]


def test_memory_evaluation_has_no_cross_user_or_agent_leakage():
    data = json.loads((ROOT / "evaluation" / "memory_golden_set.json").read_text(encoding="utf-8"))

    semantic = evaluate_semantic_memory(data, top_k=2)
    agent = evaluate_agent_memory()

    assert semantic["hit_at_k"] >= 0.90
    assert semantic["cross_user_leakage_rate"] == 0
    assert semantic["metadata_filter_applied_rate"] == 1
    assert agent["isolation_pass_rate"] == 1
