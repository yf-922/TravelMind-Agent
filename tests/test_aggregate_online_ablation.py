import json

import pytest

from scripts.aggregate_online_ablation import aggregate, render_markdown


def _report(tmp_path, name, fingerprint="same"):
    payload = {
        "execution": {
            "configuration": {
                "provider": "grok", "model": "model", "prompt_code_fingerprint": fingerprint,
            },
            "cases": [name],
        },
        "results": [
            {"mode": "planner_only", "llm_usage": {"total_tokens": 100, "latency_sum_ms": 10, "calls_total": 2}},
            {"mode": "planner_reviewer_time_check", "llm_usage": {"total_tokens": 200, "latency_sum_ms": 20, "calls_total": 4}},
        ],
        "trials": {
            "planner_only": [{"overall_pass": True, "judge": {"avg": 4.5}, "elapsed_ms": 100, "node_calls": {"planner": 1}, "error": None}],
            "planner_reviewer_time_check": [{"overall_pass": True, "judge": {"avg": 4.2}, "elapsed_ms": 200, "node_calls": {"planner": 1, "reviewer": 1, "time_check": 1}, "error": None}],
        },
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregate_preserves_cost_and_reports_no_unearned_gain(tmp_path):
    report = aggregate([_report(tmp_path, "a"), _report(tmp_path, "b")])

    assert report["conclusion"] == "no_observed_multi_agent_quality_gain"
    assert report["modes"][0]["estimated_tokens_mean"] == 100
    assert report["modes"][1]["estimated_tokens_mean"] == 200
    assert "One run per case" in render_markdown(report)


def test_aggregate_rejects_mismatched_prompt_fingerprints(tmp_path):
    with pytest.raises(ValueError, match="do not share"):
        aggregate([_report(tmp_path, "a", "one"), _report(tmp_path, "b", "two")])
