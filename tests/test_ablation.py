from tests.eval.ablation import MODES, run_ablation


def test_ablation_shows_incremental_quality_gain_and_call_cost():
    report = run_ablation()
    metrics = report["metrics"]

    assert tuple(metrics) == MODES
    assert metrics["planner_only"]["overall_pass_rate"] < metrics["planner_reviewer"]["overall_pass_rate"]
    assert metrics["planner_reviewer"]["overall_pass_rate"] < metrics["planner_reviewer_time_check"]["overall_pass_rate"]
    assert metrics["planner_only"]["avg_calls"] < metrics["planner_reviewer"]["avg_calls"] < metrics["planner_reviewer_time_check"]["avg_calls"]


def test_ablation_transcript_exposes_failure_mode_per_case():
    report = run_ablation()
    rows = {row["case_id"]: row for row in report["records"]["planner_only"]}

    assert rows["unknown_and_duplicate"]["checks"]["closed_pool"] is False
    assert rows["unknown_and_duplicate"]["checks"]["no_duplicate"] is False
    assert rows["opening_time_conflict"]["checks"]["time_valid"] is False
    assert report["records"]["planner_reviewer_time_check"][1]["checks"]["time_valid"] is True
