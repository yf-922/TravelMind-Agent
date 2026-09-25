from scripts.evaluate_adaptive_routing import evaluate


def test_adaptive_policy_replay_preserves_saved_objective_results():
    report = evaluate()

    assert len(report["cases"]) == 6
    assert [row["route"] for row in report["cases"][-3:]] == [
        "planner_reviewer_time_check", "planner_reviewer_time_check", "time_check_only",
    ]
    assert report["policies"]["single"]["passed"] == 3
    assert report["policies"]["full"]["passed"] == 6
    assert report["policies"]["adaptive"]["passed"] == 6
    assert report["policies"]["adaptive"]["elapsed_total_ms"] < report["policies"]["full"]["elapsed_total_ms"]
    assert len(report["routing_matrix"]) == 32
    assert report["metrics"]["routing_matrix_pass_rate"] == 1.0
