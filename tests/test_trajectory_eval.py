from tests.eval.run_trajectory_eval import evaluate_scenarios, render_report


def test_offline_trajectory_scenarios_cover_replan_and_failures():
    results = evaluate_scenarios()

    assert [item["contract"] for item in results] == [
        "happy_path",
        "reviewer_replan",
        "transient_tool_retry",
        "persistent_tool_failure",
    ]
    assert all(item["passed"] for item in results)
    assert len(results[2]["tool_trace"]) == 2
    assert [item["status"] for item in results[2]["tool_trace"]] == ["failed", "succeeded"]
    assert results[3]["checks"][6]["passed"] is True

    report = render_report(results)
    assert "4/4 scenarios passed" in report
    assert "No LLM, AMap, or other external API was called" in report
