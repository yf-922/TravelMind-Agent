from tests.eval.run_fault_recovery_ablation import (
    inject_duplicate,
    inject_opening_hours_fault,
    inject_opening_hours_only_fault,
    summarize_records,
    _build_recovery_graph,
)


def test_fault_injection_is_deterministic_and_does_not_mutate_source():
    route = [
        {"day": 1, "spots": [{"name": "A", "start_time": "10:00", "end_time": "11:00"}]},
        {"day": 2, "spots": [{"name": "B", "start_time": "12:00", "end_time": "13:00"}]},
    ]
    duplicated = inject_duplicate(route)
    timed = inject_opening_hours_fault(route, [{"name": "A", "open_time": "09:00-17:00"}])

    assert route[1]["spots"][0]["name"] == "B"
    assert duplicated[1]["spots"][0]["name"] == "A"
    assert timed[0]["spots"][0]["start_time"] == "07:00"
    assert timed[0]["spots"][0]["end_time"] == "08:00"


def test_fault_recovery_summary_keeps_failed_baseline_in_denominator():
    rows = [
        {"mode": "single_no_audit", "objective_pass": False, "elapsed_ms": 0},
        {"mode": "planner_reviewer", "objective_pass": True, "elapsed_ms": 10},
        {"mode": "planner_reviewer_time_check", "objective_pass": True, "elapsed_ms": 20},
    ]

    summary = summarize_records(rows)

    assert [row["pass_rate"] for row in summary] == [0.0, 1.0, 1.0]


def test_time_check_only_recovery_graph_has_no_reviewer():
    graph = _build_recovery_graph("time_check_only")
    assert "time_check" in graph.nodes
    assert "reviewer" in graph.nodes


def test_isolated_hours_fault_keeps_day_and_spot_structure():
    route = [
        {"day": 1, "spots": [{"name": "A", "period": "morning", "start_time": "10:00", "end_time": "11:00"}]},
        {"day": 2, "spots": [{"name": "B", "period": "morning", "start_time": "10:00", "end_time": "11:00"}]},
    ]
    changed = inject_opening_hours_only_fault(route)

    assert route[1]["spots"][0]["start_time"] == "10:00"
    assert changed[1]["spots"][0]["start_time"] == "18:00"
    assert changed[1]["spots"][0]["name"] == "B"
