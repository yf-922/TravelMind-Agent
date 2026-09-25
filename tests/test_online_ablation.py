from app.planning.schemas import TravelPlanState
from tests.eval import harness
from tests.eval.run_online_ablation import aggregate_mode, render_report


def _patch_eval_nodes(monkeypatch):
    monkeypatch.setattr(harness, "make_planner_node", lambda *a, **k: lambda state: {
        "route": [{"day": 1, "spots": []}],
        "review_round": state.review_round + 1,
        "history": state.history + ["planner"],
    })
    monkeypatch.setattr(harness, "make_reviewer_node", lambda *a, **k: lambda state: {
        "approved": True,
        "reviewer_issues": [],
        "route_modify_opinion": None,
        "history": state.history + ["reviewer"],
    })
    monkeypatch.setattr(harness, "make_time_check_node", lambda *a, **k: lambda state: {
        "time_check_done": True,
        "time_violations": [],
        "time_check_round": state.time_check_round + 1,
        "history": state.history + ["time_check"],
    })


def test_evaluation_modes_execute_only_selected_agents(monkeypatch):
    _patch_eval_nodes(monkeypatch)
    initial = TravelPlanState(query="test", days=1)

    planner = harness.build_evaluation_graph("planner_only").invoke(initial)
    reviewed = harness.build_evaluation_graph("planner_reviewer").invoke(initial)
    full = harness.build_evaluation_graph("planner_reviewer_time_check").invoke(initial)

    assert planner["history"] == ["planner"]
    assert reviewed["history"] == ["planner", "reviewer"]
    assert reviewed["time_check_round"] == 0
    assert full["history"] == ["planner", "reviewer", "time_check"]
    assert full["time_check_round"] == 1


def test_online_ablation_report_exposes_quality_and_cost():
    trial = {
        "overall_pass": True,
        "rounds": 1,
        "elapsed_ms": 1500,
        "error": None,
        "node_calls": {"planner": 1, "reviewer": 1, "time_check": 1},
        "judge": {"scores": {
            "preference_fit": 4,
            "habit_fit": 5,
            "theme_coherence": 4,
            "route_reasonableness": 4,
            "weather_adaptation": 3,
        }},
        "reliability": {
            "applicable": True,
            "false_approval": False,
            "false_rejection": False,
        },
        "rebuttal": {"rebuttal_rate": 0.0, "ignore_rate": 0.0},
        "case_id": "case-a",
    }
    usage = {"total_tokens": 600, "latency_sum_ms": 1200}
    row = aggregate_mode("planner_reviewer_time_check", [trial], usage)
    report = render_report([row], {"llm_calls_upper_bound": 45})

    assert row["pass_rate"] == 1.0
    assert row["pass_pow_k_rate"] == 1.0
    assert row["tokens_per_run"] == 600
    assert row["node_calls_per_run"] == 3
    assert "Tokens/run" in report
    assert "Provider-attempt budget ceiling: 45" in report
