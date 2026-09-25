import argparse

import pytest

from app.core.eval_safety import (
    estimate_online_ablation_calls,
    estimate_planner_reviewer_calls,
    require_call_budget,
    require_external_calls,
)
from tests.eval.report import aggregate_case
from tests.eval.run_eval import _failed_trial


def test_external_call_guard_fails_closed(capsys):
    parser = argparse.ArgumentParser(prog="online-eval")

    with pytest.raises(SystemExit) as exc_info:
        require_external_calls(parser, allowed=False, operation="online evaluation")

    assert exc_info.value.code == 2
    assert "--allow-external-calls" in capsys.readouterr().err


def test_external_call_guard_accepts_explicit_opt_in():
    parser = argparse.ArgumentParser(prog="online-eval")

    require_external_calls(parser, allowed=True, operation="online evaluation")


def test_planner_reviewer_budget_is_a_conservative_upper_bound():
    estimate = estimate_planner_reviewer_calls(
        [{"id": "a", "max_review_rounds": 3}, {"id": "b", "max_review_rounds": 1}],
        trials_per_case=2,
        use_judge=True,
    )

    assert estimate["cases"] == 2
    assert estimate["trials"] == 4
    assert estimate["logical_invocations_upper_bound"] == 32
    assert estimate["per_case"][0]["provider_attempts_upper_bound"] == 60
    assert estimate["per_case"][1]["provider_attempts_upper_bound"] == 36
    assert estimate["provider_attempts_upper_bound"] == 96
    assert estimate["llm_calls_upper_bound"] == 96


def test_call_budget_rejects_missing_or_insufficient_limit(capsys):
    parser = argparse.ArgumentParser(prog="online-eval")

    with pytest.raises(SystemExit):
        require_call_budget(parser, estimated_upper_bound=10, maximum=None)
    assert "--dry-run" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        require_call_budget(parser, estimated_upper_bound=10, maximum=9)
    assert "exceeds" in capsys.readouterr().err

    require_call_budget(parser, estimated_upper_bound=10, maximum=10)


def test_provider_failure_counts_as_failed_trial():
    trial = _failed_trial(TimeoutError("provider timeout"))
    aggregate = aggregate_case("timeout-case", "capability", [trial])
    assert aggregate["k"] == 1
    assert aggregate["failed_trials"] == 1
    assert aggregate["pass_rate"] == 0.0
    assert aggregate["pass_pow_k"] == 0


def test_online_ablation_budget_separates_three_modes():
    estimate = estimate_online_ablation_calls(
        [{"id": "a", "max_review_rounds": 3, "max_time_check_rounds": 3}],
        trials_per_case=1,
        modes=["planner_only", "planner_reviewer", "planner_reviewer_time_check"],
        use_judge=True,
    )

    assert estimate["by_mode"]["planner_only"]["logical_invocations_upper_bound"] == 2
    assert estimate["by_mode"]["planner_reviewer"]["logical_invocations_upper_bound"] == 10
    assert estimate["by_mode"]["planner_reviewer_time_check"]["logical_invocations_upper_bound"] == 15
    assert estimate["logical_invocations_upper_bound"] == 27
    assert estimate["provider_attempts_upper_bound"] == 81
