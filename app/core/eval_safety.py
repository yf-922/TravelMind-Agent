"""Explicit opt-in guard for evaluation commands that spend external API quota."""

from __future__ import annotations

import argparse
from typing import Any


def require_external_calls(
    parser: argparse.ArgumentParser,
    *,
    allowed: bool,
    operation: str,
) -> None:
    if not allowed:
        parser.error(
            f"{operation} calls external APIs and may incur cost; "
            "re-run with --allow-external-calls after confirming the budget"
        )


def estimate_planner_reviewer_calls(
    cases: list[dict[str, Any]],
    *,
    trials_per_case: int,
    use_judge: bool,
    include_time_check: bool = False,
    attempts_per_invocation: int = 3,
) -> dict[str, Any]:
    """Return logical-invocation and provider-attempt ceilings."""
    per_case: list[dict[str, Any]] = []
    logical_total = 0
    provider_total = 0
    attempts_per_invocation = max(1, int(attempts_per_invocation))
    for case in cases:
        review_rounds = max(0, int(case.get("max_review_rounds", 3)))
        # The graph may run max_review_rounds + 1 planner/reviewer pairs.
        graph_calls = 2 * (review_rounds + 1)
        # After the reviewer loop production enters time_check.  A violating
        # check sends the route back to planner and then time_check again, so
        # max_time_check_rounds checks imply at most (2*tc - 1) extra calls.
        # Keep this opt-in for backwards compatibility with older callers that
        # evaluate the historical planner/reviewer-only subgraph.
        if include_time_check:
            time_rounds = max(1, int(case.get("max_time_check_rounds", 3)))
            graph_calls += 2 * time_rounds - 1
        judge_calls = 2 if use_judge else 0
        upper_per_trial = graph_calls + judge_calls
        case_logical = upper_per_trial * trials_per_case
        case_provider = case_logical * attempts_per_invocation
        logical_total += case_logical
        provider_total += case_provider
        per_case.append({
            "id": str(case.get("id") or "unknown"),
            "trials": trials_per_case,
            "max_review_rounds": review_rounds,
            "include_time_check": include_time_check,
            "logical_invocations_upper_bound": case_logical,
            "provider_attempts_upper_bound": case_provider,
        })
    return {
        "cases": len(cases),
        "trials": len(cases) * trials_per_case,
        "judge_enabled": use_judge,
        "logical_invocations_upper_bound": logical_total,
        "provider_attempts_upper_bound": provider_total,
        "llm_calls_upper_bound": provider_total,
        "attempts_per_invocation": attempts_per_invocation,
        "assumption": "worst-case logical graph invocations multiplied by invoke_structured retry attempts",
        "per_case": per_case,
    }


def estimate_online_ablation_calls(
    cases: list[dict[str, Any]],
    *,
    trials_per_case: int,
    modes: list[str] | tuple[str, ...],
    use_judge: bool,
    attempts_per_invocation: int = 3,
) -> dict[str, Any]:
    """Conservative request ceiling for comparable online ablation modes."""
    valid_modes = {
        "planner_only",
        "planner_reviewer",
        "planner_reviewer_time_check",
    }
    selected = list(dict.fromkeys(modes))
    unknown = [mode for mode in selected if mode not in valid_modes]
    if unknown:
        raise ValueError(f"unknown evaluation modes: {', '.join(unknown)}")
    attempts = max(1, int(attempts_per_invocation))
    mode_totals: dict[str, dict[str, int]] = {}
    logical_total = 0
    for mode in selected:
        mode_logical = 0
        for case in cases:
            review_rounds = max(0, int(case.get("max_review_rounds", 3)))
            time_rounds = max(1, int(case.get("max_time_check_rounds", 3)))
            if mode == "planner_only":
                graph_calls = 1
                judge_calls = 1 if use_judge else 0
            elif mode == "planner_reviewer":
                graph_calls = 2 * (review_rounds + 1)
                judge_calls = 2 if use_judge else 0
            else:
                graph_calls = 2 * (review_rounds + 1) + 2 * time_rounds - 1
                judge_calls = 2 if use_judge else 0
            mode_logical += (graph_calls + judge_calls) * trials_per_case
        mode_totals[mode] = {
            "logical_invocations_upper_bound": mode_logical,
            "provider_attempts_upper_bound": mode_logical * attempts,
        }
        logical_total += mode_logical
    return {
        "cases": len(cases),
        "trials_per_case": trials_per_case,
        "modes": selected,
        "total_runs": len(cases) * trials_per_case * len(selected),
        "judge_enabled": use_judge,
        "logical_invocations_upper_bound": logical_total,
        "provider_attempts_upper_bound": logical_total * attempts,
        "llm_calls_upper_bound": logical_total * attempts,
        "attempts_per_invocation": attempts,
        "by_mode": mode_totals,
        "assumption": (
            "worst-case graph and judge invocations multiplied by "
            "invoke_structured retry attempts"
        ),
    }


def require_call_budget(
    parser: argparse.ArgumentParser,
    *,
    estimated_upper_bound: int,
    maximum: int | None,
    flag: str = "--max-llm-calls",
    resource: str = "LLM call",
) -> None:
    if maximum is None:
        parser.error(
            f"set {flag} to an explicit approved budget; "
            "use --dry-run to inspect the upper bound without external calls"
        )
    if maximum < 1:
        parser.error(f"{flag} must be at least 1")
    if estimated_upper_bound > maximum:
        parser.error(
            f"estimated {resource} upper bound {estimated_upper_bound} exceeds "
            f"approved {flag} {maximum}"
        )
