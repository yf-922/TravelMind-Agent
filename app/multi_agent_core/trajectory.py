"""Deterministic contracts for grading multi-Agent and tool trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrajectoryContract:
    name: str
    expected_dispatches: tuple[tuple[str, str, int], ...]
    expected_destination: str
    required_tools: frozenset[str] = frozenset({"poi_search"})
    allowed_tools: dict[str, frozenset[str]] = field(default_factory=lambda: {
        "intent_agent": frozenset(),
        "poi_research_agent": frozenset({"poi_search"}),
        "planner_agent": frozenset(),
        "reviewer_agent": frozenset(),
    })
    max_tool_attempts: int = 1
    max_retry_attempt: int = 0
    expect_failure: bool = False


def _check(check_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"id": check_id, "passed": passed, "detail": detail}


def grade_trajectory(result: dict[str, Any], contract: TrajectoryContract) -> dict[str, Any]:
    """Score routing, permissions, parameters, retries and grounded handoffs."""
    dispatches = [
        (str(item.get("task_type")), str(item.get("to")), int(item.get("attempt") or 0))
        for item in result.get("dispatch_log", [])
        if item.get("from") == "supervisor"
    ]
    tools = list(result.get("tool_trace") or [])
    checks: list[dict[str, Any]] = []
    checks.append(_check(
        "T1_dispatch_sequence",
        tuple(dispatches) == contract.expected_dispatches,
        f"observed={dispatches}",
    ))

    unauthorized = [
        item for item in tools
        if item.get("tool") not in contract.allowed_tools.get(str(item.get("agent")), frozenset())
        or item.get("status") == "blocked"
    ]
    checks.append(_check(
        "T2_tool_permission",
        not unauthorized,
        f"unauthorized_calls={len(unauthorized)}",
    ))

    observed_tools = {str(item.get("tool")) for item in tools}
    missing_tools = sorted(contract.required_tools - observed_tools)
    checks.append(_check(
        "T3_required_tools",
        not missing_tools,
        f"missing={missing_tools}",
    ))

    bad_parameters: list[str] = []
    for item in tools:
        if item.get("tool") != "poi_search":
            continue
        params = item.get("parameters") or {}
        query = params.get("query") or {}
        if params.get("city") != contract.expected_destination:
            bad_parameters.append("destination")
        if not isinstance(query, dict) or not query.get("present"):
            bad_parameters.append("query")
        if isinstance(params.get("query"), str):
            bad_parameters.append("raw_query_exposed")
    checks.append(_check(
        "T4_sanitized_parameters",
        not bad_parameters,
        f"violations={bad_parameters}",
    ))

    successful_fingerprints = [
        (item.get("agent"), item.get("tool"), repr(item.get("parameters")))
        for item in tools if item.get("status") == "succeeded"
    ]
    duplicate_successes = len(successful_fingerprints) - len(set(successful_fingerprints))
    checks.append(_check(
        "T5_no_redundant_success",
        duplicate_successes == 0,
        f"duplicate_successes={duplicate_successes}",
    ))

    max_attempt = max((attempt for _, _, attempt in dispatches), default=0)
    retry_ok = max_attempt <= contract.max_retry_attempt and len(tools) <= contract.max_tool_attempts
    checks.append(_check(
        "T6_bounded_retry",
        retry_ok,
        f"max_dispatch_attempt={max_attempt}, tool_attempts={len(tools)}",
    ))

    failed = result.get("status") == "failed"
    terminal_ok = failed == contract.expect_failure
    if failed:
        terminal_ok = terminal_ok and bool(result.get("failed_agent")) and bool(result.get("error_code"))
    checks.append(_check(
        "T7_structured_terminal_state",
        terminal_ok,
        f"status={result.get('status', 'succeeded')}, error_code={result.get('error_code')}",
    ))

    if failed:
        grounded = True
        grounded_detail = "not_applicable_to_failed_run"
    else:
        candidate_names = {item.get("name") for item in result.get("candidates", [])}
        itinerary_names = [item.get("name") for item in result.get("itinerary", [])]
        grounded = bool(itinerary_names) and all(name in candidate_names for name in itinerary_names)
        grounded_detail = f"itinerary_items={len(itinerary_names)}"
    checks.append(_check("T8_grounded_handoff", grounded, grounded_detail))

    return {
        "contract": contract.name,
        "passed": all(item["passed"] for item in checks),
        "passed_checks": sum(bool(item["passed"]) for item in checks),
        "total_checks": len(checks),
        "checks": checks,
        "dispatches": dispatches,
        "tool_trace": tools,
    }
