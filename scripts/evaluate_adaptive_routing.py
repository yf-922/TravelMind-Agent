"""Compare saved single, always-full, and adaptive audit outcomes per draft."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.planning.nodes import route_risk_gate_node  # noqa: E402
from tests.eval.graders.code_graders import grade_code  # noqa: E402
from tests.eval.harness import build_state_from_fixture, load_fixtures  # noqa: E402
from tests.eval.run_fault_recovery_ablation import build_scenarios  # noqa: E402
from scripts.evaluate_risk_gate import constructed_cases  # noqa: E402

NATURAL = {
    "nanjing-3d-sunny-history": "real_ablation_sunny.json",
    "nanjing-3d-singlerain-history": "real_ablation_rain.json",
    "nanjing-3d-tighthours-negative": "real_ablation_tight.json",
}
FAULT_SOURCE = {
    "duplicate_poi": "nanjing-3d-singlerain-history",
    "opening_hours": "nanjing-3d-tighthours-negative",
    "opening_hours_only": "nanjing-3d-tighthours-negative",
}


def _read(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / "evaluation" / name).read_text(encoding="utf-8"))


def _decision(fixture: dict[str, Any], route: list[dict[str, Any]]) -> str:
    state = build_state_from_fixture(fixture)
    state.route = route
    result = route_risk_gate_node(state)
    if result["review_required"]:
        return "planner_reviewer_time_check"
    if result["time_check_required"]:
        return "time_check_only"
    return "planner_only"


def evaluate(root: Path = ROOT) -> dict[str, Any]:
    fixtures = {fx["id"]: fx for fx in load_fixtures()}
    natural_reports = {case_id: _read(root, filename) for case_id, filename in NATURAL.items()}
    fault_report = _read(root, "real_fault_recovery_ablation.json")
    isolated_report = _read(root, "real_isolated_hours_recovery_fixed.json")
    fault_records = {(row["scenario"], row["mode"]): row for row in fault_report["records"]}
    isolated_records = {(row["scenario"], row["mode"]): row for row in isolated_report["records"]}
    injected_routes = {scenario: route for scenario, _, route in build_scenarios(root)}
    rows: list[dict[str, Any]] = []

    for case_id, report in natural_reports.items():
        trials = report["trials"]
        single = trials["planner_only"][0]
        full = trials["planner_reviewer_time_check"][0]
        routed = _decision(fixtures[case_id], single["_state"]["route"])
        if routed != "planner_only":
            raise ValueError(f"natural draft unexpectedly routes to {routed}: {case_id}")
        rows.append({
            "case_id": case_id, "source": "saved-natural-provider-output", "route": routed,
            "single_pass": bool(single["overall_pass"]), "full_pass": bool(full["overall_pass"]),
            "adaptive_pass": bool(single["overall_pass"]),
            "single_ms": single["elapsed_ms"], "full_ms": full["elapsed_ms"],
            "adaptive_ms": single["elapsed_ms"],
        })

    for scenario, source_id in FAULT_SOURCE.items():
        fixture = fixtures[source_id]
        state = build_state_from_fixture(fixture)
        state.route = injected_routes[scenario]
        single_pass = grade_code(state, fixture)["objective_pass"]
        records = isolated_records if scenario == "opening_hours_only" else fault_records
        full = records[(scenario, "planner_reviewer_time_check")]
        routed = _decision(fixture, state.route)
        recovery = full if routed == "planner_reviewer_time_check" else records[(scenario, "time_check_only")]
        if routed not in {"planner_reviewer_time_check", "time_check_only"}:
            raise ValueError(f"fault draft unexpectedly skips audit: {scenario}")
        planner_ms = natural_reports[source_id]["trials"]["planner_only"][0]["elapsed_ms"]
        rows.append({
            "case_id": f"injected:{scenario}", "source": "saved-real-draft-with-deterministic-fault",
            "route": routed,
            "single_pass": bool(single_pass),
            "full_pass": bool(full["objective_pass"]),
            "adaptive_pass": bool(recovery["objective_pass"]),
            "single_ms": planner_ms,
            "full_ms": planner_ms + full["elapsed_ms"],
            "adaptive_ms": planner_ms + recovery["elapsed_ms"],
        })

    policies = {}
    for policy in ("single", "full", "adaptive"):
        elapsed = sum(float(row[f"{policy}_ms"]) for row in rows)
        policies[policy] = {
            "passed": sum(bool(row[f"{policy}_pass"]) for row in rows),
            "total": len(rows),
            "elapsed_total_ms": round(elapsed, 3),
            "elapsed_mean_ms": round(elapsed / len(rows), 3),
        }
    routing_matrix = []
    for case in constructed_cases():
        fixture = fixtures[case["fixture_id"]]
        state = build_state_from_fixture(fixture)
        state.route = case["route"]
        for key, value in case["state_updates"].items():
            setattr(state, key, value)
        actual = route_risk_gate_node(state)
        if actual["review_required"]:
            actual_mode = "planner_reviewer_time_check"
        elif actual["time_check_required"]:
            actual_mode = "time_check_only"
        else:
            actual_mode = "planner_only"
        expected_mode = {
            "skip": "planner_only",
            "escalate": "planner_reviewer_time_check",
        }[case["expected_decision"]]
        if case["expected_flags"] == ["opening_time_conflict"]:
            expected_mode = "time_check_only"
        routing_matrix.append({
            "case_id": case["case_id"], "expected_mode": expected_mode,
            "actual_mode": actual_mode, "passed": actual_mode == expected_mode,
            "flags": actual["route_risk_flags"],
        })
    matrix_passed = sum(row["passed"] for row in routing_matrix)
    natural_single_tokens = sum(
        int(next(row for row in report["results"] if row["mode"] == "planner_only")["llm_usage"]["total_tokens"])
        for report in natural_reports.values()
    )
    natural_full_tokens = sum(
        int(next(row for row in report["results"] if row["mode"] == "planner_reviewer_time_check")["llm_usage"]["total_tokens"])
        for report in natural_reports.values()
    )
    return {
        "measurement_type": "offline-policy-replay-of-saved-real-provider-outcomes",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": rows,
        "routing_matrix": routing_matrix,
        "policies": policies,
        "metrics": {
            "adaptive_latency_reduction_vs_full": round(1 - policies["adaptive"]["elapsed_total_ms"] / policies["full"]["elapsed_total_ms"], 3),
            "natural_only_estimated_token_reduction": round(1 - natural_single_tokens / natural_full_tokens, 3),
            "natural_only_estimated_tokens": {"single": natural_single_tokens, "full": natural_full_tokens},
            "isolated_hours_estimated_tokens": {
                mode: (isolated_records[("opening_hours_only", mode)].get("llm_usage") or {}).get("total_tokens")
                for mode in ("time_check_only", "planner_reviewer_time_check")
            },
            "routing_matrix_pass_rate": round(matrix_passed / len(routing_matrix), 3),
            "routing_matrix_category_counts": {
                mode: sum(row["expected_mode"] == mode for row in routing_matrix)
                for mode in ("planner_only", "time_check_only", "planner_reviewer_time_check")
            },
        },
        "boundary": (
            "Six selected saved drafts, including three injected faults; one provider run per route. "
            "Timing adds measured Planner and recovery runs, not a new end-to-end adaptive execution. "
            "Fault-mode tokens are not separately attributable in the old report, so no all-case Token saving is claimed. "
            "Fault prevalence and statistical quality stability cannot be inferred."
        ),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Single vs Full vs Adaptive Audit Replay", "",
        "> Saved real-provider outcomes and deterministic fault injection; not a new online A/B.", "",
        "| Policy | Hard-constraint pass | Total latency | Mean latency |",
        "|---|---:|---:|---:|",
    ]
    for policy, row in report["policies"].items():
        lines.append(f"| {policy} | {row['passed']}/{row['total']} | {row['elapsed_total_ms']/1000:.1f}s | {row['elapsed_mean_ms']/1000:.1f}s |")
    lines += [
        "",
        f"- Adaptive latency reduction vs full: {report['metrics']['adaptive_latency_reduction_vs_full']:.1%} on this selected replay.",
        f"- Natural-case estimated Token reduction: {report['metrics']['natural_only_estimated_token_reduction']:.1%} (3 cases only).",
        "- Isolated opening-hours fault estimated Tokens: "
        + ", ".join(f"{mode}={tokens}" for mode, tokens in report["metrics"]["isolated_hours_estimated_tokens"].items()) + ".",
        "- Overall Token reduction: unavailable; old fault report does not attribute usage per mode.",
        f"- Offline routing matrix: {sum(row['passed'] for row in report['routing_matrix'])}/{len(report['routing_matrix'])} mode decisions correct; no LLM calls.",
        "",
        "| Draft | Routed mode | Single | Full | Adaptive |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['case_id']} | {row['route']} | "
            f"{'pass' if row['single_pass'] else 'fail'} | {'pass' if row['full_pass'] else 'fail'} | "
            f"{'pass' if row['adaptive_pass'] else 'fail'} |"
        )
    lines += ["", f"Boundary: {report['boundary']}", ""]
    return "\n".join(lines)


def main() -> int:
    report = evaluate()
    output = ROOT / "evaluation"
    (output / "adaptive_routing_replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "adaptive_routing_replay.md").write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0 if report["policies"]["adaptive"]["passed"] == len(report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
