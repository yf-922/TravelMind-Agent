"""Real-LLM fault-recovery ablation from identical corrupted planner drafts.

This is deliberately separate from natural-quality ablation. It measures
whether independent review agents can recover when an upstream draft contains
a known duplicate-POI or opening-hours fault.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.core.eval_safety import require_call_budget, require_external_calls
from app.core.llm_usage import snapshot as usage_snapshot
from app.core.llm_usage import usage_delta
from app.planning.nodes import (
    make_planner_node,
    make_reviewer_node,
    make_time_check_node,
    route_after_planner,
    route_after_review,
    route_after_time_check,
)
from app.planning.schemas import TravelPlanState
from tests.eval.graders.code_graders import grade_code
from tests.eval.harness import build_state_from_fixture, load_fixtures
from tests.eval.run_online_ablation import _configuration

Mode = Literal["single_no_audit", "time_check_only", "planner_reviewer", "planner_reviewer_time_check"]
MODES: tuple[Mode, ...] = (
    "single_no_audit", "planner_reviewer", "planner_reviewer_time_check",
)
ALL_MODES: tuple[Mode, ...] = ("single_no_audit", "time_check_only", "planner_reviewer", "planner_reviewer_time_check")


def _source_route(path: Path, mode: str = "planner_only") -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trials = payload.get("trials", {}).get(mode) or []
    if not trials:
        raise ValueError(f"{path}: no {mode} trial")
    return json.loads(json.dumps(trials[0]["_state"]["route"], ensure_ascii=False))


def inject_duplicate(route: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = json.loads(json.dumps(route, ensure_ascii=False))
    source = next((day["spots"][0] for day in result if day.get("spots")), None)
    target = next((day for day in reversed(result) if day.get("spots")), None)
    if source is None or target is None:
        raise ValueError("route cannot accept duplicate injection")
    target["spots"][-1] = dict(source)
    return result


def inject_opening_hours_fault(
    route: list[dict[str, Any]], pois: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = json.loads(json.dumps(route, ensure_ascii=False))
    opening = {
        str(item.get("name")): str(item.get("open_time") or "") for item in pois
    }
    for day in result:
        for spot in day.get("spots", []):
            if "09:00-17:00" in opening.get(str(spot.get("name")), ""):
                spot.update(period="morning", start_time="07:00", end_time="08:00")
                return result
    raise ValueError("fixture has no route POI with a clear 09:00-17:00 window")


def inject_opening_hours_only_fault(route: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Move the final day's only visit after closing without adding another risk."""
    result = json.loads(json.dumps(route, ensure_ascii=False))
    final_spots = result[-1].get("spots") or []
    if len(final_spots) != 1:
        raise ValueError("expected one final-day spot for isolated hours fault")
    final_spots[0].update(period="evening", start_time="18:00", end_time="19:00")
    return result


def _build_recovery_graph(mode: Mode, model_name: str | None = None):
    graph = StateGraph(TravelPlanState)
    graph.add_node("planner", make_planner_node(model_name))
    if mode == "time_check_only":
        graph.add_node("time_check", make_time_check_node(model_name))
        graph.add_node("reviewer", make_reviewer_node(model_name))
        graph.add_edge(START, "time_check")
        graph.add_edge("planner", "time_check")
        graph.add_conditional_edges(
            "time_check",
            route_after_time_check,
            {"planner": "planner", "reviewer": "reviewer", "meal_search": END, "spot_tips": END},
        )
        graph.add_conditional_edges(
            "reviewer", route_after_review,
            {"planner": "planner", "time_check": "time_check"},
        )
        return graph.compile()
    graph.add_node("reviewer", make_reviewer_node(model_name))
    graph.add_edge(START, "reviewer")
    if mode == "planner_reviewer":
        graph.add_conditional_edges(
            "reviewer", route_after_review, {"planner": "planner", "time_check": END},
        )
        graph.add_edge("planner", "reviewer")
        return graph.compile()

    graph.add_node("time_check", make_time_check_node(model_name))
    graph.add_conditional_edges(
        "reviewer", route_after_review,
        {"planner": "planner", "time_check": "time_check"},
    )
    graph.add_conditional_edges(
        "planner", route_after_planner,
        {"reviewer": "reviewer", "time_check": "time_check"},
    )
    graph.add_conditional_edges(
        "time_check",
        lambda state: "done" if not state.time_violations or state.time_check_round >= state.max_time_check_rounds else "planner",
        {"planner": "planner", "done": END},
    )
    return graph.compile()


def run_recovery(state: TravelPlanState, mode: Mode) -> TravelPlanState:
    if mode == "single_no_audit":
        return state
    graph = _build_recovery_graph(mode, state.model_name)
    result = graph.invoke(state, config={"recursion_limit": 24})
    return TravelPlanState(**result)


def build_scenarios(root: Path) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    fixture_by_id = {item["id"]: item for item in load_fixtures()}
    rain = fixture_by_id["nanjing-3d-singlerain-history"]
    tight = fixture_by_id["nanjing-3d-tighthours-negative"]
    rain_route = _source_route(root / "evaluation" / "real_ablation_rain.json")
    tight_route = _source_route(root / "evaluation" / "real_ablation_tight.json")
    return [
        ("duplicate_poi", rain, inject_duplicate(rain_route)),
        ("opening_hours", tight, inject_opening_hours_fault(tight_route, tight["pois"])),
        ("opening_hours_only", tight, inject_opening_hours_only_fault(tight_route)),
    ]


def execute(root: Path, modes: tuple[Mode, ...] = MODES, only: str | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    before = usage_snapshot()
    for scenario, fixture, corrupted_route in build_scenarios(root):
        if only and scenario != only:
            continue
        for mode in modes:
            state = build_state_from_fixture(fixture)
            state.route = json.loads(json.dumps(corrupted_route, ensure_ascii=False))
            state.review_round = 1
            state.approved = False
            state.review_required = mode != "time_check_only"
            state.planner_reviewer_dialogue = [
                "[第1轮] Planner：故障恢复评测的共同冻结草案。"
            ]
            mode_usage_before = usage_snapshot()
            started = time.perf_counter()
            final = run_recovery(state, mode)
            grade = grade_code(final, fixture)
            records.append({
                "scenario": scenario,
                "fixture_id": fixture["id"],
                "mode": mode,
                "objective_pass": grade["objective_pass"],
                "checks": grade["results"],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "llm_usage": usage_delta(mode_usage_before, usage_snapshot()),
                "review_round": final.review_round,
                "time_check_round": final.time_check_round,
                "approved": final.approved,
                "time_check_status": final.time_check_status,
                "dialogue": final.planner_reviewer_dialogue,
                "route": final.route,
            })
    usage = usage_delta(before, usage_snapshot())
    mode_summary = summarize_records(records)
    return {
        "measurement_type": "real-provider-injected-fault-recovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": _configuration(),
        "modes": mode_summary,
        "records": records,
        "llm_usage": usage,
        "boundary": (
            "Both modes start from identical real-Planner drafts with deterministic injected faults. "
            "This measures recovery robustness, not natural-request quality or failure prevalence."
        ),
    }


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    modes = []
    for mode in ALL_MODES:
        rows = [row for row in records if row["mode"] == mode]
        if not rows:
            continue
        modes.append({
            "mode": mode,
            "passed": sum(bool(row["objective_pass"]) for row in rows),
            "runs": len(rows),
            "pass_rate": round(sum(bool(row["objective_pass"]) for row in rows) / len(rows), 3),
            "elapsed_total_ms": round(sum(float(row["elapsed_ms"]) for row in rows), 3),
            "estimated_tokens_total": sum(int((row.get("llm_usage") or {}).get("total_tokens") or 0) for row in rows),
            "llm_attempts": sum(int((row.get("llm_usage") or {}).get("calls_total") or 0) for row in rows),
        })
    return modes


def regrade_saved_report(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Re-run only deterministic graders against saved model outputs."""
    fixture_by_id = {item["id"]: item for item in load_fixtures()}
    fallback_ids = {
        "duplicate_poi": "nanjing-3d-singlerain-history",
        "opening_hours": "nanjing-3d-tighthours-negative",
    }
    for record in report.get("records", []):
        fixture_id = record.get("fixture_id") or fallback_ids[str(record["scenario"])]
        fixture = fixture_by_id[str(fixture_id)]
        state = build_state_from_fixture(fixture)
        state.route = record["route"]
        state.review_round = int(record.get("review_round") or 0)
        state.time_check_round = int(record.get("time_check_round") or 0)
        state.approved = record["mode"] != "single_no_audit"
        state.time_check_done = record["mode"] == "planner_reviewer_time_check"
        grade = grade_code(state, fixture)
        record["fixture_id"] = fixture_id
        record["objective_pass"] = grade["objective_pass"]
        record["checks"] = grade["results"]
    report["modes"] = summarize_records(report.get("records", []))
    if not report.get("configuration"):
        source = root / "evaluation" / "real_ablation_rain.json"
        if source.exists():
            report["configuration"] = json.loads(
                source.read_text(encoding="utf-8")
            )["execution"]["configuration"]
    report["regraded_at"] = datetime.now(timezone.utc).isoformat()
    report["regrade_external_calls_made"] = False
    return report


def render(report: dict[str, Any]) -> str:
    configuration = report.get("configuration") or {}
    usage = report.get("llm_usage") or {}
    lines = [
        "# Real Multi-Agent Fault-Recovery Ablation",
        "",
        "> Real LLM Reviewer/Planner/Time Check calls; deterministic faults injected into identical saved real-Planner drafts.",
        f"> Provider: {configuration.get('provider', 'unknown')} / {configuration.get('model', 'unknown')}; "
        f"prompt fingerprint: `{configuration.get('prompt_code_fingerprint', 'unknown')}`.",
        "",
        "| Mode | Passed | Recovery rate | E2E total | Est. Tokens | LLM attempts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["modes"]:
        lines.append(
            f"| {row['mode']} | {row['passed']}/{row['runs']} | {row['pass_rate']:.1%} | "
            f"{row['elapsed_total_ms'] / 1000:.1f}s | "
            f"{row['estimated_tokens_total'] or 'n/a'} | {row['llm_attempts'] or 'n/a'} |"
        )
    lines.extend([
        "",
        f"- Recorded LLM attempts: {usage.get('calls_total', 0)} "
        f"({usage.get('failed_total', 0)} failed provider attempts).",
        f"- Estimated tokens across selected modes: {usage.get('total_tokens', 0)}; "
        f"exact usage ratio: {float(usage.get('usage_exact_ratio', 0)):.1%}.",
        f"- Provider latency across both multi-Agent modes: "
        f"{float(usage.get('latency_sum_ms', 0)) / 1000:.1f}s.",
        "- Regrading only re-runs deterministic checks on saved routes; it does not call the provider.",
        f"Boundary: {report['boundary']}",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-external-calls", action="store_true")
    parser.add_argument("--only", choices=("duplicate_poi", "opening_hours", "opening_hours_only"))
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=list(MODES))
    parser.add_argument("--max-llm-calls", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--regrade-only", action="store_true",
        help="re-run deterministic graders on the saved JSON without external calls",
    )
    parser.add_argument("--out", type=Path, default=Path("evaluation/real_fault_recovery_ablation.md"))
    parser.add_argument("--json-out", type=Path, default=Path("evaluation/real_fault_recovery_ablation.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.regrade_only:
        report = regrade_saved_report(
            root, json.loads(args.json_out.read_text(encoding="utf-8")),
        )
        args.out.write_text(render(report), encoding="utf-8")
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(render(report))
        return 0
    # Conservative bounds include retries inside each structured model call.
    selected_modes = tuple(args.modes)
    scenario_count = 1 if args.only else 3
    provider_attempt_ceiling = scenario_count * sum({
        "single_no_audit": 0, "time_check_only": 24,
        "planner_reviewer": 24, "planner_reviewer_time_check": 39,
    }[mode] for mode in selected_modes)
    print(json.dumps({"preflight": {"cases": scenario_count, "modes": selected_modes, "provider_attempts_upper_bound": provider_attempt_ceiling}}))
    if args.dry_run:
        return 0
    require_external_calls(parser, allowed=args.allow_external_calls, operation="fault-recovery ablation")
    require_call_budget(parser, estimated_upper_bound=provider_attempt_ceiling, maximum=args.max_llm_calls)
    report = execute(root, selected_modes, args.only)
    args.out.write_text(render(report), encoding="utf-8")
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
