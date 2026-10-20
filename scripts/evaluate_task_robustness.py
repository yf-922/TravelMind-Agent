"""30 task cases + 30 resilience cases; offline probes and opt-in real planning."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.evaluation.rag_protocol import fingerprint
from app.evaluation.task_robustness import build_cases, load_assets, offline_probe, grade, fault_distance
from app.evaluation.route_interest import run_final_route
from app.planning import nodes
from app.planning.schemas import TravelPlanState
from scripts.compare_route_interest import exclusive_ledger, implementation_fingerprint


def summary(cases, rows, real):
    result = {}
    for suite in ("task", "robustness"):
        planned = [c for c in cases if c["suite"] == suite]
        actual = [r for r in rows if r["suite"] == suite]
        result[suite] = {"planned": len(planned), "recorded": len(actual),
                         "not_run": len(planned) - len(actual), "errors": sum(bool(r.get("error")) for r in actual)}
        if real:
            successful = sum(r.get("grade", {}).get("task_success", False) for r in actual)
            safe = sum(r.get("grade", {}).get("safe_explained_rejection", False) for r in actual)
            robust = sum(r.get("grade", {}).get("robustness_ok", False) for r in actual)
            result[suite].update(task_success=successful, safe_rejections=safe,
                                 unsafe_releases=sum(r.get("grade", {}).get("unsafe_release", False) for r in actual),
                                 task_success_rate=successful / len(planned) if planned else None,
                                 resilience_rate=robust / len(planned) if planned and suite == "robustness" else None)
        else:
            result[suite].update(node_execution_ok=sum(r.get("probe", {}).get("node_execution_ok", False) for r in actual),
                                 task_success_rate=None, resilience_rate=None)
    return result


def real_trial(case, factory, model, maps):
    counters = {}
    injected = fault_distance(maps, case.get("fault") if case["category"] == "map" else None, counters)
    state, diagnostics = run_final_route(case["state"], model, lambda *a: [], factory, maps,
                                         distance_override=injected)
    initial = grade(state, case, maps)
    if case["category"] not in ("weather", "modification"):
        return state, {"initial": initial, "map_calls": counters.get("calls", 0), **diagnostics}
    # A mid-session change is made AFTER initial planning, not baked into a
    # convenient first prompt. Reuse verified candidates, reset derived verdicts.
    update = {"approved": False, "final_plan": None, "reviewer_issues": [], "time_violations": [],
              "route_distance_legs": [], "time_check_done": False, "time_check_round": 0,
              "review_round": 0, "route_risk_flags": [], "review_skipped": False,
              "risk_gate_rechecked": False, "modification_notes": case["modification"],
              "route_modify_opinion": "【用户修改意见】" + case["modification"]}
    if case["category"] == "weather":
        update["weather_forecast"] = case["weather_after"]
        update["weather_note"] = "天气暂不可用，请出发前复核" if not case["weather_after"] else None
    state = state.model_copy(update=update)
    from app.planning.graph import build_modification_graph
    with ExitStack() as stack:
        for name, replacement in {"build_structured_llm": factory, "invoke_structured": lambda llm, messages, **kw: llm.invoke(messages),
                                  "amap_key": lambda: "replay-only", "plan_route_distance": injected,
                                  "plan_transport_leg": lambda *a, **kw: None, "recommend_chain_hotel": lambda *a, **kw: None,
                                  "lookup_live_ticket_prices": lambda *a, **kw: {}, "search_planning_knowledge": lambda *a: []}.items():
            stack.enter_context(patch.object(nodes, name, replacement))
        # Optional tips do not affect selected POIs or the criteria under test.
        stack.enter_context(patch("app.planning.graph.make_spot_tips_node", return_value=lambda state: {"spot_tips_status": "skipped"}))
        output = build_modification_graph(model).invoke(state, {"recursion_limit": 60})
    return TravelPlanState(**output), {"initial": initial, "map_calls": counters.get("calls", 0),
                                     "change": case["modification"], "scope": "production_modification_graph_fixed_checkpoint"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Real paid model; default is offline node probes only")
    parser.add_argument("--suite", choices=("all", "task", "robustness"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-calls", type=int, default=60)
    parser.add_argument("--out", type=Path, default=ROOT / "data/eval/task_robustness_offline.json")
    parser.add_argument("--export-cases", type=Path, default=ROOT / "data/eval/task_robustness_cases.json")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1: parser.error("limit must be positive")
    if args.max_calls < 1: parser.error("max-calls must be positive")
    load_local_env()
    assets = build_cases()
    args.export_cases.parent.mkdir(parents=True, exist_ok=True)
    args.export_cases.write_text(json.dumps(assets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases = [c for c in assets["tasks"] + assets["robustness"] if args.suite == "all" or c["suite"] == args.suite]
    if args.limit: cases = cases[:args.limit]
    _, maps = load_assets()
    model = os.getenv("GROK_MODEL", "grok-4.6")
    binding = fingerprint({"assets": cases, "maps": maps, "implementation": implementation_fingerprint(),
                           "evaluator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "rubric": hashlib.sha256((ROOT / "app/evaluation/task_robustness.py").read_bytes()).hexdigest(),
                           "mode": args.execute, "model": model})
    report = {"mode": "real_joint_planner_final_route" if args.execute else "offline_provider_and_rule_probes",
              "binding": binding, "model": model if args.execute else None, "label_status": "author_draft",
              "scope": "fixed_upstream_candidates_and_road_replay_not_live_search_or_online_SLA",
              "cost_status": "unpriced", "max_calls": args.max_calls, "cases": cases,
              "calls": [], "rows": [], "status": "running"}
    def save():
        report["summary"] = summary(cases, report["rows"], args.execute)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"planned={len(cases)}; real_model={args.execute}; call_cap={args.max_calls}; cost=unpriced", flush=True)
    with exclusive_ledger(args.out):
        if args.out.exists():
            previous = json.loads(args.out.read_text(encoding="utf-8"))
            if previous["binding"] != binding: raise ValueError("configuration changed; use a new output path")
            if any(c["status"] == "in_flight" for c in previous["calls"]): raise RuntimeError("unsettled call; no automatic paid retry")
            report = previous
        current = {"case": ""}
        def factory(schema, **kwargs):
            from app.llm.grok import build_chat_grok, use_grok_responses_api, parse_structured_response
            client = build_chat_grok(model=model, temperature=kwargs.get("temperature", 0))
            client.max_tokens = 4096
            client.max_retries = 0
            structured = client.with_structured_output(schema, method="json_schema" if use_grok_responses_api() else "function_calling", strict=True, include_raw=True)
            class Metered:
                def invoke(self, messages):
                    if len(report["calls"]) >= args.max_calls: raise RuntimeError("evaluation_call_cap_reached")
                    call = {"case": current["case"], "task": kwargs.get("task_type"), "status": "in_flight"}
                    report["calls"].append(call); save()
                    started = time.perf_counter()
                    try:
                        response = structured.invoke(messages)
                        usage = getattr(response.get("raw"), "usage_metadata", None)
                        call.update(status="completed", usage=usage)
                        if not usage: raise RuntimeError("actual_usage_missing_stop")
                        return parse_structured_response(response, schema)
                    except Exception as exc:
                        call.update(status="failed", error=type(exc).__name__)
                        raise
                    finally:
                        call["elapsed_ms"] = (time.perf_counter() - started) * 1000
                        save()
            return Metered()
        if args.execute:
            from scripts.compare_route_interest import read_gateway_usage
            report["gateway_usage_before"] = read_gateway_usage(os.environ["GROK_BASE_URL"].rstrip("/"), os.environ["GROK_API_KEY"])
        for case in cases:
            if any(r["id"] == case["id"] for r in report["rows"]): continue
            if args.execute and len(report["calls"]) >= args.max_calls:
                report["status"] = "stopped_call_cap"; break
            current["case"] = case["id"]
            row = {"id": case["id"], "suite": case["suite"], "category": case["category"], "fault": case.get("fault")}
            started = time.perf_counter()
            try:
                if args.execute:
                    # Disable hidden application retries; optional tips explicitly
                    # skipped so this experiment measures itinerary quality only.
                    with patch.object(nodes, "invoke_structured", side_effect=lambda llm, messages, **kw: llm.invoke(messages)), patch.object(nodes, "make_spot_tips_node", return_value=lambda state: {"spot_tips_status": "skipped"}):
                        state, diagnostics = real_trial(case, factory, model, maps)
                    row.update(grade=grade(state, case, maps), diagnostics=diagnostics,
                               final_plan=state.final_plan, final_state=state.model_dump(mode="json"))
                else: row["probe"] = offline_probe(case)
            except Exception as exc:
                # No provider URL, auth headers, or exception bodies in public reports.
                row["error"] = type(exc).__name__
            row["elapsed_ms"] = (time.perf_counter() - started) * 1000
            report["rows"].append(row); save()
            print(case["id"], row.get("grade", {}).get("task_success"), row.get("error", ""), flush=True)
            if args.execute and any(c.get("error") == "RuntimeError" for c in report["calls"]):
                report["status"] = "stopped_budget_or_usage_error"; break
        else: report["status"] = "complete_pending_human_review" if args.execute else "offline_complete"
        save()
    print(json.dumps(report["summary"], ensure_ascii=True), flush=True)


if __name__ == "__main__": main()
