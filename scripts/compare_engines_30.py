"""Compare the main LangGraph and private-memory Supervisor on 30 frozen cases.

This is an offline orchestration benchmark. Both engines receive the same frozen
POI/weather facts and the same deterministic worker behavior. No LLM, AMap, or
Chroma call is made. Token values are a transparent ``chars / 4`` estimate, so
the report is suitable for relative engineering comparison, not billing or SLA
claims.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.multi_agent_core.memory import InMemoryAgentMemoryStore
from app.multi_agent_core.runtime import TravelSupervisor
from app.planning import graph
from app.planning.helpers import open_time_violations
from app.planning.nodes import route_risk_gate_node
from app.planning.schemas import TravelPlanState
from tests.eval.graders.code_graders import grade_code
from tests.eval.harness import build_state_from_fixture, load_fixtures


MODEL_ROLES = {
    "intent", "query_rewrite", "candidate_react", "planner", "reviewer", "time_check"
}


class Meter:
    def __init__(self, delay_ms: float) -> None:
        self.delay_ms = delay_ms
        self.rows: list[dict[str, Any]] = []
        self.lock = Lock()

    def record(self, role: str, state: TravelPlanState, update: dict[str, Any]) -> None:
        input_payload = state.model_dump(mode="json")
        output_payload = update
        chars = len(json.dumps(input_payload, ensure_ascii=False, default=str))
        chars += len(json.dumps(output_payload, ensure_ascii=False, default=str))
        row = {
            "role": role,
            "estimated_tokens": max(1, math.ceil(chars / 4)),
            "input_chars": chars,
        }
        with self.lock:
            self.rows.append(row)
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)


def _wrap(role: str, meter: Meter, fn: Callable[[TravelPlanState], dict[str, Any]]):
    def wrapped(state: TravelPlanState) -> dict[str, Any]:
        update = fn(state)
        if not isinstance(update, dict):
            raise TypeError(f"{role} returned a non-dict update")
        meter.record(role, state, update)
        return update
    return wrapped


def _safe_poi(poi: dict[str, Any]) -> bool:
    opening = str(poi.get("open_time") or "")
    return any(token in opening for token in ("00:00-24:00", "08:", "09:", "10:"))


def _route_for(fx: dict[str, Any], state: TravelPlanState, *, faulty: bool) -> list[dict[str, Any]]:
    pois = list(state.pois)
    if not pois:
        return [{"day": day, "spots": []} for day in range(1, max(1, state.days) + 1)]
    selected: list[dict[str, Any]] = []
    for index in range(max(1, state.days)):
        forecast = (state.weather_forecast or [])[index] if index < len(state.weather_forecast or []) else {}
        pool = [p for p in pois if _safe_poi(p)] or pois
        # The deterministic repair policy treats a bad-weather day as an
        # indoor preference whenever such a candidate exists. Both engines use
        # this same policy; it prevents the benchmark from measuring a
        # difference in fake Planner behavior instead of orchestration.
        if forecast.get("is_bad"):
            indoor = [p for p in pool if p.get("indoor") is True]
            pool = indoor or pool
        selected.append(pool[index % len(pool)])

    days = []
    for index, poi in enumerate(selected, start=1):
        spot = {
            "name": poi.get("name"),
            "period": "morning",
            "start_time": "10:00",
            "end_time": "11:00",
        }
        days.append({"day": index, "spots": [spot]})

    if faulty:
        mode = int(fx.get("_fault_mode", 0))
        if mode == 0 and days:
            days[0]["spots"].append(dict(days[0]["spots"][0]))
        elif mode == 1 and days:
            days[0]["spots"][0]["name"] = "未核验景点"
        elif mode == 2 and days:
            days[0]["spots"][0]["start_time"] = "06:00"
        elif mode == 3 and days:
            outdoor = next((p for p in pois if p.get("indoor") is False), None)
            if outdoor:
                days[0]["spots"][0]["name"] = outdoor.get("name")
        elif mode == 4 and days:
            days[0]["spots"] = []
    return days


def _case_nodes(fx: dict[str, Any], meter: Meter) -> dict[str, Callable]:
    source_pois = list(fx.get("pois") or [])
    fault_case = int(fx["_fault_mode"]) >= 0

    def intent(state):
        return {
            "destination": fx.get("destination"),
            "travel_start_date": date.fromisoformat(fx["travel_start_date"]),
            "travel_end_date": date.fromisoformat(fx["travel_end_date"]),
            "days": int(fx.get("days", 1)),
            "attraction_preference": fx.get("attraction_preference"),
            "habit_preference": fx.get("habit_preference"),
            "missing_fields": [],
            "rain_indoor_priority": "雨" in str(fx.get("query", "")) and "室内" in str(fx.get("query", "")),
        }

    def candidate_react(state):
        return {"candidate_search_actions": [{"action": "search_history", "query": "fixture"}]}

    def candidate_search(state):
        return {
            "pois": source_pois,
            "candidate_search_round": state.candidate_search_round + 1,
            "candidate_api_calls": state.candidate_api_calls + 1,
            "candidate_search_trace": [*state.candidate_search_trace, {"action": "search_history", "result_count": len(source_pois)}],
        }

    def candidate_validator(state):
        return {
            "candidate_pool_status": "ready",
            "candidate_missing_coverage": [],
            "candidate_coverage": {"count": len(source_pois), "fixture": True},
        }

    def planner(state):
        # The first draft is faulty for five deterministic slices; the next
        # planner call repairs it. This makes rework measurable without an LLM.
        first_round = state.review_round == 0
        return {
            "route": _route_for(fx, state, faulty=fault_case and first_round),
            "review_round": state.review_round + 1,
        }

    def reviewer(state):
        bad = bool(route_risk_gate_node(state).get("route_risk_flags"))
        return {
            "approved": not bad or state.review_round > 1,
            "reviewer_issues": [] if not bad or state.review_round > 1 else ["deterministic injected draft fault"],
            "route_modify_opinion": None if not bad or state.review_round > 1 else "修复硬约束问题并重新规划",
        }

    def time_check(state):
        return {"approved": True, "time_check_done": True, "time_violations": [], "time_check_round": state.time_check_round + 1}

    def risk(state):
        return route_risk_gate_node(state)

    def finalize(state):
        return {"final_plan": {"approved": bool(state.approved), "days": state.route}}

    raw = {
        "intent": intent,
        "query_rewrite": lambda state: {"rewritten_query": fx.get("query", "")},
        "weather_search": lambda state: {"weather_forecast": list(fx.get("weather_forecast") or []), "weather_note": "fixture"},
        "candidate_react": candidate_react,
        "candidate_search": candidate_search,
        "attraction_search": candidate_search,
        "candidate_validator": candidate_validator,
        "main_meal_search": lambda state: {"main_meal_candidates": [], "main_meal_status": "partial"},
        "planner": planner,
        "route_distance_check": lambda state: {"route_distance_legs": []},
        "risk_gate": risk,
        "reviewer": reviewer,
        "time_check": time_check,
        "main_meal_output": lambda state: {"meal_slots": [], "main_meal_status": "partial"},
        "meal_enrichment": lambda state: {"meal_candidates": [], "meals": []},
        "spot_tips": lambda state: {"spot_tips": {}},
        "finalize": finalize,
    }
    return {name: _wrap(name, meter, fn) for name, fn in raw.items()}


def _run_main(fx: dict[str, Any], delay_ms: float) -> dict[str, Any]:
    meter = Meter(delay_ms)
    nodes = _case_nodes(fx, meter)
    patchers = [
        patch.object(graph, "make_intent_node", return_value=nodes["intent"]),
        patch.object(graph, "make_query_rewrite_node", return_value=nodes["query_rewrite"]),
        patch.object(graph, "weather_search_node", nodes["weather_search"]),
        patch.object(graph, "attraction_search_node", lambda state: {}),
        patch.object(graph, "make_candidate_react_node", return_value=nodes["candidate_react"]),
        patch.object(graph, "candidate_search_node", nodes["candidate_search"]),
        patch.object(graph, "candidate_validator_node", nodes["candidate_validator"]),
        patch.object(graph, "main_meal_candidate_search_node", nodes["main_meal_search"]),
        patch.object(graph, "make_planner_node", return_value=nodes["planner"]),
        patch.object(graph, "route_distance_check_node", nodes["route_distance_check"]),
        patch.object(graph, "route_risk_gate_node", nodes["risk_gate"]),
        patch.object(graph, "make_reviewer_node", return_value=nodes["reviewer"]),
        patch.object(graph, "make_time_check_node", return_value=nodes["time_check"]),
        patch.object(graph, "main_meal_output_node", nodes["main_meal_output"]),
        patch.object(graph, "make_meal_enrichment_node", return_value=nodes["meal_enrichment"]),
        patch.object(graph, "make_spot_tips_node", return_value=nodes["spot_tips"]),
        patch.object(graph, "make_finalize_node", return_value=nodes["finalize"]),
    ]
    for patcher in patchers:
        patcher.start()
    try:
        app = graph.build_graph()
        initial = build_state_from_fixture(fx).model_copy(update={"pois": [], "candidate_pool_status": "pending"})
        started = time.perf_counter()
        result = app.invoke(initial, config={"recursion_limit": 80})
        elapsed = (time.perf_counter() - started) * 1000
        state = TravelPlanState(**result)
    finally:
        for patcher in reversed(patchers):
            patcher.stop()
    return _summarize_run(state, fx, meter, elapsed)


def _run_supervisor(fx: dict[str, Any], delay_ms: float) -> dict[str, Any]:
    meter = Meter(delay_ms)
    nodes = _case_nodes(fx, meter)
    runtime_nodes = {name: nodes[name] for name in (
        "intent", "query_rewrite", "weather_search", "candidate_react", "candidate_search",
        "candidate_validator", "main_meal_search", "planner", "route_distance_check", "risk_gate",
        "reviewer", "time_check", "meal_enrichment", "spot_tips", "finalize",
    )}
    initial = build_state_from_fixture(fx).model_copy(update={"pois": [], "candidate_pool_status": "pending"})
    started = time.perf_counter()
    async def consume():
        return [event async for event in TravelSupervisor(
            runtime_nodes, InMemoryAgentMemoryStore(), f"compare-{fx['id']}"
        ).stream(initial)]
    events = asyncio.run(consume())
    elapsed = (time.perf_counter() - started) * 1000
    state = TravelPlanState(**events[-1]["checkpoint"])
    return _summarize_run(state, fx, meter, elapsed)


def _summarize_run(state: TravelPlanState, fx: dict[str, Any], meter: Meter, elapsed: float) -> dict[str, Any]:
    graded = grade_code(state, fx)
    planner_calls = sum(row["role"] == "planner" for row in meter.rows)
    return {
        "passed": bool(graded["objective_pass"]),
        "approved": bool(state.approved),
        "elapsed_ms": round(elapsed, 3),
        "estimated_tokens": sum(row["estimated_tokens"] for row in meter.rows if row["role"] in MODEL_ROLES),
        "planner_calls": planner_calls,
        "rework": max(0, planner_calls - 1),
        "candidate_api_calls": state.candidate_api_calls,
        "risk_flags": list(state.route_risk_flags),
        "grade": graded,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["elapsed_ms"] for row in rows]
    tokens = [row["estimated_tokens"] for row in rows]
    reworks = [row["rework"] for row in rows]
    return {
        "cases": len(rows),
        "accuracy": round(sum(row["passed"] for row in rows) / len(rows), 3),
        "approval_rate": round(sum(row["approved"] for row in rows) / len(rows), 3),
        "mean_elapsed_ms": round(statistics.mean(values), 3),
        "p50_elapsed_ms": round(statistics.median(values), 3),
        "p95_elapsed_ms": round(sorted(values)[max(0, math.ceil(len(values) * .95) - 1)], 3),
        "mean_estimated_tokens": round(statistics.mean(tokens), 3),
        "total_estimated_tokens": sum(tokens),
        "rework_rate": round(sum(value > 0 for value in reworks) / len(rows), 3),
        "mean_rework_rounds": round(statistics.mean(reworks), 3),
        "candidate_api_calls": sum(row["candidate_api_calls"] for row in rows),
    }


def evaluate(delay_ms: float = 0.2) -> dict[str, Any]:
    fixtures = load_fixtures()
    if len(fixtures) != 30:
        raise RuntimeError(f"expected exactly 30 fixtures, found {len(fixtures)}")
    for index, fixture in enumerate(fixtures):
        fixture["_fault_mode"] = index % 6 if index % 5 == 0 else -1
    main_rows = []
    supervisor_rows = []
    for fixture in fixtures:
        main_rows.append({"case_id": fixture["id"], **_run_main(fixture, delay_ms)})
        supervisor_rows.append({"case_id": fixture["id"], **_run_supervisor(fixture, delay_ms)})
    return {
        "schema_version": 1,
        "case_count": 30,
        "measurement": "offline deterministic orchestration comparison",
        "external_calls": False,
        "token_method": "estimated chars/4 over model-role input and output payloads; not provider billing",
        "delay_ms_per_node": delay_ms,
        "main_langgraph": {"summary": _aggregate(main_rows), "cases": main_rows},
        "supervisor_private_memory": {"summary": _aggregate(supervisor_rows), "cases": supervisor_rows},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay-ms", type=float, default=0.2)
    parser.add_argument("--json-out", type=Path, default=ROOT / "evaluation" / "engine_comparison_30.json")
    parser.add_argument("--markdown-out", type=Path, default=ROOT / "evaluation" / "engine_comparison_30.md")
    args = parser.parse_args()
    report = evaluate(args.delay_ms)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    main_summary = report["main_langgraph"]["summary"]
    supervisor_summary = report["supervisor_private_memory"]["summary"]
    def delta(key: str) -> str:
        before = main_summary[key]
        after = supervisor_summary[key]
        if not before:
            return "n/a"
        return f"{(after - before) / before:+.1%}"
    markdown = "\n".join([
        "# Main LangGraph vs Supervisor 30-case comparison",
        "",
        "> Offline deterministic comparison. No LLM, AMap, or Chroma calls were made.",
        "> Token values use chars/4 estimation and are not provider billing; latency is local fake-node time, not online SLA.",
        "",
        "| Metric | Main LangGraph | Supervisor + private memory | Relative change |",
        "|---|---:|---:|---:|",
        f"| Objective hard-constraint accuracy | {main_summary['accuracy']:.1%} | {supervisor_summary['accuracy']:.1%} | {delta('accuracy')} |",
        f"| Mean elapsed time | {main_summary['mean_elapsed_ms']:.2f} ms | {supervisor_summary['mean_elapsed_ms']:.2f} ms | {delta('mean_elapsed_ms')} |",
        f"| P50 elapsed time | {main_summary['p50_elapsed_ms']:.2f} ms | {supervisor_summary['p50_elapsed_ms']:.2f} ms | {delta('p50_elapsed_ms')} |",
        f"| P95 elapsed time | {main_summary['p95_elapsed_ms']:.2f} ms | {supervisor_summary['p95_elapsed_ms']:.2f} ms | {delta('p95_elapsed_ms')} |",
        f"| Mean estimated tokens | {main_summary['mean_estimated_tokens']:.1f} | {supervisor_summary['mean_estimated_tokens']:.1f} | {delta('mean_estimated_tokens')} |",
        f"| Rework rate | {main_summary['rework_rate']:.1%} | {supervisor_summary['rework_rate']:.1%} | {delta('rework_rate')} |",
        f"| Mean rework rounds | {main_summary['mean_rework_rounds']:.2f} | {supervisor_summary['mean_rework_rounds']:.2f} | {delta('mean_rework_rounds')} |",
        "",
        "## Interpretation",
        "",
        "The Supervisor path is stricter on risky drafts and therefore can return an unapproved draft where the main graph's deterministic fixture worker approves. This is a routing/acceptance-policy observation, not proof that one model is more accurate.",
        "The extra token estimate and tail latency represent private-memory projection, message-contract validation, and explicit candidate-loop stages. A real quality/cost conclusion requires repeated runs with the same real provider and exact usage metadata.",
        "",
    ])
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(markdown, encoding="utf-8")
    for name in ("main_langgraph", "supervisor_private_memory"):
        print(name, json.dumps(report[name]["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
