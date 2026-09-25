"""Run free, deterministic engineering benchmarks without external API calls.

The road-distance benchmark executes the production concurrent node with a
delayed fake route provider and compares it with an equivalent serial baseline.
The replanning benchmark is a deterministic stage-cost model because this
repository has no historical revision that can be executed as an old version.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Any, Callable
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.planning import nodes  # noqa: E402
from app.planning import graph as graph_module  # noqa: E402
from app.planning.schemas import TravelPlanState  # noqa: E402


FULL_REPLAY_PHASES: tuple[tuple[str, ...], ...] = (
    ("intent",),
    ("query_rewrite", "weather_search"),
    ("attraction_search",),
    ("planner",),
    ("route_distance_check",),
    ("reviewer",),
    ("time_check",),
    ("meal_enrichment", "spot_tips"),
    ("finalize",),
)
LOCAL_REPLAN_PHASES: tuple[tuple[str, ...], ...] = FULL_REPLAY_PHASES[3:]

# Fixed synthetic work per stage. These are controlled workload units, not
# production measurements and not model-provider usage data.
STAGE_WORK_UNITS = {
    "intent": 3,
    "query_rewrite": 4,
    "weather_search": 5,
    "attraction_search": 6,
    "planner": 7,
    "route_distance_check": 3,
    "reviewer": 5,
    "time_check": 4,
    "meal_enrichment": 9,
    "spot_tips": 3,
    "finalize": 2,
}

# Explicit assumptions used only for estimating which LLM input budget is
# avoided when checkpoint data lets us skip intent extraction and rewriting.
SIMULATED_LLM_INPUT_TOKENS = {
    "intent": 900,
    "query_rewrite": 600,
    "planner": 2500,
    "reviewer": 1500,
    "time_check": 1000,
    "meal_recommend": 1200,
    "spot_tips": 1400,
    "finalize": 800,
}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(mean(values), 3) if values else 0.0,
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3) if values else 0.0,
    }


def _sample_state(leg_count: int) -> TravelPlanState:
    pois = [
        {"name": f"POI-{index}", "location": {"lat": 30.0 + index / 100, "lng": 120.0}}
        for index in range(leg_count + 1)
    ]
    route = [{
        "day": 1,
        "spots": [{"name": poi["name"]} for poi in pois],
    }]
    return TravelPlanState(query="offline benchmark", pois=pois, route=route)


def _delayed_route_provider(delay_seconds: float) -> Callable[..., dict[str, Any]]:
    def provider(origin, destination, api_key, mode="drive"):
        time.sleep(delay_seconds)
        return {
            "mode": mode,
            "distance_km": 1.2,
            "duration_min": 8,
            "source": "deterministic-fake",
        }

    return provider


def _serial_distance_run(state: TravelPlanState, provider: Callable[..., dict[str, Any]]) -> int:
    locations = nodes.spot_location_map(state.pois)
    completed = 0
    for day in state.route:
        for previous, current in zip(day.get("spots") or [], (day.get("spots") or [])[1:]):
            origin = locations.get(previous.get("name"))
            destination = locations.get(current.get("name"))
            if origin and destination and provider(origin, destination, "offline", mode="drive"):
                completed += 1
    return completed


def benchmark_road_distance(runs: int, leg_count: int, delay_ms: float) -> dict[str, Any]:
    state = _sample_state(leg_count)
    provider = _delayed_route_provider(delay_ms / 1000)
    serial_ms: list[float] = []
    concurrent_ms: list[float] = []

    for _ in range(runs):
        started = time.perf_counter()
        serial_count = _serial_distance_run(state, provider)
        serial_ms.append((time.perf_counter() - started) * 1000)

        with patch.object(nodes, "amap_key", return_value="offline"), patch.object(
            nodes, "plan_route_distance", side_effect=provider
        ):
            started = time.perf_counter()
            update = nodes.route_distance_check_node(state)
            concurrent_ms.append((time.perf_counter() - started) * 1000)

        if serial_count != leg_count or len(update["route_distance_legs"]) != leg_count:
            raise RuntimeError("road-distance benchmark did not complete every route leg")

    serial = summarize_ms(serial_ms)
    concurrent = summarize_ms(concurrent_ms)
    speedup = serial["p50"] / concurrent["p50"] if concurrent["p50"] else 0.0
    return {
        "measurement_type": "production-node-with-delayed-fake-provider",
        "runs": runs,
        "route_legs": leg_count,
        "fake_provider_delay_ms_per_leg": delay_ms,
        "serial_baseline_ms": serial,
        "current_concurrent_ms": concurrent,
        "p50_speedup": round(speedup, 3),
        "pass": speedup >= 2.0,
        "pass_criterion": "p50 speedup >= 2.0x with identical completed route legs",
    }


def benchmark_post_intent_fanout(runs: int, delay_ms: float) -> dict[str, Any]:
    """Measure the production LangGraph fan-out with deterministic node delays."""

    serialize = [False]
    serial_lock = Lock()

    def delayed(update: dict[str, Any]):
        def node(state):
            if serialize[0]:
                with serial_lock:
                    time.sleep(delay_ms / 1000)
            else:
                time.sleep(delay_ms / 1000)
            return update
        return node

    patches = (
        patch.object(graph_module, "make_intent_node", return_value=lambda state: {
            "destination": "南京",
            "travel_start_date": date(2026, 9, 20),
            "travel_end_date": date(2026, 9, 20),
            "days": 1,
            "missing_fields": [],
        }),
        patch.object(graph_module, "make_query_rewrite_node", return_value=delayed({
            "rewritten_query": "南京一日游",
        })),
        patch.object(graph_module, "weather_search_node", delayed({
            "weather_forecast": [], "weather_note": "offline",
        })),
        patch.object(graph_module, "attraction_search_node", lambda state: {"pois": []}),
        patch.object(graph_module, "make_planner_node", return_value=lambda state: {
            "route": [{"day": 1, "spots": []}], "review_round": 1,
        }),
        patch.object(graph_module, "route_distance_check_node", lambda state: {}),
        patch.object(graph_module, "make_reviewer_node", return_value=lambda state: {
            "approved": True, "reviewer_issues": [], "route_modify_opinion": None,
        }),
        patch.object(graph_module, "make_time_check_node", return_value=lambda state: {
            "time_check_done": True, "time_violations": [], "time_check_round": 1,
        }),
        patch.object(graph_module, "make_meal_enrichment_node", return_value=lambda state: {
            "meal_candidates": [], "meals": [],
        }),
        patch.object(graph_module, "make_spot_tips_node", return_value=lambda state: {"spot_tips": {}}),
        patch.object(graph_module, "make_finalize_node", return_value=lambda state: {"final_plan": {}}),
    )

    for active_patch in patches:
        active_patch.start()
    try:
        app = graph_module.build_graph()
        serial_ms: list[float] = []
        concurrent_ms: list[float] = []
        for _ in range(runs):
            serialize[0] = True
            started = time.perf_counter()
            serial_result = app.invoke(
                TravelPlanState(query="南京一日游"), config={"recursion_limit": 30}
            )
            serial_ms.append((time.perf_counter() - started) * 1000)

            serialize[0] = False
            started = time.perf_counter()
            result = app.invoke(TravelPlanState(query="南京一日游"), config={"recursion_limit": 30})
            concurrent_ms.append((time.perf_counter() - started) * 1000)
            if serial_result.get("final_plan") != {} or result.get("final_plan") != {}:
                raise RuntimeError("fan-out benchmark graph did not finish")
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()

    serial = summarize_ms(serial_ms)
    concurrent = summarize_ms(concurrent_ms)
    speedup = serial["p50"] / concurrent["p50"] if concurrent["p50"] else 0.0
    return {
        "measurement_type": "production-langgraph-with-delayed-fake-nodes",
        "runs": runs,
        "parallel_nodes": ["query_rewrite", "weather_search"],
        "fake_delay_ms_per_node": delay_ms,
        "serial_baseline_ms": serial,
        "current_fanout_ms": concurrent,
        "p50_speedup": round(speedup, 3),
        "pass": speedup >= 1.5,
        "pass_criterion": "p50 speedup >= 1.5x through the compiled production graph",
    }


def benchmark_enrichment_fanout(runs: int, delay_ms: float) -> dict[str, Any]:
    """Measure full meal enrichment and spot-tip overlap in the compiled graph."""

    serialize = [False]
    serial_lock = Lock()

    def delayed(update: dict[str, Any]):
        def node(state):
            if serialize[0]:
                with serial_lock:
                    time.sleep(delay_ms / 1000)
            else:
                time.sleep(delay_ms / 1000)
            return update
        return node

    patches = (
        patch.object(graph_module, "make_intent_node", return_value=lambda state: {
            "destination": "南京", "travel_start_date": date(2026, 9, 20),
            "travel_end_date": date(2026, 9, 20), "days": 1, "missing_fields": [],
        }),
        patch.object(graph_module, "make_query_rewrite_node", return_value=lambda state: {
            "rewritten_query": state.query,
        }),
        patch.object(graph_module, "weather_search_node", lambda state: {
            "weather_forecast": [], "weather_note": "offline",
        }),
        patch.object(graph_module, "attraction_search_node", lambda state: {"pois": []}),
        patch.object(graph_module, "make_planner_node", return_value=lambda state: {
            "route": [{"day": 1, "spots": []}], "review_round": 1,
        }),
        patch.object(graph_module, "route_distance_check_node", lambda state: {}),
        patch.object(graph_module, "make_reviewer_node", return_value=lambda state: {
            "approved": True, "reviewer_issues": [], "route_modify_opinion": None,
        }),
        patch.object(graph_module, "make_time_check_node", return_value=lambda state: {
            "time_check_done": True, "time_violations": [], "time_check_round": 1,
        }),
        patch.object(graph_module, "make_meal_enrichment_node", return_value=delayed({
            "meal_candidates": [], "meals": [],
        })),
        patch.object(graph_module, "make_spot_tips_node", return_value=delayed({"spot_tips": {}})),
        patch.object(graph_module, "make_finalize_node", return_value=lambda state: {"final_plan": {}}),
    )
    for active_patch in patches:
        active_patch.start()
    try:
        app = graph_module.build_graph()
        serial_ms: list[float] = []
        concurrent_ms: list[float] = []
        for _ in range(runs):
            serialize[0] = True
            started = time.perf_counter()
            serial_result = app.invoke(
                TravelPlanState(query="南京一日游"), config={"recursion_limit": 30}
            )
            serial_ms.append((time.perf_counter() - started) * 1000)

            serialize[0] = False
            started = time.perf_counter()
            result = app.invoke(TravelPlanState(query="南京一日游"), config={"recursion_limit": 30})
            concurrent_ms.append((time.perf_counter() - started) * 1000)
            if serial_result.get("final_plan") != {} or result.get("final_plan") != {}:
                raise RuntimeError("enrichment benchmark graph did not finish")
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()

    serial = summarize_ms(serial_ms)
    concurrent = summarize_ms(concurrent_ms)
    speedup = serial["p50"] / concurrent["p50"] if concurrent["p50"] else 0.0
    return {
        "measurement_type": "production-langgraph-with-delayed-fake-nodes",
        "runs": runs,
        "parallel_nodes": ["meal_enrichment", "spot_tips"],
        "fake_delay_ms_per_node": delay_ms,
        "serial_baseline_ms": serial,
        "current_fanout_ms": concurrent,
        "p50_speedup": round(speedup, 3),
        "pass": speedup >= 1.5,
        "pass_criterion": "p50 speedup >= 1.5x through the compiled production graph",
    }


def _run_stage_model(phases: tuple[tuple[str, ...], ...], unit_delay_ms: float) -> None:
    def run_stage(stage: str) -> None:
        time.sleep(STAGE_WORK_UNITS[stage] * unit_delay_ms / 1000)

    for phase in phases:
        if len(phase) == 1:
            run_stage(phase[0])
        else:
            with ThreadPoolExecutor(max_workers=len(phase)) as pool:
                list(pool.map(run_stage, phase))


def _phase_nodes(phases: tuple[tuple[str, ...], ...]) -> list[str]:
    return [node for phase in phases for node in phase]


def benchmark_checkpoint_replan(runs: int, unit_delay_ms: float) -> dict[str, Any]:
    full_ms: list[float] = []
    local_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        _run_stage_model(FULL_REPLAY_PHASES, unit_delay_ms)
        full_ms.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        _run_stage_model(LOCAL_REPLAN_PHASES, unit_delay_ms)
        local_ms.append((time.perf_counter() - started) * 1000)

    full = summarize_ms(full_ms)
    local = summarize_ms(local_ms)
    avoided_nodes = [node for node in _phase_nodes(FULL_REPLAY_PHASES) if node not in _phase_nodes(LOCAL_REPLAN_PHASES)]
    full_token_budget = sum(SIMULATED_LLM_INPUT_TOKENS.values())
    avoided_token_budget = sum(SIMULATED_LLM_INPUT_TOKENS.get(node, 0) for node in avoided_nodes)
    latency_saved = 1 - local["p50"] / full["p50"] if full["p50"] else 0.0
    return {
        "measurement_type": "deterministic-stage-cost-simulation",
        "baseline_label": "simulated full replay; not historical production data",
        "runs": runs,
        "unit_delay_ms": unit_delay_ms,
        "full_replay_ms": full,
        "checkpoint_local_replan_ms": local,
        "p50_latency_reduction_ratio": round(latency_saved, 3),
        "avoided_nodes": avoided_nodes,
        "simulated_llm_input_tokens": {
            "full_replay": full_token_budget,
            "checkpoint_local_replan": full_token_budget - avoided_token_budget,
            "avoided": avoided_token_budget,
            "reduction_ratio": round(avoided_token_budget / full_token_budget, 3),
            "disclaimer": "fixed budget assumptions, not provider-reported token usage",
        },
        "pass": latency_saved >= 0.20,
        "pass_criterion": "modeled p50 latency reduction >= 20%",
    }


def render_markdown(report: dict[str, Any]) -> str:
    fanout = report["post_intent_fanout"]
    enrichment = report["enrichment_fanout"]
    road = report["road_distance"]
    replan = report["checkpoint_replan"]
    tokens = replan["simulated_llm_input_tokens"]
    return "\n".join([
        "# Offline Engineering Benchmark",
        "",
        "> This report uses no real LLM or map API calls. It must not be presented as online quality data.",
        "",
        "## Post-intent LangGraph fan-out",
        "",
        f"- Workload: {', '.join(fanout['parallel_nodes'])}, {fanout['runs']} repeated runs, "
        f"{fanout['fake_delay_ms_per_node']} ms deterministic delay per node",
        f"- Serial baseline P50/P95: {fanout['serial_baseline_ms']['p50']} / {fanout['serial_baseline_ms']['p95']} ms",
        f"- Current graph P50/P95: {fanout['current_fanout_ms']['p50']} / {fanout['current_fanout_ms']['p95']} ms",
        f"- P50 speedup: {fanout['p50_speedup']}x; threshold result: {'PASS' if fanout['pass'] else 'FAIL'}",
        "",
        "## Meal and spot-tip fan-out",
        "",
        f"- Workload: {', '.join(enrichment['parallel_nodes'])}, {enrichment['runs']} repeated runs, "
        f"{enrichment['fake_delay_ms_per_node']} ms deterministic delay per branch",
        f"- Serial baseline P50/P95: {enrichment['serial_baseline_ms']['p50']} / {enrichment['serial_baseline_ms']['p95']} ms",
        f"- Current graph P50/P95: {enrichment['current_fanout_ms']['p50']} / {enrichment['current_fanout_ms']['p95']} ms",
        f"- P50 speedup: {enrichment['p50_speedup']}x; threshold result: {'PASS' if enrichment['pass'] else 'FAIL'}",
        "",
        "## Road-distance concurrency",
        "",
        f"- Workload: {road['route_legs']} route legs, {road['runs']} repeated runs, "
        f"{road['fake_provider_delay_ms_per_leg']} ms deterministic provider delay per leg",
        f"- Serial baseline P50/P95: {road['serial_baseline_ms']['p50']} / {road['serial_baseline_ms']['p95']} ms",
        f"- Current concurrent node P50/P95: {road['current_concurrent_ms']['p50']} / {road['current_concurrent_ms']['p95']} ms",
        f"- P50 speedup: {road['p50_speedup']}x; threshold result: {'PASS' if road['pass'] else 'FAIL'}",
        "",
        "## Checkpoint local replanning",
        "",
        f"- Comparison boundary: {replan['baseline_label']}",
        f"- Full replay P50/P95: {replan['full_replay_ms']['p50']} / {replan['full_replay_ms']['p95']} ms",
        f"- Local replan P50/P95: {replan['checkpoint_local_replan_ms']['p50']} / {replan['checkpoint_local_replan_ms']['p95']} ms",
        f"- Modeled P50 latency reduction: {replan['p50_latency_reduction_ratio']:.1%}",
        f"- Avoided stages: {', '.join(replan['avoided_nodes'])}",
        f"- Simulated LLM input budget avoided: {tokens['avoided']} / {tokens['full_replay']} "
        f"({tokens['reduction_ratio']:.1%})",
        f"- Token boundary: {tokens['disclaimer']}",
        "",
        "## Interpretation boundary",
        "",
        "The concurrency result measures the production node around a deterministic fake provider. "
        "It proves scheduling behavior, not real AMap network latency. The checkpoint result is a "
        "controlled stage-cost simulation because no runnable historical version exists. Real end-to-end "
        "latency, exact token usage, answer quality and external-service failure rates require the online suite.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--route-legs", type=int, default=8)
    parser.add_argument("--fanout-delay-ms", type=float, default=40.0)
    parser.add_argument("--provider-delay-ms", type=float, default=20.0)
    parser.add_argument("--stage-unit-delay-ms", type=float, default=2.0)
    parser.add_argument("--json-out", type=Path, default=ROOT / "evaluation" / "offline_engineering_benchmark.json")
    parser.add_argument("--markdown-out", type=Path, default=ROOT / "evaluation" / "offline_engineering_benchmark.md")
    args = parser.parse_args()
    if args.runs < 5:
        parser.error("--runs must be at least 5")
    if not 1 <= args.route_legs <= 100:
        parser.error("--route-legs must be between 1 and 100")
    if args.fanout_delay_ms <= 0 or args.provider_delay_ms <= 0 or args.stage_unit_delay_ms <= 0:
        parser.error("delay values must be positive")

    report = {
        "schema_version": 1,
        "external_calls": False,
        "post_intent_fanout": benchmark_post_intent_fanout(args.runs, args.fanout_delay_ms),
        "enrichment_fanout": benchmark_enrichment_fanout(args.runs, args.fanout_delay_ms),
        "road_distance": benchmark_road_distance(args.runs, args.route_legs, args.provider_delay_ms),
        "checkpoint_replan": benchmark_checkpoint_replan(args.runs, args.stage_unit_delay_ms),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    passed = all(report[key]["pass"] for key in (
        "post_intent_fanout", "enrichment_fanout", "road_distance", "checkpoint_replan",
    ))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
