"""Explicit live calibration (paid LLM + Amap); never imported by pytest.

Run: python scripts/evaluate_live_calibration.py --concurrency 2
Each case runs once, without runner-level retries. Production node budgets,
including any instrumented provider retries, remain active. All artifacts go
to an ignored, timestamped data/ directory; older evidence is not overwritten.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.env import load_local_env
from app.core.llm_usage import bind_run, take_run_snapshot, unbind_run
from app.planning.candidate_react import effective_api_budget, requested_names
from app.planning.graph import build_graph
from app.planning.quality import result_envelope
from app.planning.schemas import TravelPlanState

CASES = [
    {"id": "case1_nanjing_slow_history", "query": "南京两日游，明天出发，慢节奏，喜欢历史文化，避开商业街。"},
    {"id": "case3_chongqing_dense_cross_area", "query": "重庆一日游，明天出发，想去“磁器口古镇”、“洪崖洞”、“解放碑”、“南山一棵树观景台”和“重庆中国三峡博物馆”，尽量少走路，单段步行不超过2公里。"},
    {"id": "case5_nanjing_time_pressure", "query": "南京一日游，明天上午出发，只有4小时，必须去“中山陵”、“南京博物院”和“夫子庙”，三个点都不能删。"},
    {"id": "case6_failed_nonexistent", "query": "上海一日游，明天出发，只去“星河幻境博物馆”，这是唯一必去景点。"},
]


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_case(case: dict, output: Path, timeout: int) -> dict:
    start = time.perf_counter()
    run_id = f"calibration-{case['id']}-{uuid.uuid4().hex[:8]}"
    token = bind_run(run_id)
    initial = TravelPlanState(
        query=case["query"], max_review_rounds=3, max_time_check_rounds=3,
        candidate_max_rounds=2, candidate_api_budget=4, candidate_min_per_day=2,
        max_per_day=3, max_spots=20, min_rating=4.2,
    )
    state = initial
    progress = []
    print(f"START {case['id']}", flush=True)
    try:
        async with asyncio.timeout(timeout):
            async for raw in build_graph().astream(initial, {"recursion_limit": 100}, stream_mode="values"):
                state = TravelPlanState.model_validate(raw)
                stamp = {
                    "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
                    "snapshots": len(state.route_history), "review_round": state.review_round,
                    "candidate_round": state.candidate_search_round,
                    "candidate_http_attempts": state.candidate_api_calls,
                }
                progress.append(stamp)
                save(output / f"{case['id']}.progress.json", {
                    "id": case["id"], "progress": progress,
                    "last_state": state.model_dump(mode="json"),
                })
        envelope = result_envelope(state)
        row = {
            **case, "execution_completed": True,
            **envelope,
            "selected_iteration": envelope.get("score_info", {}).get("selected_iteration"),
        }
    except Exception as exc:
        row = {**case, "execution_completed": False,
               "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        unbind_run(token)
    row.update({
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
        "candidate_search_round": state.candidate_search_round,
        "candidate_api_calls": state.candidate_api_calls,
        "effective_api_budget": effective_api_budget(state),
        "requested_names": requested_names(state),
        "candidate_names": [p.get("name") for p in state.pois],
        "candidate_search_trace": state.candidate_search_trace,
        "candidate_pool_status": state.candidate_pool_status,
        "route_history": state.route_history,
        "planner_reworks": max(0, len(state.route_history) - 1),
        "relaxed_constraints": state.relaxed_constraints,
        "route_distance_legs": state.route_distance_legs,
        "llm_usage": take_run_snapshot(run_id),
        "cost_status": "unpriced",
    })
    save(output / f"{case['id']}.json", row)
    print(json.dumps({k: row.get(k) for k in (
        "id", "execution_completed", "status", "selected_iteration", "elapsed_ms",
        "candidate_search_round", "candidate_api_calls", "effective_api_budget",
        "planner_reworks", "error_type",
    )}, ensure_ascii=False), flush=True)
    return row


async def main(args) -> None:
    load_local_env()
    ready = {
        "llm_key_configured": bool(os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROK_API_KEY")),
        "amap_key_configured": bool(os.getenv("AMAP_API_KEY")),
    }
    if not all(ready.values()):
        raise SystemExit(json.dumps(ready))
    # Fresh real lookups, process-local only; .env.local is never modified.
    os.environ["REDIS_URL"] = ""
    output = ROOT / "data" / "live_calibration" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        **ready, "provider": os.getenv("AI_PROVIDER") or os.getenv("LLM_PROVIDER"),
        "model": os.getenv("AI_MODEL") or os.getenv("OPENAI_MODEL"),
        "redis_cache_disabled": True, "runner_retries": 0,
        "concurrency": args.concurrency, "case_timeout_seconds": args.timeout,
        "max_review_rounds": 3, "max_snapshots": 4, "candidate_max_rounds": 2,
        "candidate_api_budget_formula": "min(10,max(4,len(requested_names)+2))",
        "cost_status": "unpriced", "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(output / "config.json", config)
    print(json.dumps({"output": str(output), "config": config}, ensure_ascii=False), flush=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    async def limited(case):
        async with semaphore:
            return await run_case(case, output, args.timeout)
    selected = [case for case in CASES if not args.cases or case["id"] in args.cases.split(",")]
    if not selected or (args.cases and set(args.cases.split(",")) - {case["id"] for case in CASES}):
        raise SystemExit("Unknown or empty case selection")
    rows = await asyncio.gather(*(limited(case) for case in selected))
    save(output / "results.json", {"config": config, "cases": rows})
    print(f"WROTE {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--cases", default="", help="Comma separated IDs; default: all four cases")
    asyncio.run(main(parser.parse_args()))
