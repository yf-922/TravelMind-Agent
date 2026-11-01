"""Revalidate saved real snapshots offline; never regenerate or overwrite a run.

This checks snapshot selection and production final output with recorded POIs,
dates and road facts. It does not measure live-model behavior or live latency.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.planning import nodes
from app.planning.quality import choose_snapshot, hard_constraint_report, result_envelope
from app.planning.schemas import TravelPlanState


def replay(source: Path) -> dict:
    raw = source.read_bytes()
    state = TravelPlanState.model_validate(json.loads(raw)["last_state"])
    started = time.perf_counter()
    snapshots = []
    for original in state.route_history:
        selected, status, warnings, failures = choose_snapshot(state.model_copy(update={"route_history": [original]}))
        plan = original.get("plan", {})
        own_state = state.model_copy(update={"route": plan.get("route", []),
            "meal_slots": plan.get("meal_slots", []), "route_distance_legs": plan.get("route_distance_legs", [])})
        snapshots.append({
            "iteration": original["iteration"],
            "recorded_l0_violations": original.get("l0_violations", []),
            "recorded_l1_violations": original.get("l1_violations", []),
            "revalidated_l1_violations": selected.get("l1_violations", []) if selected else [],
            "current_full_hard_report": hard_constraint_report(own_state),
            "recorded_soft_score": original.get("soft_score"),
            "recorded_score_breakdown": original.get("score_breakdown"),
            "isolated_status": status, "warnings": warnings, "failure_reasons": failures,
        })

    # External enrichment is explicitly disabled. A network attempt fails the
    # replay rather than silently reaching real APIs or accruing new charges.
    with ExitStack() as stack:
        for name in ("socket.socket.connect", "socket.create_connection"):
            stack.enter_context(patch(name, side_effect=AssertionError("offline replay forbids network")))
        stack.enter_context(patch.object(nodes, "amap_key", return_value=None))
        stack.enter_context(patch.object(nodes, "lookup_live_ticket_prices", return_value={}))
        stack.enter_context(patch.object(nodes, "recommend_chain_hotel", side_effect=AssertionError("offline replay forbids hotel API")))
        final = result_envelope(state.model_copy(update=nodes.finalize_node(state)))

    closure_messages = [message for message in final["warnings"] if "景点可能闭馆" in message]
    # Acceptance checks for the actual source used for this regression.
    selected, _, _, _ = choose_snapshot(state)
    selected_closures = [v["message"] for v in (selected or {}).get("l1_violations", [])
                         if v.get("code") == "closed_on_visit_day"]
    assert selected_closures and all(message in final["warnings"] for message in selected_closures)
    assert all(message in final["plan"]["warnings"] and message in final["plan"]["route_issues"]
               for message in selected_closures)
    assert final["status"] == "PARTIAL_SUCCESS" and final["plan"]["days"] and not final["failure_reasons"]
    return {
        "mode": "offline_revalidation_of_saved_live_snapshots", "external_calls": 0,
        "source": str(source.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "code_sha256": {str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in ("app/planning/helpers.py", "app/planning/quality.py", "app/planning/nodes.py")},
        "travel_start_date": state.travel_start_date.isoformat() if state.travel_start_date else None,
        "snapshots": snapshots, "status": final["status"],
        "selected_iteration": final["score_info"].get("selected_iteration"),
        "closure_warnings": closure_messages, "final_result": final,
        "local_replay_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "scope": "Existing route versions and scores retained; closure findings revalidated before selection. No new model planning or live E2E latency measured.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/live_calibration/20261031T165143Z/case3_chongqing_dense_cross_area.progress.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "data/weekday_closure_replay" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") / "report.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing artifact: {output}")
    report = replay(args.source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "selected_iteration", "closure_warnings", "local_replay_elapsed_ms", "external_calls")}, ensure_ascii=False))
    print(f"Report: {output.resolve()}")


if __name__ == "__main__":
    main()
