"""Deterministic adversarial risk-gate probe.

This is a discovery set, not a pass-rate benchmark: some cases are expected to
be rejected.  The safety contract is that malformed provider/LLM output never
crashes the gate and is never silently approved.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.planning.nodes import route_risk_gate_node
from tests.eval.harness import build_state_from_fixture, load_fixtures


SEED = 20260925


def generate_cases(seed: int = SEED, count: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    fixture = next(item for item in load_fixtures() if item["id"] == "nanjing-1d-sunny-nightlife")
    names = [str(p["name"]) for p in fixture["pois"][:4]]
    cases: list[dict[str, Any]] = []
    for index in range(count):
        spot = {"name": rng.choice(names), "period": "morning", "start_time": "10:00", "end_time": "11:00"}
        kind = rng.choice(("none_spot", "scalar_spots", "none_day", "bad_time", "unknown", "duplicate"))
        if kind == "none_spot":
            route = [{"day": 1, "spots": [None]}]
            malformed = True
        elif kind == "scalar_spots":
            route = [{"day": 1, "spots": "not-a-list"}]
            malformed = True
        elif kind == "none_day":
            route = [None]
            malformed = True
        elif kind == "bad_time":
            spot["start_time"] = rng.choice(("99:99", "", None))
            route = [{"day": 1, "spots": [spot]}]
            malformed = False
        elif kind == "unknown":
            spot["name"] = f"随机虚构景点-{index}"
            route = [{"day": 1, "spots": [spot]}]
            malformed = False
        else:
            route = [{"day": 1, "spots": [spot, dict(spot, period="afternoon", start_time="14:00", end_time="15:00")]}]
            malformed = False
        cases.append({"case_id": f"adv-{index:03d}-{kind}", "route": route, "malformed": malformed})
    return cases


def evaluate(count: int = 40) -> dict[str, Any]:
    fixture = next(item for item in load_fixtures() if item["id"] == "nanjing-1d-sunny-nightlife")
    records: list[dict[str, Any]] = []
    for case in generate_cases(count=count):
        state = build_state_from_fixture(fixture)
        state.route = case["route"]
        try:
            result = route_risk_gate_node(state)
            crashed = False
            skipped = bool(result.get("review_skipped"))
            flags = result.get("route_risk_flags", [])
            unsafe_skip = case["malformed"] and skipped
            records.append({**case, "crashed": crashed, "skipped": skipped, "unsafe_skip": unsafe_skip, "flags": flags})
        except Exception as exc:  # discovery output is intentionally retained
            records.append({**case, "crashed": True, "skipped": False, "unsafe_skip": False,
                            "error": f"{type(exc).__name__}: {exc}"})
    crashes = sum(r["crashed"] for r in records)
    unsafe = sum(r["unsafe_skip"] for r in records)
    return {
        "seed": SEED,
        "count": len(records),
        "crash_rate": crashes / len(records),
        "unsafe_skip_rate": unsafe / len(records),
        "known_failures": [r for r in records if r["crashed"] or r["unsafe_skip"]],
        "records": records,
        "contract": "adversarial discovery; not expected to have 100% case pass rate",
    }


if __name__ == "__main__":
    root = ROOT
    report = evaluate()
    path = root / "evaluation" / "adversarial_risk_gate.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("seed", "count", "crash_rate", "unsafe_skip_rate")}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["crash_rate"] == 0 and report["unsafe_skip_rate"] == 0 else 1)
