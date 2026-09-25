"""Replay the risk gate on saved provider outputs and labelled offline cases."""

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
from tests.eval.harness import build_state_from_fixture, load_fixtures  # noqa: E402


NATURAL_REPORTS = {
    "nanjing-3d-sunny-history": "real_ablation_sunny.json",
    "nanjing-3d-singlerain-history": "real_ablation_rain.json",
    "nanjing-3d-tighthours-negative": "real_ablation_tight.json",
}


def _spot(name: str, start: str = "10:00", end: str = "11:00", period: str = "morning") -> dict[str, str]:
    return {"name": name, "period": period, "start_time": start, "end_time": end}


def _route(*days: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [{"day": index, "spots": spots} for index, spots in enumerate(days, 1)]


def constructed_cases() -> list[dict[str, Any]]:
    """Hand-labelled contrasts using the frozen AMap POI pools, not LLM outputs."""
    museum = "中国科举博物馆(江南贡院)"
    lake = "玄武湖景区"
    morning = _spot(museum)
    afternoon = _spot("南京夫子庙", "14:00", "15:00", "afternoon")
    base = _route([morning])
    pair = _route([morning, afternoon])
    sunny = "nanjing-1d-sunny-nightlife"
    rainy = "nanjing-3d-singlerain-history"
    rainy_indoor = _route([morning], [_spot("总统府")], [_spot("江苏江宁汤山方山国家地质公园博物馆")])
    rainy_route = _route([morning], [_spot(lake)], [_spot("江苏江宁汤山方山国家地质公园博物馆")])
    cases = [
        ("sunny_indoor", sunny, base, {}, "skip", []),
        ("sunny_outdoor", sunny, _route([_spot(lake)]), {}, "skip", []),
        ("valid_two_spots", sunny, pair, {}, "skip", []),
        ("no_forecast", sunny, base, {"weather_forecast": []}, "skip", []),
        ("valid_walking_leg", sunny, pair, {"max_walking_km": 3, "route_distance_legs": [{"day": 1, "from": museum, "to": "南京夫子庙", "mode": "walk", "distance_km": 2}]}, "skip", []),
        ("unknown_hours", sunny, base, {"pois": [{"name": museum, "open_time": "", "indoor": True}]}, "skip", ["opening_time_unknown"]),
        ("rain_indoor", rainy, rainy_indoor, {}, "skip", []),
        ("rain_outdoor", rainy, rainy_route, {}, "escalate", ["weather_outdoor_conflict"]),
        ("duplicate_poi", sunny, _route([morning, _spot(museum, "14:00", "15:00", "afternoon")]), {}, "escalate", ["duplicate_poi"]),
        ("unknown_poi", sunny, _route([_spot("虚构景点")]), {}, "escalate", ["unknown_poi"]),
        ("missing_day", sunny, [], {}, "escalate", ["route_structure"]),
        ("empty_day", sunny, _route([]), {}, "escalate", ["route_structure"]),
        ("duplicate_day_number", sunny, [{"day": 2, "spots": [morning]}], {}, "escalate", ["route_structure"]),
        ("overlap", sunny, _route([morning, _spot("南京夫子庙", "10:30", "11:30")]), {}, "escalate", ["route_structure"]),
        ("reverse_period", sunny, _route([_spot(museum, "10:00", "11:00", "afternoon"), _spot("南京夫子庙", "14:00", "15:00")]), {}, "escalate", ["route_structure"]),
        ("missing_time", sunny, _route([_spot(museum, "", "11:00")]), {}, "escalate", ["route_structure"]),
        ("closed_hours", sunny, _route([_spot(museum, "07:00", "08:00")]), {}, "escalate", ["opening_time_conflict"]),
        ("late_start", sunny, _route([_spot(museum, "09:00", "10:00")]), {"habit_preference": "不喜欢早起"}, "escalate", ["habit_constraint"]),
        ("slow_pace", sunny, _route([morning, afternoon, _spot(lake, "16:00", "17:00", "evening")]), {"habit_preference": "慢节奏"}, "escalate", ["habit_constraint"]),
        ("walking_too_far", sunny, pair, {"max_walking_km": 3, "route_distance_legs": [{"day": 1, "from": museum, "to": "南京夫子庙", "mode": "walk", "distance_km": 4}]}, "escalate", ["walking_constraint"]),
        ("walking_leg_missing", sunny, pair, {"max_walking_km": 3}, "escalate", ["walking_constraint"]),
        ("long_drive", sunny, pair, {"route_distance_legs": [{"day": 1, "from": museum, "to": "南京夫子庙", "mode": "drive", "distance_km": 30}]}, "escalate", ["long_road_leg"]),
        ("user_change", sunny, base, {"modification_notes": "改成室内"}, "escalate", ["user_modification"]),
        ("compound_fault", sunny, _route([_spot("虚构景点"), _spot("虚构景点", "14:00", "15:00", "afternoon")]), {}, "escalate", ["unknown_poi", "duplicate_poi"]),
        ("valid_evening", sunny, _route([_spot("南京夫子庙", "19:00", "20:00", "evening")]), {}, "skip", []),
        ("too_many_per_day", sunny, _route([morning, afternoon, _spot(lake, "16:00", "17:00", "evening"), _spot("古鸡鸣寺", "18:00", "19:00", "evening")]), {"max_per_day": 3}, "escalate", ["route_structure"]),
        ("day_number_gap", sunny, [{"day": 1, "spots": [morning]}, {"day": 3, "spots": [afternoon]}], {"days": 2}, "escalate", ["route_structure"]),
        ("zero_length_visit", sunny, _route([_spot(museum, "10:00", "10:00")]), {}, "escalate", ["route_structure"]),
        ("period_missing", sunny, _route([_spot(museum, "10:00", "11:00", "night")]), {}, "escalate", ["route_structure"]),
        ("opening_conflict_with_clean_structure", sunny, _route([_spot(museum, "07:00", "08:00")]), {}, "escalate", ["opening_time_conflict"]),
        ("unknown_hours_plus_duplicate", sunny, _route([_spot("临时展馆"), _spot("临时展馆", "14:00", "15:00", "afternoon")]), {"pois": [{"name": "临时展馆", "open_time": "", "indoor": True}]}, "escalate", ["duplicate_poi", "opening_time_unknown"]),
        ("transit_leg_not_walk", sunny, pair, {"max_walking_km": 3, "route_distance_legs": [{"day": 1, "from": museum, "to": "南京夫子庙", "mode": "transit", "distance_km": 8}]}, "escalate", ["walking_constraint"]),
        ("walk_limit_exact_boundary", sunny, pair, {"max_walking_km": 2, "route_distance_legs": [{"day": 1, "from": museum, "to": "南京夫子庙", "mode": "walk", "distance_km": 2}]}, "skip", []),
        ("drive_leg_below_threshold", sunny, pair, {"route_distance_legs": [{"day": 1, "from": museum, "to": "南京夫子庙", "mode": "drive", "distance_km": 20}]}, "skip", []),
        ("route_modify_opinion", sunny, base, {"route_modify_opinion": "【用户修改意见】替换下午景点"}, "escalate", ["user_modification"]),
        ("late_start_valid", sunny, _route([_spot(museum, "10:00", "11:00")]), {"habit_preference": "不喜欢早起"}, "skip", []),
    ]
    rainy_cross_city = [
        ("lijiang_all_rain_indoor", "lijiang-3d-allrain-history", "丽江千古情景区", "15:00", "16:00", "skip", []),
        ("lijiang_all_rain_outdoor", "lijiang-3d-allrain-history", "丽江古城", "10:00", "11:00", "escalate", ["weather_outdoor_conflict"]),
        ("sanya_all_rain_outdoor", "sanya-3d-allrain-outdoor-negative", "亚龙湾海滩", "10:00", "11:00", "escalate", ["weather_outdoor_conflict"]),
        ("shanghai_all_rain_indoor", "shanghai-3d-allrain-history", "上海失恋博物馆", "10:00", "11:00", "skip", []),
        ("shanghai_all_rain_outdoor", "shanghai-3d-allrain-history", "上海动物园", "10:00", "11:00", "escalate", ["weather_outdoor_conflict"]),
    ]
    for case_id, fixture_id, poi_name, start, end, decision, flags in rainy_cross_city:
        cases.append((case_id, fixture_id, _route([_spot(poi_name, start, end)]), {"days": 1}, decision, flags))
    city_contrasts = [
        ("jingdezhen", "jingdezhen-1d-sunny-history", "景德镇古窑民俗博览区", "07:00", "08:00"),
        ("lijiang", "lijiang-1d-sunny-nightlife", "丽江千古情景区", "10:00", "11:00"),
        ("sanya", "sanya-1d-sunny-nightlife", "三亚千古情景区", "10:00", "11:00"),
        ("shanghai", "shanghai-1d-sunny-nightlife", "上海失恋博物馆", "07:00", "08:00"),
    ]
    for city, fixture_id, name, closed_start, closed_end in city_contrasts:
        cases.extend([
            (f"{city}_open", fixture_id, _route([_spot(name, "15:00", "16:00")]), {}, "skip", []),
            (f"{city}_closed", fixture_id, _route([_spot(name, closed_start, closed_end)]), {}, "escalate", ["opening_time_conflict"]),
        ])
    return [
        {"case_id": case_id, "fixture_id": fixture_id, "route": route, "state_updates": updates,
         "expected_decision": decision, "expected_flags": flags}
        for case_id, fixture_id, route, updates, decision, flags in cases
    ]


def _decision(fixture: dict[str, Any], route: list[dict[str, Any]], **state_updates: Any) -> dict[str, Any]:
    state = build_state_from_fixture(fixture)
    state.route = route
    for key, value in state_updates.items():
        setattr(state, key, value)
    result = route_risk_gate_node(state)
    return {
        "flags": result["route_risk_flags"],
        "review_required": result["review_required"],
        "time_check_required": result["time_check_required"],
        "skipped": result["review_skipped"],
    }


def evaluate(root: Path = ROOT) -> dict[str, Any]:
    fixtures = {item["id"]: item for item in load_fixtures()}
    natural: list[dict[str, Any]] = []
    for fixture_id, filename in NATURAL_REPORTS.items():
        payload = json.loads((root / "evaluation" / filename).read_text(encoding="utf-8"))
        route = payload["trials"]["planner_only"][0]["_state"]["route"]
        natural.append({"case_id": fixture_id, **_decision(fixtures[fixture_id], route)})

    fault_payload = json.loads(
        (root / "evaluation" / "real_fault_recovery_ablation.json").read_text(encoding="utf-8")
    )
    faults: list[dict[str, Any]] = []
    for record in fault_payload["records"]:
        if record["mode"] != "single_no_audit":
            continue
        fixture_id = str(record["fixture_id"])
        faults.append({
            "scenario": record["scenario"],
            **_decision(fixtures[fixture_id], record["route"]),
        })

    constructed = []
    for case in constructed_cases():
        actual = _decision(fixtures[case["fixture_id"]], case["route"], **case["state_updates"])
        expected_skip = case["expected_decision"] == "skip"
        constructed.append({
            "case_id": case["case_id"], "fixture_id": case["fixture_id"],
            "expected_decision": case["expected_decision"], "expected_flags": case["expected_flags"],
            **actual,
            "passed": actual["skipped"] == expected_skip
            and all(flag in actual["flags"] for flag in case["expected_flags"]),
        })

    natural_ablation = json.loads(
        (root / "evaluation" / "real_single_vs_multi_ablation.json").read_text(encoding="utf-8")
    )
    modes = {row["mode"]: row for row in natural_ablation["modes"]}
    single = modes["planner_only"]
    full = modes["planner_reviewer_time_check"]
    token_reduction = 1 - single["estimated_tokens_mean"] / full["estimated_tokens_mean"]
    latency_reduction = 1 - single["elapsed_mean_ms"] / full["elapsed_mean_ms"]
    return {
        "measurement_type": "offline-policy-replay-on-real-provider-outputs",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "natural": natural,
        "faults": faults,
        "constructed": constructed,
        "metrics": {
            "natural_skip_rate": round(sum(row["skipped"] for row in natural) / len(natural), 3),
            "fault_escalation_recall": round(sum(not row["skipped"] for row in faults) / len(faults), 3),
            "constructed_pass_rate": round(sum(row["passed"] for row in constructed) / len(constructed), 3),
            "constructed_clean_skip_rate": round(sum(row["skipped"] for row in constructed if row["expected_decision"] == "skip") / sum(row["expected_decision"] == "skip" for row in constructed), 3),
            "constructed_fault_recall": round(sum(not row["skipped"] for row in constructed if row["expected_decision"] == "escalate") / sum(row["expected_decision"] == "escalate" for row in constructed), 3),
            "constructed_category_recall": {
                flag: round(sum(flag in row["flags"] for row in constructed if flag in row["expected_flags"]) / sum(flag in row["expected_flags"] for row in constructed), 3)
                for flag in sorted({flag for row in constructed for flag in row["expected_flags"]})
            },
            "projected_token_reduction_vs_always_full": round(token_reduction, 3),
            "projected_latency_reduction_vs_always_full": round(latency_reduction, 3),
        },
        "boundary": (
            "The natural/fault groups replay saved real-provider routes; constructed cases are hand-labelled routes over frozen real POI pools, not model generations. "
            "Savings are projections from the three-case online ablation, not a new online A/B."
        ),
    }


def render(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Risk-Gated Audit Policy Replay",
        "",
        "> Deterministic replay over saved real-provider routes; no external calls.",
        "",
        f"- Natural-route audit skip rate: {metrics['natural_skip_rate']:.1%}",
        f"- Injected-fault escalation recall: {metrics['fault_escalation_recall']:.1%}",
        f"- Constructed labelled cases: {sum(row['passed'] for row in report['constructed'])}/{len(report['constructed'])} pass; clean skip {metrics['constructed_clean_skip_rate']:.1%}, fault recall {metrics['constructed_fault_recall']:.1%}",
        f"- Projected token reduction vs always-full audit: {metrics['projected_token_reduction_vs_always_full']:.1%}",
        f"- Projected latency reduction vs always-full audit: {metrics['projected_latency_reduction_vs_always_full']:.1%}",
        "",
        "| Input | Decision | Flags |",
        "|---|---|---|",
    ]
    for row in report["natural"]:
        lines.append(f"| {row['case_id']} | {'skip' if row['skipped'] else 'escalate'} | {', '.join(row['flags']) or '-'} |")
    for row in report["faults"]:
        lines.append(f"| injected:{row['scenario']} | {'skip' if row['skipped'] else 'escalate'} | {', '.join(row['flags']) or '-'} |")
    for row in report["constructed"]:
        lines.append(f"| constructed:{row['case_id']} {'OK' if row['passed'] else 'FAIL'} | {'skip' if row['skipped'] else 'escalate'} | {', '.join(row['flags']) or '-'} |")
    lines.extend(["", "Constructed category recall: " + ", ".join(f"{flag}={rate:.0%}" for flag, rate in metrics["constructed_category_recall"].items())])
    lines.extend(["", f"Boundary: {report['boundary']}", ""])
    return "\n".join(lines)


def main() -> int:
    report = evaluate()
    json_path = ROOT / "evaluation" / "risk_gate_replay.json"
    markdown_path = ROOT / "evaluation" / "risk_gate_replay.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render(report), encoding="utf-8")
    print(render(report))
    metrics = report["metrics"]
    return 0 if metrics["natural_skip_rate"] == 1 and metrics["fault_escalation_recall"] == 1 and metrics["constructed_pass_rate"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
