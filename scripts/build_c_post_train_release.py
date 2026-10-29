"""Build an immutable, locally verified C post-training data candidate.

No teacher API calls. Historical source releases are read-only; all outputs go
under data/post_train/c_release_20261028. A failed gate is reported as failed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.planning.schemas import TravelPlanState, TravelRoute
from app.evaluation.route_interest import route_key
from scripts.prepare_heldout_evidence import opening_window, clock
from scripts.post_train_runtime import capture_messages, replay_check, quality_faults
from scripts.regen_v1_v2_current_protocol import make_tokenizer, saved_maps

DATA = ROOT / "data/post_train"
OUT = DATA / "c_release_20261028_final"
POOLS = json.loads((DATA / "v2_large_pool_teacher_v1/training_pools.json").read_text(encoding="utf-8"))["pools"]
MAPS = json.loads((DATA / "v2_large_pool_teacher_v1/maps.json").read_text(encoding="utf-8"))["routes"]


def read_rows(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def parse_clock(value):
    h, m = map(int, value.split(":"))
    return h * 60 + m


def time_constraints(query):
    constraints = []
    patterns = [
        (r"(\d{1,2}:\d{2})后开始", "gt"), (r"严格晚于\s*(\d{1,2}:\d{2})", "gt"),
        (r"不早于\s*(\d{1,2}:\d{2})", "ge"), (r"(\d{1,2}:\d{2})前开始", "lt"),
        (r"严格早于\s*(\d{1,2}:\d{2})", "lt"), (r"不晚于\s*(\d{1,2}:\d{2})", "le"),
    ]
    for pattern, op in patterns:
        constraints.extend((parse_clock(m), op) for m in re.findall(pattern, query))
    return constraints


def strict_time_faults(plan, query):
    faults = []
    for boundary, op in time_constraints(query):
        for day in plan.get("days", []):
            for spot in day.get("spots", []):
                start = parse_clock(spot["start_time"])
                if op == "gt" and start <= boundary or op == "ge" and start < boundary or op == "lt" and start >= boundary or op == "le" and start > boundary:
                    faults.append("user_time_window")
    for lo, hi in re.findall(r"(?:只允许在|允许在)(\d{1,2}:\d{2})\s*[-至~]\s*(\d{1,2}:\d{2})", query):
        lower, upper = parse_clock(lo), parse_clock(hi)
        if any(parse_clock(s["start_time"]) < lower or parse_clock(s["end_time"]) > upper
               for d in plan.get("days", []) for s in d.get("spots", [])):
            faults.append("user_time_window")
    return faults


def repair_schedule(state, plan, maps):
    """Reschedule existing POIs without changing the selected set or durations."""
    by_name = {p["name"]: p for p in state.get("pois", [])}
    constraints = time_constraints(state.get("query", ""))
    for day in plan.get("days", []):
        spots = sorted(day.get("spots", []), key=lambda s: s["start_time"])
        cursor = None
        previous = None
        for spot in spots:
            duration = parse_clock(spot["end_time"]) - parse_clock(spot["start_time"])
            poi = by_name.get(spot["name"], {})
            window = opening_window(poi)
            if duration <= 0 or not window:
                return False
            start = max(parse_clock(spot["start_time"]), window[0], cursor or 0)
            for boundary, op in constraints:
                if op == "gt": start = max(start, boundary + 5)
                elif op == "ge": start = max(start, boundary)
                elif op == "lt" and start + duration >= boundary: return False
                elif op == "le" and start + duration > boundary: return False
            state_obj = TravelPlanState(**state)
            from app.planning.constraints import hot_day, indoor_value
            if hot_day(state_obj, day["day"]) and indoor_value(poi) is not True and start < 900 and start + duration > 660:
                alternatives = [x for x in (max(start, 900), min(start, 660 - duration))
                    if x >= window[0] and x + duration <= min(window[1], window[2])]
                if not alternatives: return False
                start = min(alternatives, key=lambda x: abs(x - start))
            if previous:
                a = by_name.get(previous["name"], {})
                leg = maps.get(route_key(a.get("location"), poi.get("location"), "drive")) if a.get("location") and poi.get("location") else None
                if leg is None: return False
                start = max(start, parse_clock(previous["end_time"]) + int(leg["duration_min"]))
            if start + duration > min(window[1], window[2]) or start >= 1440:
                return False
            spot["start_time"], spot["end_time"] = clock(start), clock(start + duration)
            cursor, previous = start + duration, spot
    return True


def validate(case, plan, maps=MAPS):
    faults = []
    try:
        plan = TravelRoute.model_validate(plan).model_dump(mode="json")
    except Exception:
        return ["schema"]
    state = case["state"]
    names = [s["name"] for d in plan["days"] for s in d["spots"]]
    pool = {p["name"]: p for p in state.get("pois", [])}
    if any(name not in pool for name in names):
        faults.append("out_of_pool")
    if len(names) != len(set(names)):
        faults.append("duplicate_poi")
    exact = re.search(r"恰好\s*(\d+)个不同景点", state.get("query", ""))
    if exact and len(names) != int(exact.group(1)):
        faults.append("count_or_membership")
    faults.extend(strict_time_faults(plan, state.get("query", "")))
    for day in plan["days"]:
        ordered = sorted(day["spots"], key=lambda s: s["start_time"])
        for a, b in zip(ordered, ordered[1:]):
            if parse_clock(a["end_time"]) > parse_clock(b["start_time"]):
                faults.append("timeline_overlap")
            pa, pb = pool.get(a["name"], {}), pool.get(b["name"], {})
            if pa.get("location") and pb.get("location"):
                leg = maps.get(route_key(pa["location"], pb["location"], "drive"))
                if leg is None:
                    faults.append("road_unverified")
                elif parse_clock(b["start_time"]) - parse_clock(a["end_time"]) < int(leg["duration_min"]):
                    faults.append("commute_time_conflict")
        for spot in day["spots"]:
            window = opening_window(pool.get(spot["name"], {}))
            if window:
                start, end = parse_clock(spot["start_time"]), parse_clock(spot["end_time"])
                if start < window[0] or end > window[1] or start > window[2]:
                    faults.append("opening_window")
            elif spot["name"] in pool:
                faults.append("opening_unverified")
    try:
        checked = TravelPlanState(**state).model_copy(update={"route": plan["days"], "meal_slots": plan.get("meal_slots", [])})
        faults.extend(quality_faults(checked))
    except Exception:
        faults.append("production_quality_check_error")
    return sorted(set(faults))


def make_plan(state, visits, note="按快照约束安排；未提供的信息不作推断。"):
    days = []
    by_day = {}
    for name, day, start, duration in visits:
        by_day.setdefault(day, []).append({"name": name, "period": "morning" if start < 720 else "afternoon", "start_time": clock(start), "end_time": clock(start + duration)})
    for day in range(1, state["days"] + 1):
        days.append({"day": day, "spots": sorted(by_day.get(day, []), key=lambda x: x["start_time"]), "theme": f"第{day}天路线"})
    names = [s["name"] for d in days for s in d["spots"]]
    return {"reasoning": "景点均来自候选池，按用户时间要求、开放窗口和道路快照安排；全程避免重复。", "days": days,
            "meal_slots": [], "selection_reasons": [{"name": n, "reason": "候选池内地点，时段依据给定快照。", "source_ids": []} for n in names],
            "notes": note, "modification_concern": ""}


def state_for(city, pois, query, days=1, weather=None):
    dt = date(2026, 11, 4)
    return TravelPlanState(query=query, destination=city, days=days, max_per_day=3, max_spots=30,
        pois=copy.deepcopy(pois), route_distance_mode="drive", route_distance_legs=[],
        travel_start_date=dt, travel_end_date=dt + timedelta(days=days - 1),
        weather_forecast=weather or [dict(date=(dt + timedelta(days=i)).isoformat(), day_temp="24", night_temp="18", day_weather="晴", night_weather="晴", is_bad=False) for i in range(days)],
        candidate_pool_status="ok", candidate_missing_coverage=[]).model_dump(mode="json")


def choose_poi(city, used, duration=60, earliest=540, latest=1020, outdoor=None):
    for poi in POOLS[city]:
        if poi["name"] in used or (outdoor is not None and poi.get("indoor") is not outdoor):
            continue
        window = opening_window(poi)
        if not window:
            continue
        start = max(earliest, window[0])
        if start + duration <= min(latest, window[1], window[2]):
            return poi, start
    return None, None


def sample_row(identity, category, state, chosen, rejected=None, round1=False, infeasible_reason=None):
    if round1:
        previous = copy.deepcopy(rejected)
        state["route"] = previous["days"]
        codes = validate({"state": state}, previous)
        if "user_time_window" in codes:
            state["route_modify_opinion"] = "【本地规则评分器】严格时间窗违规：用户要求严格晚于09:00开始，09:00不合规；将首个景点顺延至少5分钟，并复核开放时间和道路转场。"
        else:
            state["route_modify_opinion"] = "【本地规则评分器】" + "; ".join(codes or ["修复原报错，并确保候选、时间轴、开放时间和道路转场均合规。"])
        state["review_round"] = 0
    messages = capture_messages(state)
    row = {"id": identity, "category": category, "state": state, "messages": messages, "chosen": chosen,
           "rejected": rejected, "expected": "infeasible" if infeasible_reason else "feasible",
           "infeasible_reason": infeasible_reason, "target_origin": "deterministic_local_witness", "teacher_api_calls": 0}
    row["target_defect"] = {"candidate_count": "count_or_membership", "opening_no_intersection": "user_time_window",
        "road_exceeds_window": "commute_time_conflict", "hot_no_indoor": "heat_exposure_or_unknown_indoor",
        "user_time_window": "user_time_window", "opening_road_joint": "commute_time_conflict",
        "round1_strict_time_repair": "user_time_window"}.get(category, category)
    row["chosen_faults"] = validate(row, chosen) if not infeasible_reason else []
    if infeasible_reason:
        text = chosen.get("reasoning", "") + chosen.get("notes", "")
        generic = any(k in text for k in ["无法", "不可行", "冲突", "不足", "不能满足"])
        evidence = any(k in text for k in ["候选", "开放", "时间", "通勤", "室内", "高温", "分钟"])
        if any(d["spots"] for d in chosen["days"]) or chosen["meal_slots"] or not generic or not evidence:
            row["chosen_faults"] = ["infeasible_rubric"]
    row["rejected_faults"] = validate(row, rejected) if rejected else []
    if rejected and row["expected"] == "feasible" and not any(d.get("spots") for d in rejected.get("days", [])):
        row["rejected_faults"] = ["false_refusal"]
    row["target_faults"] = row["rejected_faults"]
    return row


def infeasible_rubric(row, plan):
    state, reason = row["state"], row.get("infeasible_reason")
    text = plan.get("reasoning", "") + plan.get("notes", "")
    if any(day.get("spots") for day in plan.get("days", [])) or plan.get("meal_slots"):
        return ["infeasible_partial_plan"]
    if not any(word in text for word in ("无法", "不可行", "冲突", "不足", "不能满足")):
        return ["infeasible_generic_reason_missing"]
    state = row["state"]
    if reason == "candidate_count":
        unique = len({p["name"] for p in state.get("pois", [])})
        match = re.search(r"恰好(\d+)个不同景点", state.get("query", ""))
        required = int(match.group(1)) if match else 0
        return [] if required > unique and str(unique) in text and str(required) in text else ["infeasible_count_evidence_mismatch"]
    if reason == "opening_no_intersection":
        match = re.search(r"只允许在(\d{1,2}:\d{2})-(\d{1,2}:\d{2})", state.get("query", ""))
        windows = [opening_window(p) for p in state.get("pois", [])]
        if match:
            lo, hi = parse_clock(match.group(1)), parse_clock(match.group(2))
            disjoint = all(not w or hi <= w[0] or lo >= w[1] for w in windows)
            if disjoint and "开放" in text and "时间窗" in text:
                return []
        return ["infeasible_opening_evidence_mismatch"]
    if reason == "road_exceeds_window":
        pois = state.get("pois", [])
        leg = MAPS.get(route_key(pois[0]["location"], pois[1]["location"], "drive")) if len(pois) >= 2 else None
        match = re.search(r"(\d{1,2}:\d{2})至(\d{1,2}:\d{2})", state.get("query", ""))
        if leg and match:
            available = parse_clock(match.group(2)) - parse_clock(match.group(1))
            required = int(leg["duration_min"]) + 30
            if required > available and str(leg["duration_min"]) in text and "通勤" in text:
                return []
        return ["infeasible_road_evidence_mismatch"]
    if reason == "hot_no_indoor":
        confirmed = [p for p in state.get("pois", []) if p.get("indoor") is True]
        hot = "38°C" in state.get("query", "") or any(float(w.get("day_temp", 0)) >= 38 for w in state.get("weather_forecast", []))
        return [] if hot and not confirmed and "室内" in text and "高温" in text else ["infeasible_indoor_evidence_mismatch"]
    return ["unsupported_infeasible_reason"]


def target_error(row, plan):
    if row.get("target_defect") == "false_refusal" and not any(d.get("spots") for d in plan.get("days", [])):
        return ["false_refusal"]
    return validate(row, plan)


def clean_legacy():
    v2 = read_rows(DATA / "release_current_protocol_v1_v2/v2/train.ledger.jsonl")
    count = read_rows(DATA / "count_infeasible_ab_20261025/train.ledger.json")
    source = v2 + count
    maps = saved_maps()
    cleaned, fixes, faults, quarantine = [], [], [], []
    for original in source:
        row = copy.deepcopy(original)
        q = row["state"].get("query", "")
        constraints = [(t, op) for t, op in time_constraints(q) if op in ("gt", "lt")]
        plans = [row.get("chosen")]
        if row.get("rejected"):
            plans.append(row["rejected"])
        for plan in plans:
            if not plan:
                continue
            before = strict_time_faults(plan, q)
            repair_schedule(row["state"], plan, maps)
            if before:
                fixes.append({"id": row["id"], "boundary_constraints": [(clock(t), op) for t, op in constraints], "repaired": not strict_time_faults(plan, q)})
        for plan in plans:
            if plan:
                repair_schedule(row["state"], plan, maps)
        # State snapshots are the scoring authority. Historical target text remains untouched.
        target = row.get("chosen")
        if target:
            row["cleaning_faults"] = validate(row, target, maps)
        else:
            row["cleaning_faults"] = ["missing_chosen"]
        if row["cleaning_faults"]:
            faults.append({"id": row["id"], "faults": row["cleaning_faults"]})
            quarantine.append(row["id"])
        cleaned.append(row)
    return cleaned, fixes, faults, quarantine


def make_time_rows():
    rows, used = [], set()
    starts = [605, 610, 615, 620, 640, 645, 665]
    cities = sorted(POOLS)
    for i in range(28):
        city = cities[i % len(cities)]
        boundary = [600, 615, 630, 660][i % 4]
        duration = 60
        poi = next((p for p in POOLS[city] if p["name"] not in used and opening_window(p)
                    and opening_window(p)[0] <= boundary and boundary + duration <= min(opening_window(p)[1], opening_window(p)[2])), None)
        if not poi:
            raise RuntimeError(f"No verified time-window candidate for {city}/{i}")
        used.add(poi["name"])
        start = max(boundary + 5, starts[i % len(starts)])
        state = state_for(city, [poi], f"严格晚于{clock(boundary)}开始，每个景点至少60分钟；不得压线。")
        chosen = make_plan(state, [(poi["name"], 1, start, duration)])
        rejected = copy.deepcopy(chosen)
        bad_start = boundary
        rejected["days"][0]["spots"][0]["start_time"] = clock(bad_start)
        rejected["days"][0]["spots"][0]["end_time"] = clock(bad_start + duration)
        rows.append(sample_row(f"c-time-{i+1:02d}", "user_time_window", state, chosen, rejected))
    return rows


def make_infeasible_rows():
    rows, used = [], set()
    cities = sorted(POOLS)
    causes = ["candidate_count", "opening_no_intersection", "road_exceeds_window", "hot_no_indoor"] * 3
    for i, cause in enumerate(causes):
        city = cities[i % len(cities)]
        if cause == "candidate_count":
            poi, start = choose_poi(city, used, 60)
            used.add(poi["name"])
            state = state_for(city, [poi], "必须安排恰好2个不同景点，每个至少60分钟；若无法满足不得输出部分路线。")
            reason = f"候选池只有1个不同景点，要求2个且禁止重复，数量不足，无法满足。"
        elif cause == "opening_no_intersection":
            poi = next((p for p in POOLS[city] if opening_window(p) and opening_window(p)[1] - opening_window(p)[0] < 1200), None)
            if not poi: raise RuntimeError(f"No bounded opening window for {city}")
            window = opening_window(poi)
            impossible_time = "05:00-06:00" if window[0] >= 420 else "23:00-23:30"
            state = state_for(city, [poi], f"只允许在{impossible_time}参观，必须安排该景点至少60分钟；严格遵守开放时间。")
            reason = f"用户时间窗{impossible_time}与{poi['name']}开放窗口{clock(window[0])}-{clock(window[1])}无交集，开放与用户时间约束冲突，不可行。"
        elif cause == "road_exceeds_window":
            pair = None
            for a in POOLS[city]:
                wa = opening_window(a)
                if not wa or wa[0] > 540 or wa[1] < 570 or wa[2] < 540:
                    continue
                for b in POOLS[city]:
                    if a["name"] == b["name"]:
                        continue
                    wb = opening_window(b)
                    if not wb or wb[0] > 555:
                        continue
                    leg = MAPS.get(route_key(a["location"], b["location"], "drive"))
                    if leg and leg["duration_min"] >= 15 and 555 + 15 <= min(wb[1], wb[2]):
                        pair = (a, b, int(leg["duration_min"]))
                        break
                if pair: break
            if not pair: raise RuntimeError(f"No road evidence for {city}")
            a, b, mins = pair
            state = state_for(city, [a, b], "必须在09:00至09:30完成两个景点各15分钟参观，且必须按道路通勤时间转场。")
            reason = f"两个景点各需15分钟，且道路快照显示{a['name']}到{b['name']}需{mins}分钟；09:00-09:30可用时间不足，通勤约束导致不可行。"
        else:
            # Unknown indoor evidence cannot be promoted to indoor. The task
            # explicitly requires an indoor-confirmed visit in the hot window.
            poi = next((p for p in POOLS[city] if opening_window(p) and p.get("indoor") is not True), None)
            if not poi: raise RuntimeError(f"No windowed POI without indoor confirmation for {city}")
            state = state_for(city, [poi], "38°C高温，11:00至15:00必须安排1个已明确标注为室内的候选景点并完成60分钟参观；未知属性不算室内。", weather=[dict(date="2026-11-04", day_temp="38", night_temp="28", day_weather="晴", night_weather="晴", is_bad=False)])
            reason = f"高温时段要求已明确标注室内，但候选{poi['name']}的室内属性为未知或室外，候选池中没有已确认室内点，无法满足。"
        empty = {"reasoning": reason, "days": [{"day": d, "spots": [], "theme": "约束不可行"} for d in range(1, state["days"] + 1)], "meal_slots": [], "selection_reasons": [], "notes": reason, "modification_concern": ""}
        feasible_state = copy.deepcopy(state)
        if cause == "candidate_count":
            feasible_state["query"] = feasible_state["query"].replace("恰好2个", "恰好1个")
            chosen = make_plan(feasible_state, [(poi["name"], 1, max(540, opening_window(poi)[0]), 60)])
        elif cause == "opening_no_intersection":
            feasible_state["query"] = f"只允许在{clock(window[0])}-{clock(min(window[1], window[0]+90))}参观，必须安排该景点至少60分钟；严格遵守开放时间。"
            chosen = make_plan(feasible_state, [(poi["name"], 1, window[0], 60)])
        elif cause == "road_exceeds_window":
            feasible_state["query"] = "必须在09:00至13:00完成两个景点各15分钟参观，且必须按道路通勤时间转场。"
            chosen = make_plan(feasible_state, [(a["name"], 1, 540, 15), (b["name"], 1, 555+mins, 15)])
        else:
            feasible_state["query"] = "25°C天气，11:00至15:00安排候选景点并完成60分钟参观；室内属性未知不影响本题。"
            feasible_state["weather_forecast"] = [dict(date="2026-11-04", day_temp="25", night_temp="18", day_weather="晴", night_weather="晴", is_bad=False)]
            window = opening_window(poi)
            start = max(660, window[0])
            chosen = make_plan(feasible_state, [(poi["name"], 1, start, 60)])
        if chosen is not None:
            rejected_feasible = copy.deepcopy(chosen)
            rejected_feasible["days"] = [{"day": d, "spots": [], "theme": "数量/约束不足"} for d in range(1, feasible_state["days"] + 1)]
            rejected_feasible["selection_reasons"] = []
            rejected_feasible["reasoning"] = rejected_feasible["notes"] = "无法完成要求，不安排景点。"
            feasible_row = sample_row(f"c-infeasible-{i+1:02d}-feasible", cause, feasible_state, chosen, rejected_feasible)
            feasible_row["target_defect"] = "false_refusal"
            feasible_row["pair_id"] = f"c-counterfactual-{i+1:02d}"
            rows.append(feasible_row)
        forced = None
        if cause == "candidate_count":
            forced = make_plan(state, [(poi["name"], 1, max(540, opening_window(poi)[0]), 60)])
        elif cause == "opening_no_intersection":
            forced = make_plan(state, [(poi["name"], 1, max(540, window[0]), 60)])
        elif cause == "road_exceeds_window":
            forced = make_plan(state, [(a["name"], 1, 540, 15), (b["name"], 1, 555, 15)])
        else:
            forced = make_plan(state, [(poi["name"], 1, 660, 60)])
        infeasible_row = sample_row(f"c-infeasible-{i+1:02d}-reject", cause, state, empty, forced, infeasible_reason=cause)
        infeasible_row["pair_id"] = f"c-counterfactual-{i+1:02d}"
        rows.append(infeasible_row)
    return rows


def make_route_rows():
    rows = []
    cities = sorted(POOLS)
    for i in range(18):
        city = cities[i % len(cities)]
        pair = None
        for a in POOLS[city]:
            for b in POOLS[city]:
                if a["name"] == b["name"]: continue
                wa, wb = opening_window(a), opening_window(b)
                leg = MAPS.get(route_key(a["location"], b["location"], "drive"))
                if wa and wb and leg and wa[0] <= 540 and wb[0] < 900:
                    gap = int(leg["duration_min"])
                    first = max(540, wa[0])
                    rejected_second = first + 60 + gap - 1
                    second = rejected_second + 2
                    if rejected_second >= wb[0] and second + 60 <= min(wb[1], wb[2], 1020):
                        pair = (a, b, int(leg["duration_min"]), wa, wb, second)
                        break
            if pair: break
        if not pair: raise RuntimeError(f"No joint opening/road witness: {city}")
        a, b, travel, wa, wb, second = pair
        state = state_for(city, [a, b], "安排两个不同景点各60分钟，遵守开放时间；道路转场必须留足快照通勤时间。")
        first = max(540, wa[0])
        second = max(first + 60 + travel + 10, wb[0])
        plan = make_plan(state, [(a["name"], 1, first, 60), (b["name"], 1, second, 60)])
        bad = copy.deepcopy(plan)
        bad["days"][0]["spots"][1]["start_time"] = clock(first + 60 + travel - 1)
        rows.append(sample_row(f"c-route-{i+1:02d}", "opening_road_joint", state, plan, bad))
    return rows


def make_round1_rows():
    rows = []
    cities = sorted(POOLS)
    for i in range(20):
        city = cities[i % len(cities)]
        poi = next((p for p in POOLS[city] if opening_window(p) and opening_window(p)[0] <= 540 and opening_window(p)[1] >= 600), None)
        if not poi: raise RuntimeError(f"No Round1 witness in {city}")
        start = 545
        state = state_for(city, [poi], "严格晚于09:00开始，每个景点至少60分钟，不能重复。")
        chosen = make_plan(state, [(poi["name"], 1, max(start, 545), 60)])
        rejected = copy.deepcopy(chosen)
        rejected["days"][0]["spots"][0]["start_time"] = "09:00"
        rejected["days"][0]["spots"][0]["end_time"] = "10:00"
        row = sample_row(f"c-round1-{i+1:02d}", "round1_strict_time_repair", state, chosen, rejected, round1=True)
        if "上一版路线" not in "\n".join(m["content"] for m in row["messages"]):
            raise RuntimeError("Round1 prompt missing previous route")
        rows.append(row)
    return rows


def holdout_prompt_hashes():
    # Read only prompt messages. Never access expected labels, plans, or rubrics.
    paths = [DATA / "online_holdout50_20261027_v1/cases.json", DATA / "online_holdout50_20261027_v2/cases.json"]
    hashes = set()
    counts = {}
    for path in paths:
        if not path.exists(): continue
        cases = read_rows(path)
        current = set()
        for case in cases:
            state = case.get("state")
            messages = case.get("messages")
            if state is not None:
                messages = capture_messages(state)
            if messages:
                current.add(digest(messages))
        hashes |= current
        counts[path.parent.name] = len(current)
    return hashes, counts


def main():
    if OUT.exists():
        raise RuntimeError(f"Immutable output already exists: {OUT}")
    old, fixes, old_faults, quarantined_ids = clean_legacy()
    old = [r for r in old if r["id"] not in set(quarantined_ids)]
    clean_by_id = {r["id"]: r for r in old}
    dpo_old = read_rows(DATA / "shared_dpo_20261026/release/train.ledger.jsonl")
    dpo_cleaned, dpo_faults = [], []
    for pair in dpo_old:
        if pair["id"] in set(quarantined_ids):
            continue
        row = copy.deepcopy(pair)
        if pair["id"] in clean_by_id:
            row["chosen"] = copy.deepcopy(clean_by_id[pair["id"]]["chosen"])
        query = row["state"].get("query", "")
        constraints = [(t, op) for t, op in time_constraints(query) if op in ("gt", "lt")]
        for side in ("chosen", "rejected"):
            plan = row.get(side)
            if not plan: continue
            for boundary, op in constraints:
                for day in plan.get("days", []):
                    spots = sorted(day.get("spots", []), key=lambda s: s["start_time"])
                    bad = [s for s in spots if (op == "gt" and parse_clock(s["start_time"]) <= boundary) or (op == "lt" and parse_clock(s["start_time"]) >= boundary)]
                    if bad:
                        delta = max(boundary + 5 - parse_clock(s["start_time"]) for s in bad) if op == "gt" else max(parse_clock(s["start_time"]) - (boundary - 5) for s in bad)
                        for spot in spots:
                            spot["start_time"] = clock(parse_clock(spot["start_time"]) + delta)
                            spot["end_time"] = clock(parse_clock(spot["end_time"]) + delta)
        # Chosen must equal the clean SFT target. Rejected is audited only for
        # accidental user-window contamination; its registered defect remains.
        if row["chosen"] != clean_by_id[pair["id"]]["chosen"]:
            dpo_faults.append({"id": row["id"], "side": "chosen", "faults": ["chosen_not_aligned_to_sft"]})
        if row.get("rejected") and strict_time_faults(row["rejected"], query):
            dpo_faults.append({"id": row["id"], "side": "rejected", "faults": strict_time_faults(row["rejected"], query)})
        dpo_cleaned.append(row)
    new = make_time_rows() + make_infeasible_rows() + make_route_rows() + make_round1_rows()
    if len(new) != 90:
        raise RuntimeError(f"Expected 90 contexts, built {len(new)}")
    all_dpo = []
    for row in new:
        if row["rejected"]:
            all_dpo.append(row)
    holdout_hashes, holdout_counts = holdout_prompt_hashes()
    leaks = [r["id"] for r in old + new if digest(r.get("messages", [])) in holdout_hashes]
    gate_counts = Counter()
    for row in new:
        gate_counts["chosen_pass" if not row["chosen_faults"] else "chosen_fail"] += 1
        if row["rejected"]:
            actual = target_error(row, row["rejected"])
            row["target_faults"] = actual
            target = row["target_defect"]
            matched = len(actual) == 1 and (actual[0] == target or target == "count_or_membership" and actual[0] in {"count_or_membership", "duplicate_poi"})
            row["rejected_target_match"] = matched
            gate_counts["rejected_target_match"] += int(matched)
            gate_counts["rejected_target_miss"] += int(not matched)
    tokenizer = make_tokenizer()
    length_rows, length_errors = [], []
    for row in dpo_cleaned + all_dpo:
        chosen = row["chosen"]
        rejected = row["rejected"]
        if not chosen or not rejected: continue
        clen = len(tokenizer.encode(json.dumps(chosen, ensure_ascii=False), add_special_tokens=False))
        rlen = len(tokenizer.encode(json.dumps(rejected, ensure_ascii=False), add_special_tokens=False))
        ratio = abs(clen-rlen)/max(clen,1)
        length_rows.append({"id": row["id"], "chosen_tokens": clen, "rejected_tokens": rlen, "delta_ratio": ratio})
    avg_c = sum(x["chosen_tokens"] for x in length_rows) / max(len(length_rows), 1)
    avg_r = sum(x["rejected_tokens"] for x in length_rows) / max(len(length_rows), 1)
    avg_delta = abs(avg_c-avg_r)/max(avg_c,1)
    mean_pair_delta = sum(x["delta_ratio"] for x in length_rows) / max(len(length_rows), 1)
    pairs_over_5pct = sum(x["delta_ratio"] > 0.05 for x in length_rows)
    report = {
        "schema_version": "c-post-train-v1", "old_rows_kept": len(old), "old_rows_quarantined": len(quarantined_ids), "old_dpo_pairs": len(dpo_cleaned), "new_contexts": len(new),
        "new_dpo_pairs": len(all_dpo), "new_category_counts": dict(Counter(r["category"] for r in new)),
        "legacy_repairs": len(fixes), "legacy_quarantined_due_to_rule_faults": len(quarantined_ids), "legacy_quarantined_ids": quarantined_ids, "legacy_dpo_unintended_fault_sides": len(dpo_faults), "legacy_dpo_fault_examples": dpo_faults[:30],
        "new_gate_counts": dict(gate_counts), "holdout_prompt_hash_counts": holdout_counts,
        "holdout_prompt_intersections": len(leaks), "holdout_intersection_ids": leaks,
        "history_sft_pass_100_percent": all(not r["cleaning_faults"] for r in old),
        "history_dpo_chosen_aligned_100_percent": not dpo_faults and len(dpo_cleaned) == len(old),
        "new_chosen_pass_100_percent": all(not r["chosen_faults"] for r in new),
        "new_rejected_target_hit_100_percent": all(r.get("rejected_target_match") for r in all_dpo),
        "dpo_pair_count_in_length_audit": len(length_rows), "mean_chosen_tokens": avg_c, "mean_rejected_tokens": avg_r,
        "aggregate_mean_token_gap_ratio": avg_delta, "mean_pair_delta_ratio": mean_pair_delta,
        "pairs_over_5pct": pairs_over_5pct, "length_gate_pass_lt_5pct": avg_delta < 0.05,
        "training_started": False, "teacher_api_calls": 0,
        "caveats": ["Rejected defect identity is checked by deterministic error codes; no subjective preference claims are made.",
                     "Historical faults are quarantined in the report; source files remain immutable."]}
    OUT.mkdir(parents=True)
    dump(OUT / f"legacy_cleaned_train_{len(old)}.json", old)
    dump(OUT / "legacy_cleaned_dpo_346.json", dpo_cleaned)
    dump(OUT / "c_new_contexts_90.json", new)
    dump(OUT / "c_dpo_pairs.json", all_dpo)
    dump(OUT / "dpo_length_audit.json", length_rows)
    dump(OUT / "legacy_time_repairs.json", fixes)
    dump(OUT / "gate_report.json", report)
    (OUT / "gate_report.md").write_text("# C 数据冻结前门禁\n\n" + "\n".join(f"- **{k}**: `{v}`" for k, v in report.items()), encoding="utf-8")
    sft_rows = old + new
    sft_export = [{"conversations": [{"from": {"system": "system", "user": "human"}[m["role"]], "value": m["content"]} for m in r["messages"]]
        + [{"from": "gpt", "value": json.dumps(r["chosen"], ensure_ascii=False)}]} for r in sft_rows]
    dpo_export = [{"conversations": [{"from": {"system": "system", "user": "human"}[m["role"]], "value": m["content"]} for m in r["messages"]],
        "chosen": {"from": "gpt", "value": json.dumps(r["chosen"], ensure_ascii=False)},
        "rejected": {"from": "gpt", "value": json.dumps(r["rejected"], ensure_ascii=False)}} for r in dpo_cleaned + all_dpo]
    dump(OUT / "c_sft_train.json", sft_export)
    dump(OUT / "c_dpo_train.json", dpo_export)
    manifest_files = [p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"]
    manifest = {"stage": "local_data_gates_passed" if all([report["history_sft_pass_100_percent"], report["history_dpo_chosen_aligned_100_percent"], report["new_chosen_pass_100_percent"], report["new_rejected_target_hit_100_percent"], report["holdout_prompt_intersections"] == 0, report["length_gate_pass_lt_5pct"]]) else "not_frozen_gate_failure",
        "training_started": False, "frozen": False, "sizes": {"sft": len(sft_export), "dpo": len(dpo_export)},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifest_files}}
    # The artifacts are immutable once built; training authorization remains separate.
    manifest["frozen"] = manifest["stage"] == "local_data_gates_passed"
    dump(OUT / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
