"""Versioned task rubrics and provider-fault probes, independent of Reviewer votes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

from app.planning import nodes
from app.planning.schemas import TravelPlanState
from app.evaluation.route_interest import route_key

ROOT = Path(__file__).resolve().parents[2]
VERSION = "task-robustness-v1"


def load_assets():
    source = json.loads((ROOT / "knowledge/route_interest_v1/cases.json").read_text(encoding="utf-8"))
    maps = json.loads((ROOT / "knowledge/route_interest_v1/maps.json").read_text(encoding="utf-8"))["routes"]
    return source, maps


def weather(bad=False, label=None):
    return [{"date": "2026-10-15", "is_bad": bad, "day_weather": label or ("雨" if bad else "晴"),
             "night_weather": label or ("雨" if bad else "晴"), "day_temp": "24", "night_temp": "20"}]


def build_cases():
    source, _ = load_assets()
    tasks, robust = [], []
    for city in ("南京", "上海", "三亚"):
        originals = [c for c in source if c["destination"] == city]
        pool = copy.deepcopy(originals[0]["pois"])
        inside = [p["name"] for p in pool if p.get("indoor") is True]
        outside = [p["name"] for p in pool if p.get("indoor") is False]
        interest = originals[0]
        acceptable = interest["acceptable_pois"]
        base = {"destination": city, "days": 1, "max_per_day": 3, "pois": pool,
                "travel_start_date": "2026-10-15", "travel_end_date": "2026-10-15",
                "weather_forecast": weather(), "candidate_pool_status": "sufficient"}
        profiles = [
            ("interest", interest["query"], {"acceptable": acceptable, "min_match": 1}, {}),
            ("excluded", f"想参观{acceptable[0]}，不要安排“{outside[0]}”", {"acceptable": acceptable, "min_match": 1, "excluded": [outside[0]]}, {}),
            ("slow", "慢节奏，最多两个景点，每个至少游玩90分钟", {"max_count": 2, "min_visit": 90}, {"habit_preference": "慢节奏"}),
            ("late", "睡到自然醒，所有景点10:00以后开始", {"earliest": 600}, {"habit_preference": "睡到自然醒"}),
            ("family", f"亲子轻松游，重点参观“{acceptable[0]}”，至少游玩90分钟", {"required": [acceptable[0]], "min_visit": 90, "max_count": 2}, {}),
            ("rain", "下雨，只安排室内景点，至少一个，不要户外", {"indoor_only": True}, {"weather_forecast": weather(True), "rain_indoor_priority": True}),
            ("nature", "喜欢自然风光，只选择户外景点，不要室内展览", {"acceptable": outside, "min_match": 1, "excluded": inside}, {}),
            ("window", "必须在10:00到16:00内游玩，至少两个不同景点，每个至少60分钟", {"earliest": 600, "latest": 960, "min_count": 2, "min_visit": 60}, {}),
            ("choice", f"喜欢“{acceptable[0]}”，同时希望安排一个户外景点，最多两个", {"required": [acceptable[0]], "acceptable": outside, "min_match": 1, "max_count": 2}, {}),
            ("elder", "带老人慢节奏，每天只安排一个景点，至少游玩两小时，10:00后开始", {"max_count": 1, "min_visit": 120, "earliest": 600}, {"habit_preference": "慢节奏", "max_per_day": 1}),
        ]
        if not inside:
            profiles[5] = ("weather_window", "晴天想看海岸景观，只安排一个景点，至少游览90分钟", {"acceptable": outside, "min_match": 1, "max_count": 1, "min_visit": 90}, {})
        for label, request, rubric, extra in profiles:
            state = copy.deepcopy({**base, **extra})
            state["query"] = request if label == "interest" else f"{city}2026年10月15日一日游。{request}。"
            state["attraction_preference"] = request
            tasks.append({"id": f"task-{city}-{label}", "suite": "task", "category": label,
                          "state": state, "rubric": {"min_count": 1, **rubric},
                          "label_status": "author_draft", "evidence_case": interest["id"],
                          "scope": "fixed_real_candidate_pool_and_saved_road_responses"})
    map_faults = ["timeout", "rate_limit", "exception", "empty", "partial", "all_missing",
                  "missing_duration", "walking_unverified", "slow_road", "recovered"]
    weather_changes = ["rain", "heavy_rain", "thunder", "snow", "unknown", "rain_indoor",
                       "rain_slow", "sun_after_rain", "mixed", "rain_with_exclusion"]
    modifications = ["exclude", "required", "late", "slow", "indoor", "short_window",
                     "replace", "history_conflict", "impossible_window", "unknown_poi"]
    for category, variants in [("map", map_faults), ("weather", weather_changes), ("modification", modifications)]:
        for i, variant in enumerate(variants):
            case = copy.deepcopy(tasks[(i % 3) * 10])
            pool = case["state"]["pois"]
            indoor = next((p["name"] for p in pool if p.get("indoor") is True), pool[0]["name"])
            outdoor = next(p["name"] for p in pool if p.get("indoor") is False)
            case.update(id=f"robust-{category}-{i+1:02d}", suite="robustness", category=category,
                        fault=variant, rubric={"min_count": 1}, expected="recover_or_explain")
            case["state"]["query"] = f'{case["state"]["destination"]}2026年10月15日一日游，最多两个不同景点。'
            if category == "map":
                case["rubric"].update(min_count=2, max_count=2)
                if variant == "walking_unverified":
                    case["state"].update(max_walking_km=2, query=case["state"]["query"]+"每段步行不超过2公里。")
                case["assertion"] = "道路失败不可当作已核验；严格通勤约束未核验不可算成功"
            elif category == "weather":
                case["assertion"] = "新天气覆盖旧天气；雨雪仅室内要求不可继续输出旧户外路线"
                case["weather_after"] = [] if variant == "unknown" else weather(variant != "sun_after_rain", {"thunder": "雷阵雨", "snow": "雪", "heavy_rain": "暴雨"}.get(variant))
                if variant != "sun_after_rain" and variant != "unknown": case["rubric"]["indoor_only"] = True
                if case["rubric"].get("indoor_only") and not any(p.get("indoor") is True for p in pool): case["expected"] = "reject"
                case["modification"] = "天气变了，下雨或下雪时只能安排室内景点，请重新规划。"
                if variant == "sun_after_rain":
                    case["state"]["weather_forecast"] = weather(True)
                    case["modification"] = f"现在转晴，请改为游览“{outdoor}”。"
                    case["rubric"]["required"] = [outdoor]
                if variant == "rain_slow": case["rubric"].update(max_count=2, min_visit=90)
                if variant == "rain_with_exclusion":
                    case["rubric"]["excluded"] = [indoor]
                    case["modification"] += f"不要安排“{indoor}”。"
                if variant == "unknown": case["modification"] = "天气预报暂不可用，不能把未知天气说成晴天。"
            else:
                mods = {
                    "exclude": (f"不要“{outdoor}”，改为室内游览", {"excluded": [outdoor], "indoor_only": True}),
                    "required": (f"必须加入“{indoor}”", {"required": [indoor]}),
                    "late": ("改为10:00后开始，不能早起", {"earliest": 600}),
                    "slow": ("改为只游览一个景点，至少两小时", {"max_count": 1, "min_visit": 120}),
                    "indoor": ("改为只能安排室内景点", {"indoor_only": True}),
                    "short_window": ("改为10:00到12:00之间游览一个景点", {"earliest": 600, "latest": 720, "max_count": 1}),
                    "replace": (f"把“{outdoor}”换成“{indoor}”", {"excluded": [outdoor], "required": [indoor]}),
                    "history_conflict": (f"之前喜欢户外，现在不要“{outdoor}”，只看室内", {"excluded": [outdoor], "indoor_only": True}),
                    "impossible_window": ("只能在凌晨02:00到03:00参观博物馆，禁止补充候选", {"earliest": 120, "latest": 180}),
                    "unknown_poi": ("必须游览“测试不存在的月球旅行馆”，禁止增加候选", {"required": ["测试不存在的月球旅行馆"]}),
                }
                case["modification"], rules = mods[variant]
                case["rubric"].update(rules)
                if rules.get("indoor_only") and not any(p.get("indoor") is True for p in pool): case["expected"] = "reject"
                case["assertion"] = "本轮修改优先；新路线重新审核，不沿用旧批准结论"
                if variant in ("unknown_poi", "impossible_window"):
                    case["expected"] = "reject"
                    if variant == "impossible_window":
                        case["state"]["pois"] = [p for p in pool if p["name"] == indoor]
            robust.append(case)
    return {"version": VERSION, "tasks": tasks, "robustness": robust}


def clock(value):
    hours, minutes = map(int, str(value).split(":"))
    if not 0 <= hours <= 24 or not 0 <= minutes < 60 or (hours == 24 and minutes):
        raise ValueError("invalid clock")
    return hours * 60 + minutes


def grade(state, case, maps, error=None):
    """All declared criteria must pass; Reviewer approval is NOT a gold label."""
    hard, soft = [], []
    final = state.final_plan or {}
    route = [{"day": d.get("day"), "spots": [s for s in d.get("timeline", []) if s.get("type") == "attraction"]}
             for d in final.get("days", [])]
    if not final: hard.append("missing_final_output")
    current = state.model_copy(update={"route": route})
    flags = nodes._route_risk_flags(current)
    hard.extend(f for f in flags if f not in {"opening_time_unknown", "user_modification", "long_road_leg"})
    names = [s.get("name") for d in route for s in d["spots"]]
    rubric = case["rubric"]
    if len(names) < rubric.get("min_count", 1): hard.append("required_count")
    if not set(rubric.get("required", [])).issubset(names): hard.append("required_poi_missing")
    if set(rubric.get("excluded", [])) & set(names): hard.append("excluded_poi")
    if len(names) > rubric.get("max_count", 100): soft.append("pace_count")
    matched = set(names) & set(rubric.get("acceptable", names))
    if len(matched) < rubric.get("min_match", 0): soft.append("interest_mismatch")
    by_name = {p["name"]: p for p in state.pois}
    if rubric.get("indoor_only") and any(by_name.get(n, {}).get("indoor") is not True for n in names): hard.append("indoor_required")
    for day in route:
        for spot in day["spots"]:
            try:
                start, end = clock(spot.get("start_time")), clock(spot.get("end_time"))
                if start < rubric.get("earliest", 0) or end > rubric.get("latest", 1440): hard.append("requested_window")
                if end - start < rubric.get("min_visit", 0): soft.append("visit_too_short")
            except (ValueError, TypeError): hard.append("invalid_time")
        for a, b in zip(day["spots"], day["spots"][1:]):
            if a.get("name") not in by_name or b.get("name") not in by_name: continue
            mode = "walk" if state.max_walking_km is not None else "drive"
            # Use delivered road evidence, not the healthy replay hidden behind a fault.
            leg = next((l for l in state.route_distance_legs if l.get("day") == day["day"] and l.get("from") == a["name"] and l.get("to") == b["name"]), None)
            if not leg or leg.get("duration_min") is None:
                hard.append("road_unverified")
                continue
            try:
                if clock(b.get("start_time")) - clock(a.get("end_time")) < float(leg["duration_min"]): hard.append("commute_gap")
            except (ValueError, TypeError): hard.append("invalid_road_time")
    if state.time_violations: hard.append("unresolved_time")
    approved = final.get("approved") is True
    messages = final.get("route_issues", [])
    explained = any("无法生成满足" in str(m) or "未生成可用路线" in str(m) for m in messages)
    success = not error and approved and not hard and not soft
    rejection = not error and bool(final) and not approved and explained
    unknown_weather_disclosed = bool(state.weather_note) if case.get("fault") == "unknown" else True
    robustness_ok = rejection if case.get("expected") == "reject" else (success or rejection)
    if case.get("fault") == "unknown": robustness_ok = robustness_ok and unknown_weather_disclosed
    return {"task_success": success, "hard_pass": not hard, "soft_pass": not soft,
            "hard_violations": sorted(set(hard)), "soft_violations": sorted(set(soft)),
            "selected": names, "approved": approved, "rework": max(0, state.review_round - 1),
            "safe_explained_rejection": rejection, "robustness_ok": robustness_ok,
            "unsafe_release": approved and bool(hard or soft), "user_messages": messages,
            "interest_precision": len(matched) / len(names) if names else 0,
            "unknown_weather_disclosed": unknown_weather_disclosed, "error": error}


def fault_distance(maps, fault, counters):
    """Inject transport failures at the provider boundary; never patch approval."""
    def distance(origin, destination, key, mode="drive"):
        counters["calls"] = counters.get("calls", 0) + 1
        index = counters["calls"]
        if fault in ("timeout", "rate_limit", "exception"):
            raise TimeoutError("injected map timeout") if fault == "timeout" else RuntimeError("injected map " + fault)
        if fault in ("empty", "all_missing", "walking_unverified") or (fault == "partial" and index % 2): return None
        if fault == "recovered" and index == 1: raise TimeoutError("injected first lookup timeout")
        result = copy.deepcopy(maps.get(route_key(origin, destination, mode)))
        if result and fault == "missing_duration": result["duration_min"] = None
        if result and fault == "slow_road": result["duration_min"] = 180
        return result
    return distance


def offline_probe(case):
    """Real rule/provider nodes with injected routes. Not a model success rate."""
    _, maps = load_assets()
    state = TravelPlanState(**case["state"])
    names = [p["name"] for p in state.pois[:2]]
    state.route = [{"day": 1, "spots": [{"name": n, "period": "morning" if i == 0 else "afternoon",
                   "start_time": "10:00" if i == 0 else "14:00", "end_time": "12:00" if i == 0 else "16:00"} for i, n in enumerate(names)]}]
    counters = {}
    with patch.object(nodes, "amap_key", return_value="replay"), patch.object(nodes, "plan_route_distance", side_effect=fault_distance(maps, case.get("fault") if case["category"] == "map" else None, counters)):
        update = nodes.route_distance_check_node(state)
    state = state.model_copy(update=update)
    if case["category"] == "weather": state.weather_forecast = case["weather_after"]
    if case.get("modification"): state.modification_notes = case["modification"]
    gate = nodes.route_risk_gate_node(state)
    return {"id": case["id"], "category": case["category"], "fault": case.get("fault"),
            "provider_calls": counters.get("calls", 0), "distance_note": update["route_distance_note"],
            "road_legs": state.route_distance_legs, "gate": gate,
            "node_execution_ok": True, "model_executed": False,
            "task_success": None}
