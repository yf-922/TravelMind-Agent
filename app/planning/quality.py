"""Deterministic route quality, snapshot and result-envelope helpers.

The Reviewer may describe soft quality, but route selection is performed here
so that retries are comparable and hard constraints cannot be traded for a
high language-model score.
"""

from __future__ import annotations

import re
from typing import Any

from app.planning.helpers import open_time_violations, unknown_spots


def _clock(value: Any) -> int | None:
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(value or ""))
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _requested_names(state: Any) -> list[str]:
    # Keep this parser local to avoid importing candidate_react (which imports
    # TravelPlanState and would create a module cycle).
    from app.planning.helpers import extract_explicit_place_requests

    text = "\n".join(filter(None, [state.query, state.modification_notes]))
    excluded = set(re.findall(r'(?:不要|不去|排除|不想去|不喜欢)\s*[“\"「]([^”\"」]{2,60})[”\"」]', text))
    return [name for name in extract_explicit_place_requests(text) if name not in excluded]


def _relaxed(state: Any, key: str, name: str | None = None) -> bool:
    value = (state.relaxed_constraints or {}).get(key, [])
    if isinstance(value, bool):
        return value
    if name is None:
        return bool(value)
    return name in value


def hard_constraint_report(state: Any) -> dict[str, list[dict[str, str]]]:
    """Return L0/L1 violations after applying only explicit relaxations."""
    route = state.route or []
    pois = state.pois or []
    pool_names = {str(item.get("name") or "") for item in pois}
    l0: list[dict[str, str]] = []
    l1: list[dict[str, str]] = []

    for name in unknown_spots(route, pois):
        l0.append({"code": "unknown_poi", "message": f"路线包含候选池外景点：{name}"})

    requested = _requested_names(state)
    placed = {str(spot.get("name") or "") for day in route for spot in (day.get("spots") or [])}
    dropped = set((state.relaxed_constraints or {}).get("dropped_must_visit", []) or [])
    unverified = set((state.relaxed_constraints or {}).get("unverified_must_visit", []) or [])
    for name in requested:
        if name not in placed and name not in dropped and name not in unverified:
            l0.append({"code": "missing_must_visit", "message": f"未安排用户点名景点：{name}"})

    if len(route) != int(state.days or 0):
        l0.append({"code": "route_structure", "message": f"路线天数为 {len(route)}，要求 {state.days} 天"})
    if any(not (day.get("spots") or []) for day in route):
        l0.append({"code": "route_structure", "message": "存在没有景点的行程日"})

    indoor_only = bool(re.search(r"(?:只|仅)(?:想|要)?(?:参观|安排|去|玩)?[^，。；\n]{0,12}室内", "\n".join(filter(None, [state.query, state.modification_notes]))))
    if indoor_only and not _relaxed(state, "relaxed_indoor_only"):
        indoor_map = {str(item.get("name")): item.get("indoor") for item in pois}
        # An indoor-only request with no confirmed indoor candidate is a
        # candidate-pool infeasibility, rather than a route-quality warning.
        # Keep this deterministic so a Reviewer cannot paper over an empty
        # feasible set with a high semantic score.
        if pois and not any(value is True for value in indoor_map.values()):
            l0.append({
                "code": "candidate_pool_preference_conflict",
                "message": "候选池中没有满足纯室内要求的景点",
            })
        outdoor = [name for name in placed if indoor_map.get(name) is not True]
        for name in outdoor:
            l0.append({"code": "indoor_constraint", "message": f"仅室内要求下安排了非确定性室内景点：{name}"})

    # A request may make the schedule physically impossible before any model
    # preference judgement is involved.  Detect the common explicit form
    # (fixed window + minimum visit time + all named places required) and put
    # it in L0.  This prevents three identical Planner retries from being
    # reported as a merely partial route.
    query_text = "\n".join(filter(None, [state.query, state.modification_notes]))
    window = re.search(r"(?:只有|仅有)\s*(\d{1,2}):(\d{2})\s*(?:到|至|-)\s*(\d{1,2}):(\d{2})\s*(?:两小时|时间窗口|窗口)?", query_text)
    # Keep the numeric part independent from the classifier wording so this
    # remains robust to colloquial variants ("每处至少", "每个至少") and
    # legacy fixture encoding.
    min_visit_match = re.search(r"至少\s*(?:游玩|参观|停留|安排)?\s*(\d+)\s*分钟", query_text)
    if window and min_visit_match and len(placed) >= 2 and ("必须" in query_text or "不能删减" in query_text):
        start = int(window.group(1)) * 60 + int(window.group(2))
        end = int(window.group(3)) * 60 + int(window.group(4))
        min_visit = int(min_visit_match.group(1))
        required_names = [name for name in requested if name in placed] or list(placed)
        route_minutes = len(required_names) * min_visit
        leg_minutes = 0.0
        leg_map = {(int(leg.get("day") or 0), str(leg.get("from") or ""), str(leg.get("to") or "")): leg
                   for leg in (state.route_distance_legs or [])}
        for day in route:
            for prev, cur in zip(day.get("spots") or [], (day.get("spots") or [])[1:]):
                leg = leg_map.get((int(day.get("day") or 0), str(prev.get("name") or ""), str(cur.get("name") or "")))
                if leg and leg.get("duration_min") is not None:
                    leg_minutes += float(leg["duration_min"])
        if route_minutes + leg_minutes > max(0, end - start):
            # A physically tight window is still repairable when the current
            # draft contains every required stop and every day has content:
            # the Planner can move times, change transport, or return a
            # PARTIAL_SUCCESS with an explicit warning.  L0 remains reserved
            # for the already-computed missing_must_visit / route_structure
            # failures above (a required stop was dropped or a day is empty).
            l1.append({
                "code": "physical_time_deadlock",
                "message": "景点间预留的通勤时间不足，当前时间窗口无法同时满足所有必去景点",
            })

    # Structural and deterministic schedule defects are L1: a new Planner
    # version can repair them without rejecting the user's request.
    names = [str(spot.get("name") or "") for day in route for spot in (day.get("spots") or [])]
    for name in sorted({name for name in names if name and names.count(name) > 1}):
        l1.append({"code": "duplicate_poi", "message": f"景点重复安排：{name}"})
    if open_time_violations(route, pois):
        l1.append({"code": "opening_time_conflict", "message": "当前时间窗口无法安排所选景点，存在开放时间或闭馆日冲突"})

    leg_map = {
        (int(leg.get("day") or 0), str(leg.get("from") or ""), str(leg.get("to") or "")): leg
        for leg in (state.route_distance_legs or [])
    }
    for day in route:
        spots = day.get("spots") or []
        for prev, cur in zip(spots, spots[1:]):
            key = (int(day.get("day") or 0), str(prev.get("name") or ""), str(cur.get("name") or ""))
            leg = leg_map.get(key)
            if state.max_walking_km is not None and (not leg or leg.get("mode") != "walk" or leg.get("distance_km") is None):
                l1.append({"code": "walking_constraint", "message": f"未核验步行路段：{prev.get('name')}→{cur.get('name')}"})
            elif state.max_walking_km is not None and float(leg.get("distance_km", 0)) > float(state.max_walking_km):
                l1.append({"code": "walking_constraint", "message": f"步行距离超限：{prev.get('name')}→{cur.get('name')}"})
            if leg and leg.get("duration_min") is not None:
                gap_start, prev_end = _clock(cur.get("start_time")), _clock(prev.get("end_time"))
                if gap_start is not None and prev_end is not None and gap_start - prev_end < float(leg["duration_min"]):
                    l1.append({"code": "commute_time_conflict", "message": f"当前时间窗口无法满足通勤时间：{prev.get('name')}→{cur.get('name')}"})

    habit = str(state.habit_preference or "")
    for day in route:
        spots = day.get("spots") or []
        if ("不喜欢早起" in habit or "睡到自然醒" in habit) and spots:
            start = _clock(spots[0].get("start_time"))
            if start is None or start < 600:
                l1.append({"code": "habit_constraint", "message": f"Day{day.get('day')} 首站早于 10:00"})
        if ("慢节奏" in habit or "每天景点别太多" in habit) and len(spots) > 2:
            l1.append({"code": "habit_constraint", "message": f"Day{day.get('day')} 景点数量偏多"})

    # Weather is a hard bottom line only when the user explicitly required
    # indoor travel. Otherwise it is a soft score dimension handled below.
    bad_dates = {str(item.get("date")) for item in (state.weather_forecast or []) if item.get("is_bad")}
    if bad_dates and bool(getattr(state, "rain_indoor_priority", False)) and not _relaxed(state, "relaxed_indoor_only"):
        date_by_day = {index + 1: str(item.get("date")) for index, item in enumerate(state.weather_forecast or [])}
        indoor_map = {str(item.get("name")): item.get("indoor") for item in pois}
        outdoor = [str(spot.get("name")) for day in route if date_by_day.get(day.get("day")) in bad_dates
                   for spot in (day.get("spots") or []) if indoor_map.get(str(spot.get("name"))) is False]
        if outdoor:
            l0.append({"code": "weather_outdoor_conflict", "message": "雨天室内优先要求下安排了室外景点"})

    return {"l0": l0, "l1": l1}


def compute_soft_score(state: Any, review: Any | None = None) -> tuple[float, dict[str, float]]:
    """Compute the 100-point deterministic hybrid score."""
    route = state.route or []
    placed = {str(spot.get("name") or "") for day in route for spot in (day.get("spots") or [])}
    requested = _requested_names(state)
    dropped = set((state.relaxed_constraints or {}).get("dropped_must_visit", []) or [])
    valid_requested = [name for name in requested if name not in dropped]
    named = 20.0 if not valid_requested else 20.0 * sum(name in placed for name in valid_requested) / len(valid_requested)

    pace = 20.0
    habit = str(state.habit_preference or "")
    for day in route:
        count = len(day.get("spots") or [])
        if ("慢节奏" in habit or "每天景点别太多" in habit) and count > 2:
            pace -= min(10.0, 5.0 * (count - 2))
        if count >= int(getattr(state, "max_per_day", 5) or 5):
            pace -= 5.0
    pace = max(0.0, pace)

    continuity = 15.0
    legs = list(state.route_distance_legs or [])
    if legs:
        durations = [float(item.get("duration_min")) for item in legs if item.get("duration_min") is not None]
        play_minutes = sum(max(0, (_clock(spot.get("end_time")) or 0) - (_clock(spot.get("start_time")) or 0))
                       for day in route for spot in (day.get("spots") or []))
        if play_minutes and sum(durations) / play_minutes > 0.35:
            continuity -= 5.0
    continuity -= min(10.0, 5.0 * max(0, len([x for x in legs if x.get("distance_km") and float(x["distance_km"]) > 20]) - 0))
    continuity = max(0.0, continuity)

    weather = 15.0
    bad_dates = {str(item.get("date")) for item in (state.weather_forecast or []) if item.get("is_bad")}
    if bad_dates:
        date_by_day = {index + 1: str(item.get("date")) for index, item in enumerate(state.weather_forecast or [])}
        indoor_map = {str(item.get("name")): item.get("indoor") for item in (state.pois or [])}
        total = outdoor = 0
        for day in route:
            if date_by_day.get(day.get("day")) in bad_dates:
                for spot in day.get("spots") or []:
                    total += 1
                    outdoor += indoor_map.get(str(spot.get("name"))) is False
        if total:
            weather = max(0.0, 15.0 - 15.0 * outdoor / total)
    if (state.relaxed_constraints or {}).get("relaxed_indoor_only"):
        weather = max(0.0, weather - 5.0)

    preference_map = {"HIGH": 25.0, "MEDIUM": 15.0, "LOW": 5.0}
    diversity_map = {"GOOD": 5.0, "FAIR": 2.0, "POOR": 0.0}
    preference_level = getattr(review, "preference_match_level", None) if review else None
    diversity_level = getattr(review, "diversity_and_dining_level", None) if review else None
    preference_level = preference_level or getattr(state, "reviewer_preference_match_level", "MEDIUM")
    diversity_level = diversity_level or getattr(state, "reviewer_diversity_and_dining_level", "FAIR")
    preference = preference_map.get(str(preference_level), 15.0)
    diversity = diversity_map.get(str(diversity_level), 2.0)
    breakdown = {"named_poi": named, "pace": pace, "continuity": continuity,
                 "weather": weather, "preference_match": preference, "diversity_and_dining": diversity}
    return round(sum(breakdown.values()), 2), breakdown


def append_route_snapshot(state: Any, review: Any | None = None) -> dict[str, Any]:
    report = hard_constraint_report(state)
    score, breakdown = compute_soft_score(state, review)
    iteration = max(1, int(state.review_round or 1))
    snapshot = {
        "iteration": iteration,
        "plan": {
            "route": state.route,
            "meal_slots": state.meal_slots,
            # Distance facts belong to this route version. Without carrying
            # them, choosing an older snapshot could accidentally validate it
            # against a later route's road legs.
            "route_distance_legs": state.route_distance_legs,
        },
        "l0_violations": report["l0"],
        "l1_violations": report["l1"],
        "soft_score": score,
        "score_breakdown": breakdown,
        "reviewer_approved": bool(state.approved),
        "reviewer_warnings": list(state.reviewer_issues or []),
        "rework_instructions": [str(state.route_modify_opinion)] if state.route_modify_opinion else [],
    }
    history = [item for item in state.route_history if int(item.get("iteration", 0)) != iteration]
    history.append(snapshot)
    history.sort(key=lambda item: int(item.get("iteration", 0)))
    logger = __import__("logging").getLogger(__name__)
    logger.info("[RouteSnapshot] iteration=%s soft_score=%.2f l0=%d l1=%d breakdown=%s",
                iteration, score, len(report["l0"]), len(report["l1"]), breakdown)
    return {"route_history": history, "score_info": {"current_score": score, "current_breakdown": breakdown}}


def choose_snapshot(state: Any) -> tuple[dict[str, Any] | None, str, list[str], list[dict[str, Any]]]:
    history = list(state.route_history or [])
    if not history:
        # Offline harnesses and legacy callers can invoke finalize directly.
        # Evaluate their current route once instead of treating the missing
        # bookkeeping field as a route failure.
        if state.route:
            report = hard_constraint_report(state)
            score, breakdown = compute_soft_score(state)
            history = [{
                "iteration": max(1, int(state.review_round or 1)),
                "plan": {"route": state.route, "meal_slots": state.meal_slots,
                         "route_distance_legs": state.route_distance_legs},
                "l0_violations": report["l0"], "l1_violations": report["l1"],
                "soft_score": score, "score_breakdown": breakdown,
                "reviewer_approved": bool(state.approved),
                "reviewer_warnings": list(state.reviewer_issues or []),
                "rework_instructions": [state.route_modify_opinion] if state.route_modify_opinion else [],
            }]
        else:
            return None, "FAILED", [], [{"code": "no_route_snapshot", "message": "当前无法生成满足全部要求的路线，请调整约束或补充候选后重试。", "stage": "L0_CONSTRAINT"}]
    tier1 = [x for x in history if not x.get("l0_violations") and not x.get("l1_violations")]
    if tier1:
        chosen = max(tier1, key=lambda x: (float(x.get("soft_score", 0)), int(x.get("iteration", 0))))
        warnings = list((state.relaxed_constraints or {}).get("relaxation_warnings", []) or [])
        warnings.extend(chosen.get("reviewer_warnings", []))
        return chosen, "SUCCESS_WITH_WARNINGS" if warnings else "SUCCESS", list(dict.fromkeys(warnings)), []
    tier2 = [x for x in history if not x.get("l0_violations")]
    if tier2:
        if int(getattr(state, "max_review_rounds", 3) or 0) == 0 and any(x.get("l1_violations") for x in tier2):
            chosen = tier2[-1]
            failures = [{**item, "stage": "L1_EXHAUSTED"} for item in chosen.get("l1_violations", [])]
            return chosen, "FAILED", [], failures or [{"code": "l1_unresolved", "message": "路线仍有未修复的排期问题", "stage": "L1_EXHAUSTED"}]
        chosen = min(tier2, key=lambda x: (len(x.get("l1_violations", [])), -float(x.get("soft_score", 0)), -int(x.get("iteration", 0))))
        warnings = list((state.relaxed_constraints or {}).get("relaxation_warnings", []) or [])
        warnings.extend(v.get("message", "") for v in chosen.get("l1_violations", []))
        warnings.extend(chosen.get("reviewer_warnings", []))
        return chosen, "PARTIAL_SUCCESS", list(dict.fromkeys(warnings)), []
    chosen = min(history, key=lambda x: (len(x.get("l0_violations", [])), -int(x.get("iteration", 0)), -float(x.get("soft_score", 0))))
    failures = [{**item, "stage": "L0_CONSTRAINT"} for item in chosen.get("l0_violations", [])]
    return chosen, "FAILED", [], failures or [{"code": "unresolvable_constraints", "message": "所有路线版本均未满足核心约束", "stage": "L0_CONSTRAINT"}]


def result_envelope(state: Any) -> dict[str, Any]:
    status = state.result_status or ("SUCCESS" if state.approved else "FAILED")
    legacy_unapproved = isinstance(state.final_plan, dict) and state.final_plan.get("approved") is False and "status" not in state.final_plan
    # Custom Supervisor test/adaptor finalizers may only return final_plan and
    # leave result_status at its default. Derive the envelope from snapshots
    # when available, and preserve the legacy approval signal otherwise.
    if state.route_history and status == "SUCCESS":
        _chosen, derived_status, derived_warnings, derived_failures = choose_snapshot(state)
        status = derived_status
        if not state.warnings:
            state.warnings = list(derived_warnings)
        if not state.failure_reasons:
            state.failure_reasons = list(derived_failures)
    elif not state.route_history and status == "SUCCESS" and not state.approved:
        status = "FAILED"
    if legacy_unapproved:
        status = "FAILED"
    if status == "SUCCESS" and (state.missing_fields or state.final_plan is None):
        status = "FAILED"
    plan = state.final_plan or {"days": [], "route_issues": []}
    plan.setdefault("days", [])
    plan["route_issues"] = list(state.warnings or plan.get("route_issues") or [])
    failure_reasons = list(state.failure_reasons or []) if status == "FAILED" else []
    warnings = list(state.warnings or []) if status != "FAILED" else []
    message = failure_reasons[0]["message"] if failure_reasons else (warnings[0] if warnings else None)
    failure_details = {"reasons": failure_reasons} if failure_reasons else None
    if status == "FAILED":
        failure_details = {**(failure_details or {}),
                           "risk_flags": list(getattr(state, "route_risk_flags", []) or []),
                           "time_violations": list(getattr(state, "time_violations", []) or []),
                           "reviewer_issues": list(getattr(state, "reviewer_issues", []) or [])}
    return {"status": status, "success": status != "FAILED", "plan": plan,
            "warnings": warnings, "failure_reasons": failure_reasons,
            "score_info": dict(state.score_info or {}), "message": message,
            "failure_details": failure_details,
            "history": list(state.history or []), "missing_fields": list(state.missing_fields or [])}
