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


def semantic_route_violations(route, case, maps):
    """Predeclared candidate labels, never supplied to the model as gold answers."""
    import itertools
    rubric = case.get('semantic_rubric', {})
    spots = [spot for day in route for spot in day.get('spots', [])]
    violations = []
    if set(rubric.get('unsuitable_pois', [])) & {s['name'] for s in spots}:
        violations.append(rubric.get('violation', 'semantic_mismatch'))
    if rubric.get('avoid_outdoor_noon'):
        outdoors = {p['name'] for p in case['state']['pois'] if p.get('indoor') is False}
        for spot in spots:
            try:
                if spot['name'] in outdoors and clock(spot['start_time']) < 840 and clock(spot['end_time']) > 660:
                    violations.append('heat_exposure')
            except (ValueError, TypeError):
                continue
    if rubric.get('max_detour_ratio') and len(spots) >= 3:
        locations = {p['name']: p['location'] for p in case['state']['pois']}
        names = [s['name'] for s in spots]
        def length(order):
            values = [maps.get(route_key(locations[a], locations[b], 'drive'), {}).get('distance_km')
                      for a, b in zip(order, order[1:])]
            if any(v is None for v in values):
                raise ValueError('semantic detour scoring requires all saved road distances')
            return sum(values)
        best = min(length(order) for order in itertools.permutations(names))
        if best and length(names) / best > rubric['max_detour_ratio']:
            violations.append('avoidable_backtracking')
    if rubric.get('physical_time_deadlock'):
        required = set(rubric.get('required', []))
        names = [str(s.get('name') or '') for s in spots]
        locations = {p['name']: p['location'] for p in case['state']['pois']}
        earliest = int(rubric.get('earliest', 0))
        latest = int(rubric.get('latest', 1440))
        min_visit = int(rubric.get('min_visit', 0))
        if required.issubset(names) and len(required) > 1:
            total_visit = len(required) * min_visit
            road_minutes = []
            for origin, destination in itertools.permutations(required, 2):
                evidence = maps.get(route_key(locations[origin], locations[destination], 'drive'), {})
                if evidence.get('duration_min') is not None:
                    road_minutes.append(float(evidence['duration_min']))
            if total_visit + (min(road_minutes) if road_minutes else 0) > latest - earliest:
                violations.append('physical_time_deadlock')
    if rubric.get('candidate_pool_indoor_only_unavailable'):
        has_indoor = any(poi.get('indoor') is True for poi in case['state'].get('pois', []))
        if not has_indoor and spots:
            violations.append('candidate_pool_preference_conflict')
    return sorted(set(violations))


def build_soft_review_cases():
    """Controlled drafts; all eight cases were confirmed by the project owner."""
    bases = {c['state']['destination']: c for c in build_cases()['tasks']}
    specs = [
        ('stairs', '南京', 'task-南京-family',
         '带75岁腿脚不便的老人，不能爬连续台阶，也不参加登高活动；只选平缓、可轻松停留的景点。慢节奏游览。',
         [('中山陵景区', '10:00', '12:00')],
         {'unsuitable_pois': ['中山陵景区'], 'violation': 'mobility_intensity_conflict'}, None),
        ('heat', '三亚', 'task-三亚-family',
         '预报晴天38度，怕热且不耐暴晒，正午想在遮阴处休息，户外观景尽量放在凉快时段。',
         [('鹿回头风景区', '12:00', '13:30')], {'avoid_outdoor_noon': True}, None),
        ('commercial', '上海', 'task-上海-choice',
         '已经对商业化街区和人造主题乐园审美疲劳，这次更想看自然和历史真迹，不想再看仿造布景。',
         [('上海迪士尼度假区', '10:00', '12:00')],
         {'unsuitable_pois': ['上海迪士尼度假区'], 'violation': 'implicit_preference_conflict'}, None),
        ('backtrack', '上海', 'task-上海-family',
         '这三个候选都想去，每个参观至少60分钟，路线按相邻区域串起来，别在城市两端来回折返。',
         [('上海博物馆东馆', '10:00', '11:00'), ('上海动物园', '13:00', '14:00'),
          ('上海海洋水族馆', '16:00', '17:00')], {'max_detour_ratio': 1.3}, 'only_seed_pois'),
        ('infeasible', '南京', 'task-南京-choice',
         '带75岁腿脚不便老人，不能爬连续台阶和登高；只允许候选中无需登高的活动，不允许补充候选或假称有无障碍替代。',
         [('中山陵景区', '10:00', '12:00')],
         {'unsuitable_pois': ['中山陵景区'], 'violation': 'mobility_intensity_conflict'}, 'only_seed_pois'),
        ('time-deadlock', '南京', None,
         '只有10:00到12:00两小时窗口，必须去南京博物院和南京欢乐谷，每处至少游玩60分钟，两个景点都不能删减，必须保留真实道路通勤时间，禁止补查和新增候选。',
         [('南京博物院', '10:00', '11:00'), ('南京欢乐谷', '11:00', '12:00')],
         {'physical_time_deadlock': True, 'required': ['南京博物院', '南京欢乐谷'],
          'earliest': 600, 'latest': 720, 'min_visit': 60, 'violation': 'physical_time_deadlock'},
         'only_seed_pois'),
        ('indoor-pool-conflict', '上海', None,
         '只安排纯室内历史场馆或博物馆，禁止安排户外景点；当前候选池只有户外自然景点，禁止补查和新增候选。',
         [('上海动物园', '10:00', '12:00')],
         {'candidate_pool_indoor_only_unavailable': True,
          'unsuitable_pois': ['上海动物园'], 'violation': 'candidate_pool_preference_conflict'},
         'only_seed_pois'),
        ('clean', '南京', None,
         '带老人轻松游，想在玄武湖平缓入口附近散步休息，只安排一个景点。',
         [('玄武湖景区', '10:00', '12:00')], {}, 'only_seed_pois'),
    ]
    output = []
    for label, city, replacement, request, schedule, semantic, restriction in specs:
        case = copy.deepcopy(bases[city])
        state = case['state']
        state.update(query=f'{city}2026年10月15日一日游。{request}',
                     attraction_preference=request, habit_preference='', max_per_day=3,
                     weather_forecast=weather(), rain_indoor_priority=False)
        if label in ('stairs', 'infeasible'):
            state['query'] += '本次活动条件：中山陵安排指登临陵寝的台阶游览，不包含山脚平地替代活动。'
        if label == 'heat':
            state['weather_forecast'][0].update(day_temp='38', night_temp='28')
        if restriction:
            state['pois'] = [p for p in state['pois'] if p['name'] in {x[0] for x in schedule}]
        route = [{'day': 1, 'spots': [{'name': name, 'start_time': start, 'end_time': end,
                  'period': 'morning' if clock(start) < 720 else 'afternoon'} for name, start, end in schedule]}]
        if label == 'time-deadlock':
            case['state']['pois'] = [p for p in case['state']['pois']
                                     if p['name'] in {'南京博物院', '南京欢乐谷'}]
        elif label == 'indoor-pool-conflict':
            case['state']['pois'] = [p for p in case['state']['pois']
                                     if p.get('indoor') is False]
        expected_hard = []
        if label == 'time-deadlock':
            expected_hard = ['commute_time_conflict']
        elif label == 'indoor-pool-conflict':
            expected_hard = ['candidate_pool_preference_conflict', 'indoor_constraint']
        case.update(id='soft-review-'+label, suite='soft_review', category=label,
                    replaces=replacement, label_status='verified_v2',
                    draft_origin='author_injected_controlled_defect' if label != 'clean' else 'author_clean_control',
                    expected='reject' if label in {'infeasible', 'time-deadlock', 'indoor-pool-conflict'} else 'repair_or_preserve',
                    expected_hard_violations=expected_hard,
                    rubric={'min_count': len(schedule), 'min_visit': 60},
                    semantic_rubric=semantic, injected_route=route,
                    scope='controlled_audit_tail_not_natural_planner_success',
                    annotation_reason=request)
        if label == 'time-deadlock':
            case['rubric'].update(required=['南京博物院', '南京欢乐谷'], earliest=600, latest=720)
            case['impossibility_reason'] = '两小时窗口不足以覆盖两个必去景点各60分钟及南京博物院至南京欢乐谷的道路通勤时间，且用户禁止删减景点。'
            case['required_refusal_phrase'] = '景点间预留的通勤时间不足'
        elif label == 'indoor-pool-conflict':
            case['rubric'].update(min_count=1, indoor_only=True)
            case['impossibility_reason'] = '候选池中没有室内历史场馆，用户又禁止补查和新增候选，因此无法生成满足要求的路线。'
            case['required_refusal_phrase'] = '候选池中没有满足纯室内要求的景点'
        elif label == 'infeasible':
            case['impossibility_reason'] = '候选池中没有满足老人不能登高要求的可排景点，且用户禁止补查和替代。'
        output.append(case)
    return output


def grade_soft_review(state, case, maps, error=None):
    result = grade(state, case, maps, error)
    route = [{'spots': [s for s in day.get('timeline', []) if s.get('type') == 'attraction']}
             for day in (state.final_plan or {}).get('days', [])]
    extra = semantic_route_violations(route, case, maps)
    result['soft_violations'] = sorted(set(result['soft_violations'] + extra))
    result['soft_pass'] = not result['soft_violations']
    result['task_success'] = result['task_success'] and result['soft_pass']
    result['unsafe_release'] = result['approved'] and bool(result['hard_violations'] or result['soft_violations'])
    if case['expected'] == 'reject':
        refusal_text = ' '.join(str(message) for message in result['user_messages'])
        result['safe_explained_rejection'] = bool(
            result['safe_explained_rejection']
            and result['approved'] is False
            and not result['selected']
            and (not case.get('required_refusal_phrase')
                 or case['required_refusal_phrase'] in refusal_text)
        )
    result['expected_behavior_pass'] = (result['safe_explained_rejection'] if case['expected'] == 'reject'
                                        else result['task_success'])
    return result


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
