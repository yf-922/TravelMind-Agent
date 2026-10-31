"""Shared, observable planning contract; never infer safety from a POI name."""
from datetime import timedelta
import math
import re

from app.planning.helpers import canonical_poi_name, extract_explicit_place_requests, normalize_poi_name, poi_name_matches

PROTOCOL = "planner-contract-v1"
HEAT_START, HEAT_END, HEAT_C = 660, 900, 38.0
_NUM = r"([1-9]\d*|十|[一二两三四五六七八九])"


def daily_limit(text):
    """Recognize explicit daily upper/exact counts, not total/visit durations."""
    values = []
    for match in re.finditer(r"(?<![第\d一二三四五六七八九十])(?:每天|每日|一天)\s*(?:最多|不超过|至多|恰好|只安排|只去)\s*(?:安排|游览|游玩|去)?\s*"+_NUM+r"\s*(?:(?:个|处)\s*(?:景点|(?=[，。；、！!？?\n]|$))|景点)", text or ""):
        value = match.group(1)
        values.append(int(value) if value.isdigit() else {"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}[value])
    return min(values) if values else None


def active_must_visit(state):
    """Current named requirements, minus explicit exclusions/verified relaxations.

    Count official aliases of one POI only once. Unconfirmed names are not
    silently dropped here; candidate validation owns that decision.
    """
    text = "\n".join(filter(None, [state.query, state.modification_notes]))
    excluded = set(re.findall(r'(?:不要|不去|排除|不想去|不喜欢)\s*[“\"「]([^”\"」]{2,60})[”\"」]', text))
    relaxed = state.relaxed_constraints or {}
    exempt = set(relaxed.get("dropped_must_visit", []) or []) | set(relaxed.get("unverified_must_visit", []) or [])
    names, identities = [], set()
    for name in extract_explicit_place_requests(text):
        if name in excluded or any(poi_name_matches(item, name, state.destination or "") for item in exempt):
            continue
        identity = canonical_poi_name(name, state.pois, state.destination or "") or name
        if identity not in identities:
            names.append(name)
            identities.add(identity)
    return names


def capacity_policy(state, route=None):
    """Derive capacity from source requirements, never trust a cached limit.

    A configured default may expand to fit up to five named stops per day.
    Explicit daily ceilings and server strict limits are never expanded.
    """
    user = daily_limit(state.modification_notes) or daily_limit(state.query)
    base = max(1, int(state.max_per_day))
    strict = bool(getattr(state, "max_per_day_strict", False)) or user is not None
    limit = min(base, user) if user is not None else base
    names = active_must_visit(state)
    days = max(1, int(state.days or 1))
    needed = math.ceil(len(names) / days)
    expanded = not strict and len(names) > days * limit and needed <= 5
    if expanded:
        limit = needed
    total = days * limit
    placed = set()
    for day in (state.route if route is None else route) or []:
        if not isinstance(day, dict) or not isinstance(day.get("spots") or [], list):
            continue  # Malformed structures are reported by the gate, not repaired here.
        for spot in day.get("spots") or []:
            if not isinstance(spot, dict):
                continue
            identity = canonical_poi_name(spot.get("name"), state.pois, state.destination or "")
            if identity:
                placed.add(identity)
    # Amap can return both a venue and its viewpoint with the same verified
    # requested-name alias. Keep their official identities distinct, but allow
    # either verified candidate to satisfy that alias when actually arranged.
    # An exact official request still refers only to that official candidate.
    options = {}
    for name in names:
        normalized = normalize_poi_name(name)
        exact = {str(p.get("name")) for p in state.pois
                 if normalize_poi_name(p.get("name")) == normalized}
        aliases = {str(p.get("name")) for p in state.pois
                   if any(normalize_poi_name(a) == normalized for a in (p.get("aliases") or []))}
        canonical = canonical_poi_name(name, state.pois, state.destination or "")
        options[name] = sorted((exact or aliases or {canonical}) & placed)
    # One actual stop can satisfy at most one distinct mandatory requirement.
    # Augmenting paths avoid order-dependent undercounting of shared aliases.
    matched = {}
    def assign(name, seen):
        for identity in options[name]:
            if identity in seen:
                continue
            seen.add(identity)
            if identity not in matched or assign(matched[identity], seen):
                matched[identity] = name
                return True
        return False
    for name in names:
        assign(name, set())
    arranged = [name for name in names if name in matched.values()]
    omitted = [name for name in names if name not in arranged]
    return {"configured_max_per_day": base, "effective_max_per_day": limit,
            "source": "user" if user is not None else "strict_config" if strict else "default",
            "expanded": expanded, "total_capacity": total, "active_must_visit": names,
            "arranged_names": arranged, "omitted_names": omitted,
            "overflow_exempt": len(names) > total and len(arranged) >= total}


def effective_daily_limit(state):
    return capacity_policy(state)["effective_max_per_day"]


def capacity_warnings(state, route=None):
    """Warnings must describe the selected version, not the latest draft."""
    policy = capacity_policy(state, route)
    warnings = []
    if policy["expanded"]:
        warnings.append("单日点名景点较多，行程较紧凑；已调整默认景点数量上限，请预留交通与休息时间。")
    if policy["overflow_exempt"]:
        warnings.append(
            f"受单日最多安排 {policy['effective_max_per_day']} 个景点的限制，已优先安排 "
            + "、".join(policy["arranged_names"]) + "，未能排入的 "
            + "、".join(policy["omitted_names"]) + " 建议留作备选。"
        )
    return warnings


def capacity_prompt(state):
    policy = capacity_policy(state)
    if not policy["expanded"] and len(policy["active_must_visit"]) <= policy["total_capacity"]:
        return ""  # Keep non-conflicting requests and historical prompts unchanged.
    text = (f"每日有效景点上限={policy['effective_max_per_day']}，上限来源={policy['source']}；"
            f"有效点名景点：{'、'.join(policy['active_must_visit']) or '无'}。")
    if policy["expanded"]:
        text += "默认上限已按点名需求扩容，必须安排全部有效点名景点；仍须核验通勤、步行、开放时间及总时间窗口，不能靠删除必去景点修复。"
    elif len(policy["active_must_visit"]) > policy["total_capacity"]:
        text += (f"点名数超过总容量{policy['total_capacity']}，必须优先安排其中{policy['total_capacity']}个不同的有效点名景点，"
                 "不要用非点名景点占用其名额；优先遵循用户点名顺序和物理可行性，其余作为备选并说明。"
                 "满额后的容量溢出不要求补回，禁止为补回而突破每日上限。")
    return text


def indoor_value(poi):
    value = poi.get("indoor")
    return value if type(value) is bool else None


def indoor_description(poi):
    label = {True:"明确室内", False:"明确室外", None:"未知/混合"}[indoor_value(poi)]
    source = poi.get("indoor_source") or "输入快照声明（未独立核实）"
    scope = poi.get("indoor_scope") or "仅限输入所声明的参观范围；不外推庭院、排队及通勤"
    return f"室内属性={label}；来源={source}；范围={scope}"


def hot_day(state, day):
    """Date alignment first; explicit simulated temperature is a conservative override."""
    temperatures = []
    for text in (state.query or "", state.modification_notes or ""):
        temperatures.extend(float(n) for n in re.findall(r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*(?:°\s*C|℃)", text, re.I))
    if any(t >= HEAT_C for t in temperatures):
        return True
    forecast = state.weather_forecast or []
    if state.travel_start_date:
        target = (state.travel_start_date+timedelta(days=day-1)).isoformat()
        items = [w for w in forecast if str(w.get("date")) == target]
    else:
        items = forecast[day-1:day]
    for item in items:
        try:
            temp = float(item.get("day_temp"))
            if math.isfinite(temp) and temp >= HEAT_C:
                return True
        except (ValueError, TypeError):
            pass
    return False


def heat_violations(state, route):
    by_name = {p.get("name"): p for p in state.pois}
    bad = []
    for day in route:
        if not isinstance(day.get("day"), int) or not hot_day(state, day["day"]):
            continue
        for spot in day.get("spots", []):
            times = []
            for key in ("start_time", "end_time"):
                match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", str(spot.get(key, "")))
                if match:
                    times.append(int(match[1])*60+int(match[2]))
            if len(times) == 2 and times[0] < HEAT_END and times[1] > HEAT_START:
                if indoor_value(by_name.get(spot.get("name"), {})) is not True:
                    bad.append(str(spot.get("name")))
    return bad


def contract_text(state):
    hot = [str(d) for d in range(1, state.days+1) if hot_day(state, d)]
    capacity = capacity_prompt(state)
    label = "配置每天上限" if capacity else "系统每天上限"
    return (
        f"统一约束协议 {PROTOCOL}：{label}{state.max_per_day}；有效每天景点上限{effective_daily_limit(state)}。"
        + (f"点名容量协议：{capacity}\n" if capacity else "") +
        "当前用户修改优先于原始请求，用户更严格上限优先于默认上限；上限不是必须填满的数量。"
        "恰好数量要求仍须满足，若与硬约束冲突则说明不可行，不得重复或虚构景点。\n"
        "室内属性为true/false/unknown三值：仅明确室内可用于室内限定时段；未知/混合不等于室内，不能凭名称自行升级标签。\n"
        "高温业务边界：当日最高温>=38°C时，11:00–15:00禁止安排室外及室内属性未知的景点；"
        "11:00结束或15:00开始均不与禁限窗口重叠。用户更严格的避热要求仍须满足；缺天气不代表安全。"
        f"本次触发高温规则的天数：{','.join(hot) or '无已知触发日'}。\n"
        "解释、景点数、去重声明和游玩时长必须与最终时间表一致；无餐馆候选时meal_slots=[]。\n"
    )
