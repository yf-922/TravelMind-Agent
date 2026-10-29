"""Shared, observable planning contract; never infer safety from a POI name."""
from datetime import timedelta
import math
import re

PROTOCOL = "planner-contract-v1"
HEAT_START, HEAT_END, HEAT_C = 660, 900, 38.0
_NUM = r"([1-9]\d*|十|[一二两三四五六七八九])"


def daily_limit(text):
    """Recognize explicit daily upper/exact counts, not total/visit durations."""
    values = []
    for match in re.finditer(r"(?:每天|每日|一天)\s*(?:最多|不超过|至多|恰好|只安排|只去)\s*"+_NUM+r"\s*(?:个|处)?\s*景点", text or ""):
        value = match.group(1)
        values.append(int(value) if value.isdigit() else {"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}[value])
    return min(values) if values else None


def effective_daily_limit(state):
    # Current user modification replaces the old request's daily limit.
    user = daily_limit(state.modification_notes) or daily_limit(state.query)
    return min(state.max_per_day, user) if user is not None else state.max_per_day


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
    return (
        f"统一约束协议 {PROTOCOL}：系统每天上限{state.max_per_day}；有效每天景点上限{effective_daily_limit(state)}。"
        "当前用户修改优先于原始请求，用户更严格上限优先于默认上限；上限不是必须填满的数量。"
        "恰好数量要求仍须满足，若与硬约束冲突则说明不可行，不得重复或虚构景点。\n"
        "室内属性为true/false/unknown三值：仅明确室内可用于室内限定时段；未知/混合不等于室内，不能凭名称自行升级标签。\n"
        "高温业务边界：当日最高温>=38°C时，11:00–15:00禁止安排室外及室内属性未知的景点；"
        "11:00结束或15:00开始均不与禁限窗口重叠。用户更严格的避热要求仍须满足；缺天气不代表安全。"
        f"本次触发高温规则的天数：{','.join(hot) or '无已知触发日'}。\n"
        "解释、景点数、去重声明和游玩时长必须与最终时间表一致；无餐馆候选时meal_slots=[]。\n"
    )
