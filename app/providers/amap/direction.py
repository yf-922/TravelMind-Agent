"""高德步行、公交/地铁和驾车路径规划。失败时返回 None，由上层降级。"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.parse
import urllib.request
from typing import Any


_PLAN_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
_PLAN_CACHE_LOCK = threading.Lock()
_PLAN_CACHE_TTL = 30 * 60


def _fetch_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "TripAgent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _minutes(seconds: Any) -> int:
    try:
        return max(1, math.ceil(float(seconds) / 60))
    except (TypeError, ValueError):
        return 0


def _km(meters: Any) -> float:
    try:
        return round(float(meters) / 1000, 2)
    except (TypeError, ValueError):
        return 0.0


def _point(location: dict[str, float]) -> str:
    return f"{location['lng']},{location['lat']}"


def _walk_plan(origin: dict[str, float], destination: dict[str, float], api_key: str) -> dict[str, Any] | None:
    params = urllib.parse.urlencode({
        "key": api_key, "origin": _point(origin), "destination": _point(destination),
        "output": "json",
    })
    data = _fetch_json(f"https://restapi.amap.com/v3/direction/walking?{params}")
    paths = (data.get("route") or {}).get("paths") or []
    if data.get("status") != "1" or not paths:
        return None
    path = paths[0]
    steps = [str(step.get("instruction") or "").strip() for step in path.get("steps") or []]
    steps = [step for step in steps if step][:5]
    duration = _minutes(path.get("duration"))
    distance = _km(path.get("distance"))
    return {
        "mode": "walk", "mode_label": "步行", "distance_km": distance,
        "duration_min": duration, "estimated_cost": 0,
        "instruction": f"步行约 {duration} 分钟（道路距离 {distance} km）。",
        "steps": steps, "source": "amap", "estimate": False,
    }


def _walk_distance(segment: dict[str, Any]) -> int:
    walking = segment.get("walking") or {}
    try:
        return int(float(walking.get("distance") or 0))
    except (TypeError, ValueError):
        return 0


def _transit_plan(
    origin: dict[str, float], destination: dict[str, float], city: str, api_key: str,
) -> dict[str, Any] | None:
    params = urllib.parse.urlencode({
        "key": api_key, "origin": _point(origin), "destination": _point(destination),
        "city": city, "cityd": city, "strategy": "0", "nightflag": "0",
        "extensions": "base", "output": "json",
    })
    data = _fetch_json(f"https://restapi.amap.com/v3/direction/transit/integrated?{params}")
    route = data.get("route") or {}
    transits = route.get("transits") or []
    if data.get("status") != "1" or not transits:
        return None

    transit = transits[0]
    parts: list[str] = []
    lines: list[dict[str, str]] = []
    pending_walk = 0
    for segment in transit.get("segments") or []:
        pending_walk += _walk_distance(segment)
        buslines = ((segment.get("bus") or {}).get("buslines") or [])
        if not buslines:
            continue
        if pending_walk:
            parts.append(f"步行约 {pending_walk}m")
            pending_walk = 0
        line = buslines[0]
        raw_name = str(line.get("name") or "公交").strip()
        line_name = raw_name.split("(", 1)[0].strip() or raw_name
        departure = str((line.get("departure_stop") or {}).get("name") or "").strip()
        arrival = str((line.get("arrival_stop") or {}).get("name") or "").strip()
        detail = line_name
        if departure and arrival:
            detail += f"（{departure}上车，{arrival}下车）"
        parts.append(detail)
        lines.append({"name": line_name, "departure_stop": departure, "arrival_stop": arrival})
    if pending_walk:
        parts.append(f"步行约 {pending_walk}m")
    if not lines:
        return None

    cost_raw = transit.get("cost")
    try:
        cost = round(float(cost_raw), 2) if cost_raw not in (None, "") else None
    except (TypeError, ValueError):
        cost = None
    distance = _km(transit.get("distance"))
    duration = _minutes(transit.get("duration"))
    if cost is None:
        cost = min(10, max(2, 2 + math.ceil(max(distance, 1) / 6)))
    mode_label = "地铁/公交" if any("地铁" in line["name"] or "轨道" in line["name"] for line in lines) else "公交"
    return {
        "mode": "transit", "mode_label": mode_label, "distance_km": distance,
        "duration_min": duration, "estimated_cost": cost,
        "instruction": " → ".join(parts), "steps": parts, "transit_lines": lines,
        "source": "amap", "estimate": cost_raw in (None, ""),
    }


def _driving_plan(origin: dict[str, float], destination: dict[str, float], api_key: str) -> dict[str, Any] | None:
    params = urllib.parse.urlencode({
        "key": api_key, "origin": _point(origin), "destination": _point(destination),
        "strategy": "0", "extensions": "base", "output": "json",
    })
    data = _fetch_json(f"https://restapi.amap.com/v3/direction/driving?{params}")
    route = data.get("route") or {}
    paths = route.get("paths") or []
    if data.get("status") != "1" or not paths:
        return None
    path = paths[0]
    distance = _km(path.get("distance"))
    duration = _minutes(path.get("duration"))
    taxi_cost = round(13 + max(0, distance - 3) * 2.3)
    return {
        "mode": "taxi_or_car", "mode_label": "打车/租车", "distance_km": distance,
        "duration_min": duration, "estimated_cost": taxi_cost,
        "instruction": f"按道路驾车约 {distance} km、{duration} 分钟；打车估算约 ¥{taxi_cost}。",
        "steps": [], "source": "amap", "estimate": True,
    }


def plan_route_distance(
    origin: dict[str, float],
    destination: dict[str, float],
    api_key: str,
    *,
    mode: str = "walk",
) -> dict[str, Any] | None:
    """Return an actual road distance for a single leg.

    ``mode='walk'`` uses Amap's walking route and ``mode='drive'`` uses its
    driving route. Unlike the legacy transport selector, this function never
    chooses a mode from straight-line distance: the caller's constraint
    determines the requested route type. Failures return ``None`` so callers
    can expose an explicit degraded state instead of treating a geometric
    distance as a road distance.
    """
    normalized_mode = "drive" if mode in {"drive", "driving", "car", "taxi_or_car"} else "walk"
    cache_key = (
        "distance", normalized_mode,
        round(origin["lng"], 5), round(origin["lat"], 5),
        round(destination["lng"], 5), round(destination["lat"], 5),
    )
    now = time.monotonic()
    with _PLAN_CACHE_LOCK:
        cached = _PLAN_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return dict(cached[1])

    for attempt in range(2):
        try:
            result = (
                _driving_plan(origin, destination, api_key)
                if normalized_mode == "drive"
                else _walk_plan(origin, destination, api_key)
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            result = None
        if result:
            with _PLAN_CACHE_LOCK:
                _PLAN_CACHE[cache_key] = (now + _PLAN_CACHE_TTL, dict(result))
            return result
        if attempt == 0:
            time.sleep(0.25)
    return None


def plan_transport_leg(
    origin: dict[str, float],
    destination: dict[str, float],
    city: str,
    api_key: str,
    straight_distance_km: float,
) -> dict[str, Any] | None:
    """按距离选择路径服务；任何接口错误均返回 None，让主流程安全降级。"""
    cache_key = (
        round(origin["lng"], 5), round(origin["lat"], 5),
        round(destination["lng"], 5), round(destination["lat"], 5), city.strip(),
    )
    now = time.monotonic()
    with _PLAN_CACHE_LOCK:
        cached = _PLAN_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return dict(cached[1])
    def request_plan() -> dict[str, Any] | None:
        if straight_distance_km <= 1.2:
            return _walk_plan(origin, destination, api_key)
        if straight_distance_km <= 25:
            return _transit_plan(origin, destination, city, api_key)
        return _driving_plan(origin, destination, api_key)

    # 页面首次打开时常会同时查询多段路线。个人 Key 偶发触发瞬时限流时，
    # 高德会返回 status=0 或空方案；短暂等待后重试一次即可恢复。
    plan = None
    for attempt in range(2):
        try:
            plan = request_plan()
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            plan = None
        if plan:
            with _PLAN_CACHE_LOCK:
                _PLAN_CACHE[cache_key] = (now + _PLAN_CACHE_TTL, dict(plan))
            return plan
        if attempt == 0:
            time.sleep(0.25)
    return None
