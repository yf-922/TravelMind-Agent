"""Tool adapters for the isolated orchestration core."""

from __future__ import annotations

from typing import Any

from app.planning.helpers import amap_key, fetch_city_spots, merge_verified_poi
from app.providers.amap.poi import ATTRACTION_TYPE, poi_to_spot, search_city_pois


class ToolPermissionError(PermissionError):
    """Raised when an Agent attempts to use a tool outside its allow-list."""


class ToolRegistry:
    """Small explicit tool boundary for classroom auditing and future adapters."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, name: str, tool: Any) -> None:
        self._tools[name] = tool

    def call(self, agent: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in agent.allowed_tools:
            raise ToolPermissionError(f"{agent.name} is not allowed to call {name}")
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name](*args, **kwargs)


class AmapPoiTool:
    """Live POI tool used in the real demo. It requires AMAP_API_KEY."""

    def search(self, city: str, query: str = "") -> list[dict[str, Any]]:
        api_key = amap_key()
        candidates = fetch_city_spots(city, api_key, max_spots=12)
        if not query.strip():
            return candidates
        targeted = search_city_pois(
            city, api_key, keywords=query.strip(), types=ATTRACTION_TYPE, offset=8
        )
        for raw in targeted:
            spot = poi_to_spot(raw)
            if spot:
                candidates = merge_verified_poi(candidates, spot)
        return candidates


class FixturePoiTool:
    """Offline fixture for tests and reproducible classroom demonstrations."""

    def search(self, city: str, query: str = "") -> list[dict[str, Any]]:
        return [
            {"name": f"{city} Museum", "address": f"{city} Center"},
            {"name": f"{city} Old Town", "address": f"{city} Riverside"},
            {"name": f"{city} Park", "address": f"{city} North District"},
        ]
