"""Tool adapters for the isolated orchestration core."""

from __future__ import annotations

from typing import Any

from app.planning.helpers import amap_key, fetch_city_spots


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
        # The existing project has a stable multi-query POI collector. Reuse it here.
        return fetch_city_spots(city, amap_key(), max_spots=12)


class FixturePoiTool:
    """Offline fixture for tests and reproducible classroom demonstrations."""

    def search(self, city: str, query: str = "") -> list[dict[str, Any]]:
        return [
            {"name": f"{city} Museum", "address": f"{city} Center"},
            {"name": f"{city} Old Town", "address": f"{city} Riverside"},
            {"name": f"{city} Park", "address": f"{city} North District"},
        ]
