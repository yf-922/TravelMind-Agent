"""Tool adapters for the isolated orchestration core."""

from __future__ import annotations

from typing import Any

from app.planning.helpers import amap_key, fetch_city_spots


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

