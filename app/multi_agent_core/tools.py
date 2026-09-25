"""Tool adapters for the isolated orchestration core."""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from app.planning.helpers import amap_key, fetch_city_spots, merge_verified_poi
from app.providers.amap.poi import ATTRACTION_TYPE, poi_to_spot, search_city_pois
from app.multi_agent_core.protocols import MCPToolManifest


class ToolPermissionError(PermissionError):
    """Raised when an Agent attempts to use a tool outside its allow-list."""


@dataclass(frozen=True)
class ToolCallRecord:
    """Privacy-safe evidence for one attempted tool call."""

    task_id: str
    trace_id: str
    agent: str
    tool: str
    parameters: dict[str, Any]
    status: str
    duration_ms: float
    error_code: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def _safe_parameters(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep assertions useful without persisting raw free-text user input."""
    safe: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in {"query", "prompt", "user_request"}:
            raw = str(value or "")
            safe[key] = {
                "present": bool(raw.strip()),
                "length": len(raw),
                "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12],
            }
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = {"type": type(value).__name__}
    return safe


class ToolRegistry:
    """Small explicit tool boundary for classroom auditing and future adapters."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}
        self._manifests: dict[str, MCPToolManifest] = {}
        self.audit_log: deque[ToolCallRecord] = deque(maxlen=1000)

    def register(self, name: str, tool: Any, *, description: str = "",
                 input_schema: dict[str, Any] | None = None,
                 allowed_agents: list[str] | None = None) -> None:
        self._tools[name] = tool
        self._manifests[name] = MCPToolManifest(
            name=name,
            description=description or f"Local tool {name}",
            input_schema=dict(input_schema or {}),
            allowed_agents=list(allowed_agents or []),
        )

    def manifest(self, agent: Any | None = None) -> list[dict[str, Any]]:
        """Return a safe MCP-like tool manifest filtered by Agent permissions."""
        allowed = set(getattr(agent, "allowed_tools", set())) if agent is not None else None
        return [
            item.model_dump()
            for name, item in self._manifests.items()
            if allowed is None or name in allowed
        ]

    def call(
        self,
        agent: Any,
        name: str,
        *args: Any,
        audit_context: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        context = audit_context or {}
        started = time.perf_counter()
        status = "succeeded"
        error_code: str | None = None
        try:
            if name not in agent.allowed_tools:
                status = "blocked"
                raise ToolPermissionError(f"{agent.name} is not allowed to call {name}")
            if name not in self._tools:
                status = "failed"
                raise KeyError(f"unknown tool: {name}")
            return self._tools[name](*args, **kwargs)
        except Exception as exc:
            error_code = type(exc).__name__
            if status == "succeeded":
                status = "failed"
            raise
        finally:
            self.audit_log.append(ToolCallRecord(
                task_id=str(context.get("task_id") or ""),
                trace_id=str(context.get("trace_id") or ""),
                agent=str(getattr(agent, "name", "unknown")),
                tool=name,
                parameters=_safe_parameters(kwargs),
                status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                error_code=error_code,
            ))


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
