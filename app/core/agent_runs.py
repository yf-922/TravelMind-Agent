"""Bounded, privacy-safe observability for agent executions."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import AsyncIterator
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"succeeded", "incomplete", "failed", "timed_out", "cancelled"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def plan_timeout_seconds() -> float:
    """Return a bounded per-run timeout configured by PLAN_TIMEOUT_SECONDS."""
    try:
        value = float(os.getenv("PLAN_TIMEOUT_SECONDS", "180"))
    except ValueError:
        value = 180.0
    return min(max(value, 5.0), 900.0)


class AgentRunRegistry:
    """Keep recent traces in memory and aggregate low-cardinality metrics."""

    def __init__(self, max_runs: int = 200):
        self._max_runs = max(1, max_runs)
        self._lock = Lock()
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._run_counts: Counter[tuple[str, str]] = Counter()
        self._node_counts: Counter[tuple[str, str]] = Counter()
        self._node_duration_ms: Counter[str] = Counter()

    def start(self, user_id: str, mode: str, request_id: str | None = None) -> str:
        run_id = uuid.uuid4().hex
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "user_id": user_id,
                "request_id": request_id,
                "mode": mode,
                "status": "running",
                "started_at": _utc_now(),
                "finished_at": None,
                "duration_ms": None,
                "nodes": [],
                "error_code": None,
                "_started_mono": time.monotonic(),
            }
            self._trim_locked()
        return run_id

    def node_started(self, run_id: str, node: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run or run["status"] != "running":
                return
            attempt = sum(1 for item in run["nodes"] if item["node"] == node) + 1
            now = time.monotonic()
            run["nodes"].append({
                "node": node,
                "attempt": attempt,
                "status": "running",
                "duration_ms": None,
                "started_offset_ms": round((now - run["_started_mono"]) * 1000, 3),
                "finished_offset_ms": None,
                "_started_mono": now,
            })

    def node_finished(self, run_id: str, node: str, status: str = "succeeded") -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            span = next(
                (item for item in reversed(run["nodes"])
                 if item["node"] == node and item["status"] == "running"),
                None,
            )
            if not span:
                return
            duration_ms = max(0.0, (time.monotonic() - span.pop("_started_mono")) * 1000)
            span["duration_ms"] = round(duration_ms, 3)
            span["finished_offset_ms"] = round(span["started_offset_ms"] + duration_ms, 3)
            span["status"] = status
            self._node_counts[(node, status)] += 1
            self._node_duration_ms[node] += duration_ms

    def finish(self, run_id: str, status: str, error_code: str | None = None) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal status: {status}")
        with self._lock:
            run = self._runs.get(run_id)
            if not run or run["status"] != "running":
                return
            now = time.monotonic()
            for span in run["nodes"]:
                if span["status"] == "running":
                    duration_ms = max(0.0, (now - span.pop("_started_mono")) * 1000)
                    span["duration_ms"] = round(duration_ms, 3)
                    span["finished_offset_ms"] = round(span["started_offset_ms"] + duration_ms, 3)
                    span["status"] = status
                    self._node_counts[(span["node"], status)] += 1
                    self._node_duration_ms[span["node"]] += duration_ms
            run["duration_ms"] = round(max(0.0, (now - run.pop("_started_mono")) * 1000), 3)
            run["finished_at"] = _utc_now()
            run["status"] = status
            run["error_code"] = error_code
            self._run_counts[(run["mode"], status)] += 1

    def set_degraded_services(self, run_id: str, services: Any) -> None:
        safe_services = sorted({
            str(item)[:80] for item in (services or [])
            if isinstance(item, str) and item.strip()
        })
        with self._lock:
            run = self._runs.get(run_id)
            if run:
                run["degraded_services"] = safe_services

    def set_llm_usage(self, run_id: str, usage: dict[str, Any]) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run:
                run["llm_usage"] = deepcopy(usage)

    def get_owned(self, run_id: str, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run or run["user_id"] != user_id:
                return None
            return self._public_copy(run)

    def status(self, run_id: str) -> str | None:
        with self._lock:
            run = self._runs.get(run_id)
            return str(run["status"]) if run else None

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._lock:
            in_flight = sum(1 for run in self._runs.values() if run["status"] == "running")
            return {
                "in_flight": in_flight,
                "run_counts": dict(self._run_counts),
                "node_counts": dict(self._node_counts),
                "node_duration_ms": dict(self._node_duration_ms),
            }

    def _trim_locked(self) -> None:
        while len(self._runs) > self._max_runs:
            completed_id = next(
                (key for key, value in self._runs.items() if value["status"] != "running"),
                None,
            )
            if completed_id is None:
                break
            del self._runs[completed_id]

    @staticmethod
    def _public_copy(run: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(run)
        result.pop("user_id", None)
        result.pop("_started_mono", None)
        for span in result["nodes"]:
            span.pop("_started_mono", None)
        result["summary"] = AgentRunRegistry._summarize(result)
        return result

    @staticmethod
    def _summarize(run: dict[str, Any]) -> dict[str, Any]:
        nodes = run.get("nodes") or []
        intervals = sorted(
            (float(node["started_offset_ms"]), float(node["finished_offset_ms"]))
            for node in nodes
            if node.get("finished_offset_ms") is not None
        )
        events = sorted(
            [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals],
            key=lambda item: (item[0], item[1]),
        )
        active = max_active = 0
        active_started: float | None = None
        observed_wall_ms = 0.0
        for offset, delta in events:
            previous = active
            active += delta
            max_active = max(max_active, active)
            if previous == 0 and active > 0:
                active_started = offset
            elif previous > 0 and active == 0 and active_started is not None:
                observed_wall_ms += max(0.0, offset - active_started)
                active_started = None

        node_work_ms = sum(float(node.get("duration_ms") or 0) for node in nodes)
        overlap_ms = max(0.0, node_work_ms - observed_wall_ms)
        slowest = sorted(
            ({
                "node": str(node.get("node") or ""),
                "attempt": int(node.get("attempt") or 1),
                "duration_ms": round(float(node.get("duration_ms") or 0), 3),
            } for node in nodes),
            key=lambda item: item["duration_ms"],
            reverse=True,
        )[:3]
        failed_nodes = sorted({
            str(node.get("node") or "") for node in nodes
            if node.get("status") not in {"running", "succeeded"}
        })
        duration_ms = run.get("duration_ms")
        return {
            "total_duration_ms": duration_ms,
            "node_executions": len(nodes),
            "unique_nodes": len({node.get("node") for node in nodes}),
            "failed_nodes": failed_nodes,
            "degraded_services": list(run.get("degraded_services") or []),
            "slowest_node_executions": slowest,
            "timing": {
                "node_work_ms": round(node_work_ms, 3),
                "observed_node_wall_ms": round(observed_wall_ms, 3),
                "parallel_overlap_ms": round(overlap_ms, 3),
                "parallel_overlap_ratio": round(overlap_ms / node_work_ms, 3) if node_work_ms else 0.0,
                "max_concurrency": max_active,
                "unattributed_ms": round(max(0.0, float(duration_ms or 0) - observed_wall_ms), 3),
            },
            "llm_usage": deepcopy(run.get("llm_usage") or {
                "calls_total": 0,
                "attribution": "request_context",
            }),
        }


agent_runs = AgentRunRegistry()


def classify_agent_error(exc: BaseException) -> tuple[str, str]:
    """Convert provider failures into actionable, secret-safe user messages."""
    error_text = str(exc).lower()
    llm_key_names = ("openai_api_key", "grok_api_key", "deepseek_api_key", "doubao_api_key")
    if "缺少" in error_text and any(name in error_text for name in llm_key_names):
        return (
            "LLM_CONFIG_MISSING",
            "尚未配置所选大模型的 API Key，请检查 .env.local 中的 LLM_PROVIDER 和对应 Key（如 GROK_API_KEY）。",
        )
    if "401" in error_text or "authentication" in error_text or "invalid api key" in error_text:
        return (
            "LLM_AUTH_FAILED",
            "大模型 API Key 无效或未生效，请检查 OPENAI_API_KEY/GROK_API_KEY/DEEPSEEK_API_KEY/DOUBAO_API_KEY 和模型提供商配置。",
        )
    if "429" in error_text or "rate limit" in error_text or "quota" in error_text:
        return "LLM_RATE_LIMITED", "大模型接口达到频率或额度限制，请稍后重试。"
    if "timeout" in error_text or "timed out" in error_text:
        return "LLM_REQUEST_TIMEOUT", "大模型接口响应超时，本次规划已停止；请重试或检查模型中转服务。"
    if "amap" in error_text or "高德" in error_text:
        return "AMAP_REQUEST_FAILED", "高德接口调用失败，请检查 AMAP_API_KEY 或稍后重试。"
    return "AGENT_EXECUTION_FAILED", "规划执行失败，请使用 run_id 查询执行状态后重试。"


async def observe_agent_events(
    source: AsyncIterator[dict[str, Any]],
    run_id: str,
    *,
    timeout_seconds: float | None = None,
    registry: AgentRunRegistry = agent_runs,
) -> AsyncIterator[dict[str, Any]]:
    """Add lifecycle events, timeout handling and trace metrics to an event stream."""
    from app.core.llm_usage import bind_run, take_run_snapshot, unbind_run

    timeout = plan_timeout_seconds() if timeout_seconds is None else timeout_seconds
    usage_token = bind_run(run_id)
    terminal_seen = False
    try:
        yield {"type": "run", "run_id": run_id, "timeout_seconds": timeout}
        async with asyncio.timeout(timeout):
            async for event in source:
                event_type = event.get("type")
                node = str(event.get("node") or "")
                if event_type == "stage" and node:
                    registry.node_started(run_id, node)
                elif event_type == "stage_summary" and node:
                    registry.node_finished(run_id, node)
                elif event_type == "result":
                    plan = event.get("plan")
                    if isinstance(plan, dict):
                        registry.set_degraded_services(run_id, plan.get("degraded_services"))
                    status = "succeeded" if event.get("success") else "incomplete"
                    registry.finish(run_id, status)
                    terminal_seen = True
                elif event_type == "error":
                    registry.finish(run_id, "failed", str(event.get("code") or "AGENT_ERROR"))
                    terminal_seen = True
                if event_type in {"result", "error"}:
                    event = {**event, "run_id": run_id}
                yield event
    except TimeoutError:
        registry.finish(run_id, "timed_out", "PLAN_TIMEOUT")
        terminal_seen = True
        yield {
            "type": "error",
            "code": "PLAN_TIMEOUT",
            "message": f"规划超过 {timeout:g} 秒，已自动终止，请稍后重试。",
            "run_id": run_id,
        }
    except asyncio.CancelledError:
        registry.finish(run_id, "cancelled", "CLIENT_DISCONNECTED")
        terminal_seen = True
        raise
    except Exception as exc:  # noqa: BLE001
        error_code, message = classify_agent_error(exc)
        registry.finish(run_id, "failed", error_code)
        terminal_seen = True
        logger.exception("agent run failed", extra={"agent_run_id": run_id})
        yield {
            "type": "error",
            "code": error_code,
            "message": message,
            "run_id": run_id,
        }
    finally:
        if not terminal_seen and registry.status(run_id) == "running":
            registry.finish(run_id, "failed", "STREAM_ENDED_WITHOUT_RESULT")
        registry.set_llm_usage(run_id, take_run_snapshot(run_id))
        unbind_run(usage_token)


def prometheus_agent_lines() -> list[str]:
    data = agent_runs.metrics_snapshot()
    lines = [
        "# HELP travelmind_agent_runs_in_flight Current agent runs in flight.",
        "# TYPE travelmind_agent_runs_in_flight gauge",
        f"travelmind_agent_runs_in_flight {data['in_flight']}",
        "# HELP travelmind_agent_runs_total Completed agent runs.",
        "# TYPE travelmind_agent_runs_total counter",
    ]
    for (mode, status), count in sorted(data["run_counts"].items()):
        lines.append(f'travelmind_agent_runs_total{{mode="{mode}",status="{status}"}} {count}')
    lines.extend([
        "# HELP travelmind_agent_node_executions_total Completed agent node executions.",
        "# TYPE travelmind_agent_node_executions_total counter",
    ])
    for (node, status), count in sorted(data["node_counts"].items()):
        lines.append(f'travelmind_agent_node_executions_total{{node="{node}",status="{status}"}} {count}')
    lines.extend([
        "# HELP travelmind_agent_node_duration_ms_sum Cumulative agent node latency in milliseconds.",
        "# TYPE travelmind_agent_node_duration_ms_sum counter",
    ])
    for node, duration in sorted(data["node_duration_ms"].items()):
        lines.append(f'travelmind_agent_node_duration_ms_sum{{node="{node}"}} {duration:.3f}')
    return lines
