"""Small dependency-free request metrics used by the API and health endpoints."""

from __future__ import annotations

from collections import Counter
import math
from threading import Lock
from time import monotonic

_lock = Lock()
_started_at = monotonic()
_requests_total = 0
_errors_total = 0
_in_flight = 0
_status_counts: Counter[int] = Counter()
_route_counts: Counter[str] = Counter()
_latency_sum_ms = 0.0
_latency_count_total = 0
_latencies_ms: list[float] = []
_LATENCY_SAMPLE_LIMIT = 2000


def request_started() -> None:
    global _requests_total, _in_flight
    with _lock:
        _requests_total += 1
        _in_flight += 1


def request_finished(path: str, status_code: int, elapsed_ms: float) -> None:
    global _errors_total, _in_flight, _latency_sum_ms, _latency_count_total
    with _lock:
        _in_flight = max(0, _in_flight - 1)
        _latency_sum_ms += elapsed_ms
        _latency_count_total += 1
        _latencies_ms.append(max(0.0, float(elapsed_ms)))
        if len(_latencies_ms) > _LATENCY_SAMPLE_LIMIT:
            del _latencies_ms[: len(_latencies_ms) - _LATENCY_SAMPLE_LIMIT]
        _status_counts[status_code] += 1
        _route_counts[path] += 1
        if status_code >= 500:
            _errors_total += 1


def snapshot() -> dict[str, object]:
    with _lock:
        ordered = sorted(_latencies_ms)
        p95 = ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)] if ordered else 0.0
        return {
            "uptime_seconds": round(monotonic() - _started_at, 3),
            "requests_total": _requests_total,
            "errors_total": _errors_total,
            "in_flight": _in_flight,
            "latency_sum_ms": round(_latency_sum_ms, 3),
            "latency_count": _latency_count_total,
            "latency_sample_count": len(_latencies_ms),
            "latency_avg_ms": round(_latency_sum_ms / _latency_count_total, 3) if _latency_count_total else 0.0,
            "latency_p95_ms": round(p95, 3),
            "latency_max_ms": round(max(_latencies_ms), 3) if _latencies_ms else 0.0,
            "status_counts": dict(_status_counts),
            "route_counts": dict(_route_counts),
        }


def prometheus_text() -> str:
    from app.core.agent_runs import prometheus_agent_lines
    from app.core.llm_usage import prometheus_lines as prometheus_llm_lines
    from app.core.risk_gate_metrics import prometheus_lines as prometheus_risk_gate_lines

    data = snapshot()
    lines = [
        "# HELP travelmind_requests_total Total HTTP requests.",
        "# TYPE travelmind_requests_total counter",
        f"travelmind_requests_total {data['requests_total']}",
        "# HELP travelmind_errors_total HTTP 5xx responses.",
        "# TYPE travelmind_errors_total counter",
        f"travelmind_errors_total {data['errors_total']}",
        "# HELP travelmind_requests_in_flight Current requests in flight.",
        "# TYPE travelmind_requests_in_flight gauge",
        f"travelmind_requests_in_flight {data['in_flight']}",
        "# HELP travelmind_request_latency_ms_sum Cumulative request latency in milliseconds.",
        "# TYPE travelmind_request_latency_ms_sum counter",
        f"travelmind_request_latency_ms_sum {data['latency_sum_ms']}",
        "# HELP travelmind_request_latency_ms_count Requests included in cumulative latency.",
        "# TYPE travelmind_request_latency_ms_count counter",
        f"travelmind_request_latency_ms_count {data['latency_count']}",
        "# HELP travelmind_request_latency_ms_avg Average request latency in milliseconds.",
        "# TYPE travelmind_request_latency_ms_avg gauge",
        f"travelmind_request_latency_ms_avg {data['latency_avg_ms']}",
        "# HELP travelmind_request_latency_ms_p95 Sampled p95 request latency in milliseconds.",
        "# TYPE travelmind_request_latency_ms_p95 gauge",
        f"travelmind_request_latency_ms_p95 {data['latency_p95_ms']}",
        "# HELP travelmind_request_latency_ms_max Maximum sampled request latency in milliseconds.",
        "# TYPE travelmind_request_latency_ms_max gauge",
        f"travelmind_request_latency_ms_max {data['latency_max_ms']}",
    ]
    for status, count in sorted(data["status_counts"].items()):
        lines.append(f'travelmind_http_responses_total{{status="{status}"}} {count}')
    lines.extend(prometheus_agent_lines())
    lines.extend(prometheus_llm_lines())
    lines.extend(prometheus_risk_gate_lines())
    return "\n".join(lines) + "\n"
