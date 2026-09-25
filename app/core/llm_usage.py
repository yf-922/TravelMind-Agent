"""Privacy-safe LLM usage and latency counters.

Provider responses do not always expose usage metadata after structured-output
parsing. The registry therefore records provider usage when available and uses
an explicitly labelled character-based estimate otherwise. It never stores
prompts or model responses.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from contextvars import ContextVar, Token
from threading import Lock
from typing import Any


_lock = Lock()
_calls_total = 0
_failed_total = 0
_input_tokens = 0
_output_tokens = 0
_latency_sum_ms = 0.0
_estimated_calls = 0
_cost_usd = 0.0
_by_model: Counter[str] = Counter()
_by_schema: Counter[str] = Counter()
_current_run_id: ContextVar[str | None] = ContextVar("llm_usage_run_id", default=None)
_by_run: dict[str, Counter[str]] = defaultdict(Counter)


def bind_run(run_id: str) -> Token:
    """Attribute calls in the current async context to one agent run."""
    return _current_run_id.set(run_id)


def unbind_run(token: Token) -> None:
    _current_run_id.reset(token)


def estimate_tokens(value: Any) -> int:
    """Conservative offline estimate; CJK text is roughly two chars/token."""
    if value is None:
        return 0
    text = value if isinstance(value, str) else str(value)
    return max(0, math.ceil(len(text) / 2))


def _number(value: Any) -> int | None:
    try:
        return max(0, int(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def extract_usage(value: Any) -> tuple[int | None, int | None]:
    """Read common LangChain/OpenAI usage metadata shapes if present."""
    candidates: list[Any] = []
    for attr in ("usage_metadata", "response_metadata"):
        data = getattr(value, attr, None)
        if isinstance(data, dict):
            candidates.append(data)
            nested = data.get("token_usage")
            if isinstance(nested, dict):
                candidates.append(nested)
    for data in candidates:
        input_tokens = _number(data.get("input_tokens", data.get("prompt_tokens")))
        output_tokens = _number(data.get("output_tokens", data.get("completion_tokens")))
        if input_tokens is not None or output_tokens is not None:
            return input_tokens, output_tokens
    return None, None


def _rate(name: str) -> float:
    try:
        return max(0.0, float(os.getenv(name, "0")))
    except ValueError:
        return 0.0


def record_call(
    *,
    schema: str,
    model: str,
    input_chars: int,
    output: Any = None,
    usage: tuple[int | None, int | None] | None = None,
    latency_ms: float = 0.0,
    success: bool = True,
) -> None:
    """Record one model attempt without retaining content."""
    global _calls_total, _failed_total, _input_tokens, _output_tokens
    global _latency_sum_ms, _estimated_calls, _cost_usd
    actual_in, actual_out = usage or (None, None)
    estimated = actual_in is None or actual_out is None
    in_tokens = actual_in if actual_in is not None else max(0, math.ceil(input_chars / 2))
    out_tokens = actual_out if actual_out is not None else estimate_tokens(output)
    model_label = (model or "unknown").strip()[:80] or "unknown"
    schema_label = (schema or "unknown").strip()[:80] or "unknown"
    cost = (
        (in_tokens / 1000) * _rate("LLM_INPUT_USD_PER_1K")
        + (out_tokens / 1000) * _rate("LLM_OUTPUT_USD_PER_1K")
    )
    with _lock:
        _calls_total += 1
        _failed_total += 0 if success else 1
        _input_tokens += in_tokens
        _output_tokens += out_tokens
        _latency_sum_ms += max(0.0, float(latency_ms))
        _estimated_calls += 1 if estimated else 0
        _cost_usd += cost
        _by_model[model_label] += 1
        _by_schema[schema_label] += 1
        run_id = _current_run_id.get()
        if run_id:
            usage = _by_run[run_id]
            usage["calls_total"] += 1
            usage["failed_total"] += 0 if success else 1
            usage["input_tokens"] += in_tokens
            usage["output_tokens"] += out_tokens
            usage["estimated_calls"] += 1 if estimated else 0
            usage["latency_micros"] += round(max(0.0, float(latency_ms)) * 1000)
            usage["estimated_cost_microusd"] += round(cost * 1_000_000)


def take_run_snapshot(run_id: str) -> dict[str, Any]:
    """Remove and return privacy-safe usage attributable to one agent run."""
    with _lock:
        data = _by_run.pop(run_id, Counter())
    calls = int(data["calls_total"])
    estimated = int(data["estimated_calls"])
    input_tokens = int(data["input_tokens"])
    output_tokens = int(data["output_tokens"])
    return {
        "calls_total": calls,
        "failed_total": int(data["failed_total"]),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_calls": estimated,
        "usage_exact_ratio": round((calls - estimated) / calls, 3) if calls else 0.0,
        "latency_sum_ms": round(data["latency_micros"] / 1000, 3),
        "estimated_cost_usd": round(data["estimated_cost_microusd"] / 1_000_000, 6),
        "cost_rates_configured": bool(_rate("LLM_INPUT_USD_PER_1K") or _rate("LLM_OUTPUT_USD_PER_1K")),
        "attribution": "request_context",
    }


def snapshot() -> dict[str, Any]:
    with _lock:
        return {
            "calls_total": _calls_total,
            "failed_total": _failed_total,
            "input_tokens": _input_tokens,
            "output_tokens": _output_tokens,
            "total_tokens": _input_tokens + _output_tokens,
            "estimated_calls": _estimated_calls,
            "usage_exact_ratio": round(
                (_calls_total - _estimated_calls) / _calls_total, 3
            ) if _calls_total else 0.0,
            "latency_sum_ms": round(_latency_sum_ms, 3),
            "latency_avg_ms": round(_latency_sum_ms / _calls_total, 3) if _calls_total else 0.0,
            "estimated_cost_usd": round(_cost_usd, 6),
            "by_model": dict(_by_model),
            "by_schema": dict(_by_schema),
        }


def usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Calculate usage for one controlled evaluation window."""
    calls = max(0, int(after.get("calls_total", 0)) - int(before.get("calls_total", 0)))
    estimated = max(
        0,
        int(after.get("estimated_calls", 0)) - int(before.get("estimated_calls", 0)),
    )
    input_tokens = max(
        0,
        int(after.get("input_tokens", 0)) - int(before.get("input_tokens", 0)),
    )
    output_tokens = max(
        0,
        int(after.get("output_tokens", 0)) - int(before.get("output_tokens", 0)),
    )
    return {
        "calls_total": calls,
        "failed_total": max(
            0,
            int(after.get("failed_total", 0)) - int(before.get("failed_total", 0)),
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_calls": estimated,
        "usage_exact_ratio": round((calls - estimated) / calls, 3) if calls else 0.0,
        "latency_sum_ms": round(max(
            0.0,
            float(after.get("latency_sum_ms", 0)) - float(before.get("latency_sum_ms", 0)),
        ), 3),
        "estimated_cost_usd": round(max(
            0.0,
            float(after.get("estimated_cost_usd", 0)) - float(before.get("estimated_cost_usd", 0)),
        ), 6),
        "cost_rates_configured": bool(
            _rate("LLM_INPUT_USD_PER_1K") or _rate("LLM_OUTPUT_USD_PER_1K")
        ),
    }


def prometheus_lines() -> list[str]:
    data = snapshot()
    return [
        "# HELP travelmind_llm_calls_total LLM invocation attempts.",
        "# TYPE travelmind_llm_calls_total counter",
        f"travelmind_llm_calls_total {data['calls_total']}",
        "# HELP travelmind_llm_failed_total Failed LLM invocation attempts.",
        "# TYPE travelmind_llm_failed_total counter",
        f"travelmind_llm_failed_total {data['failed_total']}",
        "# HELP travelmind_llm_input_tokens_total Input tokens (provider usage or estimate).",
        "# TYPE travelmind_llm_input_tokens_total counter",
        f"travelmind_llm_input_tokens_total {data['input_tokens']}",
        "# HELP travelmind_llm_output_tokens_total Output tokens (provider usage or estimate).",
        "# TYPE travelmind_llm_output_tokens_total counter",
        f"travelmind_llm_output_tokens_total {data['output_tokens']}",
        "# HELP travelmind_llm_latency_ms_sum Cumulative LLM latency in milliseconds.",
        "# TYPE travelmind_llm_latency_ms_sum counter",
        f"travelmind_llm_latency_ms_sum {data['latency_sum_ms']}",
        "# HELP travelmind_llm_estimated_cost_usd_total Estimated LLM cost in USD.",
        "# TYPE travelmind_llm_estimated_cost_usd_total counter",
        f"travelmind_llm_estimated_cost_usd_total {data['estimated_cost_usd']}",
        "# HELP travelmind_llm_usage_exact_ratio Ratio of calls with provider token metadata.",
        "# TYPE travelmind_llm_usage_exact_ratio gauge",
        f"travelmind_llm_usage_exact_ratio {data['usage_exact_ratio']}",
    ]


def reset_for_tests() -> None:
    """Reset process counters for isolated unit tests."""
    global _calls_total, _failed_total, _input_tokens, _output_tokens
    global _latency_sum_ms, _estimated_calls, _cost_usd
    with _lock:
        _calls_total = _failed_total = _input_tokens = _output_tokens = 0
        _latency_sum_ms = _cost_usd = 0.0
        _estimated_calls = 0
        _by_model.clear()
        _by_schema.clear()
        _by_run.clear()
