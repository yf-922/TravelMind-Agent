"""Benchmark the real SSE planning pipeline without printing prompts or plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from provider_smoke import run
from app.core.env import load_local_env
from app.core.eval_safety import require_external_calls


TRACKED_PREFIXES = (
    "travelmind_llm_",
    "travelmind_agent_node_duration_ms_sum",
    "travelmind_agent_node_executions_total",
)


def _run_configuration() -> dict[str, object]:
    load_local_env()
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    model_key = {
        "grok": "GROK_MODEL",
        "deepseek": "DEEPSEEK_MODEL",
        "doubao": "DOUBAO_MODEL",
        "openai": "OPENAI_MODEL",
    }.get(provider, "OPENAI_MODEL")
    root = Path(__file__).resolve().parents[1]
    fingerprint = hashlib.sha256()
    for relative in ("app/planning/nodes.py", "app/planning/graph.py"):
        path = root / relative
        if path.exists():
            fingerprint.update(path.read_bytes())
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": os.getenv(model_key, "unknown"),
        "protocol": (
            "responses" if provider == "grok" and os.getenv("GROK_USE_RESPONSES_API", "1") == "1"
            else "chat_completions"
        ),
        "request_timeout_seconds": os.getenv(f"{provider.upper()}_REQUEST_TIMEOUT_SECONDS", "unknown"),
        "prompt_graph_fingerprint": fingerprint.hexdigest()[:16],
        "python": platform.python_version(),
        "pricing_rates_configured": bool(
            os.getenv("LLM_INPUT_USD_PER_1K") or os.getenv("LLM_OUTPUT_USD_PER_1K")
        ),
    }


def _metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/metrics", timeout=15) as response:
        text = response.read().decode("utf-8")
    values: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name, value = line.rsplit(None, 1)
            values[name] = float(value)
        except (ValueError, TypeError):
            continue
    return values


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return round(ordered[index], 3)


def _metric_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in sorted(after.items()):
        if name.startswith(TRACKED_PREFIXES) and not name.endswith("usage_exact_ratio"):
            delta = value - before.get(name, 0.0)
            if delta:
                result[name] = round(delta, 6)
    exact_ratio = after.get("travelmind_llm_usage_exact_ratio")
    if exact_ratio is not None:
        result["travelmind_llm_usage_exact_ratio_final"] = exact_ratio
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-review-rounds", type=int, default=2)
    parser.add_argument(
        "--allow-external-calls",
        action="store_true",
        help="confirm that real provider calls and API cost are intentional",
    )
    parser.add_argument(
        "--query",
        default="Nanjing one-day trip on 2026-10-01 with historical attractions and local food",
    )
    args = parser.parse_args()
    require_external_calls(
        parser,
        allowed=args.allow_external_calls,
        operation="online pipeline benchmark",
    )
    if not 1 <= args.runs <= 20:
        parser.error("--runs must be between 1 and 20")

    before = _metrics(args.base_url)
    results = [
        run(args.base_url, args.query, args.timeout, args.max_review_rounds)
        for _ in range(args.runs)
    ]
    after = _metrics(args.base_url)
    latencies = [float(item["elapsed_seconds"]) for item in results]
    success_count = sum(
        item.get("result_success") is True and item.get("error_code") is None
        for item in results
    )
    summary = {
        "configuration": _run_configuration(),
        "runs": args.runs,
        "successes": success_count,
        "success_rate": round(success_count / args.runs, 3),
        "latency_seconds": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3),
        },
        "metric_deltas": _metric_deltas(before, after),
        "error_codes": [item.get("error_code") for item in results if item.get("error_code")],
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0 if success_count == args.runs else 1


if __name__ == "__main__":
    sys.exit(main())
