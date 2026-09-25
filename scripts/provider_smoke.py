"""Run a real provider smoke test without printing tokens or itinerary content.

The API container must already be running with the selected provider configured
in ``.env.local``. The script creates a disposable local account, consumes the
SSE stream, and prints only safe execution metadata for interview verification.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.eval_safety import require_external_calls


def _json_request(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def run(base_url: str, query: str, timeout: int, max_review_rounds: int) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    username = f"smoke_{uuid.uuid4().hex[:12]}"
    auth = _json_request(
        f"{base_url}/api/auth/register",
        {"username": username, "password": "SmokePass_2026!"},
    )
    token = str(auth.get("token") or "")
    if not token:
        raise RuntimeError("registration did not return a token")

    body = json.dumps(
        {
            "query": query,
            "max_per_day": 5,
            "min_rating": 4.2,
            "max_spots": 20,
            "max_review_rounds": max_review_rounds,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/plan/stream",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)

    result = next((event for event in reversed(events) if event.get("type") == "result"), None)
    error = next((event for event in reversed(events) if event.get("type") == "error"), None)
    types = [str(event.get("type")) for event in events if event.get("type")]
    return {
        "event_count": len(events),
        "event_types": types,
        "result_success": result.get("success") if result else None,
        "error_code": error.get("code") if error else None,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "run_id_present": any(event.get("type") == "run" and event.get("run_id") for event in events),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--query",
        default="Nanjing one-day trip on 2026-10-01 with historical attractions and local food",
    )
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-review-rounds", type=int, default=2)
    parser.add_argument(
        "--allow-external-calls",
        action="store_true",
        help="confirm that one real provider run and API cost are intentional",
    )
    args = parser.parse_args()
    require_external_calls(
        parser,
        allowed=args.allow_external_calls,
        operation="provider smoke test",
    )
    try:
        summary = run(args.base_url, args.query, args.timeout, args.max_review_rounds)
    except (OSError, urllib.error.URLError, RuntimeError, ValueError) as exc:
        print(json.dumps({"success": False, "error": type(exc).__name__}, ensure_ascii=True))
        return 1

    summary["success"] = summary["result_success"] is True and summary["error_code"] is None
    print(json.dumps(summary, ensure_ascii=True))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
