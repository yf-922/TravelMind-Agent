import json
from pathlib import Path
import subprocess
import sys

import httpx

from scripts.benchmark_http import summarize, validate_response


def test_failure_is_in_latency_denominator_and_throughput():
    rows = [
        {"ok": True, "elapsed_ms": 1, "status_code": 200, "error": None},
        {"ok": False, "elapsed_ms": 99, "status_code": 503, "error": "HTTP_STATUS"},
    ]
    report = summarize(rows, 2)
    assert report["requests"] == 2 and report["failed"] == 1
    assert report["error_rate"] == .5
    assert report["completed_rps"] == 1 and report["successful_rps"] == .5
    assert report["latency_ms_including_failures"]["p95"] == 99


def test_http_200_does_not_hide_wrong_user_data():
    wrong_profile = httpx.Response(200, json={"habit_prefs": ["marker-1"]})
    assert validate_response("/api/profile", wrong_profile, 0, 20) == "PROFILE_OR_OWNER_MISMATCH"
    wrong_history = httpx.Response(200, json=[{"id": "plan-1-0"}])
    assert validate_response("/api/history", wrong_history, 0, 1) == "HISTORY_OR_OWNER_MISMATCH"


def test_isolated_loopback_benchmark_has_real_http_and_all_samples(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "http.json"
    result = subprocess.run([
        sys.executable, str(root / "scripts/benchmark_http.py"),
        "--requests", "4", "--concurrency", "1", "2", "--warmups", "1", "--repetitions", "1",
        "--json-out", str(output), "--markdown-out", str(tmp_path / "http.md"),
    ], cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["production_database_used"] and not report["external_calls"]
    assert len(report["scenarios"]) == 6
    assert sum(row["requests"] for row in report["scenarios"]) == 24
    assert all(row["failed"] == 0 and len(row["samples"]) == 4 for row in report["scenarios"])
