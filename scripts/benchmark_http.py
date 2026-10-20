"""Measure real loopback HTTP/SQLite reads in an isolated application process.

Never uses the running personal instance, .env.local, or external providers.
This is a closed-loop local API benchmark, not LLM throughput or online SLA.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENDPOINTS = ("/api/health", "/api/profile", "/api/history")


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def summarize(rows, elapsed):
    latency = [row["elapsed_ms"] for row in rows]
    succeeded = sum(row["ok"] for row in rows)
    return {
        "requests": len(rows), "succeeded": succeeded, "failed": len(rows) - succeeded,
        "error_rate": (len(rows) - succeeded) / len(rows) if rows else None,
        "wall_seconds": round(elapsed, 4),
        "completed_rps": round(len(rows) / elapsed, 2) if elapsed else None,
        "successful_rps": round(succeeded / elapsed, 2) if elapsed else None,
        "latency_ms_including_failures": {
            "p50": round(percentile(latency, .5), 3) if latency else None,
            "p95": round(percentile(latency, .95), 3) if latency else None,
            "p99": round(percentile(latency, .99), 3) if latency else None,
            "mean": round(statistics.mean(latency), 3) if latency else None,
            "max": round(max(latency), 3) if latency else None,
        },
        "status_codes": dict(Counter(str(row["status_code"]) for row in rows)),
        "errors": dict(Counter(row["error"] for row in rows if row["error"])),
    }


def validate_response(endpoint, response, user, history_per_user):
    if response.status_code != 200:
        return "HTTP_STATUS"
    try:
        body = response.json()
        if endpoint == "/api/health":
            return None if body.get("database") == "ok" else "DATABASE_NOT_READY"
        if endpoint == "/api/profile":
            return None if body.get("habit_prefs") == [f"marker-{user}"] else "PROFILE_OR_OWNER_MISMATCH"
        expected = {f"plan-{user}-{i}" for i in range(history_per_user)}
        actual = {item["id"] for item in body}
        return None if len(body) == history_per_user and actual == expected else "HISTORY_OR_OWNER_MISMATCH"
    except (ValueError, TypeError, KeyError, AttributeError):
        return "INVALID_RESPONSE"


async def measure(base_url, endpoint, concurrency, requests, warmups, tokens, history_per_user):
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, trust_env=False, timeout=10, limits=limits) as client:
        async def request(index):
            user = index % len(tokens)
            started = time.perf_counter()
            status = None
            error = None
            try:
                response = await client.get(endpoint, headers={"Authorization": "Bearer " + tokens[user]})
                status = response.status_code
                error = validate_response(endpoint, response, user, history_per_user)
            except httpx.HTTPError as exc:
                error = type(exc).__name__
            return {"index": index, "status_code": status, "ok": error is None,
                    "error": error, "elapsed_ms": round((time.perf_counter() - started) * 1000, 4)}

        for index in range(warmups):
            row = await request(index)
            if not row["ok"]:
                raise RuntimeError("warmup failed: " + str(row["error"]))
        queue = asyncio.Queue()
        for index in range(requests):
            queue.put_nowait(index)
        rows = []

        async def worker():
            while True:
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                rows.append(await request(index))

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - started
        return {"endpoint": endpoint, "concurrency": concurrency, "warmup_requests": warmups,
                **summarize(rows, elapsed), "samples": sorted(rows, key=lambda r: r["index"])}


def serve(port, directory):
    from app.core import database
    from app.core import env
    # No personal configuration or existing database is read by this process.
    with patch.object(env, "load_local_env", lambda: None), \
         patch.object(database, "_DB_PATH", directory / "application.db"):
        import app.main as main_module
        from app.core.memory import save_itinerary, set_user_profile
        with database.get_conn() as conn:
            for user in range(50):
                user_id = f"bench-{user}"
                conn.execute("INSERT INTO users VALUES (?,?,?,?)", (user_id, user_id, "unused", "2026-10-20"))
                set_user_profile(user_id, {"habit_prefs": [f"marker-{user}"]}, conn)
                for index in range(20):
                    plan_id = save_itinerary(user_id, {
                        "destination": "南京", "days": [{"day": day, "timeline": []} for day in (1, 2, 3)],
                    }, "南京三日游", conn)
                    conn.execute("UPDATE itineraries SET id=? WHERE id=?", (f"plan-{user}-{index}", plan_id))
        import uvicorn
        uvicorn.run(main_module.app, host="127.0.0.1", port=port, workers=1,
                    access_log=False, log_level="warning")


@contextmanager
def isolated_server(directory):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    child_env = {key: value for key, value in os.environ.items()
                 if not any(word in key.upper() for word in ("API_KEY", "TOKEN", "PASSWORD", "SECRET"))}
    secret = secrets.token_hex(32)
    child_env.update(REDIS_URL="", SEMANTIC_MEMORY_ENABLED="0", TRAVEL_KNOWLEDGE_ENABLED="0",
                     TRAVEL_INTEREST_KNOWLEDGE_ENABLED="0", JWT_SECRET=secret)
    log = (directory / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve", str(port),
                                "--directory", str(directory)], cwd=ROOT, env=child_env,
                               stdout=log, stderr=log)
    try:
        base_url = f"http://127.0.0.1:{port}"
        with httpx.Client(trust_env=False, timeout=.5) as client:
            deadline = time.monotonic() + 45
            while True:
                if process.poll() is not None:
                    raise RuntimeError("isolated benchmark server exited; " + str(directory / "server.log"))
                try:
                    if client.get(base_url + "/api/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("isolated benchmark startup timeout")
                time.sleep(.1)
        import jwt
        tokens = [jwt.encode({"sub": f"bench-{user}"}, secret, algorithm="HS256") for user in range(50)]
        yield base_url, tokens
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()


def markdown(report):
    lines = ["# 本机 HTTP 并发压测", "", f"执行时间：{report['captured_at']}。", "",
             "真实 loopback TCP → Uvicorn 单进程 → FastAPI → JWT → 临时 SQLite。",
             "50 个合成用户，每用户 20 条行程；响应校验检查用户画像与历史归属。",
             "不加载 .env.local，不读取个人数据库；关闭 Redis、Chroma 与外部模型 / 地图调用。", "",
             "闭环负载：固定数量 worker 完成请求后再发送下一条；预热排除，失败保留在分母。",
             "P50/P95/P99 为客户端从发出请求到完整读取响应的时延；不包含 worker 外排队时间。",
             "客户端与服务端同机，有限样本，不能代表线上 SLA、模型规划吞吐或最大容量。", "",
             f"环境：Python {report['environment']['python']}，{report['environment']['platform']}；"
             f"逻辑 CPU {report['environment']['logical_cpus']}。", "",
             f"每接口 / 并发组合重复 {report['configuration']['repetitions']} 次，"
             f"每次 {report['configuration']['requests_per_repetition']} 请求、预热 {report['configuration']['warmups']} 请求。",
             "每个重复独立连接池；表中时延为全部实测样本合并，吞吐为总请求 / 各重复测量时间之和。", "",
             "| 接口 | 并发 | 请求数 | 成功/失败 | 吞吐 req/s | P50 ms | P95 ms | P99 ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["scenarios"]:
        lat = row["latency_ms_including_failures"]
        lines.append(f"| `{row['endpoint']}` | {row['concurrency']} | {row['requests']} | "
                     f"{row['succeeded']}/{row['failed']} | {row['completed_rps']} | "
                     f"{lat['p50']} | {lat['p95']} | {lat['p99']} |")
    lines.extend(["", "`completed_rps` 包含失败；`successful_rps`、HTTP 状态、错误原因及逐请求时延见同名 JSON。",
                  "并发档位独立建连接池，每档先预热；执行次序为接口顺序、并发由小到大。", "",
                  "## 复现", "", "```bash",
                  "python scripts/benchmark_http.py --requests 300 --concurrency 1 8 32 --warmups 12 --repetitions 3",
                  "```", "", "原始结果：`evaluation/http_load_20261020.json`。",
                  "源码指纹绑定应用代码与本脚本；结果没有用户密钥、Bearer Token 或个人行程。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--warmups", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=ROOT / "evaluation/http_load_20261020.json")
    parser.add_argument("--markdown-out", type=Path, default=ROOT / "evaluation/http_load_20261020.md")
    parser.add_argument("--serve", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve, args.directory)
        return 0
    if (not 1 <= args.requests <= 10000 or any(not 1 <= c <= 128 for c in args.concurrency)
            or not 0 <= args.warmups <= 100 or not 1 <= args.repetitions <= 10):
        parser.error("requests: 1..10000; concurrency: 1..128; warmups: 0..100; repetitions: 1..10")
    sources = sorted((ROOT / "app").rglob("*.py")) + [Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    report = {
        "schema_version": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
        "measurement_type": "real-loopback-http-single-process-temporary-sqlite",
        "external_calls": False, "production_database_used": False,
        "source_fingerprint": digest.hexdigest(), "load_model": "closed-loop",
        "configuration": {"requests_per_repetition": args.requests, "concurrency": args.concurrency,
                          "warmups": args.warmups, "repetitions": args.repetitions},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "logical_cpus": os.cpu_count(),
                        "dependencies": {name: importlib.metadata.version(name) for name in ("fastapi", "uvicorn", "httpx")}},
        "dataset": {"synthetic_users": 50, "itineraries_per_user": 20, "total_itineraries": 1000},
        "scenarios": [],
    }
    with tempfile.TemporaryDirectory(prefix="travelmind-http-benchmark-") as directory:
        with isolated_server(Path(directory)) as (base_url, tokens):
            for endpoint in ENDPOINTS:
                for concurrency in args.concurrency:
                    trials = []
                    samples = []
                    for repetition in range(args.repetitions):
                        trial = asyncio.run(measure(base_url, endpoint, concurrency, args.requests, args.warmups, tokens, 20))
                        samples.extend({**sample, "repetition": repetition + 1} for sample in trial.pop("samples"))
                        trials.append(trial)
                    row = {"endpoint": endpoint, "concurrency": concurrency, "trials": trials, "samples": samples,
                           **summarize(samples, sum(trial["wall_seconds"] for trial in trials))}
                    report["scenarios"].append(row)
                    print(endpoint, concurrency, row["succeeded"], row["failed"], row["latency_ms_including_failures"], flush=True)
    for path, content in ((args.json_out, json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
                          (args.markdown_out, markdown(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return int(any(row["failed"] for row in report["scenarios"]))


if __name__ == "__main__":
    raise SystemExit(main())
