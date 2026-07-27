"""运行离线端到端评测并生成可版本对比的 JSON 报告。"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.multi_agent_core.agents import IntentAgent, POIResearchAgent, PlannerAgent, ReviewerAgent
from app.multi_agent_core.supervisor import Supervisor
from app.multi_agent_core.tools import FixturePoiTool

CASES = [
    ("E2E-01", "帮我规划北京文化一日游", "北京"), ("E2E-02", "重庆慢节奏一日游，想吃本地小吃", "重庆"),
    ("E2E-03", "北京亲子轻松游，少走路", "北京"), ("E2E-04", "重庆雨天也能去的地方", "重庆"),
    ("E2E-05", "北京第一次旅行，安排经典景点", "北京"), ("E2E-06", "重庆周末半日游，不要排得太满", "重庆"),
    ("E2E-07", "北京预算有限的城市漫步", "北京"), ("E2E-08", "重庆朋友同行，安排有代表性的景点", "重庆"),
    ("E2E-09", "北京历史建筑参观路线", "北京"), ("E2E-10", "重庆第一次来，给我一条可执行路线", "重庆"),
]


def run_case(case_id: str, user_request: str, destination: str) -> dict:
    supervisor = Supervisor({
        "intent_agent": IntentAgent(), "poi_research_agent": POIResearchAgent(FixturePoiTool()),
        "planner_agent": PlannerAgent(), "reviewer_agent": ReviewerAgent(),
    })
    started = time.perf_counter()
    result = supervisor.run_trip(user_request, destination, session_id=case_id)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    passed = result.get("status") != "failed" and bool(result.get("itinerary")) and bool(result.get("review", {}).get("approved"))
    return {
        "case_id": case_id, "input": user_request, "passed": passed, "latency_ms": latency_ms,
        "agent_trace": [entry["to"] for entry in result.get("dispatch_log", []) if entry.get("from") == "supervisor"],
        "failure_category": None if passed else (result.get("error_code") or "business_assertion_failed"),
    }


def main() -> None:
    records = [run_case(*case) for case in CASES]
    passed = [item for item in records if item["passed"]]
    failures: dict[str, int] = {}
    for item in records:
        if not item["passed"]:
            key = item["failure_category"] or "unknown"
            failures[key] = failures.get(key, 0) + 1
    report = {
        "report_version": "v1-offline-fixture", "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "集中式 Supervisor 多 Agent 核心；Fixture POI，未调用外部 LLM/API",
        "total_cases": len(records), "passed_cases": len(passed),
        "success_rate": round(len(passed) / len(records), 4),
        "average_latency_ms": round(statistics.mean(item["latency_ms"] for item in records), 2),
        "failure_categories": failures, "records": records,
        "next_optimization": [
            "接入真实 LLM 后，对同一 Golden Set 记录结构化输出校验与 LLM-as-Judge 分数。",
            "将高德 POI、天气、票价接口的超时、限流和空结果分别注入，统计降级成功率。",
            "补充真实用户失败请求，并按目的地缺失、日期缺失、候选池不足、工具失败分类回归。",
        ],
    }
    out = PROJECT_ROOT / "evaluation" / "latest_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total_cases", "passed_cases", "success_rate", "average_latency_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
