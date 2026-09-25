"""对固定 Golden Set 跑 LLM-as-Judge，结果写入 evaluation/llm_judge_report.json。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.llm_judge import judge_itinerary, judge_metadata
from app.core.eval_safety import require_external_calls
from app.multi_agent_core.agents import IntentAgent, POIResearchAgent, PlannerAgent, ReviewerAgent
from app.multi_agent_core.supervisor import Supervisor
from app.multi_agent_core.tools import FixturePoiTool
from tests.test_e2e_evaluation import CASES


def make_supervisor() -> Supervisor:
    return Supervisor({
        "intent_agent": IntentAgent(), "poi_research_agent": POIResearchAgent(FixturePoiTool()),
        "planner_agent": PlannerAgent(), "reviewer_agent": ReviewerAgent(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="可选：覆盖 .env.local 中的默认模型")
    parser.add_argument("--limit", type=int, default=None, help="仅跑前 N 条，便于小额试运行")
    parser.add_argument("--allow-external-calls", action="store_true",
                        help="确认调用真实 LLM Judge 并消耗 API 额度")
    args = parser.parse_args()
    require_external_calls(
        parser,
        allowed=args.allow_external_calls,
        operation="LLM-as-Judge evaluation",
    )
    records = []
    for case_id, request, destination in CASES[:args.limit]:
        result = make_supervisor().run_trip(request, destination, session_id=case_id)
        started = time.perf_counter()
        try:
            score = judge_itinerary(
                user_request=request, destination=destination, candidates=result["candidates"],
                itinerary=result["itinerary"], review=result["review"], model=args.model,
            )
            record = {"case_id": case_id, "judge": score.model_dump(), "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:  # 保留失败，不让单个 Judge 错误中断整轮评估
            record = {"case_id": case_id, "judge_error": type(exc).__name__, "detail": str(exc), "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        records.append(record)
        print(f"{case_id}: {'done' if 'judge' in record else 'failed'}")

    judged = [item["judge"] for item in records if "judge" in item]
    pass_count = sum(1 for item in judged if item["pass_case"])
    categories: dict[str, int] = {}
    for item in judged:
        for category in item["failure_categories"]:
            categories[category] = categories.get(category, 0) + 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_metadata": judge_metadata(args.model), "total_cases": len(records),
        "judged_cases": len(judged), "judge_pass_rate": round(pass_count / len(judged), 4) if judged else None,
        "average_judge_latency_ms": round(statistics.mean(item["latency_ms"] for item in records), 2) if records else None,
        "failure_categories": categories, "records": records,
    }
    output = PROJECT_ROOT / "evaluation" / "llm_judge_report.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total_cases", "judged_cases", "judge_pass_rate", "average_judge_latency_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
