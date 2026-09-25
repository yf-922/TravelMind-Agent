"""Run deterministic multi-Agent routing and tool-contract evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.multi_agent_core.agents import (
    IntentAgent,
    PlannerAgent,
    POIResearchAgent,
    ReviewerAgent,
)
from app.multi_agent_core.messages import AgentMessage
from app.multi_agent_core.supervisor import Supervisor
from app.multi_agent_core.tools import FixturePoiTool
from app.multi_agent_core.trajectory import TrajectoryContract, grade_trajectory


BASE_DISPATCHES = (
    ("intent_extract", "intent_agent", 0),
    ("poi_research", "poi_research_agent", 0),
    ("itinerary_plan", "planner_agent", 0),
    ("itinerary_review", "reviewer_agent", 0),
)


class RejectFirstDraftPlanner(PlannerAgent):
    def run(self, message: AgentMessage) -> AgentMessage:
        if message.task_type == "itinerary_plan":
            return self._reply(message, {
                "itinerary": [],
                "planning_note": "Deterministic rejected draft for trajectory evaluation.",
            })
        return super().run(message)


class FailNTimesPoiTool(FixturePoiTool):
    def __init__(self, failures: int) -> None:
        self.remaining_failures = failures

    def search(self, city: str, query: str = "") -> list[dict[str, Any]]:
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise TimeoutError("deterministic fixture timeout")
        return super().search(city, query)


def _supervisor(tool: FixturePoiTool, planner: PlannerAgent | None = None) -> Supervisor:
    return Supervisor({
        "intent_agent": IntentAgent(),
        "poi_research_agent": POIResearchAgent(tool),
        "planner_agent": planner or PlannerAgent(),
        "reviewer_agent": ReviewerAgent(),
    }, max_attempts=2)


def evaluate_scenarios() -> list[dict[str, Any]]:
    request = "Plan a relaxed cultural day without too much walking"
    destination = "Beijing"
    cases: list[tuple[Supervisor, TrajectoryContract]] = [
        (
            _supervisor(FixturePoiTool()),
            TrajectoryContract("happy_path", BASE_DISPATCHES, destination),
        ),
        (
            _supervisor(FixturePoiTool(), RejectFirstDraftPlanner()),
            TrajectoryContract(
                "reviewer_replan",
                BASE_DISPATCHES + (
                    ("itinerary_revise", "planner_agent", 1),
                    ("itinerary_review", "reviewer_agent", 1),
                ),
                destination,
                max_retry_attempt=1,
            ),
        ),
        (
            _supervisor(FailNTimesPoiTool(1)),
            TrajectoryContract(
                "transient_tool_retry",
                BASE_DISPATCHES[:2] + (
                    ("poi_research", "poi_research_agent", 1),
                ) + BASE_DISPATCHES[2:],
                destination,
                max_tool_attempts=2,
                max_retry_attempt=1,
            ),
        ),
        (
            _supervisor(FailNTimesPoiTool(2)),
            TrajectoryContract(
                "persistent_tool_failure",
                BASE_DISPATCHES[:2] + (("poi_research", "poi_research_agent", 1),),
                destination,
                max_tool_attempts=2,
                max_retry_attempt=1,
                expect_failure=True,
            ),
        ),
    ]
    results: list[dict[str, Any]] = []
    for supervisor, contract in cases:
        result = supervisor.run_trip(request, destination, session_id=f"eval-{contract.name}")
        results.append(grade_trajectory(result, contract))
    return results


def render_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Multi-Agent Trajectory Evaluation",
        "",
        "> Offline deterministic fixtures. No LLM, AMap, or other external API was called.",
        "",
        "| Scenario | Result | Checks | Dispatches | Tool attempts |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['contract']} | {'PASS' if result['passed'] else 'FAIL'} | "
            f"{result['passed_checks']}/{result['total_checks']} | "
            f"{len(result['dispatches'])} | {len(result['tool_trace'])} |"
        )
    failed = [
        f"{result['contract']}:{check['id']}"
        for result in results for check in result["checks"] if not check["passed"]
    ]
    lines.extend([
        "",
        f"- Overall: {sum(bool(item['passed']) for item in results)}/{len(results)} scenarios passed.",
        f"- Failed checks: {', '.join(failed) if failed else 'none'}.",
        "- Free-text queries are represented only by presence, length, and a 12-character SHA-256 prefix.",
        "- This suite proves orchestration contracts and failure handling, not semantic plan quality or online latency.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evaluation/agent_trajectory_report.md"))
    parser.add_argument("--json-out", type=Path, default=Path("evaluation/agent_trajectory_report.json"))
    args = parser.parse_args()
    results = evaluate_scenarios()
    payload = {
        "report_version": "v1-offline-trajectory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "external_calls_made": False,
        "scenarios": results,
        "summary": {
            "passed": sum(bool(item["passed"]) for item in results),
            "total": len(results),
        },
    }
    report = render_report(results)
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
