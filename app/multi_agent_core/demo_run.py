"""Run an end-to-end Experiment 3 demonstration and print evidence logs."""

from __future__ import annotations

import argparse
import json
import logging

from app.multi_agent_core.agents import IntentAgent, POIResearchAgent, PlannerAgent, ReviewerAgent
from app.multi_agent_core.supervisor import Supervisor
from app.multi_agent_core.tools import AmapPoiTool, FixturePoiTool


def build_supervisor(offline: bool) -> Supervisor:
    tool = FixturePoiTool() if offline else AmapPoiTool()
    return Supervisor({
        "intent_agent": IntentAgent(),
        "poi_research_agent": POIResearchAgent(tool),
        "planner_agent": PlannerAgent(),
        "reviewer_agent": ReviewerAgent(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Beijing")
    parser.add_argument("--request", default="Plan a relaxed one-day trip with cultural places.")
    parser.add_argument("--offline", action="store_true", help="Use fixture POIs instead of the live Amap tool.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    supervisor = build_supervisor(args.offline)
    result = supervisor.run_trip(args.request, args.city)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n=== PRIVATE MEMORY COUNTS ===")
    for name, agent in supervisor.agents.items():
        print(f"{name}: {len(agent.private_memory)} entries")


if __name__ == "__main__":
    main()
