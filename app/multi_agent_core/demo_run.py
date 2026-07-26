"""Run an end-to-end Experiment 3 demonstration and print evidence logs."""

from __future__ import annotations

import argparse
import json
import logging

from app.multi_agent_core.agents import IntentAgent, POIResearchAgent, PlannerAgent, ReviewerAgent
from app.multi_agent_core.supervisor import Supervisor
from app.multi_agent_core.tools import AmapPoiTool, FixturePoiTool
from app.multi_agent_core.memory import SQLiteAgentMemoryStore


def build_supervisor(offline: bool, memory_store: SQLiteAgentMemoryStore) -> Supervisor:
    tool = FixturePoiTool() if offline else AmapPoiTool()
    return Supervisor({
        "intent_agent": IntentAgent(memory_store=memory_store),
        "poi_research_agent": POIResearchAgent(tool, memory_store=memory_store),
        "planner_agent": PlannerAgent(memory_store=memory_store),
        "reviewer_agent": ReviewerAgent(memory_store=memory_store),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Beijing")
    parser.add_argument("--request", default="Plan a relaxed one-day trip with cultural places.")
    parser.add_argument("--offline", action="store_true", help="Use fixture POIs instead of the live Amap tool.")
    parser.add_argument("--session-id", default="demo-session", help="Session key for persistent, Agent-isolated memory.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    memory_store = SQLiteAgentMemoryStore()
    supervisor = build_supervisor(args.offline, memory_store)
    result = supervisor.run_trip(args.request, args.city, session_id=args.session_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n=== PRIVATE MEMORY COUNTS ===")
    for name, agent in supervisor.agents.items():
        print(f"{name}: {len(agent.private_memory)} entries")
    print("\n=== PRIVATE MEMORY CONTENTS ===")
    for name, agent in supervisor.agents.items():
        print(f"\n[{name}] system_prompt={agent.system_prompt}")
        print(json.dumps(agent.private_memory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
