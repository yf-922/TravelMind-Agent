"""Centralized dispatch with an auditable structured-message log."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from app.multi_agent_core.agents import BaseAgent
from app.multi_agent_core.messages import AgentMessage

logger = logging.getLogger(__name__)


class Supervisor:
    """Owns task state and logs; worker private memories remain inaccessible here."""

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self.agents = agents
        self.dispatch_log: list[AgentMessage] = []

    def _dispatch(self, task_id: str, task_type: str, to: str, content: dict[str, Any], attempt: int = 0) -> AgentMessage:
        message = AgentMessage(
            task_id=task_id,
            task_type=task_type,
            **{"from": "supervisor", "to": to},
            content=content,
            status="running",
            attempt=attempt,
        )
        self.dispatch_log.append(message)
        logger.info("[dispatch] task=%s to=%s type=%s attempt=%d", task_id, to, task_type, attempt)
        result = self.agents[to].run(message)
        self.dispatch_log.append(result)
        logger.info("[result] task=%s from=%s status=%s", task_id, result.from_agent, result.status)
        return result

    def run_trip(self, user_request: str, destination_hint: str) -> dict[str, Any]:
        # A dispatch trace belongs to one user task. Worker memories remain private
        # on their own instances and are never copied into this task log.
        self.dispatch_log = []
        task_id = f"trip-{uuid4()}"
        intent = self._dispatch(task_id, "intent_extract", "intent_agent", {
            "user_request": user_request,
            "destination_hint": destination_hint,
        })
        research = self._dispatch(task_id, "poi_research", "poi_research_agent", {
            "destination": intent.content["destination"],
            "place_query": user_request,
        })
        plan = self._dispatch(task_id, "itinerary_plan", "planner_agent", {
            "intent": intent.content,
            "candidates": research.content["candidates"],
        })
        review = self._dispatch(task_id, "itinerary_review", "reviewer_agent", {
            "itinerary": plan.content["itinerary"],
            "candidates": research.content["candidates"],
        })

        # Advanced conditional route: a rejected draft is repaired once, then reviewed again.
        if not review.content["approved"]:
            plan = self._dispatch(task_id, "itinerary_revise", "planner_agent", {
                "intent": intent.content,
                "candidates": research.content["candidates"],
                "review_instruction": review.content["revision_instruction"],
            }, attempt=1)
            review = self._dispatch(task_id, "itinerary_review", "reviewer_agent", {
                "itinerary": plan.content["itinerary"],
                "candidates": research.content["candidates"],
            }, attempt=1)

        return {
            "task_id": task_id,
            "intent": intent.content,
            "candidates": research.content["candidates"],
            "itinerary": plan.content["itinerary"],
            "review": review.content,
            "dispatch_log": [message.model_dump(by_alias=True) for message in self.dispatch_log],
        }
