"""Centralized dispatch with an auditable structured-message log."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from app.multi_agent_core.agents import BaseAgent
from app.multi_agent_core.messages import AgentMessage
from app.multi_agent_core.protocols import AgentCard, agent_card

logger = logging.getLogger(__name__)


class Supervisor:
    """Owns task state and logs; worker private memories remain inaccessible here."""

    def __init__(self, agents: dict[str, BaseAgent], *, max_attempts: int = 2) -> None:
        self.agents = agents
        self.max_attempts = max(1, max_attempts)
        self.dispatch_log: list[AgentMessage] = []

    def route_after(self, result: AgentMessage, *, review_passed: bool | None = None) -> str:
        """Central routing policy; workers never decide which worker runs next."""
        if result.status == "failed":
            return "stop"
        if result.task_type == "intent_extract":
            return "poi_research_agent"
        if result.task_type == "poi_research":
            return "planner_agent"
        if result.task_type == "itinerary_plan":
            return "reviewer_agent"
        if result.task_type == "itinerary_review":
            return "done" if review_passed else "planner_agent"
        return "stop"

    def agent_cards(self) -> list[AgentCard]:
        """Expose A2A-style capability cards without exposing private memory."""
        return [
            agent_card(
                agent.name,
                agent.system_prompt,
                skills=sorted(agent.allowed_tools),
                accepted_task_types=[
                    task_type for task_type, target in {
                        "intent_extract": "intent_agent",
                        "poi_research": "poi_research_agent",
                        "itinerary_plan": "planner_agent",
                        "itinerary_review": "reviewer_agent",
                    }.items() if target == agent.name
                ],
            )
            for agent in self.agents.values()
        ]

    def _dispatch(self, task_id: str, session_id: str, task_type: str, to: str, content: dict[str, Any], attempt: int = 0) -> AgentMessage:
        message = AgentMessage(
            task_id=task_id,
            session_id=session_id,
            task_type=task_type,
            **{"from": "supervisor", "to": to},
            content=content,
            status="running",
            attempt=attempt,
        )
        self.dispatch_log.append(message)
        logger.info("[dispatch] task=%s to=%s type=%s attempt=%d", task_id, to, task_type, attempt)
        agent = self.agents[to]
        for current_attempt in range(attempt, self.max_attempts):
            if current_attempt > attempt:
                message = message.model_copy(update={"attempt": current_attempt, "status": "retrying"})
                self.dispatch_log.append(message)
            try:
                result = agent.run(message)
                self.dispatch_log.append(result)
                logger.info("[result] task=%s from=%s status=%s", task_id, result.from_agent, result.status)
                return result
            except Exception as exc:  # noqa: BLE001 - supervisor converts worker failures to messages
                logger.warning("[worker-error] task=%s agent=%s attempt=%d error=%s", task_id, to, current_attempt, exc)
                if current_attempt + 1 >= self.max_attempts:
                    failed = AgentMessage(
                        task_id=task_id,
                        session_id=session_id,
                        task_type=task_type,
                        **{"from": to, "to": "supervisor"},
                        content={"error": "worker failed after retries"},
                        status="failed",
                        attempt=current_attempt,
                        trace_id=message.trace_id,
                        error_code=type(exc).__name__,
                    )
                    self.dispatch_log.append(failed)
                    return failed
        raise RuntimeError("unreachable supervisor retry state")

    def run_trip(self, user_request: str, destination_hint: str, *, session_id: str | None = None) -> dict[str, Any]:
        # A dispatch trace belongs to one user task. Worker memories remain private
        # on their own instances and are never copied into this task log.
        self.dispatch_log = []
        task_id = f"trip-{uuid4()}"
        active_session_id = session_id or task_id
        intent = self._dispatch(task_id, active_session_id, "intent_extract", "intent_agent", {
            "user_request": user_request,
            "destination_hint": destination_hint,
        })
        if intent.status == "failed":
            return self._failure_result(task_id, "intent_agent", "intent extraction failed", intent)
        research = self._dispatch(task_id, active_session_id, "poi_research", "poi_research_agent", {
            "destination": intent.content["destination"],
            "place_query": user_request,
        })
        if research.status == "failed":
            return self._failure_result(task_id, "poi_research_agent", "POI research failed", research)
        plan = self._dispatch(task_id, active_session_id, "itinerary_plan", "planner_agent", {
            "intent": intent.content,
            "candidates": research.content["candidates"],
        })
        if plan.status == "failed":
            return self._failure_result(task_id, "planner_agent", "planning failed", plan)
        review = self._dispatch(task_id, active_session_id, "itinerary_review", "reviewer_agent", {
            "itinerary": plan.content["itinerary"],
            "candidates": research.content["candidates"],
        })
        if review.status == "failed":
            return self._failure_result(task_id, "reviewer_agent", "review failed", review)

        # Advanced conditional route: a rejected draft is repaired once, then reviewed again.
        next_agent = self.route_after(review, review_passed=bool(review.content.get("approved")))
        if next_agent == "planner_agent":
            plan = self._dispatch(task_id, active_session_id, "itinerary_revise", "planner_agent", {
                "intent": intent.content,
                "candidates": research.content["candidates"],
                "review_instruction": review.content["revision_instruction"],
            }, attempt=1)
            review = self._dispatch(task_id, active_session_id, "itinerary_review", "reviewer_agent", {
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
            "tool_trace": self._tool_trace(task_id),
        }

    def _tool_trace(self, task_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for agent in self.agents.values():
            registry = getattr(agent, "tool_registry", None)
            for record in getattr(registry, "audit_log", []):
                if record.task_id == task_id:
                    records.append(record.model_dump())
        return records

    def _failure_result(self, task_id: str, agent: str, message: str, failure: AgentMessage) -> dict[str, Any]:
        """Return an actionable partial result instead of crashing on a worker failure."""
        return {
            "task_id": task_id,
            "status": "failed",
            "failed_agent": agent,
            "error": message,
            "error_code": failure.error_code,
            "dispatch_log": [entry.model_dump(by_alias=True) for entry in self.dispatch_log],
            "tool_trace": self._tool_trace(task_id),
        }
