"""Independent Agent instances for the Experiment 3 orchestration core."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Protocol

from app.multi_agent_core.messages import AgentMessage


class PoiTool(Protocol):
    def search(self, city: str, query: str = "") -> list[dict[str, Any]]: ...


class BaseAgent(ABC):
    """Each instance owns its prompt and memory. No global message list is used."""

    def __init__(self, name: str, system_prompt: str) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.private_memory: list[dict[str, str]] = []

    def _remember(self, role: str, content: str) -> None:
        self.private_memory.append({"role": role, "content": content})

    def _reply(self, message: AgentMessage, payload: dict[str, Any]) -> AgentMessage:
        self._remember("user", json.dumps(message.content, ensure_ascii=False))
        self._remember("assistant", json.dumps(payload, ensure_ascii=False))
        return AgentMessage(
            task_id=message.task_id,
            task_type=message.task_type,
            **{"from": self.name, "to": "supervisor"},
            content=payload,
            status="done",
            attempt=message.attempt,
            trace_id=message.trace_id,
        )

    @abstractmethod
    def run(self, message: AgentMessage) -> AgentMessage:
        """Process a single structured message using only this agent's memory."""


class IntentAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            "intent_agent",
            "Extract the destination and stable travel constraints. Do not plan an itinerary or search POIs.",
        )

    def run(self, message: AgentMessage) -> AgentMessage:
        request = str(message.content.get("user_request", ""))
        destination = str(message.content.get("destination_hint", "")).strip()
        return self._reply(message, {
            "user_request": request,
            "destination": destination,
            "constraints": "Use the request as the source of truth; ask for missing dates in the UI.",
        })


class POIResearchAgent(BaseAgent):
    def __init__(self, tool: PoiTool) -> None:
        super().__init__(
            "poi_research_agent",
            "Verify travel places with the POI tool. Never invent a place, coordinate, or address.",
        )
        self.tool = tool

    def run(self, message: AgentMessage) -> AgentMessage:
        city = str(message.content.get("destination", "")).strip()
        query = str(message.content.get("place_query", "")).strip()
        if not city:
            return self._reply(message, {"candidates": [], "error": "destination is required"})
        candidates = self.tool.search(city, query)
        return self._reply(message, {
            "destination": city,
            "candidates": candidates,
            "tool_used": self.tool.__class__.__name__,
        })


class PlannerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            "planner_agent",
            "Create an itinerary only from verified POI candidates. Do not judge your own plan.",
        )

    def run(self, message: AgentMessage) -> AgentMessage:
        candidates = list(message.content.get("candidates", []))
        selected = candidates[:3]
        itinerary = [
            {"order": index + 1, "name": item.get("name", "unknown"), "address": item.get("address", "")}
            for index, item in enumerate(selected)
        ]
        return self._reply(message, {
            "itinerary": itinerary,
            "planning_note": "The draft uses only POIs returned by the research tool.",
        })


class ReviewerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            "reviewer_agent",
            "Review a draft against verified candidates. Do not rewrite the itinerary.",
        )

    def run(self, message: AgentMessage) -> AgentMessage:
        itinerary = list(message.content.get("itinerary", []))
        candidate_names = {item.get("name") for item in message.content.get("candidates", [])}
        unknown = [item.get("name") for item in itinerary if item.get("name") not in candidate_names]
        approved = bool(itinerary) and not unknown
        return self._reply(message, {
            "approved": approved,
            "issues": ([] if approved else ["Draft is empty or includes an unverified POI."]),
            "revision_instruction": ("" if approved else "Keep only verified POI candidates and create a non-empty route."),
        })

