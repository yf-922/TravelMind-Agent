"""Independent Agent instances for the Experiment 3 orchestration core."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol

from app.multi_agent_core.messages import AgentMessage
from app.multi_agent_core.memory import AgentMemoryStore, InMemoryAgentMemoryStore


class PoiTool(Protocol):
    def search(self, city: str, query: str = "") -> list[dict[str, Any]]: ...


class BaseAgent(ABC):
    """Each instance owns its prompt and memory. No global message list is used."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        *,
        model: Callable[[str, list[dict[str, str]], dict[str, Any]], dict[str, Any]] | None = None,
        allowed_tools: set[str] | None = None,
        memory_store: AgentMemoryStore | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.private_memory: list[dict[str, str]] = []
        self.memory_store = memory_store or InMemoryAgentMemoryStore()
        self.model = model
        self.allowed_tools = frozenset(allowed_tools or set())

    def _session_id(self, message: AgentMessage) -> str:
        return message.session_id or message.task_id

    def _activate_memory(self, message: AgentMessage) -> str:
        session_id = self._session_id(message)
        self.private_memory = self.memory_store.load(session_id, self.name)
        return session_id

    def _remember(self, session_id: str, role: str, content: str) -> None:
        entry = {"role": role, "content": content}
        self.memory_store.append(session_id, self.name, entry)
        self.private_memory.append(entry)

    def _reply(self, message: AgentMessage, payload: dict[str, Any]) -> AgentMessage:
        session_id = self._activate_memory(message)
        self._remember(session_id, "user", json.dumps(message.content, ensure_ascii=False))
        self._remember(session_id, "assistant", json.dumps(payload, ensure_ascii=False))
        return AgentMessage(
            task_id=message.task_id,
            task_type=message.task_type,
            **{"from": self.name, "to": "supervisor"},
            content=payload,
            status="done",
            attempt=message.attempt,
            trace_id=message.trace_id,
        )

    def _model_payload(self, message: AgentMessage) -> dict[str, Any] | None:
        """Optional model seam: production callers can inject an LLM adapter; tests stay offline."""
        if self.model is None:
            return None
        self._activate_memory(message)
        payload = self.model(self.system_prompt, list(self.private_memory), message.content)
        if not isinstance(payload, dict):
            raise TypeError(f"{self.name} model must return a dict")
        return payload

    @abstractmethod
    def run(self, message: AgentMessage) -> AgentMessage:
        """Process a single structured message using only this agent's memory."""


class IntentAgent(BaseAgent):
    def __init__(self, *, model=None, memory_store: AgentMemoryStore | None = None) -> None:
        super().__init__(
            "intent_agent",
            "Extract the destination and stable travel constraints. Do not plan an itinerary or search POIs.",
            model=model,
            memory_store=memory_store,
        )

    def run(self, message: AgentMessage) -> AgentMessage:
        model_payload = self._model_payload(message)
        if model_payload is not None:
            return self._reply(message, model_payload)
        request = str(message.content.get("user_request", ""))
        destination = str(message.content.get("destination_hint", "")).strip()
        return self._reply(message, {
            "user_request": request,
            "destination": destination,
            "constraints": "Use the request as the source of truth; ask for missing dates in the UI.",
        })


class POIResearchAgent(BaseAgent):
    def __init__(self, tool: PoiTool, *, model=None, memory_store: AgentMemoryStore | None = None) -> None:
        super().__init__(
            "poi_research_agent",
            "Verify travel places with the POI tool. Never invent a place, coordinate, or address.",
            model=model,
            allowed_tools={"poi_search"},
            memory_store=memory_store,
        )
        self.tool = tool

    def use_poi_tool(self, city: str, query: str = "") -> list[dict[str, Any]]:
        if "poi_search" not in self.allowed_tools:
            raise PermissionError(f"{self.name} cannot use poi_search")
        return self.tool.search(city, query)

    def run(self, message: AgentMessage) -> AgentMessage:
        city = str(message.content.get("destination", "")).strip()
        query = str(message.content.get("place_query", "")).strip()
        if not city:
            return self._reply(message, {"candidates": [], "error": "destination is required"})
        candidates = self.use_poi_tool(city, query)
        return self._reply(message, {
            "destination": city,
            "candidates": candidates,
            "tool_used": self.tool.__class__.__name__,
            "allowed_tools": sorted(self.allowed_tools),
        })


class PlannerAgent(BaseAgent):
    def __init__(self, *, model=None, memory_store: AgentMemoryStore | None = None) -> None:
        super().__init__(
            "planner_agent",
            "Create an itinerary only from verified POI candidates. Do not judge your own plan.",
            model=model,
            memory_store=memory_store,
        )

    def run(self, message: AgentMessage) -> AgentMessage:
        model_payload = self._model_payload(message)
        if model_payload is not None:
            return self._reply(message, model_payload)
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
    def __init__(self, *, model=None, memory_store: AgentMemoryStore | None = None) -> None:
        super().__init__(
            "reviewer_agent",
            "Review a draft against verified candidates. Do not rewrite the itinerary.",
            model=model,
            memory_store=memory_store,
        )

    def run(self, message: AgentMessage) -> AgentMessage:
        model_payload = self._model_payload(message)
        if model_payload is not None:
            return self._reply(message, model_payload)
        itinerary = list(message.content.get("itinerary", []))
        candidate_names = {item.get("name") for item in message.content.get("candidates", [])}
        unknown = [item.get("name") for item in itinerary if item.get("name") not in candidate_names]
        approved = bool(itinerary) and not unknown
        return self._reply(message, {
            "approved": approved,
            "issues": ([] if approved else ["Draft is empty or includes an unverified POI."]),
            "revision_instruction": ("" if approved else "Keep only verified POI candidates and create a non-empty route."),
        })
