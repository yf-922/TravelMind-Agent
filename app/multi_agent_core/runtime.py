"""Travel Supervisor using production nodes and bounded role-local memory."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from app.multi_agent_core.memory import AgentMemoryStore
from app.planning.schemas import TravelPlanState

logger = logging.getLogger(__name__)


INPUTS = {
    "intent": {"query", "profile_hint"},
    "query_rewrite": {"query", "profile_hint", "destination", "days", "travel_start_date", "travel_end_date", "attraction_preference", "food_preference", "habit_preference"},
    "weather_search": {"destination", "travel_start_date", "travel_end_date", "days"},
    "attraction_search": {"destination", "query", "rewritten_query", "max_spots", "min_rating"},
    "planner": {"query", "destination", "days", "travel_start_date", "travel_end_date", "pois", "weather_forecast", "weather_note", "attraction_preference", "food_preference", "habit_preference", "max_walking_km", "rain_indoor_priority", "max_per_day", "route", "route_modify_opinion", "route_stale_warning", "review_round", "profile_hint", "modification_notes"},
    "route_distance_check": {"pois", "route", "max_walking_km"},
    "risk_gate": {"pois", "route", "days", "max_per_day", "travel_start_date", "weather_forecast", "habit_preference", "max_walking_km", "rain_indoor_priority", "route_distance_legs", "modification_notes", "route_modify_opinion"},
    "reviewer": {"destination", "days", "travel_start_date", "travel_end_date", "pois", "route", "habit_preference", "max_per_day", "weather_forecast", "max_walking_km", "rain_indoor_priority", "route_distance_legs", "route_distance_note", "review_round"},
    "time_check": {"pois", "route", "travel_start_date", "travel_end_date", "days", "time_check_round", "max_time_check_rounds", "approved", "review_required", "habit_preference", "max_walking_km", "rain_indoor_priority", "weather_forecast", "route_distance_legs", "max_per_day", "modification_notes"},
}


def production_nodes(model_name=None, profile_hint="", user_id=None):
    from app.planning import nodes
    return {
        "intent": nodes.make_intent_node(model_name, profile_hint=profile_hint),
        "query_rewrite": nodes.make_query_rewrite_node(model_name, user_id),
        "weather_search": nodes.weather_search_node,
        "attraction_search": nodes.attraction_search_node,
        "planner": nodes.make_planner_node(model_name),
        "route_distance_check": nodes.route_distance_check_node,
        "risk_gate": nodes.route_risk_gate_node,
        "reviewer": nodes.make_reviewer_node(model_name),
        "time_check": nodes.make_time_check_node(model_name),
        "meal_enrichment": nodes.make_meal_enrichment_node(model_name),
        "spot_tips": nodes.make_spot_tips_node(model_name),
        "finalize": nodes.make_finalize_node(),
    }


class TravelSupervisor:
    """One instance per request; orchestration owns facts, roles own history."""

    def __init__(self, nodes: dict[str, Callable], memory: AgentMemoryStore,
                 session_id: str, *, memory_chars: int = 4000):
        self.nodes = nodes
        self.memory = memory
        self.session_id = session_id
        self.memory_chars = memory_chars

    def _history(self, role: str) -> list[str]:
        try:
            entries = self.memory.load(self.session_id, role)
        except Exception:
            logger.warning("private memory lookup failed role=%s", role, exc_info=True)
            return []
        selected = []
        remaining = self.memory_chars
        for entry in reversed(entries):
            content = entry["content"]
            if len(content) > remaining:
                break
            selected.append(content)
            remaining -= len(content)
        return list(reversed(selected))

    async def _call(self, role: str, state: TravelPlanState) -> dict[str, Any]:
        fields = INPUTS.get(role)
        data = state.model_dump()
        if fields is not None:
            data = {key: value for key, value in data.items() if key in fields}
            data.setdefault("query", state.query)
        # Only this role's persisted context is exposed to its model node.
        if role in {"planner", "reviewer", "time_check"}:
            data["planner_reviewer_dialogue"] = self._history(role)
        local = TravelPlanState(**data)
        update = await asyncio.to_thread(self.nodes[role], local)
        if not isinstance(update, dict):
            raise TypeError(f"{role} must return a state update")
        safe = {key: value for key, value in update.items()
                if key not in {"history", "planner_reviewer_dialogue", "final_plan", "pois"}}
        serialized = json.dumps(safe, ensure_ascii=False, default=str)
        if len(serialized) <= self.memory_chars:
            try:
                await asyncio.to_thread(self.memory.append, self.session_id, role,
                                        {"role": "assistant", "content": serialized})
            except Exception:
                logger.warning("private memory write failed role=%s", role, exc_info=True)
        return update

    async def stream(self, state: TravelPlanState, *, modification=False, confirmed=False) -> AsyncIterator[dict[str, Any]]:
        async def execute(role):
            update = await self._call(role, state)
            merged = state.model_dump()
            merged.update({k: v for k, v in update.items()
                           if k not in {"history", "planner_reviewer_dialogue"}})
            return TravelPlanState(**merged)

        async def parallel(roles):
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(self._call(role, state)) for role in roles]
            updates = [task.result() for task in tasks]
            merged = state.model_dump()
            for update in updates:
                merged.update({k: v for k, v in update.items() if k not in {"history", "planner_reviewer_dialogue"}})
            return TravelPlanState(**merged)

        if not modification:
            yield {"type": "stage", "node": "intent"}
            state = await execute("intent")
            yield {"type": "stage_summary", "node": "intent", "summary": "completed"}
            if state.missing_fields:
                yield {"type": "result", "success": False, "missing_fields": state.missing_fields, "plan": None}
                return
            for role in ("query_rewrite", "weather_search"):
                yield {"type": "stage", "node": role}
            state = await parallel(("query_rewrite", "weather_search"))
            for role in ("query_rewrite", "weather_search"):
                yield {"type": "stage_summary", "node": role, "summary": "completed"}
            yield {"type": "stage", "node": "attraction_search"}
            state = await execute("attraction_search")
            yield {"type": "stage_summary", "node": "attraction_search", "summary": "completed"}

        for revision in range(state.max_review_rounds + 1):
            # Derived conclusions never survive a new route generation.
            state.approved = False
            state.reviewer_issues = []
            state.time_violations = []
            state.route_distance_legs = []
            state.route_distance_note = None
            state.time_check_done = False
            state.time_check_status = "skipped"
            roles = ("route_distance_check", "risk_gate") if confirmed and revision == 0 else ("planner", "route_distance_check", "risk_gate")
            for role in roles:
                yield {"type": "stage", "node": role, "revision": revision}
                state = await execute(role)
                yield {"type": "stage_summary", "node": role, "summary": "completed"}
            if state.modification_concern and modification and not confirmed:
                yield {"type": "modification_warning", "message": state.modification_concern,
                       "pending_state": state.model_dump(mode="json")}
                return
            if state.review_required:
                yield {"type": "stage", "node": "reviewer"}
                state = await execute("reviewer")
                yield {"type": "stage_summary", "node": "reviewer", "summary": "completed"}
            if state.time_check_required and state.time_check_round < state.max_time_check_rounds:
                yield {"type": "stage", "node": "time_check"}
                state = await execute("time_check")
                yield {"type": "stage_summary", "node": "time_check", "summary": "completed"}
            elif state.time_check_required:
                state.approved = False
                state.time_check_status = "partial"
            # Model approval cannot override a known structural or time fault.
            from app.planning.nodes import _route_risk_flags
            flags = _route_risk_flags(state)
            hard_faults = [flag for flag in flags if flag in {
                "duplicate_poi", "unknown_poi", "route_structure", "opening_time_conflict",
                "walking_constraint", "weather_outdoor_conflict", "long_road_leg",
            }]
            if hard_faults:
                state.approved = False
                feedback = "【规则校验未通过】" + ", ".join(hard_faults)
                state.route_modify_opinion = "\n".join(filter(None, [state.route_modify_opinion, feedback]))
            if state.approved and not state.time_violations:
                break
        for role in ("meal_enrichment", "spot_tips"):
            yield {"type": "stage", "node": role}
        state = await parallel(("meal_enrichment", "spot_tips"))
        for role in ("meal_enrichment", "spot_tips"):
            yield {"type": "stage_summary", "node": role, "summary": "completed"}
        yield {"type": "stage", "node": "finalize"}
        state = await execute("finalize")
        yield {"type": "stage_summary", "node": "finalize", "summary": "completed"}
        plan = state.final_plan
        if plan is not None:
            plan["unresolved_time_violations"] = state.time_violations
        yield {"type": "result", "success": bool(plan) and state.approved,
               "plan": plan, "missing_fields": [], "checkpoint": state.model_dump(mode="json")}


async def run_stream(query, profile_hint="", memory_writer=None, user_id=None, **overrides):
    """API-compatible entry point with owner-scoped role memories."""
    from app.multi_agent_core.memory import SQLiteAgentMemoryStore
    session_id = str(overrides.pop("session_id", None) or user_id or "anonymous")
    runtime = TravelSupervisor(production_nodes(overrides.get("model_name"), profile_hint, user_id),
                               SQLiteAgentMemoryStore(), session_id)
    async for event in runtime.stream(TravelPlanState(query=query, profile_hint=profile_hint or None, **overrides)):
        checkpoint = event.pop("checkpoint", None)
        if checkpoint and event.get("success") and memory_writer:
            await asyncio.to_thread(memory_writer, event["plan"], TravelPlanState(**checkpoint))
        yield event


async def run_modification_stream(checkpoint, modification_notes, memory_writer=None, **overrides):
    from app.multi_agent_core.memory import SQLiteAgentMemoryStore
    user_id = overrides.pop("user_id", None)
    session_id = str(overrides.pop("session_id", None) or user_id or "anonymous")
    data = dict(checkpoint)
    data.update(overrides)
    data.update(approved=False, reviewer_issues=[], time_violations=[], review_round=0,
                time_check_round=0, time_check_done=False, final_plan=None,
                modification_notes=modification_notes,
                route_modify_opinion=f"【用户修改意见】{modification_notes}")
    runtime = TravelSupervisor(production_nodes(data.get("model_name")), SQLiteAgentMemoryStore(), session_id)
    async for event in runtime.stream(TravelPlanState(**data), modification=True):
        if event.get("type") == "modification_warning":
            event["pending_state"]["_engine"] = "supervisor"
        current = event.pop("checkpoint", None)
        if current and event.get("success") and memory_writer:
            await asyncio.to_thread(memory_writer, event["plan"], TravelPlanState(**current))
        yield event


async def run_confirm_stream(checkpoint, memory_writer=None, user_id=None):
    from app.multi_agent_core.memory import SQLiteAgentMemoryStore
    data = {key: value for key, value in checkpoint.items() if not key.startswith("_")}
    data.update(approved=False, reviewer_issues=[], time_violations=[],
                time_check_round=0, time_check_done=False, final_plan=None,
                modification_concern=None)
    runtime = TravelSupervisor(production_nodes(data.get("model_name")),
                               SQLiteAgentMemoryStore(), str(user_id or "anonymous"))
    async for event in runtime.stream(TravelPlanState(**data), modification=True, confirmed=True):
        current = event.pop("checkpoint", None)
        if current and event.get("success") and memory_writer:
            await asyncio.to_thread(memory_writer, event["plan"], TravelPlanState(**current))
        yield event
