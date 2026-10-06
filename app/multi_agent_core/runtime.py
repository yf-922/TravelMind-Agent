"""Travel Supervisor using production nodes and bounded role-local memory."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from app.multi_agent_core.memory import AgentMemoryStore
from app.planning.schemas import TravelPlanState

logger = logging.getLogger(__name__)


INPUTS = {
    "modification_intent": {"modification_notes", "destination", "days", "pois", "attraction_preference", "food_preference", "habit_preference"},
    "candidate_refresh": {"destination", "pois", "modification_search_keywords"},
    "intent": {"query", "profile_hint"},
    "query_rewrite": {"query", "profile_hint", "destination", "days", "travel_start_date", "travel_end_date", "attraction_preference", "food_preference", "habit_preference", "max_walking_km", "rain_indoor_priority"},
    "weather_search": {"destination", "travel_start_date", "travel_end_date", "days"},
    "attraction_search": {"destination", "query", "rewritten_query", "max_spots", "min_rating"},
    "planner": {"query", "rewritten_query", "destination", "days", "travel_start_date", "travel_end_date", "pois", "weather_forecast", "weather_note", "attraction_preference", "food_preference", "habit_preference", "max_walking_km", "rain_indoor_priority", "max_per_day", "max_review_rounds", "route", "route_modify_opinion", "route_stale_warning", "review_round", "profile_hint", "modification_notes", "modification_search_status", "repair_feedback", "route_distance_legs", "route_distance_note"},
    "route_distance_check": {"pois", "route", "max_walking_km"},
    "risk_gate": {"pois", "route", "days", "max_per_day", "travel_start_date", "weather_forecast", "habit_preference", "max_walking_km", "rain_indoor_priority", "route_distance_legs", "modification_notes", "route_modify_opinion"},
    "reviewer": {"destination", "days", "travel_start_date", "travel_end_date", "pois", "route", "habit_preference", "attraction_preference", "modification_notes", "max_per_day", "weather_forecast", "max_walking_km", "rain_indoor_priority", "route_distance_legs", "route_distance_note", "review_round"},
    "time_check": {"destination", "pois", "route", "travel_start_date", "travel_end_date", "days", "time_check_round", "max_time_check_rounds", "approved", "review_required", "habit_preference", "max_walking_km", "rain_indoor_priority", "weather_forecast", "route_distance_legs", "max_per_day", "modification_notes"},
    "meal_enrichment": {"destination", "pois", "route", "food_preference"},
    "spot_tips": {"destination", "route", "travel_start_date", "weather_forecast"},
    "finalize": set(TravelPlanState.model_fields) - {"planner_reviewer_dialogue", "agent_private_context"},
}


OUTPUTS = {
    "modification_intent": {"attraction_preference", "food_preference", "habit_preference", "modification_search_keywords", "modification_search_status", "history"},
    "candidate_refresh": {"pois", "modification_search_status", "history"},
    "intent": {"destination", "travel_start_date", "travel_end_date", "days", "attraction_preference", "food_preference", "habit_preference", "max_walking_km", "rain_indoor_priority", "missing_fields", "history"},
    "query_rewrite": {"rewritten_query", "attraction_preference", "food_preference", "habit_preference", "history"},
    "weather_search": {"weather_forecast", "weather_note", "history"},
    "attraction_search": {"pois", "history"},
    "planner": {"route", "review_round", "history", "planner_reviewer_dialogue", "modification_concern", "route_stale_warning", "rag_sources"},
    "route_distance_check": {"route_distance_legs", "route_distance_mode", "route_distance_note"},
    "risk_gate": {"route_risk_flags", "route_risk_score", "review_required", "time_check_required", "review_skipped", "approved", "need_modify_route", "time_check_done", "time_check_status"},
    "reviewer": {"approved", "need_modify_route", "route_modify_opinion", "reviewer_issues", "history", "planner_reviewer_dialogue"},
    "time_check": {"time_violations", "time_check_done", "time_check_round", "time_check_status", "approved", "route_risk_flags", "review_required", "risk_gate_rechecked", "route_modify_opinion", "history", "planner_reviewer_dialogue"},
    "meal_enrichment": {"meal_candidates", "meals", "meal_search_status", "meal_recommend_status", "history"},
    "spot_tips": {"spot_tips", "spot_guides", "spot_tips_status"},
    "finalize": {"final_plan", "history"},
}


class RoleContractError(ValueError):
    pass


INPUTS["planner"].add("repair_feedback")


def production_nodes(model_name=None, profile_hint="", user_id=None):
    from app.planning import nodes
    from app.multi_agent_core.modification import make_modification_intent_node, candidate_refresh_node
    finalize = nodes.make_finalize_node()
    def finalize_result(state):
        if not state.approved:
            return {"final_plan": {
                "approved": False, "destination": state.destination, "days_count": state.days,
                "draft_route": state.route, "route_issues": state.reviewer_issues,
                "unresolved_time_violations": state.time_violations,
                "unresolved_risk_flags": state.route_risk_flags,
                "draft_only": True,
            }, "history": state.history + ["finalize: unapproved draft, external enrichment skipped"]}
        return finalize(state)
    return {
        "modification_intent": make_modification_intent_node(model_name),
        "candidate_refresh": candidate_refresh_node,
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
        "finalize": finalize_result,
    }


class TravelSupervisor:
    """One instance per request; orchestration owns facts, roles own history."""

    def __init__(self, nodes: dict[str, Callable], memory: AgentMemoryStore,
                 session_id: str, *, memory_chars: int = 4000):
        self.nodes = nodes
        self.memory = memory
        self.session_id = session_id
        self.memory_chars = memory_chars
        self._role_histories: dict[str, list[str]] = {}
        self._memory_degraded = bool(getattr(memory, "durability_degraded", False))

    def _history(self, role: str) -> list[str]:
        if role in self._role_histories:
            return list(self._role_histories[role])
        try:
            entries = self.memory.load(self.session_id, role)
        except Exception:
            self._memory_degraded = True
            logger.warning("private memory lookup failed role=%s", role, exc_info=True)
            self._role_histories[role] = []
            return []
        selected = []
        remaining = self.memory_chars
        for entry in reversed(entries):
            content = entry["content"]
            if len(content) > remaining:
                continue
            selected.append(content)
            remaining -= len(content)
        history = list(reversed(selected))
        self._role_histories[role] = history
        return list(history)

    async def _call(self, role: str, state: TravelPlanState) -> dict[str, Any]:
        fields = INPUTS.get(role)
        data = state.model_dump()
        if fields is not None:
            data = {key: value for key, value in data.items() if key in fields}
            data.setdefault("query", state.query)
        # Only this role's persisted context is exposed to its model node.
        if role in {"planner", "reviewer", "time_check"}:
            data["planner_reviewer_dialogue"] = self._history(role)
            data["agent_private_context"] = self._history(role)
        local = TravelPlanState(**data)
        try:
            update = await asyncio.to_thread(self.nodes[role], local)
        except Exception:
            fallback = {
                "modification_intent": {
                    "modification_search_keywords": [],
                    "modification_search_status": "failed",
                    "history": ["modification_intent: unavailable; candidate refresh could not be verified"],
                },
                "query_rewrite": {"rewritten_query": state.query},
                "weather_search": {"weather_forecast": [], "weather_note": "Weather unavailable; forecast not verified."},
                "spot_tips": {"spot_tips": {}, "spot_guides": {}, "spot_tips_status": "degraded"},
                "meal_enrichment": {"meals": [], "meal_candidates": [], "meal_search_status": "degraded", "meal_recommend_status": "degraded"},
            }
            if role not in fallback:
                raise
            logger.warning("optional supervisor node failed role=%s", role, exc_info=True)
            update = fallback[role]
        if not isinstance(update, dict):
            raise TypeError(f"{role} must return a state update")
        unexpected = set(update) - OUTPUTS[role]
        if unexpected:
            raise RoleContractError(f"{role} cannot update fields: {', '.join(sorted(unexpected))}")
        # Invalid role outputs must not contaminate persisted private context.
        TravelPlanState(**{**state.model_dump(), **update})
        safe = {key: value for key, value in update.items()
                if key not in {"history", "planner_reviewer_dialogue", "agent_private_context", "final_plan", "pois"}}
        serialized = json.dumps(safe, ensure_ascii=False, default=str)
        if len(serialized) > self.memory_chars:
            # Preserve useful findings even when a long itinerary cannot fit.
            compact = {key: value for key, value in safe.items() if key not in {"route", "meals", "meal_candidates", "spot_tips", "spot_guides", "route_distance_legs"}}
            for key, value in list(compact.items()):
                if isinstance(value, str):
                    compact[key] = value[:1000]
                elif isinstance(value, list):
                    compact[key] = value[:8]
            compact["memory_summary"] = {"role": role, "destination": state.destination,
                                         "route_days": len(update.get("route") or [])}
            serialized = json.dumps(compact, ensure_ascii=False, default=str)
        if len(serialized) <= self.memory_chars:
            history = self._history(role) + [serialized]
            while sum(map(len, history)) > self.memory_chars:
                history.pop(0)
            self._role_histories[role] = history
            try:
                await asyncio.to_thread(self.memory.append, self.session_id, role,
                                        {"role": "assistant", "content": serialized})
            except Exception:
                self._memory_degraded = True
                logger.warning("private memory write failed role=%s", role, exc_info=True)
        return update

    async def stream(self, state: TravelPlanState, *, modification=False, confirmed=False) -> AsyncIterator[dict[str, Any]]:
        state = state.model_copy(deep=True)
        state.planner_reviewer_dialogue = []
        state.agent_private_context = []
        # Freeze every model role before any node writes this request's history.
        roles = ("planner", "reviewer", "time_check")
        if hasattr(self.memory, "load_many"):
            try:
                snapshot = await asyncio.to_thread(self.memory.load_many, self.session_id, roles)
                for role in roles:
                    selected = []
                    remaining = self.memory_chars
                    for entry in reversed(snapshot[role]):
                        content = entry["content"]
                        if len(content) <= remaining:
                            selected.append(content)
                            remaining -= len(content)
                    self._role_histories[role] = list(reversed(selected))
            except Exception:
                self._memory_degraded = True
                logger.warning("private memory snapshot failed", exc_info=True)
                self._role_histories.update({role: [] for role in roles})
        else:
            for role in roles:
                await asyncio.to_thread(self._history, role)

        async def execute(role):
            update = await self._call(role, state)
            merged = state.model_dump()
            merged.update({k: v for k, v in update.items()
                           if k not in {"history", "planner_reviewer_dialogue", "agent_private_context"}})
            merged["history"] = state.history + list(update.get("history", [])) if role != "finalize" else update.get("history", state.history)
            return TravelPlanState(**merged)

        async def parallel_stream(roles):
            async def invoke(role):
                return role, await self._call(role, state)
            tasks = [asyncio.create_task(invoke(role)) for role in roles]
            merged = state.model_dump()
            try:
                for completed in asyncio.as_completed(tasks):
                    role, update = await completed
                    merged.update({k: v for k, v in update.items()
                                   if k not in {"history", "planner_reviewer_dialogue", "agent_private_context"}})
                    merged["history"] += list(update.get("history", []))
                    yield {"type": "stage_summary", "node": role, "summary": "completed"}
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            yield {"_state": TravelPlanState(**merged)}

        if not modification:
            yield {"type": "stage", "node": "intent"}
            state = await execute("intent")
            yield {"type": "stage_summary", "node": "intent", "summary": "completed"}
            if state.missing_fields:
                yield {"type": "result", "success": False, "missing_fields": state.missing_fields, "plan": None}
                return
            for role in ("query_rewrite", "weather_search"):
                yield {"type": "stage", "node": role}
            async for event in parallel_stream(("query_rewrite", "weather_search")):
                if "_state" in event:
                    state = event["_state"]
                else:
                    yield event
            yield {"type": "stage", "node": "attraction_search"}
            state = await execute("attraction_search")
            yield {"type": "stage_summary", "node": "attraction_search", "summary": "completed"}

        if modification and not confirmed and "modification_intent" in self.nodes:
            yield {"type": "stage", "node": "modification_intent"}
            state = await execute("modification_intent")
            yield {"type": "stage_summary", "node": "modification_intent", "summary": "completed"}
            if state.modification_search_keywords:
                yield {"type": "stage", "node": "candidate_refresh"}
                state = await execute("candidate_refresh")
                yield {"type": "stage_summary", "node": "candidate_refresh", "summary": "completed"}

        for revision in range(state.max_review_rounds + 1):
            # Derived conclusions never survive a new route generation.
            state.approved = False
            state.reviewer_issues = []
            state.time_violations = []
            state.time_check_done = False
            state.time_check_status = "skipped"
            roles = ("route_distance_check", "risk_gate") if confirmed and revision == 0 else ("planner", "route_distance_check", "risk_gate")
            for role in roles:
                yield {"type": "stage", "node": role, "revision": revision}
                state = await execute(role)
                yield {"type": "stage_summary", "node": role, "summary": "completed"}
            if (state.modification_concern and modification and state.modification_notes
                    and revision == 0 and not confirmed):
                yield {"type": "modification_warning", "concern": state.modification_concern,
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
            if state.modification_search_status in {"failed", "partial", "empty", "pending"}:
                flags.append("candidate_refresh_unverified")
            state.route_risk_flags = flags
            state.route_risk_score = len(flags)
            hard_faults = [flag for flag in flags if flag in {
                "duplicate_poi", "unknown_poi", "route_structure", "opening_time_conflict",
                "walking_constraint", "weather_outdoor_conflict", "long_road_leg",
                "habit_constraint",
                "modification_time_unfulfilled",
                "indoor_constraint",
                "candidate_refresh_unverified",
            }]
            if hard_faults:
                state.approved = False
                feedback = "【规则校验未通过】" + ", ".join(hard_faults)
                state.route_modify_opinion = "\n".join(filter(None, [state.route_modify_opinion, feedback]))
            if state.time_violations:
                details = json.dumps(state.time_violations, ensure_ascii=False)
                state.route_modify_opinion = "\n".join(filter(None, [state.route_modify_opinion, details]))
            if not state.approved or state.time_violations:
                feedback = {
                    "revision": revision,
                    "risk_flags": hard_faults,
                    "time_violations": state.time_violations[:8],
                    "reviewer_issues": state.reviewer_issues[:8],
                    "instruction": (state.route_modify_opinion or "")[-1600:],
                }
                state.repair_feedback = [*state.repair_feedback[-2:], feedback]
            if state.approved and not state.time_violations:
                break
        if state.approved:
            for role in ("meal_enrichment", "spot_tips"):
                yield {"type": "stage", "node": role}
            async for event in parallel_stream(("meal_enrichment", "spot_tips")):
                if "_state" in event:
                    state = event["_state"]
                else:
                    yield event
        else:
            state.meals = []
            state.meal_candidates = []
            state.spot_tips = {}
            state.spot_guides = {}
            state.meal_search_status = "skipped"
            state.meal_recommend_status = "skipped"
            state.spot_tips_status = "skipped"
        yield {"type": "stage", "node": "finalize"}
        state = await execute("finalize")
        yield {"type": "stage_summary", "node": "finalize", "summary": "completed"}
        plan = state.final_plan
        if plan is not None:
            plan["unresolved_time_violations"] = state.time_violations
            plan["unresolved_risk_flags"] = state.route_risk_flags if not state.approved else []
            if self._memory_degraded:
                plan["degraded_services"] = list(dict.fromkeys([*(plan.get("degraded_services") or []), "private_memory"]))
        yield {"type": "result", "success": bool(plan) and state.approved,
               "plan": plan, "missing_fields": [], "history": state.history,
               "message": None if state.approved else "行程在修复次数上限内未通过审核，草稿仅供参考。",
               "failure_details": None if state.approved else {
                   "risk_flags": state.route_risk_flags,
                   "time_violations": state.time_violations,
                   "reviewer_issues": state.reviewer_issues,
               },
               "checkpoint": state.model_dump(mode="json")}


async def run_stream(query, profile_hint="", memory_writer=None, user_id=None, **overrides):
    """API-compatible entry point with owner-scoped role memories."""
    session = str(overrides.pop("session_id", None) or uuid.uuid4().hex)
    state = TravelPlanState(query=query, profile_hint=profile_hint or None, **overrides)
    state.memory_session_id = session
    runtime = TravelSupervisor(production_nodes(overrides.get("model_name"), profile_hint, user_id),
                               build_memory_store(), memory_scope(user_id, session))
    async for event in runtime.stream(state):
        checkpoint = event.pop("checkpoint", None)
        if checkpoint and event.get("success") and memory_writer:
            await asyncio.to_thread(memory_writer, event["plan"], TravelPlanState(**checkpoint))
        yield _label_event(event)


async def run_modification_stream(checkpoint, modification_notes, memory_writer=None, **overrides):
    user_id = overrides.pop("user_id", None)
    session = str(overrides.pop("session_id", None) or checkpoint.get("memory_session_id") or uuid.uuid4().hex)
    data = dict(checkpoint)
    data.update(overrides)
    # API checkpoints serialize optional dates as empty strings; Pydantic state
    # expects absent dates to be None so modification streams can resume them.
    for field in ("travel_start_date", "travel_end_date"):
        if data.get(field) in ("", None):
            data[field] = None
    data.update(approved=False, reviewer_issues=[], time_violations=[], review_round=0,
                time_check_round=0, time_check_done=False, final_plan=None,
                modification_notes=modification_notes,
                route_modify_opinion=f"【用户修改意见】{modification_notes}", repair_feedback=[])
    data.update(modification_search_keywords=[], modification_search_status="not_required")
    data["memory_session_id"] = session
    runtime = TravelSupervisor(production_nodes(data.get("model_name"), data.get("profile_hint") or "", user_id),
                               build_memory_store(), memory_scope(user_id, session))
    async for event in runtime.stream(TravelPlanState(**data), modification=True):
        if event.get("type") == "modification_warning":
            event["pending_state"]["_engine"] = "supervisor"
        current = event.pop("checkpoint", None)
        if current and event.get("success") and memory_writer:
            await asyncio.to_thread(memory_writer, event["plan"], TravelPlanState(**current))
        yield _label_event(event)


async def run_confirm_stream(checkpoint, memory_writer=None, user_id=None):
    data = {key: value for key, value in checkpoint.items() if not key.startswith("_")}
    for field in ("travel_start_date", "travel_end_date"):
        if data.get(field) in ("", None):
            data[field] = None
    data.update(approved=False, reviewer_issues=[], time_violations=[],
                time_check_round=0, time_check_done=False, final_plan=None,
                modification_concern=None)
    session = str(data.get("memory_session_id") or uuid.uuid4().hex)
    data["memory_session_id"] = session
    runtime = TravelSupervisor(production_nodes(data.get("model_name"), data.get("profile_hint") or "", user_id),
                               build_memory_store(), memory_scope(user_id, session))
    async for event in runtime.stream(TravelPlanState(**data), modification=True, confirmed=True):
        current = event.pop("checkpoint", None)
        if current and event.get("success") and memory_writer:
            await asyncio.to_thread(memory_writer, event["plan"], TravelPlanState(**current))
        yield _label_event(event)


def _label_event(event):
    from app.planning.graph import _NODE_LABELS
    if event.get("type") == "stage":
        event = {**event, "label": _NODE_LABELS.get(event.get("node"), event.get("node"))}
    return event


def memory_scope(user_id, session_id):
    return json.dumps([str(user_id or "anonymous"), str(session_id)], separators=(",", ":"))


def build_memory_store():
    from app.multi_agent_core.memory import SQLiteAgentMemoryStore, InMemoryAgentMemoryStore
    try:
        return SQLiteAgentMemoryStore()
    except Exception:
        logger.warning("private memory initialization failed; using request-local memory", exc_info=True)
        store = InMemoryAgentMemoryStore()
        store.durability_degraded = True
        return store
