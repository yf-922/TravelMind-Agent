import asyncio
import pytest

from app.multi_agent_core.memory import InMemoryAgentMemoryStore
from app.multi_agent_core.runtime import TravelSupervisor
from app.multi_agent_core.runtime import memory_scope
from app.planning.schemas import TravelPlanState


def make_nodes(*, reject=False):
    return {
        "intent": lambda s: {"destination": "Nanjing", "days": 1},
        "query_rewrite": lambda s: {"rewritten_query": "museum"},
        "weather_search": lambda s: {"weather_note": "fixture"},
        "attraction_search": lambda s: {"pois": [{"name": "Museum"}]},
        "planner": lambda s: {"route": [{"day": 1, "spots": [{"name": "Museum", "period": "morning", "start_time": "10:00", "end_time": "11:00"}]}], "review_round": s.review_round + 1},
        "route_distance_check": lambda s: {"route_distance_legs": []},
        "risk_gate": lambda s: {"review_required": reject, "time_check_required": False, "approved": not reject},
        "reviewer": lambda s: {"approved": False, "route_modify_opinion": "revise", "reviewer_issues": ["unresolved"]},
        "time_check": lambda s: {},
        "meal_enrichment": lambda s: {"meals": []},
        "spot_tips": lambda s: {"spot_tips": {}},
        "finalize": lambda s: {"final_plan": {"approved": s.approved, "days": s.route}},
    }


def collect(runtime, state):
    async def run():
        return [event async for event in runtime.stream(state)]
    return asyncio.run(run())


def test_supervisor_runs_full_pipeline_with_role_isolation():
    memory = InMemoryAgentMemoryStore()
    memory.append("s", "reviewer", {"role": "assistant", "content": "REVIEWER_PRIVATE"})
    nodes = make_nodes()
    original = nodes["planner"]
    def planner(state):
        assert "REVIEWER_PRIVATE" not in str(state.planner_reviewer_dialogue)
        assert state.pois == [{"name": "Museum"}]
        return original(state)
    nodes["planner"] = planner
    events = collect(TravelSupervisor(nodes, memory, "s"), TravelPlanState(query="trip"))
    assert events[-1]["success"] is True
    stages = [e["node"] for e in events if e["type"] == "stage"]
    assert stages == ["intent", "query_rewrite", "weather_search", "attraction_search", "planner", "route_distance_check", "risk_gate", "meal_enrichment", "spot_tips", "finalize"]
    assert memory.load("s", "planner")
    assert memory.load("another", "planner") == []


def test_rejected_route_stops_with_incomplete_result():
    events = collect(TravelSupervisor(make_nodes(reject=True), InMemoryAgentMemoryStore(), "s"),
                     TravelPlanState(query="trip", max_review_rounds=2))
    assert sum(e.get("node") == "planner" for e in events if e["type"] == "stage") == 3
    assert events[-1]["success"] is False
    assert events[-1]["plan"]["approved"] is False
    assert not any(e.get("node") in {"meal_enrichment", "spot_tips"} for e in events)


def test_parallel_completion_reports_fast_node_before_slow_node_finishes():
    import threading
    release = threading.Event()
    nodes = make_nodes()
    def slow(state):
        assert release.wait(2), "fast completion event was not streamed"
        return {"rewritten_query": "museum"}
    nodes["query_rewrite"] = slow
    async def run():
        events = []
        async for event in TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s").stream(TravelPlanState(query="trip")):
            events.append(event)
            if event.get("type") == "stage_summary" and event.get("node") == "weather_search":
                release.set()
        return events
    events = asyncio.run(run())
    summaries = [e["node"] for e in events if e["type"] == "stage_summary"]
    assert summaries.index("weather_search") < summaries.index("query_rewrite")


def test_initial_fact_queries_fan_out_meals_with_query_and_weather():
    """Supervisor keeps the same first-stage fan-out as the main graph."""
    import threading

    nodes = make_nodes()
    started = []
    release = threading.Event()

    def delayed(name, result):
        def run(state):
            started.append(name)
            assert release.wait(2), f"{name} did not receive release"
            return result
        return run

    nodes["query_rewrite"] = delayed("query_rewrite", {"rewritten_query": "museum"})
    nodes["weather_search"] = delayed("weather_search", {"weather_note": "fixture"})
    nodes["main_meal_search"] = delayed("main_meal_search", {
        "main_meal_candidates": [{"name": "Local Kitchen"}],
        "main_meal_status": "ok",
    })

    async def run():
        task = asyncio.create_task(
            collect_async(TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"), TravelPlanState(query="trip"))
        )
        for _ in range(20):
            await asyncio.sleep(0.01)
            if len(started) == 3:
                break
        assert set(started) == {"query_rewrite", "weather_search", "main_meal_search"}
        release.set()
        return await task

    async def collect_async(runtime, state):
        return [event async for event in runtime.stream(state)]

    events = asyncio.run(run())
    stages = [event["node"] for event in events if event["type"] == "stage"]
    assert stages[:4] == ["intent", "query_rewrite", "weather_search", "main_meal_search"]
    assert events[-1]["success"] is True
    assert events[-1]["checkpoint"]["main_meal_status"] == "ok"


def test_optional_initial_meal_failure_is_degraded_but_not_falsely_used():
    nodes = make_nodes()
    nodes["main_meal_search"] = lambda state: (_ for _ in ()).throw(TimeoutError("meal provider timeout"))

    events = collect(
        TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"),
        TravelPlanState(query="trip"),
    )
    assert events[-1]["success"] is True
    checkpoint = events[-1]["checkpoint"]
    assert checkpoint["main_meal_candidates"] == []
    assert checkpoint["main_meal_status"] == "degraded"


def test_initial_attraction_failure_cannot_be_approved_from_empty_pool():
    nodes = make_nodes()
    nodes["attraction_search"] = lambda state: (_ for _ in ()).throw(TimeoutError("poi provider timeout"))

    events = collect(
        TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"),
        TravelPlanState(query="trip", max_review_rounds=0),
    )
    assert events[-1]["success"] is False
    assert "unknown_poi" in events[-1]["failure_details"]["risk_flags"]


def test_owner_and_trip_memory_scopes_cannot_collide():
    memory = InMemoryAgentMemoryStore()
    memory.append(memory_scope("owner", "trip-1"), "planner", {"role": "assistant", "content": "secret"})
    assert memory.load(memory_scope("owner", "trip-2"), "planner") == []
    assert memory.load(memory_scope("other", "trip-1"), "planner") == []
    assert memory_scope("a/b", "c") != memory_scope("a", "b/c")


def test_large_planner_output_still_saves_bounded_private_summary():
    memory = InMemoryAgentMemoryStore()
    nodes = make_nodes()
    nodes["planner"] = lambda s: {"route": [{"day": 1, "spots": [{"name": "Museum", "notes": "x" * 10000}]}],
                                  "review_round": 1}
    runtime = TravelSupervisor(nodes, memory, "s")
    asyncio.run(runtime._call("planner", TravelPlanState(query="trip", destination="Nanjing")))
    entries = memory.load("s", "planner")
    assert len(entries) == 1
    assert len(entries[0]["content"]) <= 4000
    assert '"route_days": 1' in entries[0]["content"]


def test_explicit_habit_violation_cannot_be_approved_by_model():
    nodes = make_nodes()
    original = nodes["planner"]
    nodes["planner"] = lambda s: {**original(s), "route": [{"day": 1, "spots": [{
        "name": "Museum", "period": "morning", "start_time": "08:00", "end_time": "09:00",
    }]}]}
    events = collect(TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"),
                     TravelPlanState(query="trip", habit_preference="不喜欢早起", max_review_rounds=0))
    assert events[-1]["success"] is False
    assert "habit_constraint" in events[-1]["checkpoint"]["route_modify_opinion"]


def test_confirmation_cannot_approve_ignored_explicit_time_change():
    nodes = make_nodes()
    nodes["reviewer"] = lambda s: {"approved": True}
    nodes["time_check"] = lambda s: {"approved": True, "time_violations": []}
    state = TravelPlanState(query="trip", days=1, pois=[{"name": "Museum", "open_time": "09:00-17:00"}],
                            route=[{"day": 1, "spots": [{"name": "Museum", "period": "morning", "start_time": "10:00", "end_time": "11:00"}]}],
                            modification_notes="坚持02:00至03:00游览", max_review_rounds=1)
    async def run():
        return [e async for e in TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s").stream(state, modification=True, confirmed=True)]
    events = asyncio.run(run())
    assert events[-1]["success"] is False
    assert "modification_time_unfulfilled" in events[-1]["checkpoint"]["route_modify_opinion"]
    assert "modification_time_unfulfilled" in events[-1]["failure_details"]["risk_flags"]
    assert "modification_time_unfulfilled" in events[-1]["plan"]["unresolved_risk_flags"]


def test_failed_memory_initialization_is_visible_degradation(monkeypatch):
    import app.multi_agent_core.memory as memory_module
    from app.multi_agent_core.runtime import build_memory_store
    def fail():
        raise OSError("unwritable SQLite path")
    monkeypatch.setattr(memory_module, "SQLiteAgentMemoryStore", fail)
    store = build_memory_store()
    events = collect(TravelSupervisor(make_nodes(), store, "s"), TravelPlanState(query="trip"))
    assert events[-1]["success"] is True
    assert events[-1]["plan"]["degraded_services"] == ["private_memory"]


def test_entrypoints_persist_trip_scope_and_reuse_it_for_modification(monkeypatch):
    import app.multi_agent_core.runtime as runtime_module
    import app.multi_agent_core.memory as memory_module
    memory = InMemoryAgentMemoryStore()
    monkeypatch.setattr(memory_module, "SQLiteAgentMemoryStore", lambda: memory)
    monkeypatch.setattr(runtime_module, "production_nodes", lambda *a, **k: make_nodes())
    saved = []
    def writer(plan, state):
        saved.append(state.model_dump(mode="json"))
    async def run():
        first = [e async for e in runtime_module.run_stream("trip", user_id="owner", memory_writer=writer)]
        session = saved[0]["memory_session_id"]
        second = [e async for e in runtime_module.run_modification_stream(saved[0], "same museum", user_id="owner", memory_writer=writer)]
        return first, second, session
    first, second, session = asyncio.run(run())
    assert first[-1]["success"] is True and second[-1]["success"] is True
    assert saved[1]["memory_session_id"] == session
    assert len(memory.load(memory_scope("owner", session), "planner")) == 2
    assert memory.load(memory_scope("other", session), "planner") == []


def test_role_history_has_character_budget():
    memory = InMemoryAgentMemoryStore()
    for i in range(10):
        memory.append("s", "planner", {"role": "assistant", "content": str(i) * 100})
    runtime = TravelSupervisor(make_nodes(), memory, "s", memory_chars=250)
    assert runtime._history("planner") == ["8" * 100, "9" * 100]


def test_modification_reuses_pool_and_skips_external_prefetch():
    nodes = make_nodes()
    def unexpected(state):
        raise AssertionError("modification must reuse existing facts")
    for name in ("intent", "query_rewrite", "weather_search", "attraction_search"):
        nodes[name] = unexpected
    runtime = TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "owner")
    async def run():
        return [e async for e in runtime.stream(TravelPlanState(query="trip", destination="Nanjing", days=1,
                      pois=[{"name": "Museum"}], approved=True), modification=True)]
    events = asyncio.run(run())
    assert events[-1]["success"] is True
    assert events[-1]["checkpoint"]["pois"] == [{"name": "Museum"}]


def test_modification_refreshes_candidates_only_when_analysis_requires_it(monkeypatch):
    from app.multi_agent_core.modification import ModificationAnalysis
    import app.multi_agent_core.modification as modification
    calls = []
    monkeypatch.setattr(modification, "build_structured_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(modification, "amap_key", lambda: "fixture")
    monkeypatch.setattr(modification, "search_city_pois", lambda *args, **kwargs: calls.append(kwargs["keywords"]) or [{
        "name": "New Museum", "location": "118.7,32.0", "type": "museum", "rating": "4.8",
    }])
    monkeypatch.setattr(modification, "invoke_structured", lambda *args, **kwargs: ModificationAnalysis(
        attraction_preference="博物馆", candidate_pool_sufficient=False,
        search_keywords=["博物馆"], reasoning="旧池没有足够室内候选"))
    nodes = {"modification_intent": modification.make_modification_intent_node(),
             "candidate_refresh": modification.candidate_refresh_node}
    runtime = TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s")
    state = TravelPlanState(query="trip", destination="Nanjing", days=1,
                            pois=[{"name": "Old Park"}], modification_notes="改成博物馆")
    async def run():
        return await runtime._call("modification_intent", state)
    update = asyncio.run(run())
    refreshed = asyncio.run(runtime._call("candidate_refresh", TravelPlanState(**{**state.model_dump(), **update})))
    assert update["modification_search_keywords"] == ["博物馆"]
    assert calls == ["博物馆"]
    assert {poi["name"] for poi in refreshed["pois"]} == {"Old Park", "New Museum"}
    assert refreshed["modification_search_status"] == "complete"


def test_candidate_refresh_marks_partial_provider_failure(monkeypatch):
    import app.multi_agent_core.modification as modification
    monkeypatch.setattr(modification, "amap_key", lambda: "fixture")
    def search(*args, **kwargs):
        if kwargs["keywords"] == "museum":
            return [{"name": "New Museum", "location": "118.7,32.0", "type": "museum"}]
        raise RuntimeError("provider timeout")
    monkeypatch.setattr(modification, "search_city_pois", search)
    state = TravelPlanState(query="trip", destination="Nanjing", pois=[{"name": "Old Park"}],
                            modification_search_keywords=["museum", "gallery"])
    result = modification.candidate_refresh_node(state)
    assert result["modification_search_status"] == "partial"
    assert {poi["name"] for poi in result["pois"]} == {"Old Park", "New Museum"}


def test_modification_intent_failure_becomes_unverified_review_finding():
    nodes = make_nodes()
    nodes["modification_intent"] = lambda state: (_ for _ in ()).throw(RuntimeError("model timeout"))
    state = TravelPlanState(query="trip", destination="Nanjing", days=1,
                            pois=[{"name": "Museum"}], modification_notes="改成自然景点")
    async def run():
        return [event async for event in TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s").stream(
            state, modification=True)]
    events = asyncio.run(run())
    assert events[-1]["success"] is False
    assert "candidate_refresh_unverified" in events[-1]["failure_details"]["risk_flags"]


def test_food_modification_refreshes_meal_candidates_but_time_edit_does_not():
    def run_for(notes):
        nodes = make_nodes()
        nodes["modification_intent"] = lambda s: {"modification_search_status": "not_required"}
        nodes["main_meal_search"] = lambda s: {
            "main_meal_candidates": [{"name": "New Restaurant"}],
            "main_meal_status": "ok",
        }
        async def run():
            state = TravelPlanState(query="trip", destination="Nanjing", days=1,
                                    pois=[{"name": "Museum"}], modification_notes=notes)
            return [event async for event in TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s").stream(
                state, modification=True)]
        return asyncio.run(run())
    food_events = run_for("换成清淡口味餐厅")
    time_events = run_for("把博物馆改到下午")
    assert "main_meal_search" in [event["node"] for event in food_events if event["type"] == "stage"]
    assert "main_meal_search" not in [event["node"] for event in time_events if event["type"] == "stage"]


def test_food_and_candidate_modification_branches_run_in_parallel():
    import threading

    nodes = make_nodes()
    started = []
    release = threading.Event()
    nodes["modification_intent"] = lambda state: {
        "modification_search_keywords": ["museum"],
        "modification_search_status": "pending",
    }

    def delayed(name, result):
        def run(state):
            started.append(name)
            assert release.wait(2), f"{name} did not receive release"
            return result
        return run

    nodes["candidate_refresh"] = delayed("candidate_refresh", {
        "pois": [{"name": "Museum"}],
        "modification_search_status": "complete",
    })
    nodes["main_meal_search"] = delayed("main_meal_search", {
        "main_meal_candidates": [{"name": "Local Kitchen"}],
        "main_meal_status": "ok",
    })

    async def collect_async(runtime, state):
        return [event async for event in runtime.stream(state, modification=True)]

    async def run():
        task = asyncio.create_task(collect_async(
            TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"),
            TravelPlanState(query="trip", destination="Nanjing", days=1,
                            pois=[{"name": "Museum"}], modification_notes="改成博物馆并换清淡餐厅"),
        ))
        for _ in range(20):
            await asyncio.sleep(0.01)
            if len(started) == 2:
                break
        assert set(started) == {"candidate_refresh", "main_meal_search"}
        release.set()
        return await task

    events = asyncio.run(run())
    assert events[-1]["success"] is True


def test_candidate_refresh_exception_is_unverified_not_a_crash():
    nodes = make_nodes()
    nodes["modification_intent"] = lambda state: {
        "modification_search_keywords": ["museum"],
        "modification_search_status": "pending",
    }
    nodes["candidate_refresh"] = lambda state: (_ for _ in ()).throw(TimeoutError("refresh timeout"))
    state = TravelPlanState(query="trip", destination="Nanjing", days=1,
                            pois=[{"name": "Museum"}], modification_notes="改成博物馆")
    async def run():
        return [event async for event in TravelSupervisor(
            nodes, InMemoryAgentMemoryStore(), "s"
        ).stream(state, modification=True)]
    events = asyncio.run(run())
    assert events[-1]["success"] is False
    assert "candidate_refresh_unverified" in events[-1]["failure_details"]["risk_flags"]


def test_confirm_entrypoint_normalizes_empty_optional_dates(monkeypatch):
    import app.multi_agent_core.runtime as runtime_module
    monkeypatch.setattr(runtime_module, "production_nodes", lambda *args, **kwargs: make_nodes())
    checkpoint = TravelPlanState(query="trip", destination="Nanjing", days=1,
                                 pois=[{"name": "Museum"}], route=[],
                                 memory_session_id="session").model_dump(mode="json")
    checkpoint["travel_start_date"] = ""
    checkpoint["travel_end_date"] = ""
    async def run():
        return [event async for event in runtime_module.run_confirm_stream(checkpoint)]
    events = asyncio.run(run())
    assert events[-1]["type"] == "result"


def test_modification_flow_runs_analysis_refresh_before_replanning():
    nodes = make_nodes()
    seen = {}
    nodes["modification_intent"] = lambda s: {
        "attraction_preference": "museum",
        "modification_search_keywords": ["museum"],
        "modification_search_status": "pending",
    }
    nodes["candidate_refresh"] = lambda s: {
        "pois": [*s.pois, {"name": "New Museum"}],
        "modification_search_status": "complete",
    }
    original_planner = nodes["planner"]
    def planner(state):
        seen.update({"names": [poi["name"] for poi in state.pois],
                     "status": state.modification_search_status})
        return original_planner(state)
    nodes["planner"] = planner
    state = TravelPlanState(query="trip", destination="Nanjing", days=1,
                            pois=[{"name": "Museum"}], modification_notes="改成博物馆")
    async def run():
        return [event async for event in TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s").stream(state, modification=True)]
    events = asyncio.run(run())
    stages = [event["node"] for event in events if event["type"] == "stage"]
    assert stages[:3] == ["modification_intent", "candidate_refresh", "planner"]
    assert seen == {"names": ["Museum", "New Museum"], "status": "complete"}


def test_modification_refresh_failure_cannot_be_reported_as_success():
    nodes = make_nodes()
    nodes["modification_intent"] = lambda s: {
        "modification_search_keywords": ["museum"],
        "modification_search_status": "pending",
    }
    nodes["candidate_refresh"] = lambda s: {
        "pois": s.pois, "modification_search_status": "empty",
    }
    state = TravelPlanState(query="trip", destination="Nanjing", days=1,
                            pois=[{"name": "Museum"}], modification_notes="改成博物馆")
    async def run():
        return [event async for event in TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s").stream(
            state, modification=True)]
    events = asyncio.run(run())
    assert events[-1]["success"] is False
    assert "candidate_refresh_unverified" in events[-1]["failure_details"]["risk_flags"]
    assert events[-1]["plan"]["approved"] is False


def test_supervisor_api_uses_authenticated_runtime(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    import app.multi_agent_core.runtime as runtime
    from app.core.auth import create_token
    captured = {}
    async def fake_stream(query, **options):
        captured.update(options)
        yield {"type": "result", "success": False, "missing_fields": ["date"], "plan": None}
    monkeypatch.setattr(runtime, "run_stream", fake_stream)
    monkeypatch.setattr(main, "search_user_memories", lambda *args: [])
    client = TestClient(main.app)
    assert client.post("/api/plan/stream", json={"query": "trip", "engine": "supervisor"}).status_code == 401
    response = client.post("/api/plan/stream", json={"query": "trip", "engine": "supervisor"},
                           headers={"Authorization": "Bearer " + create_token("supervisor-test-owner")})
    assert response.status_code == 200
    assert captured["user_id"] == "supervisor-test-owner"
    assert "thread_id" in response.text
    assert "X-Agent-Run-ID" in response.headers


def test_memory_failure_does_not_break_planning():
    class BrokenMemory:
        def load(self, *args):
            raise OSError("storage unavailable")
        def append(self, *args):
            raise OSError("storage unavailable")
    events = collect(TravelSupervisor(make_nodes(), BrokenMemory(), "s"), TravelPlanState(query="trip"))
    assert events[-1]["success"] is True
    assert events[-1]["plan"]["degraded_services"] == ["private_memory"]


@pytest.mark.parametrize("failure", ["snapshot", "write"])
def test_runtime_memory_failure_is_visible_and_request_local(failure):
    class IntermittentMemory(InMemoryAgentMemoryStore):
        broken = True

        def load_many(self, *args):
            if self.broken and failure == "snapshot":
                raise OSError("snapshot unavailable")
            return super().load_many(*args)

        def append(self, *args):
            if self.broken and failure == "write":
                raise OSError("write unavailable")
            return super().append(*args)

    memory = IntermittentMemory()
    first = collect(TravelSupervisor(make_nodes(), memory, "s"), TravelPlanState(query="trip"))
    assert first[-1]["success"] is True
    assert first[-1]["plan"]["degraded_services"] == ["private_memory"]
    memory.broken = False
    second = collect(TravelSupervisor(make_nodes(), memory, "s"), TravelPlanState(query="trip"))
    assert second[-1]["success"] is True
    assert "private_memory" not in second[-1]["plan"].get("degraded_services", [])


def test_rule_fault_cannot_be_overridden_by_model_approval():
    nodes = make_nodes()
    nodes["planner"] = lambda s: {"route": [{"day": 1, "spots": [{"name": "Invented", "start_time": "10:00", "end_time": "11:00"}]}]}
    events = collect(TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"),
                     TravelPlanState(query="trip", max_review_rounds=0))
    assert events[-1]["success"] is False
    assert "unknown_poi" in events[-1]["checkpoint"]["route_modify_opinion"]


def test_confirm_checks_existing_draft_before_replanning():
    nodes = make_nodes()
    nodes["planner"] = lambda s: (_ for _ in ()).throw(AssertionError("accepted draft should be checked first"))
    state = TravelPlanState(query="trip", days=1, pois=[{"name": "Museum"}],
                            route=[{"day": 1, "spots": [{"name": "Museum", "period": "morning", "start_time": "10:00", "end_time": "11:00"}]}])
    async def run():
        return [e async for e in TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s").stream(state, modification=True, confirmed=True)]
    assert asyncio.run(run())[-1]["success"] is True


def test_private_history_snapshot_ignores_other_inflight_run():
    memory = InMemoryAgentMemoryStore()
    memory.append("s", "planner", {"role": "assistant", "content": "previous"})
    runtime = TravelSupervisor(make_nodes(), memory, "s")
    assert runtime._history("planner") == ["previous"]
    memory.append("s", "planner", {"role": "assistant", "content": "other request"})
    assert runtime._history("planner") == ["previous"]


def test_all_private_histories_are_frozen_before_first_node():
    memory = InMemoryAgentMemoryStore()
    memory.append("s", "reviewer", {"role": "assistant", "content": "previous review"})
    nodes = make_nodes(reject=True)
    original = nodes["intent"]
    def intent(state):
        memory.append("s", "reviewer", {"role": "assistant", "content": "other inflight review"})
        return original(state)
    nodes["intent"] = intent
    def reviewer(state):
        assert state.planner_reviewer_dialogue == ["previous review"]
        return {"approved": True}
    nodes["reviewer"] = reviewer
    events = collect(TravelSupervisor(nodes, memory, "s"), TravelPlanState(query="trip"))
    assert events[-1]["success"] is True


def test_modification_warning_matches_frontend_contract():
    nodes = make_nodes()
    original = nodes["planner"]
    nodes["planner"] = lambda s: {**original(s), "modification_concern": "Requires user confirmation"}
    async def run():
        return [e async for e in TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s").stream(
            TravelPlanState(query="trip", days=1, pois=[{"name": "Museum"}],
                            modification_notes="replace the museum"), modification=True)]
    event = asyncio.run(run())[-1]
    assert event["type"] == "modification_warning"
    assert event["concern"] == "Requires user confirmation"


def test_internal_repair_concern_does_not_interrupt_audit_loop():
    nodes = make_nodes(reject=True)
    original = nodes["planner"]
    nodes["planner"] = lambda s: {**original(s), "modification_concern": "Internal repair is difficult"}
    async def run():
        return [e async for e in TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s").stream(
            TravelPlanState(query="trip", days=1, pois=[{"name": "Museum"}], max_review_rounds=1),
            modification=True)]
    events = asyncio.run(run())
    assert all(e["type"] != "modification_warning" for e in events)
    assert events[-1]["type"] == "result"
    assert events[-1]["success"] is False


def test_optional_parallel_failures_return_explicit_degradation():
    nodes = make_nodes()
    def unavailable(state):
        raise TimeoutError("provider timeout")
    for name in ("weather_search", "query_rewrite", "spot_tips", "meal_enrichment"):
        nodes[name] = unavailable
    events = collect(TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"), TravelPlanState(query="trip"))
    final = events[-1]
    assert final["success"] is True
    assert final["checkpoint"]["weather_forecast"] == []
    assert final["checkpoint"]["spot_tips_status"] == "degraded"
    assert final["checkpoint"]["meal_search_status"] == "degraded"


def test_cancellation_does_not_save_late_worker_output():
    import threading
    memory = InMemoryAgentMemoryStore()
    started = threading.Event()
    release = threading.Event()
    nodes = make_nodes()
    def blocked(state):
        started.set()
        release.wait(2)
        return {"destination": "Nanjing", "days": 1}
    nodes["intent"] = blocked
    async def run():
        runtime = TravelSupervisor(nodes, memory, "s")
        async def consume():
            return [event async for event in runtime.stream(TravelPlanState(query="trip"))]
        task = asyncio.create_task(consume())
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation was swallowed")
    asyncio.run(run())
    assert memory.load("s", "intent") == []


def test_planner_receives_review_feedback_and_configured_limit():
    nodes = make_nodes(reject=True)
    rounds = []
    original = nodes["planner"]
    def planner(state):
        rounds.append((state.route_modify_opinion, state.max_review_rounds))
        return original(state)
    nodes["planner"] = planner
    collect(TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"), TravelPlanState(query="trip", max_review_rounds=1))
    assert rounds == [(None, 1), ("revise", 1)]


def test_time_checker_receives_destination_but_not_reviewer_history():
    memory = InMemoryAgentMemoryStore()
    memory.append("s", "reviewer", {"role": "assistant", "content": "reviewer secret"})
    memory.append("s", "time_check", {"role": "assistant", "content": "own check"})
    captured = []
    nodes = make_nodes()
    def check(state):
        captured.append((state.destination, state.planner_reviewer_dialogue))
        return {}
    nodes["time_check"] = check
    runtime = TravelSupervisor(nodes, memory, "s")
    asyncio.run(runtime._call("time_check", TravelPlanState(query="trip", destination="Nanjing")))
    assert captured == [("Nanjing", ["own check"])]


def test_parallel_enrichment_does_not_receive_private_context():
    nodes = make_nodes()
    seen = []
    def enrichment(state):
        seen.append((state.planner_reviewer_dialogue, state.agent_private_context, state.reviewer_issues))
        return {}
    nodes["meal_enrichment"] = enrichment
    nodes["spot_tips"] = enrichment
    collect(TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"),
            TravelPlanState(query="trip", planner_reviewer_dialogue=["secret"], agent_private_context=["secret"], reviewer_issues=["old"]))
    assert seen == [([], [], []), ([], [], [])]


def test_public_execution_logs_survive_without_exposing_private_dialogue():
    nodes = make_nodes()
    original = nodes["planner"]
    nodes["planner"] = lambda s: {**original(s), "history": ["new planner draft"],
                                  "planner_reviewer_dialogue": ["private internal dialogue"]}
    events = collect(TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"), TravelPlanState(query="trip"))
    assert "new planner draft" in events[-1]["history"]
    assert "private internal dialogue" not in str(events[-1]["checkpoint"])


@pytest.mark.parametrize("role,update", [
    ("planner", {"approved": True}),
    ("planner", {"pois": [{"name": "Fabricated"}]}),
    ("weather_search", {"route": []}),
    ("reviewer", {"route": []}),
    ("spot_tips", {"approved": True}),
    ("finalize", {"memory_session_id": "other"}),
])
def test_role_output_contract_blocks_cross_role_writes_before_memory(role, update):
    from app.multi_agent_core.runtime import RoleContractError
    memory = InMemoryAgentMemoryStore()
    nodes = make_nodes()
    nodes[role] = lambda s: update
    runtime = TravelSupervisor(nodes, memory, "s")
    with pytest.raises(RoleContractError):
        asyncio.run(runtime._call(role, TravelPlanState(query="trip")))
    assert memory.load("s", role) == []


def test_invalid_owned_field_does_not_enter_private_memory():
    from pydantic import ValidationError
    memory = InMemoryAgentMemoryStore()
    nodes = make_nodes()
    nodes["planner"] = lambda s: {"route": "not a route"}
    runtime = TravelSupervisor(nodes, memory, "s")
    with pytest.raises(ValidationError):
        asyncio.run(runtime._call("planner", TravelPlanState(query="trip")))
    assert memory.load("s", "planner") == []


def test_finalize_cannot_read_role_private_context_even_when_called_directly():
    nodes = make_nodes()
    def finalize(state):
        assert state.planner_reviewer_dialogue == []
        assert state.agent_private_context == []
        return {"final_plan": {}}
    nodes["finalize"] = finalize
    runtime = TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s")
    asyncio.run(runtime._call("finalize", TravelPlanState(query="trip", agent_private_context=["secret"], planner_reviewer_dialogue=["secret"])))


def test_replanning_receives_bounded_feedback_and_preserves_user_request():
    nodes = make_nodes(reject=True)
    original = nodes["planner"]
    seen = []
    def planner(state):
        seen.append((state.modification_notes, list(state.repair_feedback)))
        return original(state)
    nodes["planner"] = planner
    events = collect(TravelSupervisor(nodes, InMemoryAgentMemoryStore(), "s"),
                     TravelPlanState(query="trip", modification_notes="keep the original museum", max_review_rounds=4))
    assert len(seen) == 5
    assert all(notes == "keep the original museum" for notes, _ in seen)
    assert seen[0][1] == []
    assert seen[1][1][0]["reviewer_issues"] == ["unresolved"]
    assert len(events[-1]["checkpoint"]["repair_feedback"]) == 3


def test_production_auditors_satisfy_contract_and_receive_role_local_context(monkeypatch):
    from app.planning import nodes as production
    from app.planning.schemas import RouteReview, TimeCheckResult
    memory = InMemoryAgentMemoryStore()
    memory.append("s", "reviewer", {"role": "assistant", "content": "reviewer previous finding"})
    memory.append("s", "time_check", {"role": "assistant", "content": "time previous finding"})
    monkeypatch.setattr(production, "build_structured_llm", lambda schema, **k: schema)
    prompts = {}
    def invoke(schema, messages, **kwargs):
        prompts[schema.__name__] = messages[1][1]
        if schema is RouteReview:
            return RouteReview(reasoning="current route verified", approved=True, score=90)
        return TimeCheckResult(reasoning="current hours verified", violations=[])
    monkeypatch.setattr(production, "invoke_structured", invoke)
    runtime = TravelSupervisor({"reviewer": production.make_reviewer_node(None),
                                "time_check": production.make_time_check_node(None)}, memory, "s")
    state = TravelPlanState(query="trip", destination="Nanjing", days=1, approved=True,
                            pois=[{"name": "Museum", "open_time": "09:00-17:00"}],
                            route=[{"day": 1, "spots": [{"name": "Museum", "period": "morning", "start_time": "10:00", "end_time": "11:00"}]}])
    review = asyncio.run(runtime._call("reviewer", state))
    time = asyncio.run(runtime._call("time_check", state))
    assert review["approved"] is True and time["approved"] is True
    assert "reviewer previous finding" in prompts["RouteReview"]
    assert "time previous finding" not in prompts["RouteReview"]
    assert "time previous finding" in prompts["TimeCheckResult"]
    assert "reviewer previous finding" not in prompts["TimeCheckResult"]


def test_production_rejected_draft_does_not_call_live_finalization(monkeypatch):
    import app.planning.nodes as nodes_module
    from app.multi_agent_core.runtime import production_nodes
    monkeypatch.setattr(nodes_module, "build_structured_llm", lambda *a, **k: object())
    calls = []
    def make_finalize():
        def finalize(state):
            calls.append(state.approved)
            return {"final_plan": {"approved": state.approved}}
        return finalize
    monkeypatch.setattr(nodes_module, "make_finalize_node", make_finalize)
    nodes = production_nodes()
    draft = nodes["finalize"](TravelPlanState(query="trip", approved=False, route_risk_flags=["unknown_poi"]))
    assert calls == []
    assert draft["final_plan"]["draft_only"] is True
    assert draft["final_plan"]["unresolved_risk_flags"] == ["unknown_poi"]
    nodes["finalize"](TravelPlanState(query="trip", approved=True))
    assert calls == [True]
