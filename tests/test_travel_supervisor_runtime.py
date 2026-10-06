import asyncio

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
