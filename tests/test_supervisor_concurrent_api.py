import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app.core.auth import create_token
from app.multi_agent_core.memory import SQLiteAgentMemoryStore
from app.multi_agent_core.runtime import TravelSupervisor, memory_scope
from app.planning.schemas import TravelPlanState
from tests.test_travel_supervisor_runtime import make_nodes


def test_same_owner_parallel_api_trips_save_distinct_private_sessions(monkeypatch, tmp_path):
    import app.main as main
    from app.core import database
    import app.multi_agent_core.runtime as runtime
    import app.multi_agent_core.memory as memory_module

    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "application.db")
    database.init_db()
    store = SQLiteAgentMemoryStore(tmp_path / "private.db")
    monkeypatch.setattr(memory_module, "SQLiteAgentMemoryStore", lambda: store)
    monkeypatch.setattr(main, "search_user_memories", lambda *a: [])
    async def no_background(*a):
        return None
    monkeypatch.setattr(main, "run_profile_update_agent", no_background)
    monkeypatch.setattr(main, "run_semantic_memory_update", no_background)
    barrier = threading.Barrier(2)
    def factory(*a, **k):
        nodes = make_nodes()
        original = nodes["planner"]
        def planner(state):
            assert state.planner_reviewer_dialogue == []
            barrier.wait(timeout=5)
            return {**original(state), "rag_sources": [{"text": state.query}]}
        nodes["planner"] = planner
        return nodes
    monkeypatch.setattr(runtime, "production_nodes", factory)
    headers = {"Authorization": "Bearer " + create_token("parallel-owner")}
    with TestClient(main.app) as client, ThreadPoolExecutor(max_workers=2) as workers:
        def request(query):
            response = client.post("/api/plan/stream", headers=headers, json={"query": query, "engine": "supervisor"})
            assert response.status_code == 200
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            result = events[-1]
            assert result["success"] is True
            checkpoint = client.get("/api/history/" + result["plan_id"], headers=headers).json()["planner_state"]
            return query, checkpoint, result["run_id"]
        results = list(workers.map(request, ("trip alpha", "trip beta")))
    assert len({checkpoint["memory_session_id"] for _, checkpoint, _ in results}) == 2
    assert len({run_id for _, _, run_id in results}) == 2
    for query, checkpoint, run_id in results:
        entries = store.load(memory_scope("parallel-owner", checkpoint["memory_session_id"]), "planner")
        assert len(entries) == 1
        assert query in entries[0]["content"]
        assert store.load(memory_scope("other-owner", checkpoint["memory_session_id"]), "planner") == []
        assert main.agent_runs.get_owned(run_id, "parallel-owner")["status"] == "succeeded"


def test_same_trip_concurrent_modifications_keep_frozen_history_and_all_writes(tmp_path):
    store = SQLiteAgentMemoryStore(tmp_path / "private.db")
    scope = memory_scope("owner", "trip")
    store.append(scope, "planner", {"role": "assistant", "content": "previous draft"})
    barrier = threading.Barrier(2)
    seen = []
    def factory():
        nodes = make_nodes()
        original = nodes["planner"]
        def planner(state):
            seen.append((state.query, list(state.planner_reviewer_dialogue)))
            barrier.wait(timeout=5)
            return {**original(state), "rag_sources": [{"text": state.query}]}
        nodes["planner"] = planner
        return nodes
    async def run():
        async def consume(query):
            state = TravelPlanState(query=query, days=1, pois=[{"name": "Museum"}])
            return [event async for event in TravelSupervisor(factory(), store, scope).stream(state, modification=True)]
        return await asyncio.gather(consume("edit alpha"), consume("edit beta"))
    results = asyncio.run(run())
    assert all(events[-1]["success"] for events in results)
    assert sorted(seen) == [("edit alpha", ["previous draft"]), ("edit beta", ["previous draft"])]
    entries = store.load(scope, "planner")
    assert len(entries) == 3
    assert "edit alpha" in str(entries) and "edit beta" in str(entries)


def test_supervisor_api_replays_saved_trip_through_modification(tmp_path, monkeypatch):
    import app.main as main
    import app.multi_agent_core.runtime as runtime
    import app.multi_agent_core.memory as memory_module
    from app.core import database

    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "application.db")
    database.init_db()
    private = SQLiteAgentMemoryStore(tmp_path / "private.db")
    monkeypatch.setattr(memory_module, "SQLiteAgentMemoryStore", lambda: private)
    monkeypatch.setattr(main, "search_user_memories_with_status", lambda *args: ([], "skipped"))
    async def no_background(*args, **kwargs):
        return None
    monkeypatch.setattr(main, "run_profile_update_agent", no_background)
    monkeypatch.setattr(main, "run_semantic_memory_update", no_background)

    from tests.test_travel_supervisor_runtime import make_nodes
    def factory(*args, **kwargs):
        nodes = make_nodes()
        nodes["modification_intent"] = lambda state: {
            "modification_search_keywords": [],
            "modification_search_status": "not_required",
            "attraction_preference": "历史文化",
        }
        return nodes
    monkeypatch.setattr(runtime, "production_nodes", factory)
    headers = {"Authorization": "Bearer " + create_token("replay-owner")}

    def events(response):
        assert response.status_code == 200
        return [json.loads(line[6:]) for line in response.text.splitlines()
                if line.startswith("data: ")]

    with TestClient(main.app) as client:
        first = events(client.post("/api/plan/stream", headers=headers,
                                   json={"query": "南京历史文化一日游", "engine": "supervisor"}))
        first_result = next(event for event in reversed(first) if event["type"] == "result")
        assert first_result["success"] is True
        parent_id = first_result["plan_id"]
        second = events(client.post("/api/plan/stream", headers=headers, json={
            "query": "南京历史文化一日游", "engine": "supervisor",
            "plan_id": parent_id, "modification_notes": "改成博物馆路线",
        }))
        second_result = next(event for event in reversed(second) if event["type"] == "result")
        assert second_result["success"] is True
        assert second_result["plan_id"] != parent_id
        assert [event["node"] for event in second if event["type"] == "stage"][:3] == [
            "modification_intent", "planner", "route_distance_check",
        ]
        detail = client.get("/api/history/" + second_result["plan_id"], headers=headers)
        assert detail.status_code == 200
        checkpoint = detail.json()["planner_state"]
        assert checkpoint["memory_session_id"] == client.get(
            "/api/history/" + parent_id, headers=headers).json()["planner_state"]["memory_session_id"]
        assert checkpoint["attraction_preference"] == "历史文化"
