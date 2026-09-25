from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.core.agent_runs import (
    AgentRunRegistry,
    agent_runs,
    classify_agent_error,
    observe_agent_events,
)


def test_registry_tracks_node_attempts_without_prompt_content():
    registry = AgentRunRegistry(max_runs=5)
    run_id = registry.start("user-1", "plan", "request-1")

    registry.node_started(run_id, "planner")
    registry.node_finished(run_id, "planner")
    registry.node_started(run_id, "planner")
    registry.node_finished(run_id, "planner")
    registry.finish(run_id, "succeeded")

    trace = registry.get_owned(run_id, "user-1")
    assert trace is not None
    assert trace["status"] == "succeeded"
    assert [node["attempt"] for node in trace["nodes"]] == [1, 2]
    assert all(node["duration_ms"] >= 0 for node in trace["nodes"])
    assert "user_id" not in trace
    assert "prompt" not in str(trace).lower()
    assert registry.get_owned(run_id, "another-user") is None


def test_trace_summary_reports_parallel_overlap_degradation_and_slowest_nodes():
    run = {
        "duration_ms": 150.0,
        "degraded_services": ["spot_tips"],
        "nodes": [
            {"node": "planner", "attempt": 1, "status": "succeeded", "duration_ms": 80.0,
             "started_offset_ms": 0.0, "finished_offset_ms": 80.0},
            {"node": "weather_search", "attempt": 1, "status": "succeeded", "duration_ms": 50.0,
             "started_offset_ms": 20.0, "finished_offset_ms": 70.0},
            {"node": "spot_tips", "attempt": 1, "status": "failed", "duration_ms": 20.0,
             "started_offset_ms": 90.0, "finished_offset_ms": 110.0},
        ],
    }

    summary = AgentRunRegistry._summarize(run)

    assert summary["node_executions"] == 3
    assert summary["unique_nodes"] == 3
    assert summary["failed_nodes"] == ["spot_tips"]
    assert summary["degraded_services"] == ["spot_tips"]
    assert summary["slowest_node_executions"][0]["node"] == "planner"
    assert summary["timing"] == {
        "node_work_ms": 150.0,
        "observed_node_wall_ms": 100.0,
        "parallel_overlap_ms": 50.0,
        "parallel_overlap_ratio": 0.333,
        "max_concurrency": 2,
        "unattributed_ms": 50.0,
    }


def test_observed_stream_emits_run_id_and_completes_metrics():
    registry = AgentRunRegistry()
    run_id = registry.start("user-1", "plan")

    async def source():
        yield {"type": "stage", "node": "intent", "label": "start"}
        await asyncio.sleep(0)
        yield {"type": "stage_summary", "node": "intent", "summary": "done"}
        yield {"type": "result", "success": True, "plan": {}}

    async def collect():
        return [event async for event in observe_agent_events(
            source(), run_id, timeout_seconds=1, registry=registry
        )]

    events = asyncio.run(collect())
    assert events[0] == {"type": "run", "run_id": run_id, "timeout_seconds": 1}
    assert events[-1]["run_id"] == run_id
    assert registry.get_owned(run_id, "user-1")["status"] == "succeeded"
    metrics = registry.metrics_snapshot()
    assert metrics["run_counts"][("plan", "succeeded")] == 1
    assert metrics["node_counts"][("intent", "succeeded")] == 1


def test_observed_stream_attributes_llm_usage_to_its_run():
    from app.core import llm_usage

    llm_usage.reset_for_tests()
    registry = AgentRunRegistry()
    run_id = registry.start("user-usage", "plan")

    async def source():
        llm_usage.record_call(
            schema="TravelRoute",
            model="test-model",
            input_chars=20,
            output="done",
            usage=(11, 7),
            latency_ms=12.5,
        )
        yield {"type": "result", "success": True, "plan": {"degraded_services": ["meal_search"]}}

    async def collect():
        return [event async for event in observe_agent_events(
            source(), run_id, timeout_seconds=1, registry=registry
        )]

    asyncio.run(collect())
    summary = registry.get_owned(run_id, "user-usage")["summary"]
    assert summary["degraded_services"] == ["meal_search"]
    assert summary["llm_usage"]["calls_total"] == 1
    assert summary["llm_usage"]["total_tokens"] == 18
    assert summary["llm_usage"]["usage_exact_ratio"] == 1.0
    assert summary["llm_usage"]["attribution"] == "request_context"


def test_observed_stream_times_out_and_closes_running_node():
    registry = AgentRunRegistry()
    run_id = registry.start("user-1", "plan")

    async def source():
        yield {"type": "stage", "node": "planner", "label": "start"}
        await asyncio.sleep(0.05)
        yield {"type": "result", "success": True}

    async def collect():
        return [event async for event in observe_agent_events(
            source(), run_id, timeout_seconds=0.005, registry=registry
        )]

    events = asyncio.run(collect())
    assert events[-1]["code"] == "PLAN_TIMEOUT"
    trace = registry.get_owned(run_id, "user-1")
    assert trace["status"] == "timed_out"
    assert trace["nodes"][0]["status"] == "timed_out"


def test_provider_auth_error_is_actionable_without_leaking_key():
    code, message = classify_agent_error(
        RuntimeError("401 Authentication Fails, Your api key: sk-secret-value is invalid")
    )
    assert code == "LLM_AUTH_FAILED"
    assert "API Key 无效" in message
    assert "OPENAI_API_KEY" in message
    assert "sk-secret-value" not in message


def test_missing_provider_key_has_specific_config_error():
    code, message = classify_agent_error(
        RuntimeError("缺少 OPENAI_API_KEY。请在 .env.local 中配置后重试。")
    )
    assert code == "LLM_CONFIG_MISSING"
    assert ".env.local" in message


def test_provider_timeout_has_actionable_error():
    code, message = classify_agent_error(TimeoutError("request timed out"))
    assert code == "LLM_REQUEST_TIMEOUT"
    assert "模型接口响应超时" in message


def test_agent_run_api_enforces_owner(monkeypatch):
    from app.core.auth import create_token
    from app.main import app

    monkeypatch.setenv("JWT_SECRET", "test-only-secret-with-sufficient-length")
    user_id = "trace-owner"
    run_id = agent_runs.start(user_id, "plan", "request-api")
    agent_runs.finish(run_id, "incomplete")
    client = TestClient(app)

    owner = {"Authorization": "Bearer " + create_token(user_id)}
    other = {"Authorization": "Bearer " + create_token("other-user")}
    response = client.get(f"/api/agent-runs/{run_id}", headers=owner)
    assert response.status_code == 200
    assert response.json()["request_id"] == "request-api"
    assert client.get(f"/api/agent-runs/{run_id}", headers=other).status_code == 404
    assert client.get(f"/api/agent-runs/{run_id}").status_code == 401


def test_plan_stream_exposes_trace_end_to_end(monkeypatch):
    import json

    import app.main as main
    from app.core.auth import create_token

    monkeypatch.setenv("JWT_SECRET", "test-only-secret-with-sufficient-length")

    async def fake_plan_stream(*args, **kwargs):
        yield {"type": "stage", "node": "intent", "label": "understanding"}
        yield {"type": "stage_summary", "node": "intent", "summary": "understood"}
        yield {
            "type": "result",
            "success": False,
            "missing_fields": ["travel_start_date"],
            "history": [],
            "plan": None,
        }

    monkeypatch.setattr(main, "run_plan_stream", fake_plan_stream)
    user_id = "stream-owner"
    headers = {"Authorization": "Bearer " + create_token(user_id)}
    client = TestClient(main.app)

    response = client.post("/api/plan/stream", json={"query": "plan a trip"}, headers=headers)
    assert response.status_code == 200
    run_id = response.headers["X-Agent-Run-ID"]
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[0]["type"] == "run"
    assert events[0]["run_id"] == run_id
    assert events[-1]["type"] == "result"
    assert events[-1]["run_id"] == run_id

    trace = client.get(f"/api/agent-runs/{run_id}", headers=headers).json()
    assert trace["status"] == "incomplete"
    assert trace["nodes"][0]["node"] == "intent"
    assert trace["nodes"][0]["status"] == "succeeded"


def test_prometheus_endpoint_includes_agent_metrics():
    from app.main import app

    response = TestClient(app).get("/api/metrics")
    assert response.status_code == 200
    assert "travelmind_agent_runs_in_flight" in response.text
    assert "travelmind_agent_node_executions_total" in response.text
