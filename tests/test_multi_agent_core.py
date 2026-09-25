from app.multi_agent_core.agents import IntentAgent, POIResearchAgent, PlannerAgent, ReviewerAgent
from app.multi_agent_core.messages import AgentMessage
from app.multi_agent_core.supervisor import Supervisor
from app.multi_agent_core.tools import FixturePoiTool
from app.multi_agent_core.tools import ToolPermissionError, ToolRegistry
from app.multi_agent_core.trajectory import TrajectoryContract, grade_trajectory
from app.multi_agent_core.memory import SQLiteAgentMemoryStore


def make_supervisor() -> Supervisor:
    return Supervisor({
        "intent_agent": IntentAgent(),
        "poi_research_agent": POIResearchAgent(FixturePoiTool()),
        "planner_agent": PlannerAgent(),
        "reviewer_agent": ReviewerAgent(),
    })


def test_supervisor_dispatches_multiple_agents_and_preserves_private_memories():
    supervisor = make_supervisor()
    result = supervisor.run_trip("Plan a cultural day trip", "Beijing")

    assert result["review"]["approved"] is True
    assert [entry["to"] for entry in result["dispatch_log"] if entry["from"] == "supervisor"] == [
        "intent_agent", "poi_research_agent", "planner_agent", "reviewer_agent",
    ]
    assert supervisor.agents["intent_agent"].private_memory
    assert supervisor.agents["planner_agent"].private_memory
    assert supervisor.agents["intent_agent"].private_memory is not supervisor.agents["planner_agent"].private_memory
    assert supervisor.agents["intent_agent"].private_memory != supervisor.agents["planner_agent"].private_memory


def test_poi_tool_output_is_grounded_in_the_planner_draft():
    supervisor = make_supervisor()
    result = supervisor.run_trip("Plan a relaxed day", "Chongqing")
    candidate_names = {candidate["name"] for candidate in result["candidates"]}

    assert result["itinerary"]
    assert all(item["name"] in candidate_names for item in result["itinerary"])


def test_supervisor_exposes_sanitized_auditable_tool_trace():
    supervisor = make_supervisor()
    result = supervisor.run_trip("Plan a private anniversary trip", "Beijing")

    assert len(result["tool_trace"]) == 1
    call = result["tool_trace"][0]
    assert call["agent"] == "poi_research_agent"
    assert call["tool"] == "poi_search"
    assert call["parameters"]["city"] == "Beijing"
    assert call["parameters"]["query"]["present"] is True
    assert "private anniversary" not in str(call)

    contract = TrajectoryContract(
        "happy",
        (
            ("intent_extract", "intent_agent", 0),
            ("poi_research", "poi_research_agent", 0),
            ("itinerary_plan", "planner_agent", 0),
            ("itinerary_review", "reviewer_agent", 0),
        ),
        "Beijing",
    )
    assert grade_trajectory(result, contract)["passed"] is True


class FirstDraftNeedsRevision(PlannerAgent):
    """Makes the conditional Supervisor route observable in an offline test."""

    def run(self, message: AgentMessage) -> AgentMessage:
        if message.task_type == "itinerary_plan":
            return self._reply(message, {"itinerary": [], "planning_note": "Intentional test draft."})
        return super().run(message)


def test_supervisor_routes_rejected_draft_back_to_planner_once():
    supervisor = Supervisor({
        "intent_agent": IntentAgent(),
        "poi_research_agent": POIResearchAgent(FixturePoiTool()),
        "planner_agent": FirstDraftNeedsRevision(),
        "reviewer_agent": ReviewerAgent(),
    })
    result = supervisor.run_trip("Plan a relaxed day", "Chongqing")
    dispatches = [entry for entry in result["dispatch_log"] if entry["from"] == "supervisor"]

    assert result["review"]["approved"] is True
    assert [(entry["task_type"], entry["to"], entry["attempt"]) for entry in dispatches][-2:] == [
        ("itinerary_revise", "planner_agent", 1),
        ("itinerary_review", "reviewer_agent", 1),
    ]


def test_supervisor_retries_worker_and_returns_structured_failure():
    class BrokenIntent(IntentAgent):
        def run(self, message):
            raise TimeoutError("simulated timeout")

    supervisor = Supervisor({
        "intent_agent": BrokenIntent(),
        "poi_research_agent": POIResearchAgent(FixturePoiTool()),
        "planner_agent": PlannerAgent(),
        "reviewer_agent": ReviewerAgent(),
    }, max_attempts=2)
    result = supervisor.run_trip("Plan a day", "Beijing")

    assert result["status"] == "failed"
    assert result["failed_agent"] == "intent_agent"
    assert result["error_code"] == "TimeoutError"
    assert [entry["attempt"] for entry in result["dispatch_log"] if entry["to"] == "intent_agent"] == [0, 1]


def test_tool_registry_rejects_cross_agent_tool_access():
    class ProtectedAgent(IntentAgent):
        pass

    registry = ToolRegistry()
    registry.register("poi_search", lambda city: [{"name": city}])

    try:
        registry.call(ProtectedAgent(), "poi_search", "Beijing")
    except ToolPermissionError:
        assert registry.audit_log[-1].status == "blocked"
        assert registry.audit_log[-1].error_code == "ToolPermissionError"
    else:
        raise AssertionError("an Agent without poi_search permission called the tool")


def test_model_adapter_receives_private_memory_and_role_prompt():
    captured = {}

    def adapter(system_prompt, memory, content):
        captured.update(system_prompt=system_prompt, memory=memory, content=content)
        return {"user_request": "x", "destination": "Beijing", "constraints": "y"}

    agent = IntentAgent(model=adapter)
    agent.run(AgentMessage(task_id="t", task_type="intent_extract", **{"from": "supervisor", "to": "intent_agent"}, content={"user_request": "x"}))

    assert "Extract the destination" in captured["system_prompt"]
    assert captured["memory"] == []


def test_sqlite_memory_is_isolated_by_session_and_agent(tmp_path):
    store = SQLiteAgentMemoryStore(tmp_path / "agent_memory.db")
    intent = IntentAgent(memory_store=store)
    planner = PlannerAgent(memory_store=store)

    intent.run(AgentMessage(
        task_id="task-a", session_id="session-a", task_type="intent_extract",
        **{"from": "supervisor", "to": "intent_agent"},
        content={"user_request": "visit Beijing", "destination_hint": "Beijing"},
    ))

    assert len(store.load("session-a", "intent_agent")) == 2
    assert store.load("session-a", "planner_agent") == []
    assert store.load("session-b", "intent_agent") == []
