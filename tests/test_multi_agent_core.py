from app.multi_agent_core.agents import IntentAgent, POIResearchAgent, PlannerAgent, ReviewerAgent
from app.multi_agent_core.messages import AgentMessage
from app.multi_agent_core.supervisor import Supervisor
from app.multi_agent_core.tools import FixturePoiTool


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
    assert supervisor.agents["intent_agent"].private_memory != supervisor.agents["planner_agent"].private_memory


def test_poi_tool_output_is_grounded_in_the_planner_draft():
    supervisor = make_supervisor()
    result = supervisor.run_trip("Plan a relaxed day", "Chongqing")
    candidate_names = {candidate["name"] for candidate in result["candidates"]}

    assert result["itinerary"]
    assert all(item["name"] in candidate_names for item in result["itinerary"])


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
