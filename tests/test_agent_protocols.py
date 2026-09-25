from app.multi_agent_core.agents import IntentAgent, POIResearchAgent
from app.multi_agent_core.messages import AgentMessage
from app.multi_agent_core.protocols import A2ATaskEnvelope, a2a_from_message
from app.multi_agent_core.supervisor import Supervisor
from app.multi_agent_core.tools import FixturePoiTool, ToolRegistry


def test_tool_registry_exposes_permission_filtered_mcp_manifest():
    registry = ToolRegistry()
    registry.register(
        "poi_search", lambda city, query="": [],
        description="Search verified POIs",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        allowed_agents=["poi_research_agent"],
    )
    agent = POIResearchAgent(FixturePoiTool())

    manifest = registry.manifest(agent)

    assert manifest[0]["name"] == "poi_search"
    assert manifest[0]["input_schema"]["type"] == "object"
    assert registry.manifest(IntentAgent()) == []


def test_existing_agent_message_adapts_to_a2a_envelope_without_private_memory():
    message = AgentMessage(
        task_id="task-1", task_type="intent_extract", **{"from": "supervisor", "to": "intent_agent"},
        content={"user_request": "南京一日游"}, status="running", attempt=1,
    )

    envelope = a2a_from_message(message)

    assert isinstance(envelope, A2ATaskEnvelope)
    assert envelope.sender == "supervisor"
    assert envelope.recipient == "intent_agent"
    assert "private_memory" not in envelope.model_dump()


def test_supervisor_agent_cards_are_discovery_only():
    supervisor = Supervisor({
        "intent_agent": IntentAgent(),
        "poi_research_agent": POIResearchAgent(FixturePoiTool()),
    })

    cards = supervisor.agent_cards()

    assert {card.name for card in cards} == {"intent_agent", "poi_research_agent"}
    poi_card = next(card for card in cards if card.name == "poi_research_agent")
    assert poi_card.skills == ["poi_search"]
    assert poi_card.accepted_task_types == ["poi_research"]
