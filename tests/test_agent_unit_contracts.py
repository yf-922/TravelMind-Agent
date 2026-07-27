"""三类核心 Agent 的离线单元测试：角色、工具权限与输出契约。"""

from app.multi_agent_core.agents import IntentAgent, POIResearchAgent, PlannerAgent
from app.multi_agent_core.messages import AgentMessage
from app.multi_agent_core.tools import FixturePoiTool


def test_intent_agent_extracts_destination_and_keeps_private_memory():
    agent = IntentAgent()
    reply = agent.run(AgentMessage(
        task_id="unit-intent", session_id="unit-session", task_type="intent_extract",
        **{"from": "supervisor", "to": "intent_agent"},
        content={"user_request": "想慢慢逛重庆", "destination_hint": "重庆"},
    ))
    assert reply.status == "done"
    assert reply.content["destination"] == "重庆"
    assert len(agent.private_memory) == 2


def test_poi_research_agent_only_returns_tool_grounded_candidates():
    agent = POIResearchAgent(FixturePoiTool())
    reply = agent.run(AgentMessage(
        task_id="unit-poi", session_id="unit-session", task_type="poi_research",
        **{"from": "supervisor", "to": "poi_research_agent"},
        content={"destination": "北京", "place_query": "历史文化"},
    ))
    assert reply.status == "done"
    assert reply.content["candidates"]
    # Fixture 与真实高德工具的共同输出契约是 name/address；坐标属于线上 POI 的可选增强字段。
    assert all("name" in poi and "address" in poi for poi in reply.content["candidates"])


def test_planner_agent_does_not_invent_places_outside_candidates():
    candidates = [
        {"name": "故宫博物院", "location": {"lng": 116.397, "lat": 39.918}},
        {"name": "景山公园", "location": {"lng": 116.397, "lat": 39.925}},
    ]
    agent = PlannerAgent()
    reply = agent.run(AgentMessage(
        task_id="unit-planner", session_id="unit-session", task_type="itinerary_plan",
        **{"from": "supervisor", "to": "planner_agent"},
        content={"intent": {"destination": "北京"}, "candidates": candidates},
    ))
    assert reply.status == "done"
    assert {item["name"] for item in reply.content["itinerary"]} <= {item["name"] for item in candidates}
