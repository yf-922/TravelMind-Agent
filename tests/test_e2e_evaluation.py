"""10 条可自动执行的端到端用例：用户请求 → Supervisor → 多 Agent → 审核结果。"""

import pytest

from app.multi_agent_core.agents import IntentAgent, POIResearchAgent, PlannerAgent, ReviewerAgent
from app.multi_agent_core.supervisor import Supervisor
from app.multi_agent_core.tools import FixturePoiTool


CASES = [
    ("E2E-01", "帮我规划北京文化一日游", "北京"),
    ("E2E-02", "重庆慢节奏一日游，想吃本地小吃", "重庆"),
    ("E2E-03", "北京亲子轻松游，少走路", "北京"),
    ("E2E-04", "重庆雨天也能去的地方", "重庆"),
    ("E2E-05", "北京第一次旅行，安排经典景点", "北京"),
    ("E2E-06", "重庆周末半日游，不要排得太满", "重庆"),
    ("E2E-07", "北京预算有限的城市漫步", "北京"),
    ("E2E-08", "重庆朋友同行，安排有代表性的景点", "重庆"),
    ("E2E-09", "北京历史建筑参观路线", "北京"),
    ("E2E-10", "重庆第一次来，给我一条可执行路线", "重庆"),
]


def make_supervisor() -> Supervisor:
    return Supervisor({
        "intent_agent": IntentAgent(),
        "poi_research_agent": POIResearchAgent(FixturePoiTool()),
        "planner_agent": PlannerAgent(),
        "reviewer_agent": ReviewerAgent(),
    })


@pytest.mark.parametrize("case_id,user_request,destination", CASES, ids=[item[0] for item in CASES])
def test_end_to_end_trip_cases(case_id, user_request, destination):
    result = make_supervisor().run_trip(user_request, destination, session_id=case_id)

    assert result.get("status") != "failed"
    assert result["intent"]["destination"] == destination
    assert result["itinerary"]
    assert result["review"]["approved"] is True
    dispatches = [item for item in result["dispatch_log"] if item["from"] == "supervisor"]
    assert [item["to"] for item in dispatches][:4] == [
        "intent_agent", "poi_research_agent", "planner_agent", "reviewer_agent",
    ]
