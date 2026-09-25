from app.planning import nodes
from app.planning.schemas import TravelPlanState


def test_finalize_exposes_degraded_services_to_the_client(monkeypatch):
    monkeypatch.setattr(nodes, "amap_key", lambda: (_ for _ in ()).throw(RuntimeError("no key")))
    monkeypatch.setattr(nodes, "recommend_chain_hotel", lambda *args, **kwargs: None)
    monkeypatch.setattr(nodes, "lookup_live_ticket_prices", lambda *args, **kwargs: {})
    state = TravelPlanState(
        query="test", destination="南京", days=1,
        route=[{"day": 1, "theme": "test", "spots": []}],
        time_check_status="degraded",
        meal_search_status="partial",
        meal_recommend_status="ok",
        spot_tips_status="degraded",
    )

    update = nodes.finalize_node(state)
    plan = update["final_plan"]

    assert plan["degraded_services"] == ["time_check", "meal_search", "spot_tips"]
    assert plan["service_status"]["meal_recommend"] == "ok"
    assert len(plan["route_issues"]) == 3
    assert any("官方渠道复核" in issue for issue in plan["route_issues"])
