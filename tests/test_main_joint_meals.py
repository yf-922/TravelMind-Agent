from app.planning import nodes
from app.planning.schemas import TravelPlanState


def test_time_check_exhaustion_stops_residual_meal_and_review_loops():
    for flag in ("meal_overlap", "candidate_pool_unverified"):
        state = TravelPlanState(query="trip", time_check_round=3, max_time_check_rounds=3,
                                review_round=4, max_review_rounds=3, risk_gate_rechecked=True,
                                review_required=True, route_risk_flags=[flag])
        assert nodes.route_after_time_check(state) == ["meal_search", "spot_tips"]


def test_main_meal_search_degrades_without_provider(monkeypatch):
    monkeypatch.setattr(nodes, "amap_key", lambda: (_ for _ in ()).throw(RuntimeError("no key")))
    result = nodes.main_meal_candidate_search_node(
        TravelPlanState(query="南京旅行", destination="南京", food_preference="淮扬菜")
    )
    assert result["main_meal_candidates"] == []
    assert result["main_meal_status"] == "degraded"


def test_joint_meal_risk_checks_budget_and_cuisine_metadata():
    state = TravelPlanState(
        query="南京旅行，人均不超过80元",
        destination="南京",
        days=1,
        food_preference="淮扬菜",
        main_meal_candidates=[{
            "name": "Other Kitchen", "cost": "120",
            "keytag": "川菜", "open_time": "11:00-14:00",
        }],
        meal_slots=[{
            "day": 1, "meal": "lunch", "restaurant_name": "Other Kitchen",
            "start_time": "12:00", "end_time": "13:00",
        }],
    )
    flags = nodes._main_meal_risk_flags(state)
    assert "meal_budget_conflict" in flags
    assert "meal_preference_conflict" in flags


def test_route_distance_checks_spot_to_meal_and_meal_to_spot(monkeypatch):
    calls = []
    monkeypatch.setattr(nodes, "amap_key", lambda: "key")

    def fake_route(origin, destination, api_key, *, mode):
        calls.append((origin, destination, mode))
        return {"mode": mode, "distance_km": 1.2, "duration_min": 12, "source": "fake-amap"}

    monkeypatch.setattr(nodes, "plan_route_distance", fake_route)
    state = TravelPlanState(
        query="trip", destination="Nanjing", days=1,
        pois=[
            {"name": "Museum", "location": {"lng": 118.78, "lat": 32.06}},
            {"name": "Park", "location": {"lng": 118.80, "lat": 32.07}},
        ],
        route=[{"day": 1, "spots": [
            {"name": "Museum", "start_time": "10:00", "end_time": "11:30"},
            {"name": "Park", "start_time": "14:00", "end_time": "15:00"},
        ]}],
        main_meal_candidates=[{"name": "Lunch House", "location": {"lng": 118.79, "lat": 32.065}}],
        meal_slots=[{"day": 1, "meal": "lunch", "restaurant_name": "Lunch House",
                     "start_time": "12:00", "end_time": "13:00"}],
    )
    result = nodes.route_distance_check_node(state)
    pairs = {
        (round(origin["lng"], 3), round(origin["lat"], 3),
         round(destination["lng"], 3), round(destination["lat"], 3))
        for origin, destination, _mode in calls
    }
    assert pairs == {
        (118.78, 32.06, 118.79, 32.065),
        (118.79, 32.065, 118.80, 32.07),
    }
    assert len(result["route_distance_legs"]) == 2
