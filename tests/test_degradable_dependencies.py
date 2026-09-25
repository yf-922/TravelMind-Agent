from app.planning import nodes
from app.planning.schemas import TravelPlanState


def _route_state() -> TravelPlanState:
    return TravelPlanState(
        query="test",
        destination="南京",
        route=[{"day": 1, "spots": [
            {"name": "A", "period": "morning", "start_time": "09:00", "end_time": "11:00"},
            {"name": "B", "period": "afternoon", "start_time": "13:00", "end_time": "15:00"},
        ]}],
        pois=[
            {"name": "A", "location": {"lat": 1.0, "lng": 1.0}},
            {"name": "B", "location": {"lat": 1.1, "lng": 1.1}},
        ],
    )


def test_meal_search_degrades_when_map_key_is_unavailable(monkeypatch):
    monkeypatch.setattr(nodes, "amap_key", lambda: (_ for _ in ()).throw(ValueError("missing")))

    update = nodes.meal_search_node(_route_state())

    assert len(update["meal_candidates"]) == 1
    assert update["meal_candidates"][0]["lunch"]["candidates"] == []
    assert update["meal_search_status"] == "degraded"
    assert "地图服务不可用" in update["history"][-1]


def test_meal_recommend_uses_rating_fallback_for_non_runtime_provider_error(monkeypatch):
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(nodes, "invoke_structured", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))
    state = TravelPlanState(query="test", meal_candidates=[{
        "day": 1,
        "lunch": {"anchor": "A", "candidates": [
            {"name": "Top Lunch", "rating": 4.9, "cost": "30", "keytag": "local"},
        ]},
        "dinner": {"anchor": "B", "candidates": [
            {"name": "Top Dinner", "rating": 4.8, "cost": "50", "keytag": "local"},
        ]},
    }])

    update = nodes.make_meal_recommend_node(None)(state)

    assert update["meals"][0]["lunch"]["name"] == "Top Lunch"
    assert update["meals"][0]["dinner"]["name"] == "Top Dinner"
    assert update["meal_recommend_status"] == "partial"


def test_spot_tips_timeout_degrades_to_empty_enrichment(monkeypatch):
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(nodes, "invoke_structured", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))

    update = nodes.make_spot_tips_node(None)(_route_state())

    assert update == {"spot_tips": {}, "spot_guides": {}, "spot_tips_status": "degraded"}


def test_time_check_timeout_marks_check_done_and_keeps_pipeline_moving(monkeypatch):
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(nodes, "invoke_structured", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))

    update = nodes.make_time_check_node(None)(_route_state())

    assert update["time_check_done"] is True
    assert update["time_check_status"] == "degraded"
    assert update["time_violations"] == []
    assert "跳过时间核查" in update["history"][-1]
