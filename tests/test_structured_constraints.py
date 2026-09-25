from app.planning import nodes
from app.planning.helpers import extract_explicit_place_requests, missing_explicit_places
from datetime import date

from app.planning.schemas import IntentExtraction, TravelPlanState
from tests.eval.graders.code_graders import g10_habit_constraints, g9_walking_distance, g6_weather


def test_extract_explicit_place_requests_is_conservative():
    text = '请把第二天换成“南京博物院”，并增加「总统府」；喜欢室内博物馆'
    assert extract_explicit_place_requests(text) == ["南京博物院", "总统府"]


def test_canonicalize_near_spelling_without_masking_hallucination():
    route = [{"day": 1, "spots": [
        {"name": "水墨大垸旅游区"},
        {"name": "不存在的景点"},
    ]}]
    pois = [{"name": "水墨大埝旅游区"}, {"name": "总统府"}]
    replacements = nodes.canonicalize_route_spot_names(route, pois)
    assert replacements == [("水墨大垸旅游区", "水墨大埝旅游区")]
    assert route[0]["spots"][0]["name"] == "水墨大埝旅游区"
    assert route[0]["spots"][1]["name"] == "不存在的景点"


def test_missing_explicit_places_only_returns_places_outside_pool():
    pois = [{"name": "南京博物院"}]
    assert missing_explicit_places('换成“南京博物院”，再加入“总统府”', pois) == ["总统府"]


def test_intent_node_extracts_explicit_mobility_and_weather_constraints(monkeypatch):
    class FakeLLM:
        pass

    monkeypatch.setattr(nodes, "build_structured_llm", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(nodes, "invoke_structured", lambda *args, **kwargs: IntentExtraction(
        destination="南京",
        travel_start_date="2026-09-20",
        travel_end_date="2026-09-20",
        travel_days=1,
        max_walking_km=1.5,
        rain_indoor_priority=True,
    ))
    result = nodes.make_intent_node(None)(TravelPlanState(query="南京一日游，步行不要超过1.5公里，雨天优先室内"))
    assert result["max_walking_km"] == 1.5
    assert result["rain_indoor_priority"] is True


def test_intent_fans_out_weather_as_independent_stage():
    state = TravelPlanState(
        query="南京一日游",
        destination="南京",
        travel_start_date=date(2026, 9, 20),
        travel_end_date=date(2026, 9, 20),
    )
    assert nodes.route_after_intent(state) == ["query_rewrite", "weather_search"]


def test_weather_search_node_is_degradable_and_returns_forecast(monkeypatch):
    monkeypatch.setattr(nodes, "amap_key", lambda: "test-key")
    monkeypatch.setattr(
        nodes,
        "fetch_weather_for_dates",
        lambda *args: ([{"date": "2026-09-20", "is_bad": True}], None),
    )
    state = TravelPlanState(
        query="南京一日游",
        destination="南京",
        travel_start_date=date(2026, 9, 20),
        travel_end_date=date(2026, 9, 20),
    )
    update = nodes.weather_search_node(state)
    assert update["weather_forecast"][0]["is_bad"] is True


def test_route_distance_check_uses_road_service_and_records_mode(monkeypatch):
    monkeypatch.setattr(nodes, "amap_key", lambda: "test-key")
    monkeypatch.setattr(
        nodes,
        "plan_route_distance",
        lambda origin, destination, key, *, mode: {
            "mode": mode,
            "distance_km": 1.8,
            "duration_min": 24,
            "source": "amap",
        },
    )
    state = TravelPlanState(
        query="南京一日游",
        destination="南京",
        max_walking_km=2,
        pois=[
            {"name": "A", "location": {"lat": 32.04, "lng": 118.78}},
            {"name": "B", "location": {"lat": 32.05, "lng": 118.80}},
        ],
        route=[{"day": 1, "spots": [{"name": "A"}, {"name": "B"}]}],
    )
    update = nodes.route_distance_check_node(state)
    assert update["route_distance_mode"] == "walk"
    assert update["route_distance_legs"][0]["distance_km"] == 1.8


def test_g9_walking_distance_checks_only_when_user_sets_limit():
    route = [{"day": 1, "spots": [{"name": "A"}, {"name": "B"}]}]
    road_legs = [{"day": 1, "from": "A", "to": "B", "mode": "walk", "distance_km": 0.8}]
    passed, detail = g9_walking_distance(route, [], 0.5, road_legs)
    assert not passed
    assert "A→B" in detail
    assert "道路步行距离" in detail
    assert g9_walking_distance(route, [], None)[0]


def test_g6_rain_indoor_priority_overrides_fixture_threshold():
    pois = [
        {"name": "Indoor", "indoor": True},
        {"name": "Outdoor", "indoor": False},
    ]
    route = [{"day": 1, "spots": [{"name": "Outdoor"}]}]
    weather = [{"date": "2026-09-20", "is_bad": True}]
    assert g6_weather(route, pois, weather, outdoor_on_bad_day_max=1, rain_indoor_priority=False)[0]
    assert not g6_weather(route, pois, weather, outdoor_on_bad_day_max=1, rain_indoor_priority=True)[0]


def test_g10_enforces_explicit_late_start_and_slow_pace():
    bad_route = [{"day": 1, "spots": [
        {"name": "A", "start_time": "08:30"},
        {"name": "B", "start_time": "12:00"},
        {"name": "C", "start_time": "15:00"},
    ]}]
    passed, detail = g10_habit_constraints(bad_route, "不喜欢早起，慢节奏")
    assert not passed
    assert "早于10:00" in detail
    assert "3 个景点" in detail

    good_route = [{"day": 1, "spots": [
        {"name": "A", "start_time": "10:00"},
        {"name": "B", "start_time": "14:00"},
    ]}]
    assert g10_habit_constraints(good_route, "睡到自然醒，每天景点别太多")[0]
