from types import SimpleNamespace

from tests.eval.graders.code_graders import g2_time_check_clean, g4_structure, grade_code
from app.planning.helpers import open_time_violations


def test_structure_rejects_duplicate_spots_across_days():
    route = [
        {"day": 1, "spots": [{
            "name": "Museum", "period": "morning",
            "start_time": "10:00", "end_time": "11:00",
        }]},
        {"day": 2, "spots": [{
            "name": "Museum", "period": "afternoon",
            "start_time": "14:00", "end_time": "15:00",
        }]},
    ]

    passed, detail = g4_structure(route, [{"name": "Museum"}], 3)

    assert passed is False
    assert "重复景点" in detail


def test_grade_code_recomputes_opening_time_instead_of_trusting_empty_agent_state():
    route = [{"day": 1, "spots": [{
        "name": "Museum", "period": "afternoon",
        "start_time": "16:30", "end_time": "19:00",
    }]}]
    state = SimpleNamespace(
        route=route,
        pois=[{"name": "Museum", "open_time": "09:00-17:00"}],
        max_per_day=3,
        weather_forecast=[],
        rain_indoor_priority=False,
        max_walking_km=None,
        route_distance_legs=[],
        habit_preference=None,
        approved=True,
        review_round=1,
        max_review_rounds=3,
        time_violations=[],
        time_check_done=True,
        time_check_round=1,
        max_time_check_rounds=3,
    )

    result = grade_code(state, {"days": 1})

    assert result["results"]["g2_time_check"]["passed"] is False
    assert result["objective_pass"] is False


def test_time_check_detail_accepts_deterministic_string_violations():
    passed, detail = g2_time_check_clean(["Day1 Museum exceeds 09:00-17:00"])

    assert passed is False
    assert "Museum" in detail


def test_opening_time_accepts_visit_inside_any_declared_window():
    route = [{"day": 1, "spots": [{
        "name": "Temple", "start_time": "09:00", "end_time": "11:30",
    }]}]
    pois = [{"name": "Temple", "open_time": "15:05-17:30,17:15停止售票；07:30-17:30"}]

    assert open_time_violations(route, pois) == []
