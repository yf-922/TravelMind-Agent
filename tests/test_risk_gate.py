from datetime import date

from app.planning.nodes import route_after_risk_gate, route_risk_gate_node
from app.planning.schemas import TravelPlanState
from app.core.risk_gate_metrics import reset_for_tests, snapshot
from app.planning import nodes
from app.planning.schemas import TimeCheckResult


def _state(**overrides):
    base = dict(
        query="南京一日游",
        destination="南京",
        travel_start_date=date(2026, 9, 22),
        travel_end_date=date(2026, 9, 22),
        days=1,
        pois=[
            {"name": "博物馆", "open_time": "09:00-17:00", "indoor": True},
            {"name": "公园", "open_time": "09:00-18:00", "indoor": False},
        ],
        route=[{"day": 1, "spots": [{
            "name": "博物馆", "period": "morning",
            "start_time": "10:00", "end_time": "12:00",
        }]}],
    )
    base.update(overrides)
    return TravelPlanState(**base)


def test_low_risk_route_skips_expensive_audits():
    state = _state()

    update = route_risk_gate_node(state)
    final = state.model_copy(update=update)

    assert update["route_risk_flags"] == []
    assert update["review_skipped"] is True
    assert update["time_check_done"] is True
    assert update["approved"] is True
    assert route_after_risk_gate(final) == ["meal_search", "spot_tips"]


def test_duplicate_route_escalates_reviewer_and_time_check():
    state = _state(route=[{"day": 1, "spots": [
        {"name": "博物馆", "period": "morning", "start_time": "10:00", "end_time": "12:00"},
        {"name": "博物馆", "period": "afternoon", "start_time": "14:00", "end_time": "16:00"},
    ]}])

    update = route_risk_gate_node(state)
    final = state.model_copy(update=update)

    assert "duplicate_poi" in update["route_risk_flags"]
    assert update["review_required"] is True
    assert update["time_check_required"] is True
    assert route_after_risk_gate(final) == "reviewer"


def test_opening_time_only_route_goes_directly_to_time_check():
    state = _state(route=[{"day": 1, "spots": [{
        "name": "博物馆", "period": "morning",
        "start_time": "07:00", "end_time": "08:00",
    }]}])

    update = route_risk_gate_node(state)
    final = state.model_copy(update=update)

    assert "opening_time_conflict" in update["route_risk_flags"]
    assert update["review_required"] is False
    assert route_after_risk_gate(final) == "time_check"


def test_unknown_opening_hours_skip_unverifiable_llm_check_but_mark_partial():
    state = _state(
        pois=[{"name": "临时展馆", "open_time": "", "indoor": True}],
        route=[{"day": 1, "spots": [{
            "name": "临时展馆", "period": "morning",
            "start_time": "10:00", "end_time": "12:00",
        }]}],
    )

    update = route_risk_gate_node(state)
    final = state.model_copy(update=update)

    assert "opening_time_unknown" in update["route_risk_flags"]
    assert update["time_check_required"] is False
    assert update["time_check_status"] == "partial"
    assert route_after_risk_gate(final) == ["meal_search", "spot_tips"]


def test_user_modification_always_forces_independent_review():
    state = _state(modification_notes="把下午景点换成室内场馆")

    update = route_risk_gate_node(state)

    assert "user_modification" in update["route_risk_flags"]
    assert update["review_required"] is True


def test_time_check_correction_never_reenters_reviewer():
    state = _state(
        time_check_done=True,
        time_check_required=True,
        review_required=True,
        route_modify_opinion="【开放时间修正】把时间改到开馆后",
    )

    assert route_after_risk_gate(state) == "time_check"


def test_habit_and_long_road_leg_escalate_review():
    state = _state(
        habit_preference="不喜欢早起",
        route_distance_legs=[{
            "day": 1, "from": "博物馆", "to": "公园", "mode": "drive",
            "distance_km": 31,
        }],
        route=[{"day": 1, "spots": [
            {"name": "博物馆", "period": "morning", "start_time": "08:00", "end_time": "10:00"},
            {"name": "公园", "period": "afternoon", "start_time": "14:00", "end_time": "16:00"},
        ]}],
    )

    update = route_risk_gate_node(state)

    assert "habit_constraint" in update["route_risk_flags"]
    assert "long_road_leg" in update["route_risk_flags"]
    assert update["review_required"] is True


def test_risk_gate_records_low_cardinality_decisions_and_flags():
    reset_for_tests()
    state = _state(route=[{"day": 1, "spots": [
        {"name": "博物馆", "period": "morning", "start_time": "10:00", "end_time": "12:00"},
        {"name": "博物馆", "period": "afternoon", "start_time": "14:00", "end_time": "16:00"},
    ]}])

    route_risk_gate_node(state)
    metrics = snapshot()

    assert metrics["decisions"] == {"reviewer": 1}
    assert metrics["flags"] == {"duplicate_poi": 1}


def test_time_only_branch_approves_only_after_verified_hours(monkeypatch):
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        nodes, "invoke_structured",
        lambda *args, **kwargs: TimeCheckResult(reasoning="checked", violations=[]),
    )
    time_check = nodes.make_time_check_node(None)
    state = _state(review_required=False, approved=False)

    valid = time_check(state)
    invalid = time_check(_state(
        review_required=False, approved=False,
        route=[{"day": 1, "spots": [{
            "name": "博物馆", "period": "morning",
            "start_time": "07:00", "end_time": "08:00",
        }]}],
    ))

    assert valid["approved"] is True
    assert valid["time_check_status"] == "ok"
    assert invalid["approved"] is False
    assert invalid["time_check_status"] == "partial"


def test_time_check_rechecks_new_non_time_risks_after_planner_revision(monkeypatch):
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        nodes, "invoke_structured",
        lambda *args, **kwargs: TimeCheckResult(reasoning="checked", violations=[]),
    )
    state = _state(
        review_required=False, approved=False,
        route=[{"day": 1, "spots": [
            {"name": "博物馆", "period": "morning", "start_time": "10:00", "end_time": "12:00"},
            {"name": "博物馆", "period": "afternoon", "start_time": "14:00", "end_time": "16:00"},
        ]}],
    )
    update = nodes.make_time_check_node(None)(state)

    assert "duplicate_poi" in update["route_risk_flags"]
    assert update["review_required"] is True
    assert nodes.route_after_time_check(state.model_copy(update=update)) == "reviewer"


def test_time_check_does_not_repeat_approved_modification_review(monkeypatch):
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *a, **k: object())
    monkeypatch.setattr(nodes, "invoke_structured", lambda *a, **k: TimeCheckResult(reasoning="checked", violations=[]))
    state = _state(approved=True, review_required=True, modification_notes="use an indoor museum")
    update = nodes.make_time_check_node(None)(state)
    assert update["approved"] is True
    assert update["review_required"] is False
    assert nodes.route_after_time_check(state.model_copy(update=update)) == ["meal_search", "spot_tips"]


def test_time_check_does_not_override_rejected_modification_review(monkeypatch):
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *a, **k: object())
    monkeypatch.setattr(nodes, "invoke_structured", lambda *a, **k: TimeCheckResult(reasoning="checked", violations=[]))
    state = _state(approved=False, review_required=True, modification_notes="use an indoor museum")
    assert nodes.make_time_check_node(None)(state)["approved"] is False


def test_time_checker_prompt_uses_private_reference_not_old_verdict(monkeypatch):
    prompts = []
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *a, **k: object())
    def invoke(llm, messages, **kwargs):
        prompts.append(messages[1][1])
        return TimeCheckResult(reasoning="current route checked", violations=[])
    monkeypatch.setattr(nodes, "invoke_structured", invoke)
    update = nodes.make_time_check_node(None)(_state(approved=True, agent_private_context=["OLD_TIME_PROBLEM"]))
    assert "OLD_TIME_PROBLEM" in prompts[0]
    assert "已修复的问题不得沿用" in prompts[0]
    assert update["time_violations"] == []


def test_explicit_modification_time_must_be_applied_to_route_json():
    state = _state(modification_notes="坚持原景点凌晨02:00至03:00游览，不接受改时间")
    assert "modification_time_unfulfilled" in route_risk_gate_node(state)["route_risk_flags"]
    state = _state(modification_notes="保持原景点，游览改为11:00开始")
    assert nodes.explicit_modification_time_violation(state) is True
    state.route[0]["spots"][0].update(start_time="11:00", end_time="12:00")
    assert nodes.explicit_modification_time_violation(state) is False


def test_requested_window_and_opening_hours_are_independent_constraints():
    state = _state(modification_notes="坚持02:00至03:00游览")
    state.route[0]["spots"][0].update(start_time="02:00", end_time="03:00")
    flags = route_risk_gate_node(state)["route_risk_flags"]
    assert "opening_time_conflict" in flags
    assert "modification_time_unfulfilled" not in flags


def test_indoor_only_request_escalates_unknown_and_outdoor_candidates():
    state = _state(query="只参观一个室内历史文化景点")
    assert "indoor_constraint" not in route_risk_gate_node(state)["route_risk_flags"]
    state.pois[0]["indoor"] = None
    update = route_risk_gate_node(state)
    assert "indoor_constraint" in update["route_risk_flags"]
    assert update["review_required"] is True
    state.pois[0]["indoor"] = False
    assert "indoor_constraint" in route_risk_gate_node(state)["route_risk_flags"]
    state.modification_notes = "游览改为11:00开始"
    assert "indoor_constraint" in route_risk_gate_node(state)["route_risk_flags"]
