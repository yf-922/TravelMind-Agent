"""Regressions for failures observed in real Planner/Amap calibration."""
import pytest

from app.planning import candidate_react as cr, nodes
from app.planning.helpers import canonical_poi_name, extract_time_constraints, poi_name_matches
from app.planning.quality import append_route_snapshot, choose_snapshot, hard_constraint_report
from app.planning.schemas import TravelPlanState


def scheduled(spots=None, **kwargs):
    spots = spots or [{"name": "甲馆", "start_time": "10:00", "end_time": "10:30"}]
    return TravelPlanState(query="南京一日游", destination="南京", days=1,
        pois=[{"name": x["name"], "indoor": True} for x in spots],
        route=[{"day": 1, "spots": spots}], **kwargs)


@pytest.mark.parametrize("start,end", [("10:00", "10:00"), ("11:32", "10:44"),
    ("10:00", "10:29"), ("24:00", "25:00"), ("10:60", "11:30"), (None, "11:00")])
def test_invalid_visit_times_are_l1_and_cannot_be_success(start, end):
    state = scheduled([{"name": "甲馆", "start_time": start, "end_time": end}], review_round=1)
    report = hard_constraint_report(state)
    assert not report["l0"]
    assert any(x["code"] == "invalid_time_slot" for x in report["l1"])
    state = state.model_copy(update=append_route_snapshot(state))
    assert choose_snapshot(state)[1] == "PARTIAL_SUCCESS"
    assert nodes.route_after_review(state) == "planner"
    assert nodes.route_risk_gate_node(state)["review_required"]


@pytest.mark.parametrize("start,end,valid", [("10:30", "11:00", True),
    ("10:29", "11:00", False), ("09:00", "10:00", False)])
def test_original_order_must_be_monotonic(start, end, valid):
    state = scheduled([{"name": "甲馆", "start_time": "10:00", "end_time": "10:30"},
                       {"name": "乙馆", "start_time": start, "end_time": end}])
    assert (not hard_constraint_report(state)["l1"]) is valid


@pytest.mark.parametrize("start,end,valid", [("11:00", "11:20", True),
    ("11:00", "11:19", False), ("10:20", "10:50", False)])
def test_meal_duration_and_visit_overlap(start, end, valid):
    state = scheduled(meal_slots=[{"day": 1, "meal": "lunch", "restaurant_name": "甲餐厅",
        "start_time": start, "end_time": end, "duration_min": 60}])
    assert (not hard_constraint_report(state)["l1"]) is valid


@pytest.mark.parametrize("minutes,valid", [(15, True), (22, False)])
def test_real_meal_transfer_is_l1_and_prevents_enum_early_stop(minutes, valid):
    state = scheduled(
        meal_slots=[{"day": 1, "restaurant_name": "甲餐厅", "start_time": "10:45", "end_time": "11:30"}],
        route_distance_legs=[{"day": 1, "from": "甲馆", "to": "甲餐厅", "duration_min": minutes}],
        reviewer_preference_match_level="HIGH", reviewer_diversity_and_dining_level="FAIR",
        approved=False, review_round=1,
    )
    report = hard_constraint_report(state)
    assert not report["l0"]
    assert bool(report["l1"]) is not valid
    assert all(x["code"] == "commute_time_conflict" for x in report["l1"])
    state = state.model_copy(update=append_route_snapshot(state))
    assert nodes.route_after_review(state) == ("finalize" if valid else "planner")


@pytest.mark.parametrize("pref,div", [("HIGH", "GOOD"), ("HIGH", "FAIR"),
    ("MEDIUM", "GOOD"), ("MEDIUM", "FAIR")])
def test_acceptable_enums_early_stop_despite_model_veto(pref, div):
    state = scheduled(approved=False, review_round=1, time_check_required=True,
        reviewer_preference_match_level=pref, reviewer_diversity_and_dining_level=div,
        route_modify_opinion="午餐可再丰富一些", reviewer_issues=["可增加午餐选择"])
    state = state.model_copy(update=append_route_snapshot(state))
    assert nodes.route_after_review(state) == "finalize"
    assert nodes.route_after_time_check(state) == "finalize"
    chosen, status, warnings, _ = choose_snapshot(state)
    assert status == "SUCCESS_WITH_WARNINGS"
    assert "午餐可再丰富一些" in warnings
    assert not any("达到返工上限" in x for x in warnings)
    # Recorded enums belong to this snapshot, not a later mutable verdict.
    assert chosen["preference_match_level"] == pref
    assert nodes.finalize_node(state)["final_plan"]["approved"] is True


@pytest.mark.parametrize("pref,div", [("LOW", "GOOD"), ("HIGH", "POOR")])
def test_low_semantics_rework_even_if_model_approves(pref, div):
    state = scheduled(approved=True, review_round=1,
        reviewer_preference_match_level=pref, reviewer_diversity_and_dining_level=div)
    state = state.model_copy(update=append_route_snapshot(state))
    assert nodes.route_after_review(state) == "planner"
    assert nodes.route_after_time_check(state) == "planner"


def test_four_snapshot_cap_applies_to_both_routers():
    state = scheduled(approved=False, review_round=4,
        reviewer_preference_match_level="LOW", route_history=[{"iteration": i} for i in range(1, 5)])
    assert nodes.route_after_review(state) == nodes.route_after_time_check(state) == "finalize"


@pytest.mark.parametrize("history", [[], [{"iteration": 4, "l0_violations": [{"code": "route_structure"}]}]])
def test_terminal_parse_failure_cannot_reopen_planner(history):
    state = TravelPlanState(query="trip", route=[], approved=False, review_round=4,
        time_check_done=True, time_check_round=3, route_history=history)
    assert nodes.route_after_review(state) == ("finalize" if history else "time_check")
    assert nodes.route_after_time_check(state) == ("finalize" if history else ["meal_search", "spot_tips"])


@pytest.mark.parametrize("count,expected", [(0, 4), (1, 4), (4, 6), (5, 7), (6, 8), (12, 10)])
def test_dynamic_budget(count, expected):
    state = TravelPlanState(query="想去" + "、".join(f'“景点{i}”' for i in range(count)))
    assert cr.effective_api_budget(state) == expected


def transport(monkeypatch, rows=None, failure=False):
    calls = []
    monkeypatch.setattr(cr, "get_cached", lambda *_: None)
    monkeypatch.setattr(cr, "set_cached", lambda *_: None)
    monkeypatch.setattr(cr, "amap_key", lambda: "offline")
    def search(*args, keywords, request_budget, **kwargs):
        assert request_budget["remaining"] > 0
        request_budget["remaining"] -= 1
        request_budget["used"] += 1
        calls.append(keywords)
        if failure:
            raise RuntimeError("timeout")
        return rows or []
    monkeypatch.setattr(cr, "search_city_pois", search)
    monkeypatch.setattr(cr, "build_structured_llm", lambda *a, **kw: pytest.fail("named gaps require no LLM"))
    return calls


def test_six_names_all_receive_exact_search_before_core_retry(monkeypatch):
    calls = transport(monkeypatch)
    names = [f"甲{i}博物馆" for i in range(6)]
    state = TravelPlanState(query="想去" + "、".join(f'“{n}”' for n in names), destination="南京")
    state = state.model_copy(update=nodes.attraction_search_node(state))
    for _ in range(2):
        state = state.model_copy(update=cr.make_candidate_react_node()(state))
        state = state.model_copy(update=cr.candidate_search_node(state))
    assert calls[:6] == names
    assert state.candidate_api_budget == 8
    assert len(calls) == state.candidate_api_calls == cr.effective_api_budget(state) == 8
    assert state.candidate_search_round == 2


def test_two_empty_named_variants_then_frontend_failure(monkeypatch):
    calls = transport(monkeypatch)
    state = TravelPlanState(query='上海只去“星河幻境博物馆”', destination="上海")
    state = state.model_copy(update=nodes.attraction_search_node(state))
    assert calls == ["星河幻境博物馆"]
    for _ in range(2):
        state = state.model_copy(update=cr.make_candidate_react_node()(state))
        state = state.model_copy(update=cr.candidate_search_node(state))
        state = state.model_copy(update=cr.candidate_validator_node(state))
    assert calls == ["星河幻境博物馆", "星河幻境"]
    assert state.relaxed_constraints["dropped_must_visit"] == ["星河幻境博物馆"]
    assert cr.route_after_candidate_validation(state) == "candidate_failure"
    assert state.route_history == []
    assert state.relaxed_constraints["failure_reasons"][0]["code"] == "named_poi_unavailable"


def test_one_empty_lookup_is_not_confirmation_even_when_exhausted(monkeypatch):
    transport(monkeypatch)
    state = TravelPlanState(query='只去“星河幻境博物馆”', destination="上海", candidate_search_round=2,
        candidate_search_trace=[{"query": "星河幻境博物馆", "result_count": 0}])
    result = cr.candidate_validator_node(state)
    assert not result["relaxed_constraints"].get("dropped_must_visit")
    assert result["relaxed_constraints"]["unverified_must_visit"] == ["星河幻境博物馆"]


def test_provider_errors_never_become_confirmed_absence(monkeypatch):
    transport(monkeypatch, failure=True)
    state = TravelPlanState(query='只去“星河幻境博物馆”', destination="上海")
    state = state.model_copy(update=nodes.attraction_search_node(state))
    state = state.model_copy(update=cr.make_candidate_react_node()(state))
    state = state.model_copy(update=cr.candidate_search_node(state))
    state.candidate_search_round = 2
    result = cr.candidate_validator_node(state)
    assert result["relaxed_constraints"]["failure_reasons"][0]["code"] == "named_poi_unverified"


def test_per_variant_retry_exhaustion_is_provider_failure(monkeypatch):
    transport(monkeypatch)
    def limited(*args, request_budget, **kwargs):
        request_budget["remaining"] -= 1
        request_budget["used"] += 1
        raise RuntimeError("candidate_api_budget_exhausted")
    monkeypatch.setattr(cr, "search_city_pois", limited)
    state = TravelPlanState(query='想去“甲博物馆”', destination="南京",
        candidate_search_actions=[{"action": "search_user_requested_poi", "query": "甲博物馆"}])
    update = cr.candidate_search_node(state)
    assert update["candidate_api_calls"] == 1
    assert update["candidate_search_trace"][-1]["error_code"] == "SEARCH_FAILED"


def test_core_search_does_not_relax_returned_poi_identity(monkeypatch):
    transport(monkeypatch, rows=[{"name": "星河幻境咖啡店", "type": "餐饮服务", "location": "121,31"}])
    state = TravelPlanState(query='只去“星河幻境博物馆”', destination="上海",
        candidate_search_trace=[{"query": "星河幻境博物馆", "query_key": cr.query_key("上海", "星河幻境博物馆", "")}])
    state = state.model_copy(update=cr.make_candidate_react_node()(state))
    result = cr.candidate_search_node(state)
    assert not result["pois"]
    assert result["candidate_search_trace"][-1]["name_mismatch_filtered"] == 1


def test_explicit_daily_time_limits_are_deterministic_and_count_meals():
    assert extract_time_constraints("南京只有4小时")['max_duration_hours'] == 4
    assert extract_time_constraints("半天游，12点前结束") == {
        "max_duration_hours": 4.0, "latest_end_time": "12:00"
    }
    state = scheduled(
        [{"name": "甲馆", "start_time": "08:00", "end_time": "08:30"}],
        max_duration_hours=4,
        meal_slots=[{"day": 1, "restaurant_name": "午餐", "start_time": "12:30", "end_time": "13:00"}],
    )
    report = hard_constraint_report(state)
    assert any(item["code"] == "time_window_exceeded" for item in report["l1"])
    state = state.model_copy(update={"max_duration_hours": None, "latest_end_time": "12:00"})
    report = hard_constraint_report(state)
    assert any(item["code"] == "time_window_exceeded" for item in report["l1"])


@pytest.mark.parametrize("requested,returned", [
    ("解放碑", "人民解放纪念碑"),
    ("三峡博物馆", "重庆中国三峡博物馆"),
    ("洪崖洞", "洪崖洞民俗风貌区"),
])
def test_official_poi_name_variants_match_and_preserve_canonical_name(requested, returned):
    poi = {"name": returned, "category": "风景名胜", "typecode": "110000", "aliases": [requested]}
    assert poi_name_matches(returned, requested, "重庆", category=poi["category"], typecode=poi["typecode"])
    assert canonical_poi_name(requested, [poi], "重庆") == returned


def test_auxiliary_or_restaurant_name_does_not_match_explicit_attraction():
    assert not poi_name_matches("解放碑停车场", "解放碑", "重庆", category="交通设施")
    assert not poi_name_matches("星河幻境咖啡店", "星河幻境博物馆", "上海", category="餐饮服务")
    assert not poi_name_matches("解放碑剧院", "解放碑", "重庆", category="科教文化服务")
    assert not poi_name_matches("解放碑步行街", "解放碑", "重庆", category="风景名胜")


def test_canonical_spelling_wins_over_earlier_checkpoint_alias():
    pool = [
        {"name": "解放碑步行街", "aliases": ["解放碑", "人民解放纪念碑"]},
        {"name": "人民解放纪念碑", "aliases": ["解放碑"]},
    ]
    assert canonical_poi_name("人民解放纪念碑", pool, "重庆") == "人民解放纪念碑"


def test_named_search_stores_alias_and_route_uses_canonical_poi(monkeypatch):
    returned_name = "人民解放纪念碑"
    transport(monkeypatch, rows=[{
        "name": returned_name, "type": "风景名胜", "typecode": "110000", "location": "106.55,29.56"
    }])
    state = TravelPlanState(
        query='重庆必须去“解放碑”', destination="重庆",
        candidate_search_actions=[{"action": "search_user_requested_poi", "query": "解放碑"}],
    )
    result = cr.candidate_search_node(state)
    assert [poi["name"] for poi in result["pois"]] == [returned_name]
    assert "解放碑" in result["pois"][0]["aliases"]
    route = [{"day": 1, "spots": [{"name": "解放碑"}]}]
    nodes.canonicalize_route_spot_names(route, result["pois"])
    assert route[0]["spots"][0]["name"] == returned_name
    assert not hard_constraint_report(state.model_copy(update={
        "days": 1, "pois": result["pois"], "route": route,
    }))['l0']
