from app.planning.candidate_react import candidate_validator_node, route_after_candidate_validation
from app.planning.quality import choose_snapshot, result_envelope
from app.planning.schemas import TravelPlanState


def snapshot(iteration, score, *, l0=None, l1=None):
    return {
        "iteration": iteration,
        "plan": {"route": [{"day": 1, "spots": [{"name": f"POI-{iteration}"}]}], "meal_slots": []},
        "l0_violations": l0 or [],
        "l1_violations": l1 or [],
        "soft_score": score,
        "score_breakdown": {"named_poi": score},
        "reviewer_approved": not bool(l0 or l1),
        "reviewer_warnings": [],
        "rework_instructions": [],
    }


def test_candidate_pool_checker_relaxes_count_after_bounded_search():
    state = TravelPlanState(
        query="南京一日游", destination="南京", days=1,
        pois=[{"name": "博物馆", "location": {"lat": 32.0, "lng": 118.0}}],
        candidate_search_round=2,
        candidate_search_actions=[{"action": "stop"}],
    )
    update = candidate_validator_node(state)
    assert update["candidate_pool_status"] == "insufficient"
    assert update["relaxed_constraints"]["relaxed_count"] is True
    assert route_after_candidate_validation(state.model_copy(update=update)) == "planner"


def test_unique_missing_candidate_short_circuits_with_failure_reason():
    state = TravelPlanState(
        query="南京去‘不存在景点’", destination="南京", days=1,
        candidate_search_round=2,
        candidate_search_actions=[{"action": "stop"}],
        candidate_search_trace=[{"query": "不存在景点", "query_key": "q", "error_code": "EMPTY"}],
    )
    update = candidate_validator_node(state)
    assert update["candidate_pool_status"] == "failed"
    assert update["relaxed_constraints"]["failure_reasons"]
    assert route_after_candidate_validation(state.model_copy(update=update)) == "candidate_failure"


def test_history_tier_one_chooses_highest_score_and_latest_tie():
    state = TravelPlanState(query="x", route_history=[
        snapshot(1, 0, l1=[{"code": "opening_time_conflict", "message": "时间冲突"}]),
        snapshot(2, 78), snapshot(3, 88), snapshot(4, 80),
    ])
    chosen, status, warnings, failures = choose_snapshot(state)
    assert chosen["iteration"] == 3
    assert status == "SUCCESS"
    assert not warnings and not failures


def test_history_tier_two_prefers_fewer_l1_defects_over_score():
    state = TravelPlanState(query="x", route_history=[
        snapshot(1, 95, l1=[{"code": "walking_constraint", "message": "步行超限"}, {"code": "commute_time_conflict", "message": "通勤不足"}]),
        snapshot(2, 70, l1=[{"code": "walking_constraint", "message": "步行超限"}]),
        snapshot(3, 88, l1=[{"code": "walking_constraint", "message": "步行超限"}]),
    ])
    chosen, status, warnings, failures = choose_snapshot(state)
    assert chosen["iteration"] == 3
    assert status == "PARTIAL_SUCCESS"
    assert "步行超限" in warnings
    assert not failures


def test_all_l0_versions_return_structured_failure():
    state = TravelPlanState(query="x", route_history=[
        snapshot(1, 80, l0=[{"code": "unknown_poi", "message": "候选池外景点"}]),
        snapshot(2, 75, l0=[{"code": "missing_must_visit", "message": "缺少点名景点"}]),
        snapshot(3, 70, l0=[{"code": "route_structure", "message": "路线结构不完整"}]),
    ])
    chosen, status, warnings, failures = choose_snapshot(state)
    assert chosen["iteration"] == 3
    assert status == "FAILED"
    assert failures[0]["stage"] == "L0_CONSTRAINT"
    assert not warnings


def test_result_envelope_keeps_failed_plan_object_shape():
    state = TravelPlanState(
        query="x", result_status="FAILED",
        failure_reasons=[{"code": "named_poi_unavailable", "message": "未找到指定景点", "stage": "PRECHECK"}],
        final_plan={"days": [], "route_issues": []},
    )
    result = result_envelope(state)
    assert result["status"] == "FAILED"
    assert result["plan"]["days"] == []
    assert result["failure_reasons"][0]["code"] == "named_poi_unavailable"



def test_hard_tiers_dominate_extreme_soft_scores():
    for tier in ("l0", "l1"):
        dirty = snapshot(2, 1000000, **{tier: [{"code": "unknown_poi" if tier == "l0" else "commute_time_conflict", "message": "bad"}]})
        chosen, status, _, _ = choose_snapshot(TravelPlanState(query="x", route_history=[snapshot(1, 0), dirty]))
        assert chosen["iteration"] == 1 and status == "SUCCESS"
    chosen, status, _, _ = choose_snapshot(TravelPlanState(query="x", route_history=[
        snapshot(1, 0, l1=[{"code": "walking_constraint", "message": "warning"}]),
        snapshot(2, 1000000, l0=[{"code": "unknown_poi", "message": "bad"}]),
    ]))
    assert chosen["iteration"] == 1 and status == "PARTIAL_SUCCESS"


def test_soft_score_70_30_split_and_enum_order():
    from app.planning.quality import compute_soft_score
    scores = []
    for pref, diversity in [("LOW", "POOR"), ("MEDIUM", "FAIR"), ("HIGH", "GOOD")]:
        state = TravelPlanState(query="x", reviewer_preference_match_level=pref, reviewer_diversity_and_dining_level=diversity)
        score, parts = compute_soft_score(state)
        assert sum(parts[k] for k in ("named_poi", "pace", "continuity", "weather")) == 70
        assert parts["preference_match"] + parts["diversity_and_dining"] <= 30
        assert score == sum(parts.values())
        scores.append(score)
    assert scores == [75, 87, 100]



def test_default_candidate_round_contract_stops_with_gap_at_two():
    state = TravelPlanState(query="旅行", candidate_search_round=1,
        candidate_search_actions=[{"action": "search_indoor", "query": "博物馆"}])
    assert state.candidate_max_rounds == 2
    first = state.model_copy(update=candidate_validator_node(state))
    assert route_after_candidate_validation(first) == "candidate_react"
    last = first.model_copy(update={"candidate_search_round": 2})
    last = last.model_copy(update=candidate_validator_node(last))
    assert last.candidate_missing_coverage
    assert route_after_candidate_validation(last) == "planner"
