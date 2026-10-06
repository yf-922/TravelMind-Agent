from app.planning import candidate_react as cr
from app.planning.schemas import TravelPlanState


def poi(name, indoor=False):
    return {"name": name, "indoor": indoor, "location": {"lat": 32, "lng": 118}, "rating": 5}


def test_weather_outdoor_preference_requires_both_coverages():
    state = TravelPlanState(query="喜欢户外", days=1, rain_indoor_priority=True,
        weather_forecast=[{"is_bad": True}], pois=[poi("公园")])
    stats, missing = cr.coverage(state)
    assert "indoor" in missing and "outdoor" in missing
    assert stats["required_count"] == 4


def test_explicit_indoor_requirement_overrides_historical_outdoor_preference():
    state = TravelPlanState(query="只去室内", attraction_preference="户外", days=1,
        pois=[poi(str(i), True) for i in range(4)])
    assert cr.coverage(state)[1] == []


def test_missing_explicit_poi_is_reported():
    state = TravelPlanState(query='想去"南京博物院"', pois=[poi("公园")])
    assert "requested:南京博物院" in cr.coverage(state)[1]


def test_single_category_requests_diversity_search():
    state = TravelPlanState(query="旅行", pois=[poi(str(i)) for i in range(4)])
    assert cr.coverage(state)[1] == ["diversity"]


def test_decision_uses_current_request_memory_weather_and_observations(monkeypatch):
    captured = []
    monkeypatch.setattr(cr, "build_structured_llm", lambda *a, **k: object())
    def invoke(llm, messages, **kwargs):
        captured.append(messages[1][1])
        return cr.CandidateDecision(actions=[cr.CandidateAction(action="search_indoor", query="博物馆")])
    monkeypatch.setattr(cr, "invoke_structured", invoke)
    result = cr.make_candidate_react_node()(TravelPlanState(query="当前需求", profile_hint="历史记忆",
        weather_forecast=[{"is_bad": True}]))
    assert result["candidate_search_actions"][0]["query"] == "博物馆"
    assert "当前需求" in captured[0] and "历史记忆" in captured[0] and "is_bad" in captured[0]


def test_invalid_model_action_never_reaches_tool(monkeypatch):
    monkeypatch.setattr(cr, "build_structured_llm", lambda *a, **k: object())
    monkeypatch.setattr(cr, "invoke_structured", lambda *a, **k: object())
    state = TravelPlanState(query="旅行")
    update = cr.make_candidate_react_node()(state)
    state = state.model_copy(update=update)
    monkeypatch.setattr(cr, "search_city_pois", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert cr.candidate_search_node(state)["candidate_api_calls"] == 0


def test_cached_results_merge_without_duplicate_or_api_calls(monkeypatch):
    monkeypatch.setattr(cr, "get_cached", lambda key: [
        {"name": "公园", "location": "118,32", "type": "公园"},
        {"name": "博物馆", "location": "118,32", "type": "博物馆", "biz_ext": {"rating": "5"}}])
    monkeypatch.setattr(cr, "search_city_pois", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    state = TravelPlanState(query="旅行", pois=[poi("公园")],
        candidate_search_actions=[{"action": "search_indoor", "query": "博物馆"}])
    update = cr.candidate_search_node(state)
    assert [p["name"] for p in update["pois"]] == ["公园", "博物馆"]
    assert update["candidate_api_calls"] == 0
    assert update["candidate_search_trace"][0]["cache_hit"]


def test_budget_and_two_round_limit_preserve_partial_pool(monkeypatch):
    monkeypatch.setattr(cr, "get_cached", lambda key: None)
    monkeypatch.setattr(cr, "amap_key", lambda: "fake")
    def fail(*a, request_budget, **k):
        request_budget["used"] += 1
        request_budget["remaining"] -= 1
        raise RuntimeError("failed")
    monkeypatch.setattr(cr, "search_city_pois", fail)
    state = TravelPlanState(query="旅行", pois=[poi("公园")], candidate_search_round=1,
        candidate_api_budget=1, candidate_search_actions=[
            {"action": "search_indoor", "query": "博物馆"},
            {"action": "search_history", "query": "古迹"}])
    state = state.model_copy(update=cr.candidate_search_node(state))
    state = state.model_copy(update=cr.candidate_validator_node(state))
    assert state.candidate_pool_status == "insufficient"
    assert state.candidate_api_calls == 1 and state.pois == [poi("公园")]
    assert cr.route_after_candidate_validation(state) == "planner"


def test_repeated_query_is_not_executed_again(monkeypatch):
    monkeypatch.setattr(cr, "get_cached", lambda key: [])
    state = TravelPlanState(query="旅行", candidate_search_actions=[{"action": "search_indoor", "query": "博物馆"}])
    state = state.model_copy(update=cr.candidate_search_node(state))
    update = cr.candidate_search_node(state)
    assert update["candidate_search_trace"][-1]["error_code"] == "REPEATED_QUERY"


def test_validator_routes_gap_to_second_search_then_stops():
    state = TravelPlanState(query="旅行", candidate_search_round=1,
        candidate_search_actions=[{"action": "search_indoor", "query": "博物馆"}])
    state = state.model_copy(update=cr.candidate_validator_node(state))
    assert cr.route_after_candidate_validation(state) == "candidate_react"
    state.candidate_search_round = 2
    state = state.model_copy(update=cr.candidate_validator_node(state))
    assert cr.route_after_candidate_validation(state) == "planner"


def test_provider_budget_counts_retry_attempts(monkeypatch):
    from app.providers.amap import poi as provider
    monkeypatch.setattr(provider.time, "sleep", lambda *a: None)
    monkeypatch.setattr(provider, "http_get_json", lambda *a, **k: {"status": "0", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT"})
    budget = {"remaining": 1, "used": 0}
    import pytest
    with pytest.raises(RuntimeError):
        provider.search_city_pois("南京", "fake", keywords="公园", types="", request_budget=budget)
    assert budget == {"remaining": 0, "used": 1}


def test_real_graph_loop_observes_new_results_and_finishes_after_two_rounds(monkeypatch):
    from langgraph.graph import StateGraph, START, END
    monkeypatch.setattr(cr, "build_structured_llm", lambda *a, **k: object())
    observations = []
    def invoke(llm, messages, **kwargs):
        observations.append(messages[1][1])
        action, query = ("search_outdoor", "公园") if len(observations) == 1 else ("search_indoor", "博物馆")
        return cr.CandidateDecision(actions=[cr.CandidateAction(action=action, query=query)])
    monkeypatch.setattr(cr, "invoke_structured", invoke)
    monkeypatch.setattr(cr, "get_cached", lambda key: None)
    monkeypatch.setattr(cr, "set_cached", lambda *a: None)
    monkeypatch.setattr(cr, "amap_key", lambda: "fake")
    def search(*a, keywords, request_budget, **k):
        request_budget["remaining"] -= 1
        request_budget["used"] += 1
        return [{"name": keywords + str(i), "location": "118,32", "type": keywords,
                 "biz_ext": {"rating": "5"}} for i in range(3)]
    monkeypatch.setattr(cr, "search_city_pois", search)
    g = StateGraph(TravelPlanState)
    g.add_node("candidate_react", cr.make_candidate_react_node())
    g.add_node("search", cr.candidate_search_node)
    g.add_node("validate", cr.candidate_validator_node)
    g.add_edge(START, "candidate_react")
    g.add_edge("candidate_react", "search")
    g.add_edge("search", "validate")
    g.add_conditional_edges("validate", cr.route_after_candidate_validation,
                           {"candidate_react": "candidate_react", "planner": END})
    result = g.compile().invoke(TravelPlanState(query="喜欢户外", days=1,
        rain_indoor_priority=True, weather_forecast=[{"is_bad": True}]))
    assert result["candidate_pool_status"] == "ready"
    assert result["candidate_search_round"] == 2 and result["candidate_api_calls"] == 2
    assert "公园0" in observations[1]
    assert result["candidate_missing_coverage"] == []


def test_unrequested_explicit_tool_action_is_blocked_before_network(monkeypatch):
    monkeypatch.setattr(cr, "get_cached", lambda *a: (_ for _ in ()).throw(AssertionError()))
    state = TravelPlanState(query="旅行", candidate_search_actions=[
        {"action": "search_user_requested_poi", "query": "虚构景点"}])
    result = cr.candidate_search_node(state)
    assert result["candidate_search_trace"][-1]["error_code"] == "UNREQUESTED_PLACE"
    assert result["candidate_api_calls"] == 0
