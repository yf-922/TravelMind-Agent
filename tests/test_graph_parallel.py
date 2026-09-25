import threading
import time
from datetime import date

from app.planning.schemas import TravelPlanState


def test_intent_post_processing_runs_weather_and_rewrite_in_parallel(monkeypatch):
    import app.planning.graph as graph_module

    intervals: dict[str, tuple[float, float]] = {}
    lock = threading.Lock()

    def timed(name, update):
        def node(state):
            started = time.perf_counter()
            time.sleep(0.08)
            finished = time.perf_counter()
            with lock:
                intervals[name] = (started, finished)
            return update(state) if callable(update) else update
        return node

    monkeypatch.setattr(graph_module, "make_intent_node", lambda *a, **k: lambda state: {
        "destination": "南京", "travel_start_date": date(2026, 9, 20),
        "travel_end_date": date(2026, 9, 20), "days": 1, "missing_fields": [],
        "history": state.history + ["intent"],
    })
    monkeypatch.setattr(graph_module, "make_query_rewrite_node", lambda *a, **k: timed(
        "query_rewrite", lambda state: {"rewritten_query": state.query, "history": state.history + ["rewrite"]}
    ))
    monkeypatch.setattr(graph_module, "weather_search_node", timed(
        "weather_search", {"weather_forecast": [], "weather_note": "mock"}
    ))
    monkeypatch.setattr(graph_module, "attraction_search_node", lambda state: {
        "pois": [], "history": state.history + ["attractions"]
    })
    monkeypatch.setattr(graph_module, "make_planner_node", lambda *a, **k: lambda state: {
        "route": [{"day": 1, "spots": []}], "review_round": 1,
    })
    monkeypatch.setattr(graph_module, "route_distance_check_node", lambda state: {})
    monkeypatch.setattr(graph_module, "make_reviewer_node", lambda *a, **k: lambda state: {
        "approved": True, "reviewer_issues": [], "route_modify_opinion": None,
    })
    monkeypatch.setattr(graph_module, "make_time_check_node", lambda *a, **k: lambda state: {
        "time_check_done": True, "time_violations": [], "time_check_round": 1,
    })
    monkeypatch.setattr(graph_module, "make_meal_enrichment_node", lambda *a, **k: lambda state: {
        "meal_candidates": [], "meals": [],
    })
    monkeypatch.setattr(graph_module, "make_spot_tips_node", lambda *a, **k: lambda state: {"spot_tips": {}})
    monkeypatch.setattr(graph_module, "make_finalize_node", lambda *a, **k: lambda state: {"final_plan": {}})

    app = graph_module.build_graph()
    result = app.invoke(TravelPlanState(query="南京一日游"), config={"recursion_limit": 30})
    assert result["final_plan"] == {}
    rewrite_start, rewrite_end = intervals["query_rewrite"]
    weather_start, weather_end = intervals["weather_search"]
    assert rewrite_start < weather_end and weather_start < rewrite_end


def test_meal_enrichment_and_spot_tips_run_in_parallel_before_finalize(monkeypatch):
    import app.planning.graph as graph_module

    intervals: dict[str, tuple[float, float]] = {}
    finalized: dict[str, object] = {}
    lock = threading.Lock()

    def delayed(name, update):
        def node(state):
            started = time.perf_counter()
            time.sleep(0.08)
            finished = time.perf_counter()
            with lock:
                intervals[name] = (started, finished)
            return update
        return node

    monkeypatch.setattr(graph_module, "make_intent_node", lambda *a, **k: lambda state: {
        "destination": "南京", "travel_start_date": date(2026, 9, 20),
        "travel_end_date": date(2026, 9, 20), "days": 1, "missing_fields": [],
    })
    monkeypatch.setattr(graph_module, "make_query_rewrite_node", lambda *a, **k: lambda state: {
        "rewritten_query": state.query,
    })
    monkeypatch.setattr(graph_module, "weather_search_node", lambda state: {
        "weather_forecast": [], "weather_note": "mock",
    })
    monkeypatch.setattr(graph_module, "attraction_search_node", lambda state: {"pois": []})
    monkeypatch.setattr(graph_module, "make_planner_node", lambda *a, **k: lambda state: {
        "route": [{"day": 1, "spots": []}], "review_round": 1,
    })
    monkeypatch.setattr(graph_module, "route_distance_check_node", lambda state: {})
    monkeypatch.setattr(graph_module, "make_reviewer_node", lambda *a, **k: lambda state: {
        "approved": True, "reviewer_issues": [], "route_modify_opinion": None,
    })
    monkeypatch.setattr(graph_module, "make_time_check_node", lambda *a, **k: lambda state: {
        "time_check_done": True, "time_violations": [], "time_check_round": 1,
    })
    monkeypatch.setattr(graph_module, "make_meal_enrichment_node", lambda *a, **k: delayed(
        "meal_enrichment", {"meals": [{"day": 1, "lunch": None, "dinner": None}]}
    ))
    monkeypatch.setattr(graph_module, "make_spot_tips_node", lambda *a, **k: delayed(
        "spot_tips", {"spot_tips": {"Museum": "tip"}}
    ))

    def finalize(state):
        finalized["meals"] = state.meals
        finalized["spot_tips"] = state.spot_tips
        return {"final_plan": {}}

    monkeypatch.setattr(graph_module, "make_finalize_node", lambda *a, **k: finalize)

    result = graph_module.build_graph().invoke(
        TravelPlanState(query="南京一日游"), config={"recursion_limit": 30}
    )

    assert result["final_plan"] == {}
    assert finalized["meals"] and finalized["spot_tips"] == {"Museum": "tip"}
    meal_start, meal_end = intervals["meal_enrichment"]
    tips_start, tips_end = intervals["spot_tips"]
    assert meal_start < tips_end and tips_start < meal_end
