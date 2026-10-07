from datetime import date

from app.providers.weather import amap
from app.planning import nodes
from app.planning.helpers import fetch_weather_for_dates
from app.planning.schemas import TravelPlanState


def test_weather_timeout_is_single_attempt_and_not_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(amap, "get_cached", lambda key: None)
    monkeypatch.setattr(amap, "set_cached", lambda *a: (_ for _ in ()).throw(AssertionError()))
    def timeout(url, **kwargs):
        calls.append(kwargs)
        raise TimeoutError()
    monkeypatch.setattr(amap, "http_get_json", timeout)
    assert amap.fetch_forecast("北京", "fake") == []
    assert calls == [{"timeout": 5, "attempts": 1}]


def test_unavailable_weather_is_not_assumed_sunny(monkeypatch):
    monkeypatch.setattr(amap, "fetch_forecast", lambda *a: [])
    forecast, note = fetch_weather_for_dates("北京", date(2026, 10, 8), date(2026, 10, 10), "fake")
    assert forecast == []
    assert "不将未知天气视为晴天" in note


def test_query_rewrite_failure_preserves_intent_and_attempts_once(monkeypatch):
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *a, **k: object())
    calls = []
    def timeout(*args, **kwargs):
        calls.append(kwargs)
        raise TimeoutError()
    monkeypatch.setattr(nodes, "invoke_structured", timeout)
    state = TravelPlanState(query="北京三日游", attraction_preference="历史", rain_indoor_priority=True)
    update = nodes.make_query_rewrite_node(None, None)(state)
    assert calls == [{"retries": 1}]
    final = state.model_copy(update=update)
    assert final.attraction_preference == "历史"
    assert final.rain_indoor_priority
    assert "失败降级" in final.history[-1]
