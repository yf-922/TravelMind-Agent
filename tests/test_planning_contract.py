import pytest
from app.planning.constraints import daily_limit, effective_daily_limit, heat_violations, hot_day
from app.planning.schemas import TravelPlanState
from app.providers.amap.poi import poi_to_spot
from scripts.post_train_runtime import capture_messages
from scripts.prepare_sft_v2_release import explanation_faults


def state(**kwargs):
    return TravelPlanState(query="测试", pois=[{"name":"A", "indoor":None}], **kwargs)


@pytest.mark.parametrize("text,expected", [("每天最多两个景点",2),("每日不超过3处景点",3),
    ("每天恰好1个景点",1),("全程至少三个景点，每个90分钟",None)])
def test_daily_parse(text, expected):
    assert daily_limit(text) == expected


def test_daily_override():
    s = state(max_per_day=3)
    s.query = "每天最多两个景点"
    assert effective_daily_limit(s) == 2
    s.modification_notes = "每天最多3个景点"
    assert effective_daily_limit(s) == 3
    s.modification_notes = "每天最多5个景点"
    assert effective_daily_limit(s) == 3


@pytest.mark.parametrize("indoor,start,end,bad", [(None,"10:00","11:00",False),
    (False,"15:00","16:00",False),(None,"10:00","11:01",True),
    (False,"14:59","16:00",True),(True,"12:00","13:00",False),
    ("true","12:00","13:00",True)])
def test_heat_boundary(indoor, start, end, bad):
    s = state(weather_forecast=[{"day_temp":"38"}])
    s.pois[0]["indoor"] = indoor
    route = [{"day":1,"spots":[{"name":"A","start_time":start,"end_time":end}]}]
    assert bool(heat_violations(s, route)) == bad


def test_heat_dates_threshold():
    s = state(travel_start_date="2026-11-03", weather_forecast=[
        {"date":"2026-11-04","day_temp":"38"},{"date":"2026-11-03","day_temp":"37.9"}])
    assert not hot_day(s, 1)
    assert hot_day(s, 2)
    assert not hot_day(s, 3)


def test_provider_name_does_not_certify_indoor():
    p = poi_to_spot({"name":"某博物馆", "location":"120,30"})
    assert p["indoor"] is None
    assert p["indoor_hint"] is True


def test_production_messages_expose_contract():
    messages = capture_messages(state().model_dump(mode="json"))
    text = messages[-1]["content"]
    assert "planner-contract-v1" in text and "未知/混合" in text
    assert "11:00" in text and "15:00" in text


def test_known_repair_leak_and_legitimate_change():
    base = {"days":[],"reasoning":"本轮将原先集中在前两天的景点调整为三天", "notes":""}
    assert "repair_context_in_round_zero_answer" in explanation_faults(base)
    base["reasoning"] = "没有餐馆候选，用餐安排改为留空。"
    assert not explanation_faults(base)
