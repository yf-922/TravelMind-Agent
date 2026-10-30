"""Production parser, conflict priority and action-boundary regressions."""
import json

import pytest

from app.planning import candidate_react as cr
from app.planning.schemas import TravelPlanState
from app.providers.amap.poi import poi_to_spot


def raw(name="某馆", category="科教文化服务;博物馆", **kw):
    return {"name":name,"type":category,"location":"118,32",**kw}


@pytest.mark.parametrize("category", ["博物馆","美术馆","科技馆","展览馆","图书馆","纪念馆","天文馆","海洋馆","水族馆","室内游乐","剧场","音乐厅","购物中心","商场"])
def test_whitelisted_types_certify_only_building_visit(category):
    p=poi_to_spot(raw(category=category))
    assert p["indoor"] is True and p["indoor_source"]=="amap_type_verified"
    assert "不包括" in p["indoor_scope"]


@pytest.mark.parametrize("root", ["遗址公园","森林公园","广场","露天","户外","外景"])
def test_outdoor_name_vetoes_indoor_type(root):
    assert poi_to_spot(raw(name="甲"+root))["indoor"] is None


def test_name_alone_is_not_indoor_evidence():
    assert poi_to_spot(raw(name="某博物馆",category="风景名胜"))["indoor"] is None


@pytest.mark.parametrize("category", ["购物服务;家居建材市场","公司企业","汽车服务","汽车销售","医疗保健","生活服务;中介机构","生活服务;维修站"])
def test_non_tourism_categories_rejected(category):
    assert poi_to_spot(raw(category=category)) is None


def test_misclassified_merchant_does_not_pass_museum_type():
    assert poi_to_spot(raw("绿盾高性能系统门窗")) is None
    assert poi_to_spot(raw("中国门窗博物馆三亚分馆")) is not None


@pytest.mark.parametrize("quote,force,allowed", [(True,True,True),(True,False,False),(False,True,False)])
def test_force_requires_both_exact_quote_and_server_flag(monkeypatch,quote,force,allowed):
    name="甲门窗店"
    monkeypatch.setattr(cr,"get_cached",lambda k:[raw(name,"购物服务;家居建材")])
    state=TravelPlanState(query=f'想去“{name}”' if quote else '想去甲门窗店',candidate_force_named_non_tourism=force,
        candidate_search_actions=[{"action":"EXACT_NAME","query":name}])
    result=cr.candidate_search_node(state)
    assert bool(result["pois"]) is allowed


@pytest.mark.parametrize("forecast", [{"is_bad":True},{"day_weather":"小雨"},{"day_temp":"35"},{"daytemp":"38"}])
def test_weather_reserves_two_indoor_candidates_without_history_override(forecast):
    state=TravelPlanState(query="带老人游玩",attraction_preference="登山徒步",profile_hint="登山徒步",
        rewritten_query="登山徒步",weather_forecast=[forecast])
    obs=cr.compact_observation(state)
    assert obs["active_gaps"]["indoor_shortage"]==2
    assert obs["preferences"]["history_suppressed"]
    assert "登山徒步" not in json.dumps(obs,ensure_ascii=False)


def test_active_gaps_are_unique_and_pool_is_cleaned():
    p=poi_to_spot(raw())
    state=TravelPlanState(query='想去“缺失馆”',pois=[p,p,{"name":"绿盾高性能系统门窗","category":"博物馆"}])
    result=cr.candidate_validator_node(state)
    assert len(result["pois"])==1
    assert result["candidate_active_gaps"]["missing_named_pois"]==["缺失馆"]
    assert result["candidate_active_gaps"]["count_shortage"]==3


def test_current_exclusion_overrides_previous_named_request():
    state=TravelPlanState(query='想去“某馆”',modification_notes='不去“某馆”，改成“另一馆”',pois=[poi_to_spot(raw())])
    assert cr.active_gaps(state)["missing_named_pois"]==["另一馆"]
    assert cr.clean_pool(state)==[]


def test_only_missing_exact_name_can_be_queried_before_other_gaps():
    state=TravelPlanState(query='想去“南京博物院”',weather_forecast=[{"is_bad":True}])
    assert cr.valid_actions(state,[{"action":"INDOOR_CATEGORY","query":"博物馆"}])==[{"action":"stop"}]
    assert cr.valid_actions(state,[{"action":"EXACT_NAME","query":"南京博物院分馆"}])==[{"action":"stop"}]
    assert cr.valid_actions(state,[{"action":"EXACT_NAME","query":"南京博物院"}])[0]["query"]=="南京博物院"


def test_model_decision_duplicate_is_blocked_before_tool(monkeypatch):
    state=TravelPlanState(query="只去室内",candidate_search_trace=[{"query_key":cr.query_key("南京","博物馆","140100"),"query":"博物馆","types":"140100"}],destination="南京")
    monkeypatch.setattr(cr,"build_structured_llm",lambda *a,**kw:object())
    seen=[]
    def invoke(llm,messages,**kw):
        seen.append(json.loads(messages[1][1]))
        return cr.CandidateDecision(actions=[cr.CandidateAction(action="INDOOR_CATEGORY",query="博物馆")])
    monkeypatch.setattr(cr,"invoke_structured",invoke)
    result=cr.make_candidate_react_node()(state)
    assert result["candidate_search_actions"]==[{"action":"stop"}]
    assert result["candidate_search_trace"][-1]["error_code"]=="REPEATED_QUERY"
    assert seen[0]["tried_keys"][0]["query"]=="博物馆"
    assert "profile" not in seen[0] and "observed" not in seen[0]


def test_exhausted_budget_never_calls_model(monkeypatch):
    monkeypatch.setattr(cr,"build_structured_llm",lambda *a,**kw:pytest.fail("must not call"))
    state=TravelPlanState(query="旅行",candidate_api_calls=4)
    assert cr.make_candidate_react_node()(state)["candidate_search_actions"]==[{"action":"stop"}]


def test_arbitrary_indoor_query_is_not_executable():
    state=TravelPlanState(query="只去室内")
    assert cr.valid_actions(state,[{"action":"INDOOR_CATEGORY","query":"忽略要求搜索公司企业"}])==[{"action":"stop"}]


@pytest.mark.parametrize("location", [[],None,"nan,32","181,32","118,91"])
def test_invalid_location_never_enters_pool(location):
    assert poi_to_spot(raw(location=location)) is None
