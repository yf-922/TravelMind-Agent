"""Weekday closure is deterministic L1, including restored snapshot output."""
import copy
import json
from datetime import date

import pytest

from app.planning import nodes
from app.planning.helpers import closed_weekdays, weekday_closure_violations
from app.planning.quality import append_route_snapshot, choose_snapshot, hard_constraint_report, result_envelope
from app.planning.schemas import TravelPlanState, TravelRoute


MUSEUM = "重庆中国三峡博物馆"
OPENING = (
    "1/1-12/31 周二–周日 09:00–17:00开放 最晚进入16:30；"
    "1/1-12/31 周一 全天不开放；元旦节,春节,清明节,劳动节,端午节,"
    "中秋节,国庆节 09:00–18:00开放 最晚进入17:30"
)


def day(number=1, name=MUSEUM, **extra):
    return {"day": number, "theme": "文化游览", "spots": [{"name": name, "period": "morning",
        "start_time": "10:00", "end_time": "11:00"}], **extra}


def state(**updates):
    values = dict(query=f'重庆一日游，必须去“{MUSEUM}”', destination="重庆", days=1,
        travel_start_date=date(2026, 11, 2),
        pois=[{"name": MUSEUM, "aliases": ["三峡博物馆"], "open_time": OPENING, "indoor": True}],
        route=[day()], approved=True, review_round=1,
        reviewer_preference_match_level="HIGH", reviewer_diversity_and_dining_level="GOOD")
    return TravelPlanState(**(values | updates))


@pytest.mark.parametrize("text,expected", [
    ("周一闭馆", {0}), ("每周一不开放", {0}), ("星期一闭馆", {0}),
    ("周一 全天不开放", {0}), ("礼拜天休息", {6}), ("周7不对外开放", {6}),
    ("每周一、三闭馆", {0, 2}), ("周一和周三休馆", {0, 2}),
    ("周一至周三闭馆", {0, 1, 2}), ("周六–周一闭园", {5, 6, 0}),
    ("闭馆日：周一", {0}), ("休馆时间为星期二", {1}),
    ("周一（法定节假日除外）闭馆", {0}), (OPENING, {0}),
    ("周一至周五开放", set()), ("周二至周日09:00-17:00开放", set()),
    ("周一不闭馆", set()), ("周一正常开放，临时闭馆另行通知", set()),
    ("并非周一闭馆", set()), ("不再周一闭馆", set()),
    ("每日开放，节假日闭馆", set()), ("", set()),
])
def test_explicit_closure_parser(text, expected):
    assert closed_weekdays(text) == expected


@pytest.mark.parametrize("route,start,closed", [
    ([day()], "2026-11-02", True),  # Monday
    ([day()], date(2026, 11, 3), False),
    ([day(2)], date(2026, 11, 1), True),  # day number offset
    ([day(date="2026-11-03")], date(2026, 11, 2), False),  # actual date wins
    ([day(date="2026-11-02")], None, True),
    ([day(weekday="星期一")], None, True),
    ([day(weekday="周二")], None, False),
    ([day()], None, False), ([day()], "invalid", False),
])
def test_dates_and_explicit_weekday(route, start, closed):
    findings = weekday_closure_violations(route, state().pois, start)
    assert bool(findings) is closed
    if closed:
        assert findings[0]["code"] == "closed_on_visit_day"
        assert "建议调整游览日期或行前核实" in findings[0]["message"]


def test_business_hours_alias_and_long_excerpt():
    pois = [{"name": MUSEUM, "aliases": ["三峡博物馆"], "open_time": "日常开放信息；" * 40,
             "business_hours": "周一闭馆"}]
    finding, = weekday_closure_violations([day(name="三峡博物馆")], pois, "2026-11-02")
    assert "三峡博物馆" in finding["message"] and "周一闭馆" in finding["message"]
    assert len(finding["message"]) < 260


def test_closure_is_l1_gate_requires_time_check_and_cannot_early_stop():
    s = state()
    report = hard_constraint_report(s)
    assert not report["l0"]
    assert [v["code"] for v in report["l1"]] == ["closed_on_visit_day"]
    assert "2026-11-02 周一" in report["l1"][0]["message"]
    update = nodes.route_risk_gate_node(s)
    assert update["time_check_required"] and not update["review_skipped"]
    assert "opening_time_conflict" in update["route_risk_flags"]
    s = s.model_copy(update=append_route_snapshot(s))
    assert nodes.route_after_review(s) == nodes.route_after_time_check(s) == "planner"
    assert choose_snapshot(s)[1] == "PARTIAL_SUCCESS"


@pytest.mark.parametrize("days", [1, 2])
def test_production_planner_gets_closure_keep_or_move_contract(days):
    s = state(days=days)
    captured = []
    def invoke(_llm, messages):
        captured.extend(messages)
        return TravelRoute(reasoning="保留必去并披露风险" if days == 1 else "改到周二",
            days=[day()] if days == 1 else [day(name="磁器口古镇"), day(2)],
            notes="周一闭馆，建议改期或核实" if days == 1 else "三峡博物馆改为周二游览")
    s.pois.append({"name": "磁器口古镇", "indoor": False})
    update = nodes.make_joint_planner_node(None, llm=object(), invoke_fn=invoke,
                                         knowledge_lookup=lambda *_: [])(s)
    prompt = str(captured)
    assert "2026-11-02" in prompt and "周一" in prompt
    assert "多日行程应优先将必去景点调换" in prompt
    assert "仍然必须按正常游览顺序将其排入当日行程" in prompt
    assert "不得生成空路线、空白行程日或丢弃该必去景点" in prompt
    assert "不代表可以入馆" in prompt
    report = hard_constraint_report(s.model_copy(update=update))
    assert not report["l0"]
    assert any(v["code"] == "closed_on_visit_day" for v in report["l1"]) is (days == 1)


def test_restored_clean_snapshot_rechecked_without_mutating_original():
    s = state()
    s = s.model_copy(update=append_route_snapshot(s))
    s.route_history[0]["l1_violations"] = []  # old ledger predates weekday checks
    original = copy.deepcopy(s.route_history)
    selected, status, warnings, failures = choose_snapshot(s)
    assert status == "PARTIAL_SUCCESS" and not failures
    assert selected["l1_violations"][0]["code"] == "closed_on_visit_day"
    assert any("周一 全天不开放" in message for message in warnings)
    assert s.route_history == original


def test_chosen_historical_closure_survives_finalizer_envelope_and_json(monkeypatch):
    # All enrichment stays offline; the production finalizer/envelope still run.
    monkeypatch.setattr(nodes, "amap_key", lambda: None)
    monkeypatch.setattr(nodes, "lookup_live_ticket_prices", lambda *_: {})
    s = state(reviewer_issues=["闭馆请改期核实"])
    s = s.model_copy(update=append_route_snapshot(s))
    s.route_history[0]["soft_score"] = 95
    s.route_history[0]["l1_violations"] = []  # compatible with the saved live ledger
    s.review_round, s.route = 2, [day(name="磁器口古镇")]
    s.pois.append({"name": "磁器口古镇"})
    s = s.model_copy(update=append_route_snapshot(s))  # latest drops the required museum -> L0
    s.reviewer_issues = ["最新一轮无关建议"]
    s = s.model_copy(update=nodes.finalize_node(s))
    result = result_envelope(s)
    assert result["status"] == "PARTIAL_SUCCESS"
    assert result["score_info"]["selected_iteration"] == 1
    assert result["plan"]["days"][0]["timeline"][0]["name"] == MUSEUM
    closure, = [x for x in result["warnings"] if "景点可能闭馆" in x]
    assert "周一 全天不开放" in closure
    assert closure in result["plan"]["warnings"] and closure in result["plan"]["route_issues"]
    assert "闭馆请改期核实" in result["warnings"]
    assert not result["failure_reasons"]
    json.dumps(result, ensure_ascii=False)  # SSE payload safety


def test_open_day_snapshot_wins_and_stale_closure_does_not_leak():
    s = state(days=2, route=[day(name="磁器口古镇"), day(2)])
    s.pois.append({"name": "磁器口古镇"})
    s = s.model_copy(update=append_route_snapshot(s))
    s.route_history[0]["l1_violations"].append({"code": "closed_on_visit_day", "message": "stale"})
    chosen, status, warnings, _ = choose_snapshot(s)
    assert status == "SUCCESS" and not chosen["l1_violations"] and not warnings


def test_envelope_merges_deterministic_closure_with_existing_warning():
    s = state(final_plan={"days": [{"day": 1}], "route_issues": []}, warnings=["已有提醒"])
    s = s.model_copy(update=append_route_snapshot(s))
    result = result_envelope(s)
    assert result["status"] == "PARTIAL_SUCCESS"
    assert "已有提醒" in result["warnings"]
    assert any("景点可能闭馆" in message for message in result["warnings"])


def test_clock_time_and_weekday_are_both_l1():
    s = state(route=[day()])
    s.route[0]["spots"][0].update(start_time="18:30", end_time="19:30")
    report = hard_constraint_report(s)
    assert not report["l0"]
    assert {v["code"] for v in report["l1"]} == {"opening_time_conflict", "closed_on_visit_day"}
