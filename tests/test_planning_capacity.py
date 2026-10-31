"""Daily capacity is a shared contract; overflow never buys off other defects."""
import json

import pytest

from app.planning.constraints import (
    active_must_visit, capacity_policy, capacity_prompt, daily_limit,
    effective_daily_limit,
)
from app.planning.graph import restore_modification_state
from app.planning.quality import (
    append_route_snapshot, choose_snapshot, compute_soft_score, hard_constraint_report,
)
from app.planning.schemas import TravelPlanState, TravelRoute


NAMES = ["磁器口古镇", "洪崖洞", "人民解放纪念碑", "南山一棵树观景台",
         "重庆中国三峡博物馆", "重庆动物园", "重庆科技馆"]


def day(names, number=1):
    return {"day": number, "spots": [
        {"name": name, "start_time": f"{9+i:02}:00", "end_time": f"{9+i:02}:30"}
        for i, name in enumerate(names)]}


def state(count=5, arranged=None, **kwargs):
    query = "重庆旅行，必须去" + "、".join(f'“{n}”' for n in NAMES[:count])
    return TravelPlanState(query=query, destination="重庆", days=1, max_per_day=3,
        pois=[{"name": n} for n in NAMES],
        route=[day(NAMES[:count] if arranged is None else arranged)], **kwargs)


def test_default_expands_five_named_stops_across_prompt_checks_and_score():
    s = state(review_round=1, approved=True)
    assert effective_daily_limit(s) == 5
    assert "每日有效景点上限=5" in capacity_prompt(s)
    assert hard_constraint_report(s) == {"l0": [], "l1": []}
    score, dims = compute_soft_score(s)
    assert dims["named_poi"] == 20 and dims["pace"] == 15
    assert score < 100  # Crowding is visible rather than hidden by expansion.
    s = s.model_copy(update=append_route_snapshot(s))
    snap, status, warnings, failures = choose_snapshot(s)
    assert snap["capacity_policy"]["expanded"]
    assert status == "SUCCESS_WITH_WARNINGS" and not failures
    assert any("行程较紧凑" in w for w in warnings)
    json.dumps(s.model_dump(mode="json"), ensure_ascii=False)


def test_day_specific_counts_or_category_quotas_are_not_global_ceiling():
    assert daily_limit("每天最多2个景点，第一天恰好1个、第二天恰好2个") == 2
    assert daily_limit("每天最多2个室内场馆") is None


@pytest.mark.parametrize("wording", ["每天最多去三个景点", "每日不超过3处景点", "一天最多安排3个景点", "每天最多去3个"])
def test_explicit_limit_is_strict_and_saturated_overflow_is_warning(wording):
    s = state(arranged=NAMES[:3])
    s.query += "。" + wording
    assert daily_limit(s.query) == effective_daily_limit(s) == 3
    assert capacity_policy(s)["overflow_exempt"]
    assert not hard_constraint_report(s)["l0"]
    _, status, warnings, _ = choose_snapshot(s)
    assert status == "SUCCESS_WITH_WARNINGS"
    assert any(NAMES[3] in w and NAMES[4] in w and "备选" in w for w in warnings)
    assert compute_soft_score(s)[1]["named_poi"] == 12  # 3/5, not 5/5.


def test_server_strict_flag_disables_expansion():
    s = state(arranged=NAMES[:3], max_per_day_strict=True)
    assert capacity_policy(s)["source"] == "strict_config"
    assert effective_daily_limit(s) == 3
    assert not hard_constraint_report(s)["l0"]


@pytest.mark.parametrize("arranged", [NAMES[:2], NAMES[:2] + [NAMES[5]],
                                     [NAMES[0], NAMES[0], NAMES[1]]])
def test_underfill_filler_or_duplicate_cannot_satisfy_mandatory_capacity(arranged):
    s = state(arranged=arranged, max_per_day_strict=True)
    assert not capacity_policy(s)["overflow_exempt"]
    assert any(v["code"] == "missing_must_visit" for v in hard_constraint_report(s)["l0"])


def test_more_than_five_per_day_cannot_expand_and_needs_full_capacity():
    s = state(count=6, arranged=NAMES[:3])
    assert not capacity_policy(s)["expanded"]
    assert effective_daily_limit(s) == 3
    assert capacity_policy(s)["overflow_exempt"]
    s.route = [day(NAMES[:2])]
    assert not capacity_policy(s)["overflow_exempt"]


def test_multiday_uses_ceil_and_keeps_per_day_l1():
    s = state(count=7)
    s.days = 2
    s.route = [day(NAMES[:4]), day(NAMES[4:], 2)]
    assert effective_daily_limit(s) == 4
    assert not hard_constraint_report(s)["l0"]
    assert not hard_constraint_report(s)["l1"]
    s.max_per_day_strict = True  # capacity 6; all seven names still over daily max
    assert capacity_policy(s)["overflow_exempt"]
    assert any(v["code"] == "daily_capacity_exceeded" for v in hard_constraint_report(s)["l1"])


def test_expansion_does_not_exempt_missing_mandatory_names():
    s = state(arranged=NAMES[:3])
    assert capacity_policy(s)["expanded"]
    assert len(hard_constraint_report(s)["l0"]) == 2


def test_aliases_exclusions_and_relaxed_names_do_not_inflate_capacity():
    s = state()
    s.pois[2]["aliases"] = ["解放碑"]
    s.query += '、“解放碑”。不要“重庆动物园”'
    s.relaxed_constraints = {"dropped_must_visit": [NAMES[0]], "unverified_must_visit": [NAMES[1]]}
    assert active_must_visit(s) == NAMES[2:5]
    assert effective_daily_limit(s) == 3
    s.route = [day(["解放碑", *NAMES[3:5]])]
    assert capacity_policy(s)["arranged_names"] == NAMES[2:5]
    assert compute_soft_score(s)[1]["named_poi"] == 20


def test_verified_alias_shared_by_parent_and_viewpoint_can_match_arranged_viewpoint():
    s = state()
    s.pois[3] = {"name": "南山一棵树景区", "aliases": [NAMES[3]]}
    s.pois.append({"name": "南山一棵树景区-观景台", "aliases": [NAMES[3]]})
    s.route[0]["spots"][3]["name"] = "南山一棵树景区-观景台"
    assert capacity_policy(s)["arranged_names"] == NAMES[:5]
    assert not hard_constraint_report(s)["l0"]
    assert compute_soft_score(s)[1]["named_poi"] == 20
    s.pois.reverse()
    assert capacity_policy(s)["arranged_names"] == NAMES[:5]


def test_exact_official_requirement_cannot_be_covered_by_another_candidates_alias():
    s = state()
    s.pois.append({"name": "同园另一处观景台", "aliases": [NAMES[3]]})
    s.route[0]["spots"][3]["name"] = "同园另一处观景台"
    assert capacity_policy(s)["omitted_names"] == [NAMES[3]]
    assert any(v["code"] == "missing_must_visit" for v in hard_constraint_report(s)["l0"])


def test_shared_aliases_cannot_count_one_arranged_identity_twice():
    s = state(arranged=NAMES[:2], max_per_day_strict=True)
    # Two distinct requested names can match one candidate, but that is one
    # actual stop and cannot fill two mandatory slots for overflow exemption.
    s.pois[1]["aliases"] = [NAMES[2], NAMES[3]]
    assert not capacity_policy(s)["overflow_exempt"]
    assert len(capacity_policy(s)["arranged_names"]) == 2


def test_overflow_does_not_exempt_unknown_spots_or_time_conflicts():
    s = state(arranged=NAMES[:3], max_per_day_strict=True, max_duration_hours=1)
    s.route[0]["spots"].append({"name": "虚构景点", "start_time": "13:00", "end_time": "13:30"})
    report = hard_constraint_report(s)
    assert capacity_policy(s)["overflow_exempt"]
    assert any(v["code"] == "unknown_poi" for v in report["l0"])
    assert not any(v["code"] == "missing_must_visit" for v in report["l0"])
    assert {v["code"] for v in report["l1"]} >= {"time_window_exceeded", "daily_capacity_exceeded"}


def test_warnings_describe_selected_historical_route_not_latest():
    s = state(arranged=NAMES[:3], max_per_day_strict=True, approved=True, review_round=1)
    s = s.model_copy(update=append_route_snapshot(s))
    s.route, s.review_round = [day(NAMES[2:5])], 2
    s = s.model_copy(update=append_route_snapshot(s))
    s.route_history[0]["soft_score"] = 99
    chosen, _, warnings, _ = choose_snapshot(s)
    assert chosen["iteration"] == 1
    assert "已优先安排 " + "、".join(NAMES[:3]) in warnings[0]
    assert "未能排入的 " + "、".join(NAMES[3:5]) in warnings[0]


def test_checkpoint_preserves_source_but_recomputes_effective_limit():
    s = state(arranged=NAMES[:3], max_per_day_strict=True)
    restored = restore_modification_state(s.model_dump(mode="json"), "调整出发时间")
    assert restored.max_per_day_strict and effective_daily_limit(restored) == 3
    s.max_per_day_strict = False
    restored = restore_modification_state(s.model_dump(mode="json"), "每天最多去两个景点")
    assert effective_daily_limit(restored) == 2
    restored.effective_max_per_day = 99
    assert effective_daily_limit(restored) == 2


def test_production_joint_planner_receives_expanded_limit_and_persists_it():
    from app.planning.nodes import make_joint_planner_node
    s = state()
    captured = []
    def invoke(_llm, messages):
        captured.extend(messages)
        return TravelRoute(reasoning="按容量安排全部点名", days=[{
            **s.route[0], "theme": "重庆地标", "spots": [
                {**p, "period": "morning"} for p in s.route[0]["spots"]]}])
    update = make_joint_planner_node(None, llm=object(), invoke_fn=invoke,
                                   knowledge_lookup=lambda *_: [])(s)
    assert update["effective_max_per_day"] == 5
    assert "每天景点上限：5" in str(captured)
    assert "必须安排全部有效点名景点" in str(captured)
    assert len(update["route"][0]["spots"]) == 5


def test_supervisor_role_contracts_keep_capacity_provenance():
    from app.multi_agent_core.runtime import INPUTS, OUTPUTS
    for role in ("planner", "reviewer", "risk_gate", "time_check"):
        assert {"query", "max_per_day_strict", "relaxed_constraints"} <= INPUTS[role]
    assert "effective_max_per_day" in OUTPUTS["planner"]
