import copy

from scripts.prepare_v2_large_pool_teacher import clean_input, reason_duration_faults, explanation_count_faults
from scripts.post_train_runtime import capture_messages


def test_fresh_teacher_input_discards_all_historical_answer_channels():
    marker = 'HISTORICAL_ANSWER_MUST_NEVER_REACH_TEACHER'
    raw = dict(query='南京两日游', destination='南京', days=2,
        pois=[dict(name='玄武湖景区', location=dict(lng=118.81, lat=32.07))],
        route=[dict(day=1, spots=[dict(name=marker)])], previous_plan_summary=marker,
        history=[marker], planner_reviewer_dialogue=[marker], agent_private_context=[marker],
        repair_feedback=[dict(text=marker)], route_modify_opinion=marker,
        selection_reasons=[dict(name=marker)], rewritten_query=marker,
        modification_notes=marker, profile_hint=marker, rag_sources=[dict(text=marker)],
        final_plan=dict(text=marker))
    state = clean_input(raw)
    messages = capture_messages(state)
    assert marker not in str(messages)
    assert state['route'] == [] and state['review_round'] == 0
    assert state['previous_plan_summary'] is None
    assert state['query'] == raw['query'] and state['pois'] == raw['pois']
    assert '上一版路线（仅供修改参考）：无' in messages[1]['content']
    assert '本轮修改/审核意见：无' in messages[1]['content']


def plan(reason):
    return dict(days=[dict(day=1, spots=[dict(name='候选甲', start_time='09:00', end_time='10:30')])],
        selection_reasons=[dict(name='候选甲', reason=reason, source_ids=[])])


def test_selection_reason_duration_claim_checked_without_rewriting_target():
    target = plan('候选池中的可选地点，安排60分钟。')
    before = copy.deepcopy(target)
    assert reason_duration_faults(target) == ['selection_reason_duration_mismatch']
    assert target == before
    assert reason_duration_faults(plan('安排90分钟。')) == []


def test_optional_duration_advice_is_not_a_scheduling_claim():
    assert reason_duration_faults(plan('输入没有展览说明；如需深度参观，可自行增加60分钟。')) == []


def test_unsupported_selection_source_and_name_are_rejected():
    target = plan('安排90分钟。')
    target['selection_reasons'][0]['source_ids'] = ['invented']
    assert reason_duration_faults(target) == ['unsupported_source_reference']
    target['selection_reasons'][0]['name'] = '不存在'
    assert reason_duration_faults(target) == ['selection_reason_not_selected']


def test_explanation_day_and_total_count_claims_match_schedule():
    target = plan('安排90分钟。')
    target.update(reasoning='第1天安排2个景点，全程3个景点。')
    assert explanation_count_faults(target) == ['explanation_day_count_mismatch', 'explanation_total_count_mismatch']
    target['reasoning'] = '第1天安排1个景点，全程1个景点。'
    assert explanation_count_faults(target) == []


def test_export_preserves_captured_production_messages_verbatim():
    from scripts.export_llamafactory_dataset import export
    state = clean_input(dict(query='南京1日游', destination='南京', days=1,
        pois=[dict(name='候选甲', open_time='09:00-17:00')]))
    messages = capture_messages(state)
    target = plan('安排90分钟。')
    target.update(reasoning='按候选与时间窗口规划。', notes='', meal_slots=[], modification_concern='')
    target['days'][0]['theme'] = '候选甲参观'
    target['days'][0]['spots'][0]['period'] = 'morning'
    before = copy.deepcopy(messages)
    sft, dpo = export([dict(id='mock-fresh', messages=messages, chosen=target,
        mock=True, status='accepted', rejected=None)])
    assert [c['value'] for c in sft[0]['conversations'][:-1]] == [m['content'] for m in messages]
    assert messages == before and not dpo
