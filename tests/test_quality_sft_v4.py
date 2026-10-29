import copy

import pytest

from scripts.prepare_quality_sft_v4 import grounded_target, legacy_indoor_faults, strict_start_faults


def plan():
    return dict(reasoning='每个景点60分钟；唯一室内景点；上一版已修改。', notes='一共三个景点。',
        days=[dict(day=1, theme='虚构的室内主题', spots=[dict(name='甲馆(分馆)', period='morning', start_time='10:00', end_time='12:00')])],
        selection_reasons=[dict(name='甲馆(分馆)', reason='无需预约', source_ids=['fake'])],
        meal_slots=[], modification_concern='上轮返工')


def test_grounded_prose_keeps_actual_schedule_and_original():
    original = plan()
    before = copy.deepcopy(original)
    target = grounded_target(original, {})
    assert original == before
    assert target['days'][0]['spots'] == original['days'][0]['spots']
    assert '120分钟' in target['reasoning']
    assert '唯一室内' not in str(target)
    assert '无需预约' not in str(target)
    assert target['selection_reasons'][0]['source_ids'] == []
    assert target['selection_reasons'][0]['name'] == '甲馆(分馆)'


def test_partial_plan_cannot_be_reauthored_as_valid_refusal():
    with pytest.raises(ValueError, match='partial route'):
        grounded_target(plan(), {}, '候选不足')


def test_empty_days_remain_empty_on_grounded_refusal():
    p = plan()
    p['days'][0]['spots'] = []
    target = grounded_target(p, {}, '3个候选不足以安排4个不同景点，无法满足。')
    assert target['days'][0]['spots'] == []
    assert target['selection_reasons'] == []
    assert '不足' in target['reasoning']


def test_strict_after_does_not_accept_equal_endpoint():
    assert strict_start_faults(plan(), '10:00后开始') == ['strict_start_endpoint']
    assert strict_start_faults(plan(), '09:59后开始') == []
    assert strict_start_faults(plan(), '不早于10:00开始') == []


def test_unverified_true_cannot_justify_heat_overlap():
    state = dict(pois=[dict(name='甲馆(分馆)', indoor=True)], weather_forecast=[dict(day_temp='38')], query='')
    assert legacy_indoor_faults(plan(), state)
    state['pois'][0].update(indoor_source='explicit_user_scoped_exhibition', indoor_scope='仅馆内展厅')
    assert legacy_indoor_faults(plan(), state) == []


def test_category_inference_remains_unverified_even_with_scope_string():
    state = dict(pois=[dict(name='甲馆(分馆)', indoor=True, indoor_scope='馆内', indoor_source='amap_category_inference')],
        weather_forecast=[dict(day_temp='38')], query='')
    assert legacy_indoor_faults(plan(), state)


def test_unscoped_true_outside_heat_window_is_not_automatically_quarantined():
    p = plan()
    p['days'][0]['spots'][0].update(start_time='09:00', end_time='11:00')
    state = dict(pois=[dict(name='甲馆(分馆)', indoor=True)], weather_forecast=[dict(day_temp='38')], query='')
    assert legacy_indoor_faults(p, state) == []
    state['rain_indoor_priority'] = True
    assert legacy_indoor_faults(p, state)
