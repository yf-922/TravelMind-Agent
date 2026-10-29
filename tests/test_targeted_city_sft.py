import copy

from scripts.prepare_targeted_city_sft import conservative_day_window, additional_faults, feasible_plan
from scripts.run_hard_post_train_synthesis import audit_plan
from app.evaluation.route_interest import route_key


def fixture_case():
    pois = [{'name': name, 'location': {'lng': 104+i*.01, 'lat': 30},
             'open_time': '09:00-17:00 最晚进入16:00', 'indoor': None}
            for i, name in enumerate(['甲', '乙', '丙', '丁'])]
    case = {'id': 'newcity-test', 'category': 'heat', 'suite': 'task', 'spots_per_day': 2,
            'state': {'destination': '成都', 'days': 2, 'pois': pois, 'main_meal_candidates': [],
                      'query': '模拟38°C高温；全程不重复', 'max_per_day': 2},
            'rubric': {'min_count': 4, 'max_count': 4, 'required': ['甲'], 'excluded': [],
                       'min_visit': 90, 'earliest': 540, 'latest': 1020}}
    maps = {route_key(a['location'], b['location'], 'drive'): {
        'duration_min': 25, 'distance_km': 8, 'mode': 'taxi_or_car', 'estimate': False}
        for a in pois for b in pois if a['name'] != b['name']}
    return case, maps


def test_day_and_night_tickets_do_not_create_impossible_intersection():
    assert conservative_day_window('冬07:30–17:00 最晚进入16:30；夏07:30–17:30；夜游:18:30-21:00') == (450, 1020, 990)
    assert conservative_day_window('08:30-19:00(18:00停止售票)') == (510, 1140, 1080)
    assert conservative_day_window('09:00–17:00(16:00 止入)') == (540, 1020, 960)
    assert conservative_day_window('未知') is None
    assert conservative_day_window('09:00-12:00,13:30-16:30') is None


def test_multiday_witness_respects_unknown_indoor_heat_and_real_roads():
    case, maps = fixture_case()
    plan = feasible_plan(case, maps)
    assert audit_plan(plan, case, maps)[0] == []
    assert len({p['name'] for d in plan['days'] for p in d['spots']}) == 4
    assert [p['start_time'] for p in plan['days'][0]['spots']] == ['09:00', '15:00']


def test_targeted_audit_rejects_duplicate_days_and_unsupported_explanations():
    case, maps = fixture_case()
    plan = feasible_plan(case, maps)
    bad = copy.deepcopy(plan)
    bad['days'][1] = copy.deepcopy(bad['days'][0])
    bad['notes'] = '本轮修正，四个景点步行可达，全程三个景点。'
    faults = additional_faults(bad, case)
    assert {'missing_or_duplicate_day', 'global_duplicate_poi', 'unsupported_explanation_fact',
            'explanation_count_mismatch', 'repair_context_in_round_zero_answer'} <= set(faults)


def test_unverified_opening_and_missing_direction_are_not_passes():
    case, maps = fixture_case()
    plan = feasible_plan(case, maps)
    case['state']['pois'][0]['open_time'] = '未知'
    assert 'heldout_opening_unverified' in additional_faults(plan, case)
    case, maps = fixture_case()
    assert 'road_unverified' in audit_plan(plan, case, {})[0]
