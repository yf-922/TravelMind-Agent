import copy
import json
from pathlib import Path
import pytest

from scripts.prepare_heldout_evidence import normalize_batch, witness, required_edges, opening_window, opening_evidence_faults
from scripts.build_verified_preferences import grounded_answer, build_pair, subtle_variants
from scripts.post_train_runtime import capture_messages
from scripts.synthesize_post_train_data import mock_plan
from scripts.evaluate_reviewer_ablation import cases_v2
from app.evaluation.task_robustness import load_assets
from app.evaluation.route_interest import route_key
from app.planning.schemas import TravelRoute
from collections import Counter


def toy_case():
    a = {'name': 'A', 'location': {'lng': 100., 'lat': 25.}, 'open_time': '10:00-18:00', 'indoor': False}
    b = {'name': 'B', 'location': {'lng': 100.1, 'lat': 25.1}, 'open_time': '10:00-18:00', 'indoor': False}
    case = {'id': 'heldout-test', 'state': {'destination': 'test', 'query': 'two stops', 'days': 1,
        'pois': [a, b], 'main_meal_candidates': []},
        'rubric': {'min_count': 2, 'max_count': 2, 'required': ['A'], 'excluded': [],
                   'earliest': 600, 'latest': 1080, 'min_visit': 90}}
    return case, a, b


def test_roads_require_real_positive_measurements_and_direction():
    case, a, b = toy_case()
    raw = {'status': '1', 'results': [{'origin_id': '1', 'distance': '5000', 'duration': '601', 'info': 'OK'}]}
    routes = normalize_batch(raw, [a], b, 'raw-id')
    key = route_key(a['location'], b['location'], 'drive')
    assert routes[key]['duration_min'] == 11
    assert routes[key]['evidence_id'] == 'raw-id'
    assert route_key(b['location'], a['location'], 'drive') not in routes
    assert len(required_edges([case])) == 2
    for bad in [None, 'nan', '-1', '0', []]:
        invalid = copy.deepcopy(raw)
        invalid['results'][0]['duration'] = bad
        assert not normalize_batch(invalid, [a], b, 'raw-id')
    assert not normalize_batch({'status': '0', **{'results': raw['results']}}, [a], b, 'x')


def test_feasibility_requires_saved_roads_and_checks_buffers_and_opening():
    case, a, b = toy_case()
    assert witness(case, {}) is None
    routes = {route_key(a['location'], b['location'], 'drive'): {'duration_min': 31, 'distance_km': 15}}
    result = witness(case, routes)
    assert result and result['rubric_grade']['task_success']
    spots = result['plan']['days'][0]['spots']
    assert spots[0]['end_time'] == '11:30'
    assert spots[1]['start_time'] == '12:16'  # 31 minutes road plus 15 buffer
    impossible = copy.deepcopy(case)
    impossible['state']['pois'][1]['open_time'] = '10:00-11:00'
    assert witness(impossible, routes) is None
    case['rubric']['excluded'] = ['B']
    assert witness(case, routes) is None


def test_seasonal_opening_uses_conservative_intersection_not_guessed_season():
    assert opening_window({'open_time': '冬季08:30-17:00 最晚进入16:30；夏季08:00-17:30 最晚进入17:00'}) == (510, 1020, 990)
    assert opening_window({'open_time': '09:00-12:00；14:00-17:00'}) is None
    assert opening_window({'open_time': '请电话确认'}) is None


def test_strict_opening_does_not_treat_unknown_or_late_admission_as_pass():
    case, a, b = toy_case()
    plan = {'days': [{'day': 1, 'spots': [{'name': 'A', 'start_time': '16:45', 'end_time': '17:00'}]}]}
    case['state']['pois'][0]['open_time'] = '10:00-18:00 最晚进入16:30'
    assert opening_evidence_faults(plan, case) == ['heldout_last_admission']
    case['state']['pois'][0]['open_time'] = None
    assert opening_evidence_faults(plan, case) == ['heldout_opening_unverified']


def test_subtle_commute_negative_has_five_minute_deficit_and_keeps_visit_duration():
    case, a, b = toy_case()
    maps = {route_key(a['location'], b['location'], 'drive'): {'duration_min': 31, 'distance_km': 15}}
    plan = witness(case, maps)['plan']
    negatives = list(subtle_variants(plan, case['state'], maps))
    negative = next(p for kind,p in negatives if kind == 'commute_deficit_5min')
    assert negative['days'][0]['spots'][1]['start_time'] == '11:56'
    assert negative['days'][0]['spots'][1]['end_time'] == '13:26'
    assert negative['days'][0]['spots'][0] == plan['days'][0]['spots'][0]


def test_heldout_loader_blocks_missing_roads_and_keeps_witness_out_of_prompt(tmp_path):
    from scripts.evaluate_post_train_ablation import load_heldout_suite, prepare
    case, a, b = toy_case()
    maps = {route_key(a['location'], b['location'], 'drive'): {'duration_min': 31, 'distance_km': 15},
            route_key(b['location'], a['location'], 'drive'): {'duration_min': 45, 'distance_km': 20}}
    case.update(category='heldout_city_multi_constraint', training_use='forbidden',
        readiness='offline_replay_ready_not_realworld_certified', feasibility_evidence=witness(case, maps))
    cases_path, maps_path = tmp_path/'cases.json', tmp_path/'maps.json'
    cases_path.write_text(json.dumps([case]), encoding='utf-8')
    maps_path.write_text(json.dumps({'routes': maps}), encoding='utf-8')
    loaded, _ = load_heldout_suite(cases_path, maps_path)
    assert prepare(loaded[0]) == case['state']
    assert '离线可行性证据' not in json.dumps(capture_messages(prepare(loaded[0])), ensure_ascii=False)
    maps.pop(route_key(b['location'], a['location'], 'drive'))
    maps_path.write_text(json.dumps({'routes': maps}), encoding='utf-8')
    with pytest.raises(ValueError, match='incomplete'):
        load_heldout_suite(cases_path, maps_path)


def test_both_sides_have_route_grounded_explanations_without_teacher_claims():
    case = cases_v2()[0]
    original = mock_plan(case['state'])
    original['reasoning'] = 'All constraints are satisfied, previous version fixed.'
    before = copy.deepcopy(original)
    chosen = grounded_answer(original, case['state'])
    rejected = copy.deepcopy(original)
    spot = rejected['days'][0]['spots'][0]
    spot['start_time'] = '06:00'
    negative = grounded_answer(rejected, case['state'])
    assert original == before
    assert 'previous version' not in chosen['reasoning']
    assert '06:00' in negative['reasoning']
    assert chosen['notes'] == negative['notes']
    assert chosen['selection_reasons'] == negative['selection_reasons'] == []
    TravelRoute.model_validate(chosen)
    TravelRoute.model_validate(negative)


def test_unverified_chosen_or_changed_messages_cannot_be_preferences():
    case = cases_v2()[0]
    row = {'id': 'test', 'state': case['state'], 'messages': [], 'chosen': mock_plan(case['state'])}
    _, maps = load_assets()
    pair, reason = build_pair(row, maps, {}, Counter())
    assert pair is None and reason == 'message_mismatch'
    row['messages'] = capture_messages(row['state'])
    row['chosen']['days'][0]['spots'][0]['name'] = 'outside-pool'
    pair, reason = build_pair(row, maps, {}, Counter())
    assert pair is None and reason == 'chosen_failed_recheck'
