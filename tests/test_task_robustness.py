import copy
from collections import Counter

import pytest

from app.evaluation.task_robustness import build_cases, grade, load_assets, offline_probe
from app.planning.schemas import TravelPlanState
from scripts.evaluate_task_robustness import summary

ASSETS = build_cases()


def test_manifest_has_thirty_per_suite_and_ten_per_failure_family():
    assert len(ASSETS['tasks']) == len(ASSETS['robustness']) == 30
    assert Counter(c['category'] for c in ASSETS['robustness']) == {'map': 10, 'weather': 10, 'modification': 10}
    assert Counter(c['state']['destination'] for c in ASSETS['tasks']) == {'南京': 10, '上海': 10, '三亚': 10}
    ids = [c['id'] for c in ASSETS['tasks'] + ASSETS['robustness']]
    assert len(set(ids)) == 60
    assert all(len(c['state']['pois']) >= 4 for c in ASSETS['tasks'])


@pytest.mark.parametrize('case', ASSETS['tasks'] + ASSETS['robustness'], ids=lambda c: c['id'])
def test_sixty_offline_provider_and_gate_probes(case):
    result = offline_probe(case)
    assert result['node_execution_ok']
    assert not result['model_executed']
    assert result['task_success'] is None
    if case['category'] == 'map' and case['fault'] in ('timeout', 'rate_limit', 'exception', 'empty', 'all_missing', 'walking_unverified'):
        assert result['road_legs'] == []
        assert '失败' in result['distance_note']
    if case['category'] == 'map' and case['fault'] == 'walking_unverified':
        assert 'walking_constraint' in result['gate']['route_risk_flags']
    if case['category'] == 'modification':
        assert result['gate']['review_required']
        assert 'user_modification' in result['gate']['route_risk_flags']


def delivered(case, names=None):
    state = TravelPlanState(**case['state'])
    names = names or [state.pois[0]['name']]
    state.approved = True
    state.route = [{'day': 1, 'spots': [{'name': n, 'period': 'morning', 'start_time': '10:00', 'end_time': '12:00'} for n in names]}]
    state.final_plan = {'approved': True, 'days': [{'day': 1, 'timeline': [{'type': 'attraction', **s} for s in state.route[0]['spots']]}], 'route_issues': []}
    return state


def test_soft_rubric_can_fail_model_approved_route():
    case = copy.deepcopy(ASSETS['tasks'][0])
    state = delivered(case)
    case['rubric'] = {'min_count': 1, 'min_visit': 180}
    result = grade(state, case, {})
    assert result['hard_pass']
    assert not result['soft_pass']
    assert not result['task_success']
    assert result['unsafe_release']


@pytest.mark.parametrize('rubric,violation', [
    ({'required': ['missing']}, 'required_poi_missing'),
    ({'excluded': ['南京博物院']}, 'excluded_poi'),
    ({'earliest': 660}, 'requested_window'),
    ({'min_count': 2}, 'required_count'),
])
def test_model_approval_does_not_override_original_request(rubric, violation):
    case = copy.deepcopy(ASSETS['tasks'][0])
    state = delivered(case)
    case['rubric'] = rubric
    result = grade(state, case, {})
    assert violation in result['hard_violations']
    assert not result['task_success']


def test_missing_road_evidence_not_rescued_by_healthy_replay():
    case = ASSETS['tasks'][0]
    state = delivered(case, [p['name'] for p in case['state']['pois'][:2]])
    _, maps = load_assets()
    result = grade(state, case, maps)
    assert 'road_unverified' in result['hard_violations']
    assert not result['task_success']


def test_rejection_is_not_task_completion_and_missing_reason_is_not_safe_rejection():
    case = ASSETS['robustness'][-1]
    state = delivered(case)
    state.final_plan['approved'] = False
    assert not grade(state, case, {})['robustness_ok']
    state.final_plan['route_issues'] = ['当前无法生成满足全部要求的路线：候选不足。']
    result = grade(state, case, {})
    assert result['robustness_ok'] and result['safe_explained_rejection']
    assert not result['task_success']


def test_failed_and_not_run_requests_stay_in_denominator():
    cases = ASSETS['tasks']
    rows = [{'id': cases[0]['id'], 'suite': 'task', 'grade': {'task_success': True}},
            {'id': cases[1]['id'], 'suite': 'task', 'error': 'TimeoutError'}]
    result = summary(cases, rows, True)['task']
    assert result['task_success_rate'] == 1 / 30
    assert result['errors'] == 1 and result['not_run'] == 28
    assert summary(cases, [], False)['task']['task_success_rate'] is None
