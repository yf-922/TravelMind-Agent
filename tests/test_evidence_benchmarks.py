import copy
import json

import pytest

from app.evaluation.task_robustness import (build_cases, build_soft_review_cases,
    grade_soft_review, load_assets, semantic_route_violations)
from app.evaluation.checkpoint_benchmark import measure_checkpoint_prompts
from app.planning.graph import restore_modification_state
from app.planning.schemas import TravelPlanState
from scripts.audit_current_risk_gate import audit
from scripts.evaluate_soft_review import preflight, summarize
from scripts.review_soft_review_labels import export_form, apply_confirmation
from scripts.summarize_soft_review import public_report


def make_ledger():
    case = build_cases()['tasks'][0]
    state = {**case['state'], 'route': [{'day': 1, 'spots': [{
        'name': case['state']['pois'][0]['name'], 'period': 'morning',
        'start_time': '10:00', 'end_time': '12:00'}]}]}
    return {'cases': [case], 'seeds': {case['id']: {'state': state}},
            'rows': [{'id': case['id'], 'suite': 'task', 'arm': 'rules_reviewer', 'paid_call_indices': [0]}],
            'calls': [{'task': 'reviewer', 'status': 'completed',
                       'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120}}]}


def test_current_gate_preserves_semantic_review_and_cannot_invent_timecheck_usage():
    report = audit(make_ledger())
    assert report['initial_skip_count'] == 0
    assert report['observed_audit_usage']['total_tokens'] == 120
    assert report['initial_reviewer_token_saving_ratio'] == 0
    assert report['full_current_policy_token_reduction_ratio'] is None
    assert report['full_current_policy_audit_tokens'] is None


def test_missing_seed_does_not_become_zero_risk_or_savings():
    ledger = make_ledger()
    ledger['seeds'] = {}
    result = audit(ledger)
    assert result['initial_call_reduction_ratio'] is None
    assert result['initial_reviewer_token_saving_ratio'] is None


def test_failed_usage_stays_explicitly_incomplete():
    ledger = make_ledger()
    ledger['calls'][0] = {'task': 'reviewer', 'status': 'failed'}
    result = audit(ledger)
    assert not result['observed_usage_complete']
    assert result['initial_reviewer_token_saving_ratio'] is None


def test_controlled_cases_are_rule_clean_and_soft_defective_before_spending():
    cases = build_soft_review_cases()
    _, maps = load_assets()
    result = preflight(cases, maps)
    assert len(result) == 8
    by_id = {row['id']: row for row in result}
    assert all(not [fault for fault in row['hard_violations']
                   if fault not in next(c for c in cases if c['id'] == row['id']).get('expected_hard_violations', [])]
               for row in result)
    assert sum(bool(row['soft_violations']) for row in result) == 7
    assert all(c['label_status'] == 'verified_v2' for c in cases)
    assert sum(bool(c.get('replaces')) for c in cases) == 5
    assert by_id['soft-review-time-deadlock']['hard_violations'] == ['commute_time_conflict']
    assert by_id['soft-review-indoor-pool-conflict']['hard_violations'] == [
        'candidate_pool_preference_conflict', 'indoor_constraint']
    assert any(c['expected'] == 'reject' for c in cases)


@pytest.mark.parametrize('arm', ['rules', 'rules_reviewer'])
@pytest.mark.parametrize('case_id', [
    'soft-review-time-deadlock', 'soft-review-indoor-pool-conflict',
])
def test_new_infeasible_cases_exit_after_three_repairs_with_specific_refusal(case_id, arm):
    from unittest.mock import patch
    from app.evaluation.route_interest import route_key
    from app.planning import nodes
    from scripts.evaluate_reviewer_ablation import run_arm

    cases = build_soft_review_cases()
    case = next(c for c in cases if c['id'] == case_id)
    _, maps = load_assets()
    initial = {**copy.deepcopy(case['state']), 'route': copy.deepcopy(case['injected_route'])}

    def planner(state):
        return {'route': copy.deepcopy(case['injected_route']), 'review_round': state.review_round + 1}

    def reviewer(state):
        # An optimistic Reviewer cannot override deterministic conflicts.
        return {'approved': True, 'route_modify_opinion': '',
                'reviewer_issues': []}

    with patch.object(nodes, 'amap_key', return_value='replay'), \
         patch.object(nodes, 'plan_route_distance', side_effect=lambda a, b, key, mode='drive':
                      copy.deepcopy(maps.get(route_key(a, b, mode)))), \
         patch.object(nodes, 'recommend_chain_hotel', return_value=None), \
         patch.object(nodes, 'lookup_live_ticket_prices', return_value={}):
        trace = []
        state, repairs = run_arm(initial, arm, planner, reviewer, max_repairs=3, trace=trace)

    assert repairs == 3
    assert len(trace) == 4
    assert state.pois == initial['pois']
    assert state.final_plan['approved'] is False
    assert state.final_plan['days'] == []
    assert any(case['required_refusal_phrase'] in message
               for message in state.final_plan['route_issues'])
    grade = grade_soft_review(state, case, maps)
    assert grade['safe_explained_rejection']
    assert grade['expected_behavior_pass']
    assert not grade['task_success']


@pytest.mark.parametrize('case', build_soft_review_cases(), ids=lambda c: c['id'])
def test_semantic_challenge_scoring_never_trusts_optimistic_model_verdict(case):
    _, maps = load_assets()
    state = TravelPlanState(**case['state'])
    state.final_plan = {'approved': True, 'days': [{'day': 1, 'timeline': [
        {'type': 'attraction', **s} for s in case['injected_route'][0]['spots']]}]}
    result = grade_soft_review(state, case, maps)
    if case['category'] != 'clean':
        assert not result['soft_pass']
        assert not result['task_success']


def test_infeasible_refusal_is_not_counted_as_completed_itinerary():
    case = next(c for c in build_soft_review_cases() if c['category'] == 'infeasible')
    state = TravelPlanState(**case['state'])
    state.review_round = 3
    state.final_plan = {'approved': False, 'days': [], 'route_issues': ['当前无法生成满足条件的路线：没有合适候选。']}
    result = grade_soft_review(state, case, {})
    assert result['expected_behavior_pass']
    assert not result['task_success']


def test_offline_infeasible_replay_preserves_unaudited_baseline_and_never_calls_providers(monkeypatch):
    from scripts.evaluate_infeasible_safety import evaluate
    from app.planning import nodes

    def unexpected_call(*args, **kwargs):
        raise AssertionError('live provider access is forbidden')

    monkeypatch.setattr(nodes, 'plan_route_distance', unexpected_call)
    monkeypatch.setattr(nodes, 'invoke_structured', unexpected_call)
    report = evaluate()
    assert report['summary']['rules'] == {'safe_rejections': 2, 'cases': 2}
    assert report['summary']['rules_reviewer'] == {'safe_rejections': 2, 'cases': 2}
    assert report['model_calls'] == report['external_calls'] == 0
    baseline = [r for r in report['rows'] if r['arm'] == 'planner_only']
    assert len(baseline) == 2
    assert all(r['grade']['unsafe_release'] for r in baseline)
    assert all(r['grade']['selected'] for r in baseline)
    assert all(not r['grade']['task_success'] for r in baseline)


def test_repaired_heat_slot_must_change_route_json_not_notes():
    case = next(c for c in build_soft_review_cases() if c['category'] == 'heat')
    route = copy.deepcopy(case['injected_route'])
    route[0]['notes'] = '已经调整到凉爽时段'
    assert semantic_route_violations(route, case, {}) == ['heat_exposure']
    route[0]['spots'][0].update(start_time='09:00', end_time='10:30')
    assert semantic_route_violations(route, case, {}) == []


def test_local_restore_preserves_facts_but_not_old_approval():
    source = next(iter(make_ledger()['seeds'].values()))['state']
    source.update(approved=True, final_plan={'approved': True}, reviewer_issues=['old'])
    restored = restore_modification_state(source, '下午改室内')
    assert restored.pois == source['pois']
    assert restored.weather_forecast == source['weather_forecast']
    assert restored.route == source['route']
    assert not restored.approved
    assert restored.final_plan is None
    assert restored.reviewer_issues == []


def test_checkpoint_count_cannot_be_filled_by_repeating_same_seed():
    with pytest.raises(ValueError, match='distinct checkpoints'):
        measure_checkpoint_prompts(make_ledger(), samples=2)


def test_prompt_tokens_use_encoding_and_can_report_negative_saving():
    ledger = make_ledger()
    state = next(iter(ledger['seeds'].values()))['state']
    state['route'][0]['spots'][0]['notes'] = '额外的历史路线信息' * 3000
    result = measure_checkpoint_prompts(ledger, samples=1, repetitions=1)
    assert result['text_input_tokens']['reduction_ratio'] < 0
    assert result['avoided_front_llm_invocations'] == 8
    assert result['end_to_end_latency_reduction_ratio'] is None


def test_composite_reuse_is_explicit_and_does_not_mutate_legacy():
    legacy = make_ledger()
    before = json.dumps(legacy, sort_keys=True)
    report = {'cases': build_soft_review_cases(), 'rows': []}
    result = summarize(report, legacy)
    assert result['rules_reviewer']['new_recorded'] == 0
    assert result['rules_reviewer']['planned_composite'] == 6
    assert result['rules_reviewer']['new_feasible_count'] == 4
    assert result['rules_reviewer']['new_reject_count'] == 1
    assert json.dumps(legacy, sort_keys=True) == before


def test_human_verification_requires_identity_all_cases_and_bound_content():
    cases = build_soft_review_cases()
    for case in cases:
        case['label_status'] = 'pending_human_review'
    report = {'binding': 'test-binding', 'status': 'complete_pending_human_review',
              'cases': cases}
    form = export_form(report)
    with pytest.raises(ValueError, match='identity'):
        apply_confirmation(report, form)
    form.update(reviewer='test-reviewer', reviewed_at='2026-10-23T12:00:00+08:00')
    for decision in form['decisions']:
        decision['confirm'] = True
    form['decisions'][-1]['case_fingerprint'] = 'tampered'
    with pytest.raises(ValueError, match='changed case'):
        apply_confirmation(report, form)
    assert all(c['label_status'] == 'pending_human_review' for c in report['cases'])


def test_confirmed_copy_records_review_provenance():
    cases = build_soft_review_cases()
    for case in cases:
        case['label_status'] = 'pending_human_review'
    report = {'binding': 'test-binding', 'status': 'complete_pending_human_review',
              'cases': cases}
    form = export_form(report)
    form.update(reviewer='test-reviewer', reviewed_at='2026-10-23T12:00:00+08:00')
    for decision in form['decisions']:
        decision['confirm'] = True
    result = apply_confirmation(report, form)
    assert result['label_status'] == 'verified_v2'
    assert all(c['human_review']['reviewer'] == 'test-reviewer' for c in result['cases'])


def test_public_report_exposes_caught_parse_failure_without_leaking_prompts():
    case = next(c for c in build_soft_review_cases() if c['category'] == 'infeasible')
    report = {'cases': [case], 'rows': [
        {'id': case['id'], 'arm': arm, 'call_indices': [0] if arm == 'rules_reviewer' else [],
         'grade': {'task_success': False, 'safe_explained_rejection': arm == 'rules_reviewer'}}
        for arm in ('planner_only', 'rules', 'rules_reviewer')],
        'calls': [{'task': 'reviewer', 'status': 'failed', 'error': 'ValueError',
                   'messages': ['private prompt'], 'usage': {'input_tokens': 5, 'output_tokens': 7, 'total_tokens': 12}}],
        'status': 'complete_pending_human_review', 'label_status': 'pending_human_review',
        'model': 'test', 'binding': 'test', 'legacy_sha256': 'test',
        'cost_status': 'unpriced', 'max_calls': 42, 'output_token_cap': 4096}
    result = public_report(report, {'rows': []})
    outcome = result['outcomes'][-1]
    assert outcome['execution_failure_observed']
    assert not outcome['semantic_refusal_without_model_failure']
    assert result['usage']['total_tokens'] == 12
    assert result['failed_calls'] == 1
    assert not result['formal_test_report']
    assert 'private prompt' not in json.dumps(result)
    report['rows'].pop()
    with pytest.raises(ValueError, match='one recorded outcome'):
        public_report(report, {'rows': []})
