import copy
import json
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.synthesize_post_train_data import mutate_state, inject_plan_defects, mock_plan
from scripts.post_train_runtime import capture_messages, replay_check, quality_faults, Meter
from scripts.export_llamafactory_dataset import export
from scripts.evaluate_reviewer_ablation import cases_v2
from app.evaluation.task_robustness import load_assets
from app.planning import nodes
from app.planning.schemas import TravelPlanState, TravelRoute


def test_capture_matches_production_boundary():
    state = TravelPlanState(**cases_v2()[0]['state'])
    actual = []
    def invoke(llm, messages):
        actual.extend(messages)
        return TravelRoute.model_validate(mock_plan(state.model_dump()))
    nodes.make_joint_planner_node(None, llm=object(), invoke_fn=invoke, knowledge_lookup=lambda *a: [])(state)
    captured = capture_messages(state.model_dump())
    assert captured == [{'role': 'user' if r == 'human' else r, 'content': t} for r, t in actual]


def test_injected_negative_is_schema_valid_and_actually_fails():
    initial = cases_v2()[0]['state']
    plan = mock_plan(initial)
    _, maps = load_assets()
    for kind in (0, 1):
        negative = inject_plan_defects(plan, initial, kind)
        TravelRoute.model_validate(negative)
        errors, _ = replay_check(negative, initial, maps)
        assert errors
    assert plan['days'][0]['spots'][0]['name'] != '__候选池外虚构景点__'


def test_export_exact_messages_pair_shape_and_dedup():
    state = cases_v2()[0]['state']
    plan = mock_plan(state)
    row = {'status': 'accepted', 'mock': False, 'messages': capture_messages(state), 'chosen': plan,
        'rejected': inject_plan_defects(plan, state), 'rejected_rule_errors': ['unknown_poi']}
    sft, dpo = export([row, copy.deepcopy(row)])
    assert len(sft) == len(dpo) == 1
    assert [m['value'] for m in sft[0]['conversations'][:-1]] == [m['content'] for m in row['messages']]
    assert dpo[0]['chosen']['from'] == dpo[0]['rejected']['from'] == 'gpt'


def test_no_live_maps_and_unverified_roads_fail_closed():
    initial = cases_v2()[0]['state']
    plan = mock_plan(initial)
    second = copy.deepcopy(plan['days'][0]['spots'][0])
    second.update(name=initial['pois'][1]['name'], start_time='14:00', end_time='16:00', period='afternoon')
    plan['days'][0]['spots'].append(second)
    with patch.object(nodes, 'plan_route_distance', side_effect=AssertionError('live map forbidden')):
        errors, _ = replay_check(plan, initial, {})
    assert 'road_unverified' in errors


def test_heat_label_unknown_not_assumed_safe():
    state = TravelPlanState(**cases_v2()[0]['state'])
    state.query = '38°C高温，正午避免室外暴晒'
    state.route = mock_plan(state.model_dump())['days']
    state.route[0]['spots'][0].update(start_time='12:00', end_time='14:00')
    state.pois[0]['indoor'] = None
    assert 'heat_exposure_or_unknown_indoor' in quality_faults(state)


def test_all_800_mutations_handle_small_or_empty_candidate_pools():
    rng = random.Random(7)
    cases = cases_v2()
    for i in range(800):
        state = mutate_state(cases[i % len(cases)]['state'], i, rng)
        assert 1 <= state['days'] <= 3
        assert len(state['pois']) <= len(cases[i % len(cases)]['state']['pois'])


def test_mock_export_and_resume(tmp_path):
    root = Path(__file__).parents[1]
    ledger = tmp_path/'ledger.jsonl'
    cmd = [sys.executable, str(root/'scripts/synthesize_post_train_data.py'), '--mock', '--samples', '4', '--output', str(ledger)]
    subprocess.run(cmd, check=True)
    before = ledger.read_bytes()
    subprocess.run(cmd, check=True)
    assert ledger.read_bytes() == before
    subprocess.run([sys.executable, str(root/'scripts/export_llamafactory_dataset.py'), '--input', str(ledger), '--output-dir', str(tmp_path/'export'), '--dry-run'], check=True)
    assert json.loads((tmp_path/'export/travelmind_planner_sft.json').read_text(encoding='utf-8'))


def test_budget_reserves_before_request_and_has_no_hidden_retry(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_API_KEY', 'test-only')
    meter = Meter(tmp_path/'calls.jsonl', 'test', max_calls=0)
    with pytest.raises(RuntimeError, match='MAX_LLM_CALLS'):
        meter.call([], {})
    assert meter.count == 0
    assert meter.client.max_retries == 0


def test_curation_rejects_unversioned_mock_and_verifies_real_negatives():
    from scripts.curate_post_train_data import curate
    state = cases_v2()[0]['state']
    chosen = mock_plan(state)
    row = {'id': 'fixture', 'mock': False, 'status': 'accepted', 'state': state,
        'chosen': chosen, 'messages': capture_messages(state), 'preference_source': 'injected_verified'}
    _, maps = load_assets()
    legacy = copy.deepcopy(row)
    legacy.pop('mock')
    rows, _, _ = curate([legacy, row, copy.deepcopy(row)], maps)
    assert len(rows) == 1
    assert rows[0]['human_review_status'] == 'pending'
    assert rows[0]['rejected_rule_errors']
    TravelRoute.model_validate(rows[0]['rejected'])
    assert rows[0]['preference_source'] != 'injected_verified'


def test_hard_audit_enforces_rubric_beyond_production_rules():
    from scripts.run_hard_post_train_synthesis import audit_plan
    state = cases_v2()[0]['state']
    plan = mock_plan(state)
    _, maps = load_assets()
    case = {'state': state, 'rubric': {'min_count': 2}, 'category': 'multi_constraint', 'suite': 'task'}
    errors, score, _ = audit_plan(plan, case, maps)
    assert 'required_count' in errors
    assert not score['task_success']
