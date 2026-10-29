import copy
import json
from pathlib import Path
import pytest

from app.evaluation.task_robustness import load_assets
from scripts.evaluate_sft_validation import score, compare
from scripts.evaluate_reviewer_ablation import cases_v2
from scripts.synthesize_post_train_data import mock_plan


def test_empty_route_does_not_pass_development_gate():
    case = cases_v2()[0]
    plan = mock_plan(case['state'])
    for day in plan['days']:
        day['spots'] = []
    _, maps = load_assets()
    assert 'empty_route' in score(plan, {'id': 'example', 'state': case['state']}, maps, {})


def test_validation_respects_hard_rubric_instead_of_just_json():
    case = cases_v2()[0]
    plan = mock_plan(case['state'])
    required = 'missing-required-stop'
    hard = {'id': 'hard-test', 'state': case['state'], 'rubric': {'required': [required], 'min_count': 1}}
    _, maps = load_assets()
    faults = score(plan, {'id': 'hard-test', 'state': case['state']}, maps, {'hard-test': hard})
    assert 'required_poi_missing' in faults


def test_paired_comparison_counts_failures_and_regressions(tmp_path):
    base = [{'id': str(i), 'passed': i < 10} for i in range(30)]
    sft = [{'id': str(i), 'passed': 1 <= i <= 15} for i in range(30)]
    for name, rows in [('base', base), ('sft', sft)]:
        (tmp_path/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    result = compare(tmp_path)
    assert result['base_passed'] == 10 and result['sft_passed'] == 15
    assert len(result['improved']) == 6 and result['regressed'] == ['0']
    assert result['dpo_auto_start'] is False
    (tmp_path/'sft.jsonl').write_text(json.dumps(sft[0])+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Incomplete'):
        compare(tmp_path)
