"""Prevent prompt experiments from silently repairing or mis-scoring outputs."""
import copy
import json

from scripts.compare_planner_prompt_candidates import PROMPTS, SUITE, score_output, strict_schema


def fixture():
    cases = json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))
    case = next(c for c in cases if c['expected']=='feasible')
    maps = json.loads((SUITE/'maps.json').read_text(encoding='utf-8'))['routes']
    return case, maps


def test_witness_scores_without_repair():
    case, maps = fixture()
    result = score_output(json.dumps(case['feasibility_witness']), case, maps)
    assert result['parse_success'] and result['passed']


def test_invalid_clock_is_failure_with_plan_preserved():
    case, maps = fixture()
    plan = copy.deepcopy(case['feasibility_witness'])
    plan['days'][0]['spots'][0]['start_time'] = '09:60'
    result = score_output(json.dumps(plan), case, maps)
    assert not result['passed']
    assert result['plan']['days'][0]['spots'][0]['start_time']=='09:60'


def test_wrong_period_is_not_normalized():
    case, maps = fixture()
    plan = copy.deepcopy(case['feasibility_witness'])
    plan['days'][0]['spots'][0]['period'] = '09:00-10:00'
    result = score_output(json.dumps(plan), case, maps)
    assert not result['passed']
    assert 'route_structure' in result['faults']


def test_json_fence_is_not_silently_stripped():
    case, maps = fixture()
    result = score_output('```json\n'+json.dumps(case['feasibility_witness'])+'\n```',case,maps)
    assert not result['parse_success']


def test_schema_is_fresh_and_candidates_are_distinct():
    schema = strict_schema()
    schema['properties'].clear()
    assert strict_schema()['properties']
    assert len(set(PROMPTS.values()))==4
