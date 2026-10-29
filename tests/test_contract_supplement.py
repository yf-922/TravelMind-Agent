from collections import Counter
import copy
import json
from scripts.supplement_planning_contract import SOURCE, check
from scripts.post_train_runtime import capture_messages


def test_supplement_witnesses_and_sources():
    # Saved input evidence is optional outside the local research workspace.
    import pytest
    from scripts.supplement_planning_contract import OUT
    if not (OUT/'cases.json').exists():
        pytest.skip('Local evidence-backed research assets not included in checkout')
    cases = json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
    maps = json.loads((SOURCE/'maps.json').read_text(encoding='utf-8'))['routes']
    assert len(cases) == 32
    assert Counter(c['category'] for c in cases) == {
        'sparse':8,'multi_day':8,'weather_boundary':8,'explanation':8}
    for case in cases:
        assert case['state']['destination'] not in {'丽江','景德镇'}
        assert not check(case['feasibility_witness'],case,maps)
        assert 'planner-contract-v1' in capture_messages(case['state'])[-1]['content']
    multi = next(c for c in cases if c['category']=='multi_day')
    broken = copy.deepcopy(multi['feasibility_witness'])
    broken['days'][1]['spots'][0]['name'] = broken['days'][0]['spots'][0]['name']
    assert 'duplicate_poi' in check(broken,multi,maps)
