import copy
from collections import Counter
import json

from scripts.build_representative_eval50 import OUT, score_case
from scripts.evaluate_representative50 import verify


def assets():
    return (json.loads((OUT/'cases.json').read_text(encoding='utf-8')),
            json.loads((OUT/'maps.json').read_text(encoding='utf-8'))['routes'])


def test_frozen_50_has_exact_balanced_city_strata_and_genuine_large_pools():
    verify()
    cases,_=assets()
    assert len({c['id'] for c in cases})==50
    assert Counter(c['city'] for c in cases)==dict.fromkeys(['南京','上海','三亚','成都','苏州'],10)
    for city in {c['city'] for c in cases}:
        subset=[c for c in cases if c['city']==city]
        assert Counter(c['stratum'] for c in subset)=={'normal':8,'hard':2}
        for c in subset:
            if c['stratum']=='normal':
                assert len(c['state']['pois'])>=18
            for p in c['state']['pois']:
                assert p['amap_id'] and p['provenance']['raw_sha256']
                assert p['indoor'] is None
    assert Counter(c['expected'] for c in cases)=={'feasible':45,'explained_infeasible':5}


def test_every_witness_passes_but_each_audited_negative_is_rejected():
    cases,maps=assets()
    for c in cases:
        assert score_case(c['feasibility_witness'],c,maps)==[]
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    assert len(audit)==50
    assert all(control['faults'] for c in audit for control in c['controls'])


def test_upper_limit_cannot_be_satisfied_by_empty_days_or_extra_visits():
    cases,maps=assets()
    c=next(c for c in cases if c['category']=='upper_limit')
    p=copy.deepcopy(c['feasibility_witness'])
    p['days'][0]['spots']=[]
    assert 'day_count' in score_case(p,c,maps)
    p=copy.deepcopy(c['feasibility_witness'])
    p['days'][0]['spots'].append(copy.deepcopy(p['days'][0]['spots'][0]))
    assert 'day_count' in score_case(p,c,maps)


def test_required_and_excluded_are_not_only_name_membership_checks():
    cases,maps=assets()
    c=next(c for c in cases if c['category']=='required_excluded')
    p=copy.deepcopy(c['feasibility_witness'])
    required=c['rubric']['required'][0]
    target=next(s for d in p['days'] for s in d['spots'] if s['name']==required)
    target['name']=c['rubric']['excluded'][0]
    faults=score_case(p,c,maps)
    assert 'required_missing' in faults and 'excluded_selected' in faults


def test_sparse_feasible_cannot_repeat_across_days():
    cases,maps=assets()
    c=next(c for c in cases if c['category']=='sparse_feasible')
    p=copy.deepcopy(c['feasibility_witness'])
    p['days'][1]['spots'][0]['name']=p['days'][0]['spots'][0]['name']
    assert 'global_duplicate' in score_case(p,c,maps)


def test_infeasible_case_requires_explanation_and_refuses_unsafe_partial_route():
    cases,maps=assets()
    c=next(c for c in cases if c.get('infeasibility_proof',{}).get('kind')=='pigeonhole')
    p=copy.deepcopy(c['feasibility_witness'])
    p['notes']='完成';p['reasoning']='完成'
    assert 'infeasible_not_explained' in score_case(p,c,maps)
    p=copy.deepcopy(c['feasibility_witness'])
    p['days'][0]['spots']=[dict(name=c['state']['pois'][0]['name'],start_time='10:00',end_time='11:00',period='morning')]
    assert 'infeasible_unsafe_schedule' in score_case(p,c,maps)


def test_missing_road_evidence_is_not_treated_as_zero_travel():
    cases,maps=assets()
    c=next(c for c in cases if c['category']=='upper_limit')
    assert 'road_unverified' in score_case(c['feasibility_witness'],c,{})
