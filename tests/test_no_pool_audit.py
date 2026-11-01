import copy
import json

from pathlib import Path
from scripts.evaluate_no_pool_base import SEEDS, strip_pool

SUITE = Path(__file__).parent / 'fixtures' / 'post_train' / 'holdout'
from scripts.verify_no_pool_entities import classify, spot_names


def test_remove_pool_keep_constraints_across_entire_frozen_suite():
    cases = json.loads((SUITE / 'cases.json').read_text(encoding='utf8'))
    original = copy.deepcopy(cases)
    assert len(cases) == 50 and sum(c['expected']=='feasible' for c in cases) == 45
    for c in cases:
        messages = strip_pool(c['messages'])
        text = '\n'.join(m['content'] for m in messages)
        assert '景点候选池' not in text
        assert '只从给定候选' not in text
        assert '现有候选' not in text
        assert '目前只检索到' not in text
        assert c['city'] in text
        assert 'source_ids' in text and 'HH:MM' in text
        for name in c['rubric'].get('required', []) + c['rubric'].get('excluded', []):
            assert name in text
    assert cases == original
    assert SEEDS == [1001,1002,1003]


POLYGON = [[(0,0),(10,0),(10,10),(0,10),(0,0)]]


def test_four_buckets_and_no_fuzzy_acceptance():
    poi={'name':'博物馆','location':'5,5','type':'科教文化服务;博物馆','typecode':'140100'}
    assert classify('城市','博物馆',[poi],POLYGON)['bucket']=='a'
    assert classify('城市','博物馆',[dict(poi,location='15,15')],POLYGON)['bucket']=='b'
    assert classify('城市','博物馆',[dict(poi,type='餐饮服务',typecode='050000')],POLYGON)['bucket']=='c'
    assert classify('城市','博物馆分馆',[poi],POLYGON)['bucket']=='d'


def test_api_failure_and_missing_coordinates_not_invention():
    assert classify('城市','地点',[],POLYGON,False)['bucket']=='pending'
    assert classify('城市','地点',[{'name':'地点'}],POLYGON)['bucket']=='pending'
    assert classify('城市','地点',[{'name':'地点','location':'5,5'}],[])['bucket']=='pending'


def test_entity_denominator_counts_occurrences_not_unique_names():
    plan={'days':[{'spots':[{'name':'甲'},{'name':'甲'}]},{'spots':[{'name':'乙'}]}]}
    assert spot_names(plan)==['甲','甲','乙']
