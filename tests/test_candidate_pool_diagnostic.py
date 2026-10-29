import copy

from scripts.evaluate_candidate_pool_diagnostic import expanded_state, summarize


def test_expansion_keeps_original_facts_and_does_not_modify_inputs():
    original = dict(query='每天最多两个', weather_forecast=[{'is_bad': True}],
                    pois=[dict(name='A', indoor=True, open_time='09:00-17:00')])
    full = [dict(name='A', indoor=None), dict(name='B', indoor=None)]
    before = copy.deepcopy((original, full))
    result = expanded_state(original, full)
    assert result['pois'] == [original['pois'][0], full[1]]
    assert result['query'] == original['query']
    assert result['weather_forecast'] == original['weather_forecast']
    assert (original, full) == before
    result['pois'][1]['indoor'] = True
    assert full[1]['indoor'] is None


def test_parse_failure_is_not_counted_as_success_or_duplicate():
    records = [dict(parse_success=False,passed=False,faults=[],explanation_faults=[]),
               dict(parse_success=True,passed=False,faults=['duplicate_poi'],explanation_faults=[])]
    result = summarize(records)
    assert result['n'] == 2
    assert result['parsed'] == 1
    assert result['joint_success'] == 0
    assert result['duplicate_cases'] == 1
