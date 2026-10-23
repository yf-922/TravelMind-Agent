"""Modification probes check real lookup routing, parity and unsupported cases."""
import copy
from unittest.mock import Mock

import pytest

from app.evaluation import checkpoint_benchmark as benchmark
from app.planning.modification import refresh_explicit_modification_candidates
from tests.test_evidence_benchmarks import make_ledger


def test_all_four_edits_run_independently_and_prompt_counts_ignore_timing_repeats():
    ledger = make_ledger()
    before = copy.deepcopy(ledger)
    result = benchmark.measure_checkpoint_prompts(ledger, samples=1, repetitions=2)
    assert ledger == before
    assert result['modification_requests'] == 4
    assert result['timed_runs_per_arm'] == 8
    assert result['mock_provider_calls']['local'] == {
        'weather_prefetch': 0, 'poi_prefetch': 0, 'poi_targeted': 1,
    }
    assert result['mock_provider_calls']['full'] == {
        'weather_prefetch': 4, 'poi_prefetch': 4, 'poi_targeted': 0,
    }
    for row, prompts in zip(result['rows'], result['_prompts']):
        assert row['id'] == prompts['id']
        assert row['checkpoint_sha256'] == benchmark.fingerprint(next(iter(before['seeds'].values()))['state'])
        assert row['identical_candidate_pool']
        assert len(row['timings_ms']['local']) == 2
        assert row['full_text_tokens'] == sum(row['full_text_tokens_by_node'].values())
    assert result['rows'][-1]['mock_lookup_trace'][0]['status'] == 'added'


def test_indoor_and_replacement_pool_gaps_are_not_claimed_as_cache_success():
    ledger = make_ledger()
    state = next(iter(ledger['seeds'].values()))['state']
    state['pois'] = [dict(state['pois'][0], indoor=False)]
    result = benchmark.measure_checkpoint_prompts(ledger, samples=1, repetitions=1)
    for row in result['rows'][1:3]:
        assert not row['applicable']
        assert row['inapplicable_reason']
        assert not row['cache_pool_sufficient']
        assert row['mocked_targeted_lookup_calls'] == 0


def test_pair_cannot_hide_broken_merge_by_reusing_local_candidate_output(monkeypatch):
    monkeypatch.setattr(benchmark, 'refresh_explicit_modification_candidates',
                        lambda state, notes, **kw: (state, notes, []))
    with pytest.raises(RuntimeError, match='pool parity'):
        benchmark.measure_checkpoint_prompts(make_ledger(), samples=1, repetitions=1)


@pytest.mark.parametrize('outcome,expected', [
    ([], 'not_verified'),
    ([{'name': '同名分馆', 'location': '118,32'}], 'not_verified'),
    ([{'name': '新增馆', 'location': ''}], 'not_verified'),
    (RuntimeError('offline simulated timeout'), 'provider_failed'),
    ([{'name': '新增馆', 'location': '118,32'}], 'added'),
])
def test_api_shared_lookup_requires_matching_name_and_valid_coordinates(outcome, expected):
    checkpoint = {'destination': '南京', 'pois': [{'name': '原景点'}]}
    before = copy.deepcopy(checkpoint)
    lookup = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
    result, notes, trace = refresh_explicit_modification_candidates(
        checkpoint, '加入“新增馆”', lookup=lookup, key_provider=lambda: 'offline',
    )
    assert lookup.call_count == 1
    assert trace == [{'name': '新增馆', 'status': expected}]
    assert checkpoint == before
    assert len(result['pois']) == (2 if expected == 'added' else 1)
    assert ('系统已核验新增景点' in notes) == (expected == 'added')


def test_lookup_skips_existing_poi_and_limits_missing_names_without_destination():
    lookup = Mock(return_value=[])
    checkpoint = {'destination': '南京', 'pois': [{'name': '已有馆'}]}
    refresh_explicit_modification_candidates(checkpoint, '换成“已有馆”', lookup=lookup)
    lookup.assert_not_called()
    refresh_explicit_modification_candidates(checkpoint, '加入“新馆一”，加入“新馆二”，加入“新馆三”，加入“新馆四”',
                                             lookup=lookup, key_provider=lambda: 'offline')
    assert lookup.call_count == 3
    lookup.reset_mock()
    _, _, trace = refresh_explicit_modification_candidates({'pois': []}, '加入“新馆一”', lookup=lookup)
    lookup.assert_not_called()
    assert trace[0]['status'] == 'missing_destination'
