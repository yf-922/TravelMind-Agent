import json
import sys

import pytest

from scripts.compare_planner_rag import main, pilot_cases, prepare, summary
from scripts.report_planner_rag_pilot import paired_report


def test_pilot_has_controls_and_no_invented_live_metadata():
    cases = pilot_cases()
    assert len(cases) == 12
    assert len({c['id'] for c in cases}) == 12
    assert {c['kind'] for c in cases} == {'interest', 'unknown_live_fact', 'insufficient_pool', 'no_evidence'}
    for case in cases:
        assert all(set(p) == {'name'} for p in case['state']['pois'])
        assert case['candidate_provenance']['kind'] == 'official_snapshot_entity_not_live_poi'


def test_paired_messages_differ_only_in_retrieved_block():
    cases = pilot_cases()
    rows = prepare(cases)
    assert len(rows) == 24
    for baseline, rag in zip(rows[::2], rows[1::2]):
        assert baseline['state'] == rag['state']
        assert baseline['messages'][0] == rag['messages'][0]
        assert baseline['messages'][2] == rag['messages'][2]
        prompt = rag['messages'][1][1]
        if rag['evidence']:
            start = prompt.index('\n\n<RETRIEVED_DATA>')
            end = prompt.index('</RETRIEVED_DATA>') + len('</RETRIEVED_DATA>')
            prompt = prompt[:start] + prompt[end:]
        assert baseline['messages'][1][1] == prompt
        if not rag['state']['pois']:
            assert not rag['evidence']
    # Keep retrieval misses as experimental observations, not guaranteed hits.
    assert any(r['evidence'] for r in rows[1::2])
    assert any(r['state']['pois'] and not r['evidence'] for r in rows[1::2])


def test_not_run_summary_does_not_fabricate_zero_scores():
    report = summary([], 36)
    assert report['with_rag']['availability'] is None
    assert report['with_rag']['actual_input_tokens'] is None
    assert report['without_rag']['p95_ms'] is None


def test_failed_calls_are_in_denominator_and_missing_usage_not_estimated():
    rows = [{'arm': 'with_rag', 'latency_ms': 10, 'usage': {'input_tokens': 20, 'output_tokens': 5},
             'basic_rules': {'passed': True}},
            {'arm': 'with_rag', 'latency_ms': 90, 'error': 'timeout'}]
    report = summary(rows, 2)['with_rag']
    assert report['availability'] == .5
    assert report['basic_rule_pass_including_failures'] == .5
    assert report['usage_measured_calls'] == 1
    assert report['actual_input_tokens'] == 20
    assert report['p95_ms'] == 90


def test_unrun_trials_are_not_provider_failures_and_pair_totals_exclude_unpaired():
    row = {'case_id': 'a', 'repetition': 0, 'arm': 'without_rag', 'latency_ms': 10,
           'usage': {'input_tokens': 20, 'output_tokens': 5}, 'basic_rules': {'passed': True}}
    other = {**row, 'arm': 'with_rag', 'latency_ms': 12}
    unpaired = {**row, 'case_id': 'b', 'usage': {'input_tokens': 999, 'output_tokens': 999}}
    report = paired_report({'results': [row, other, unpaired], 'status': 'budget_stopped',
                            'fingerprint': 'test', 'model': 'fake', 'scope': 'offline',
                            'preflight': {'requests': 12, 'repetitions': 1}})
    assert report['completed_pairs_only']['without_rag']['input_tokens'] == 20
    assert len(report['trials']) == 3
    summary_row = report['all_recorded_summary']['without_rag']
    assert summary_row['availability'] == 1
    assert summary_row['execution_completion'] == 2/12


def test_preflight_does_not_run_and_preserves_previous_report(monkeypatch, tmp_path):
    out = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['compare', '--out', str(out)])
    main()
    saved = out.read_bytes()
    report = json.loads(saved)
    assert report['status'] == 'not_run' and report['results'] == []
    assert report['preflight']['calls_ceiling'] == 72
    assert report['preflight']['cost_ceiling'] is None
    assert report['retrieval_diagnostic']['with_evidence'] == 5
    with pytest.raises(SystemExit):
        main()
    assert out.read_bytes() == saved


def test_execution_requires_budget_before_client_creation(monkeypatch, tmp_path):
    out = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['compare', '--out', str(out), '--execute',
                                    '--model', 'test', '--url', 'https://example.org',
                                    '--input-price', '2', '--output-price', '3'])
    with pytest.raises(SystemExit):
        main()
    report = json.loads(out.read_text(encoding='utf-8'))
    assert report['status'] == 'not_run' and report['results'] == []
    assert report['preflight']['cost_ceiling'] > 0


def test_execution_records_failures_usage_transport_and_completed_output(monkeypatch, tmp_path):
    import langchain_openai
    from types import SimpleNamespace
    class FakeClient:
        calls = 0
        total_calls = 0
        def __init__(self, **kwargs):
            assert kwargs['reasoning_effort'] == 'low'
            assert kwargs['use_responses_api'] is True
            assert kwargs['max_retries'] == 0
        def invoke(self, messages):
            self.calls += 1
            type(self).total_calls += 1
            if self.calls == 1:
                raise RuntimeError('simulated failure')
            return SimpleNamespace(content=[{'type': 'text', 'text': json.dumps({'reasoning': 'test', 'days': []})}],
                                   usage_metadata={'input_tokens': 100, 'output_tokens': 10})
    monkeypatch.setattr(langchain_openai, 'ChatOpenAI', FakeClient)
    monkeypatch.setenv('PILOT_TEST_KEY', 'fake-offline-key')
    out = tmp_path / 'results.json'
    monkeypatch.setattr(sys, 'argv', ['compare', '--out', str(out), '--execute', '--model', 'test',
                                    '--url', 'https://example.org', '--key-env', 'PILOT_TEST_KEY',
                                    '--input-price', '2', '--output-price', '6', '--max-cost', '1',
                                    '--max-calls', '24', '--repetitions', '1',
                                    '--reasoning-effort', 'low', '--responses-api'])
    main()
    report = json.loads(out.read_text(encoding='utf-8'))
    assert report['status'] == 'completed_pending_human_review'
    assert len(report['results']) == 24 and 'in_flight' not in report
    assert report['budget']['unknown_usage_calls'] == 1
    assert report['summary']['without_rag']['availability'] == 11/12
    assert report['summary']['with_rag']['actual_input_tokens'] == 1200
    assert all(r['human_scores'] is None for r in report['results'])
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--resume'])
    main()
    assert FakeClient.total_calls == 24
    resumed = json.loads(out.read_text(encoding='utf-8'))
    assert len(resumed['results']) == 24
    assert resumed['budget']['attempted_calls'] == 24
