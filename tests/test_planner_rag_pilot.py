import json
import sys

import pytest

from scripts.compare_planner_rag import main, pilot_cases, prepare, summary


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
