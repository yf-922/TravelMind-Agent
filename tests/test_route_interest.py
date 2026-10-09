"""Contracts for interest evidence and finalized production itinerary comparisons."""
import copy
import json
from pathlib import Path
import pytest
from app.planning import nodes
from app.planning.schemas import TravelPlanState, TravelRoute, RouteReview, SpotTipsResult, TimeCheckResult
from app.evaluation.route_interest import validate_assets, freeze_assets, scoped_lookup, grade_final, summarize, run_final_route, paired_diagnostics
from scripts.review_route_interest import import_reviews, validate_snapshots
from scripts.compare_route_interest import verify_billing, exclusive_ledger

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'knowledge/route_interest_v1'


def load_assets():
    return tuple(json.loads((ASSETS/f'{name}.json').read_text(encoding='utf-8')) for name in ('cases', 'facts'))


def test_collector_keeps_exact_quote_despite_unrelated_invalid_bytes(tmp_path, monkeypatch):
    import httpx
    from scripts import build_route_interest_assets as builder
    monkeypatch.setattr(builder, 'SOURCES', [('南京', ['景点A'], 'https://official.example/fact', '古代教育', '历史')])
    monkeypatch.setattr(httpx, 'get', lambda *a, **k: httpx.Response(200, content=b'<html>'+ '古代教育'.encode()+b'<aside>\xff</aside></html>', request=httpx.Request('GET', a[0])))
    facts, records = builder.collect_sources(tmp_path)
    assert len(facts) == 1 and facts[0]['text'] == '古代教育'
    assert not validate_snapshots(tmp_path, facts)
    (tmp_path/'facts.json').write_text(json.dumps(facts), encoding='utf-8')
    monkeypatch.setattr(httpx, 'get', lambda *a, **k: pytest.fail('verified evidence must be reused'))
    reused, _ = builder.collect_sources(tmp_path, only_missing=True)
    assert reused == facts


def test_collector_rejects_quote_corrupted_by_invalid_bytes(tmp_path, monkeypatch):
    import httpx
    from scripts import build_route_interest_assets as builder
    monkeypatch.setattr(builder, 'SOURCES', [('南京', ['景点A'], 'https://official.example/fact', '古代教育', '历史')])
    monkeypatch.setattr(httpx, 'get', lambda *a, **k: httpx.Response(200, content='古代'.encode()+b'\xff'+'教育'.encode(), request=httpx.Request('GET', a[0])))
    facts, records = builder.collect_sources(tmp_path)
    assert facts == [] and records[0]['status'] == 'unavailable'


def test_collector_uses_identifiable_html_headers(tmp_path, monkeypatch):
    import httpx
    from scripts import build_route_interest_assets as builder
    monkeypatch.setattr(builder, 'SOURCES', [('上海', ['上海豫园'], 'https://official.example/yuyuan', '官方销售渠道购票', '购票')])
    seen = {}
    def fake_get(*args, **kwargs):
        seen.update(kwargs.get('headers', {}))
        return httpx.Response(200, content='官方销售渠道购票'.encode(), request=httpx.Request('GET', args[0]))
    monkeypatch.setattr(httpx, 'get', fake_get)
    facts, _ = builder.collect_sources(tmp_path)
    assert facts and 'TravelMind-eval-source-capture' in seen['User-Agent']
    assert seen['Accept'].startswith('text/html')


def test_assets_have_30_cases_18_distinct_real_candidates_and_provenance():
    cases, facts = load_assets()
    assert not validate_assets(cases, facts)
    assert not validate_snapshots(ASSETS, facts)
    assert len({(c['destination'], p['name']) for c in cases for p in c['pois']}) == 18
    assert all(p.get('location') for c in cases for p in c['pois'])


def test_frozen_test_refuses_unsigned_labels_and_incomplete_knowledge():
    cases, facts = load_assets()
    with pytest.raises(ValueError, match='human confirmation'):
        freeze_assets(cases, facts, {'model': 'fake'})
    for c in cases:
        c.update(annotation_status='human_reviewed', annotator='test', reviewed_at='2026-10-08', change_log=['test only'])
    # Simulate an incomplete corpus in the gate test; the checked-in corpus is
    # currently complete for all three cities.
    facts = [f for f in facts if '上海豫园' not in f.get('entities', [])]
    for f in facts:
        f.update(review_status='human_reviewed', annotator='test', reviewed_at='2026-10-08', change_log=['test only'])
    with pytest.raises(ValueError, match='coverage below six'):
        freeze_assets(cases, facts, {})


def test_group_leak_and_duplicate_source_are_rejected():
    cases, facts = load_assets()
    cases[-1]['intent_group'] = cases[0]['intent_group']
    assert 'intent group crosses splits' in validate_assets(cases, facts)
    assert 'duplicate source IDs' in validate_assets(cases, facts+[facts[0]])


@pytest.mark.parametrize('city,pool', [('上海', ['南京博物院']), ('南京', ['上海自然博物馆']), ('上海', ['上海博物馆']), ('南京', [])])
def test_scoped_interest_lookup_never_crosses_city_or_branch(city, pool):
    _, facts = load_assets()
    assert scoped_lookup(facts)(city, [{'name': n} for n in pool], '历史文化自然博物馆') == []


def test_missing_evidence_returns_empty_instead_of_nearest_forced_match():
    _, facts = load_assets()
    assert scoped_lookup(facts)('南京', [{'name': '南京博物院'}], '量子机器人新展名额') == []


def test_interest_lookup_uses_curated_topic_for_short_official_facts():
    _, facts = load_assets()
    rows = scoped_lookup(facts)('上海', [{'name': '上海自然博物馆'}], '生命演化的科学展示')
    assert rows and rows[0]['entities'] == ['上海自然博物馆']
    assert rows[0]['text']


def test_retrieval_diagnostic_scores_only_canonical_candidate_names(tmp_path):
    import subprocess, sys
    out = tmp_path / 'diagnostic.json'
    subprocess.run(
        [sys.executable, 'scripts/compare_route_interest.py', '--split', 'dev',
         '--retrieval-only', '--limit', '15', '--k', '3', '--retriever', 'keyword',
         '--out', str(out)], check=True, capture_output=True, text=True,
    )
    report = json.loads(out.read_text(encoding='utf-8'))
    rows = report['retrieval_diagnostics']
    assert all(set(row['candidate_entities']) <= {
        p['name'] for case in json.loads((ASSETS / 'cases.json').read_text(encoding='utf-8'))
        if case['id'] == row['case_id'] for p in case['pois']
    } for row in rows)
    assert report['retrieval_summary']['answerable_candidate_recall_mean'] is not None


@pytest.mark.parametrize('duplicate_day', [1, 2])
def test_joint_planner_removes_lunch_split_and_cross_day_duplicate(monkeypatch, duplicate_day):
    monkeypatch.setattr(nodes, 'build_structured_llm', lambda *a, **kw: object())
    spots = [{'name': 'A', 'period': 'morning', 'start_time': '10:00', 'end_time': '11:00'}]
    second = {'name': 'A', 'period': 'afternoon', 'start_time': '14:00', 'end_time': '15:00'}
    days = [{'day': 1, 'theme': 'test', 'spots': spots+([second] if duplicate_day == 1 else [])}]
    if duplicate_day == 2: days.append({'day': 2, 'theme': 'test', 'spots': [second]})
    result = TravelRoute(reasoning='test', days=days)
    planner = nodes.make_joint_planner_node(None, knowledge_lookup=lambda *a: [], invoke_fn=lambda *a: result)
    update = planner(TravelPlanState(query='test', days=duplicate_day, pois=[{'name': 'A'}]))
    assert [s['name'] for d in update['route'] for s in d['spots']] == ['A']
    if duplicate_day == 2:
        state = TravelPlanState(query='test', days=2, pois=[{'name': 'A'}], route=update['route'])
        assert 'route_structure' in nodes._route_risk_flags(state)  # empty second day cannot become a pass


@pytest.mark.parametrize('mode', ['empty', 'failure', 'fresh'])
def test_old_evidence_cleared_and_citations_bound_to_selected_entity(monkeypatch, mode):
    captured = []
    result = TravelRoute(reasoning='test', days=[{'day': 1, 'theme': 'test', 'spots': [
        {'name': 'A', 'period': 'morning', 'start_time': '10:00', 'end_time': '11:00'}]}],
        selection_reasons=[{'name': 'A', 'reason': 'test', 'source_ids': ['old', 'new', 'wrong', 'new']},
                           {'name': 'B', 'reason': 'unselected', 'source_ids': ['wrong']}])
    def lookup(*a):
        if mode == 'failure': raise RuntimeError('unavailable')
        if mode == 'empty': return []
        return [{'source': sid, 'entities': [name], 'chunk_id': sid+'-0', 'text': 'fact'} for sid, name in [('new', 'A'), ('wrong', 'B')]]
    def invoke(llm, messages): captured.extend(messages); return result
    planner = nodes.make_joint_planner_node(None, llm=object(), knowledge_lookup=lookup, invoke_fn=invoke)
    update = planner(TravelPlanState(query='现在只想历史，不要游乐园', rewritten_query='历史偏好游乐园', pois=[{'name': 'A'}, {'name': 'B'}],
                                    rag_sources=[{'source': 'old'}], selection_reasons=[{'name': 'OLD'}]))
    assert '原始需求：现在只想历史，不要游乐园' in captured[1][1]
    assert update['selection_reasons'][0]['source_ids'] == (['new'] if mode == 'fresh' else [])
    assert len(update['selection_reasons']) == 1
    assert all(r.get('source') != 'old' for r in update['rag_sources'])


def case_and_state(names):
    case = {'acceptable_pois': ['A', 'B'], 'forbidden_pois': ['C'], 'answerable': True, 'annotation_status': 'draft', 'days': 1}
    spots = [{'name': n, 'period': 'morning' if i == 0 else 'afternoon', 'start_time': '10:00' if i == 0 else '14:00', 'end_time': '11:00' if i == 0 else '15:00'} for i, n in enumerate(names)]
    state = TravelPlanState(query='test', days=1, pois=[{'name': n, 'open_time': '09:00-18:00'} for n in 'ABC'],
                            route=[{'day': 1, 'spots': []}], final_plan={'days': [{'day': 1, 'timeline': [{'type': 'attraction', **s} for s in spots]}]})
    return case, state


def test_scoring_ignores_notes_and_old_draft_route():
    case, state = case_and_state(['C'])
    state.route = [{'day': 1, 'spots': [{'name': 'A'}]}]
    state.final_plan['notes'] = 'A B highly suitable'
    metrics = grade_final(state, case)
    assert metrics['selected_pois'] == ['C'] and metrics['interest_precision'] == 0
    assert metrics['forbidden_selected'] == ['C']
    assert not metrics['hard_constraint_pass']


def test_final_scoring_enforces_indoor_requirement_beyond_legacy_graders():
    case, state = case_and_state(['A'])
    state.query = '只去室内景点'
    state.pois[0]['indoor'] = False
    metrics = grade_final(state, case)
    assert metrics['code_graders']['objective_pass']
    assert 'indoor_constraint' in metrics['hard_constraint_violations']
    assert not metrics['hard_constraint_pass']


def test_review_signal_alone_does_not_fail_final_hard_constraints():
    case, state = case_and_state(['A'])
    state.modification_notes = '选择A'
    metrics = grade_final(state, case)
    assert 'user_modification' in metrics['hard_flags']
    assert not metrics['hard_constraint_violations']
    assert metrics['hard_constraint_pass']


def test_precision_recall_dedup_and_empty_final_are_not_success():
    case, state = case_and_state(['A', 'A'])
    metrics = grade_final(state, case)
    assert metrics['interest_precision'] == 1 and metrics['interest_recall'] == .5
    assert not metrics['hard_constraint_pass']
    state.final_plan = {'days': []}
    assert not grade_final(state, case)['hard_constraint_pass']


def test_failures_remain_in_overall_quality_denominator():
    case, state = case_and_state(['A'])
    rows = [{'answerable': True, 'metrics': grade_final(state, case), 'elapsed_ms': 10}, {'answerable': True, 'error': 'timeout'}]
    result = summarize(rows, 3)
    assert result['interest_precision_including_failures'] == .5
    assert result['interest_precision_success_only'] == 1
    assert result['availability_attempted'] == .5 and result['execution_coverage'] == 2/3


def test_annotation_import_cannot_rewrite_query_or_forge_unsigned_review():
    cases, _ = load_assets()
    edited = copy.deepcopy(cases); edited[0]['query'] = 'tuned test query'
    with pytest.raises(ValueError, match='cannot be changed'): import_reviews(cases, edited)
    edited = copy.deepcopy(cases); edited[0]['annotation_status'] = 'human_reviewed'
    with pytest.raises(ValueError, match='signed'): import_reviews(cases, edited)


def test_unknown_billing_blocks_before_calls(tmp_path):
    with pytest.raises(ValueError, match='multiplier unknown'): verify_billing(None, 'grok-4.6', 'url')
    path = tmp_path/'billing.json'; path.write_text('{}')
    with pytest.raises(ValueError): verify_billing(path, 'grok-4.6', 'url')


def test_monitor_retries_rate_limit_but_not_auth_or_invalid_usage(monkeypatch):
    import httpx
    from scripts import compare_route_interest as runner
    sleeps = []
    monkeypatch.setattr(runner.time, 'sleep', sleeps.append)
    statuses = iter([429, 200])
    def get(url, **kw):
        return httpx.Response(next(statuses), json={'data': {'total_used': 5}}, request=httpx.Request('GET', url))
    monkeypatch.setattr(httpx, 'get', get)
    assert runner.read_gateway_usage('https://example.org/v1', 'fake') == 5
    assert sleeps == [5]
    statuses = iter([401])
    with pytest.raises(httpx.HTTPStatusError): runner.read_gateway_usage('https://example.org/v1', 'fake')
    assert sleeps == [5]


@pytest.mark.parametrize('unpriced', [False, True])
def test_paid_entry_pairs_trials_and_records_usage_without_external_calls(tmp_path, monkeypatch, unpriced):
    import sys
    import httpx
    from types import SimpleNamespace
    from scripts import compare_route_interest as runner
    from app.llm import grok
    monkeypatch.setenv('GROK_BASE_URL', 'https://www.micuapi.ai/v1')
    monkeypatch.setenv('GROK_API_KEY', 'offline-test-key')
    monkeypatch.setattr(runner, 'load_local_env', lambda: None)
    monkeypatch.setattr(runner, 'verify_billing', lambda *a: {
        'input_per_million': 1, 'output_per_million': 1,
        'account_multiplier': 1, 'quota_units_per_CNY': 1000000,
    })
    def monitor(url, **kwargs):
        assert url.endswith('/api/usage/token/')
        return httpx.Response(200, json={'data': {'total_used': 0}}, request=httpx.Request('GET', url))
    monkeypatch.setattr(httpx, 'get', monitor)
    class Client:
        def with_structured_output(self, *a, **kw): return self
        def invoke(self, messages):
            return {'raw': SimpleNamespace(usage_metadata={'input_tokens': 10, 'output_tokens': 5}),
                    'parsed': 'offline-result', 'parsing_error': None}
    monkeypatch.setattr(grok, 'build_chat_grok', lambda **kw: Client())
    def route(case, model, lookup, factory, replay):
        assert factory(TravelRoute, task_type='joint_planner').invoke([('human', 'test')]) == 'offline-result'
        return TravelPlanState(query=case['query']), {'retrieval': {'failed': False}}
    monkeypatch.setattr(runner, 'run_final_route', route)
    out = tmp_path / 'report.json'
    argv = ['compare', '--execute', '--limit', '5' if unpriced else '1', '--out', str(out), '--ledger', str(tmp_path / 'ledger.json')]
    if unpriced:
        argv.append('--allow-unpriced')
        monkeypatch.setattr(runner, 'verify_billing', lambda *a: pytest.fail('unpriced pilot must not invent billing'))
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    report = json.loads(out.read_text(encoding='utf-8'))
    assert report['status'] == 'exploratory_complete'
    assert len(report['calls']) == len(report['results']) == (10 if unpriced else 2)
    assert {c['trial'] for c in report['calls']} == {
        f"{r['case_id']}/{r['arm']}/0" for r in report['results']}
    if unpriced:
        assert report['preflight']['billing_status'] == 'unpriced_pilot'
        assert report['preflight']['cost_estimate'] is None
    assert all(r['actual_model_calls'] == 1 and r['actual_usage_complete'] for r in report['results'])
    assert all(r['actual_usage'] == {'input_tokens': 10, 'output_tokens': 5} for r in report['results'])
    # Changes to read-only metering must not replay successful paid trials.
    monkeypatch.setattr(sys, 'argv', [*argv[:], '--out', str(tmp_path/'resumed.json')])
    monkeypatch.setattr(runner, 'run_final_route', lambda *a, **kw: pytest.fail('completed trial replayed'))
    runner.main()
    resumed = json.loads((tmp_path/'resumed.json').read_text(encoding='utf-8'))
    assert not resumed['calls']
    assert all(r['reused_paid_trial'] for r in resumed['results'])


def test_paid_ledger_rejects_concurrent_process_and_releases_after_error(tmp_path):
    ledger = tmp_path/'ledger.json'
    with pytest.raises(ValueError):
        with exclusive_ledger(ledger):
            with pytest.raises(RuntimeError, match='locked'):
                with exclusive_ledger(ledger): pass
            raise ValueError('simulated failure')
    with exclusive_ledger(ledger): pass


def test_legacy_import_rejects_changed_inputs_before_migration(monkeypatch):
    from scripts import compare_route_interest as runner
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: type('Result', (), {'stdout': b'changed code'})())
    ledger = {'trials': {}}
    with pytest.raises(ValueError, match='behavior changed'):
        runner.import_legacy_trials(ledger, {}, 'fake', [], [], {}, {})
    assert ledger['trials'] == {}


def test_interrupted_call_is_audited_not_assumed_zero_or_replayed(tmp_path):
    from scripts.reconcile_route_interest_interruption import reconcile
    ledger = tmp_path/'ledger.json'
    report = tmp_path/'report.json'
    ledger.write_text(json.dumps({'in_flight': True, 'budget': {'unknown_usage_calls': 0}, 'trials': {}}))
    report.write_text(json.dumps({'calls': [{'status': 'in_flight', 'usage': None}]}))
    reconcile(ledger, report)
    value = json.loads(ledger.read_text())
    assert not value['in_flight']
    assert value['budget']['unknown_usage_calls'] == 1
    assert value['interruption_audits'][0]['replayed'] is False
    assert not value['trials']
    assert json.loads(report.read_text())['status'] == 'interrupted'
    with pytest.raises(ValueError, match='no matching'):
        reconcile(ledger, report)


def test_wrong_city_or_entity_evidence_is_rejected():
    cases, facts = load_assets()
    cases[0]['required_evidence'] = [next(f['source'] for f in facts if f['city'] == '上海')]
    assert any('does not support' in e for e in validate_assets(cases, facts))


def test_budget_counts_failed_attempts_and_unknown_usage():
    from scripts.evaluate_rag_generation import CallBudget
    budget = CallBudget(1, 10, 2, 6, 2048)
    reservation = budget.reserve([('user', '历史兴趣')])
    budget.reconcile(reservation, None)
    assert budget.unknown_usage_calls == 1 and budget.calls == 1
    with pytest.raises(RuntimeError, match='exhausted'):
        budget.reserve([('user', '重试')])


def test_paired_changes_include_failures_and_keep_repetitions_in_same_group():
    cases = [{'id': 'a', 'answerable': True, 'intent_group': 'history'}, {'id': 'b', 'answerable': True, 'intent_group': 'animals'}]
    rows = [
        {'case_id': 'a', 'repetition': 0, 'arm': 'without_knowledge', 'error': 'timeout'},
        {'case_id': 'a', 'repetition': 0, 'arm': 'with_knowledge', 'metrics': {'interest_precision': 1, 'selected_pois': ['A']}},
        {'case_id': 'b', 'repetition': 0, 'arm': 'without_knowledge', 'metrics': {'interest_precision': 1, 'selected_pois': ['B']}},
        {'case_id': 'b', 'repetition': 0, 'arm': 'with_knowledge', 'metrics': {'interest_precision': 0, 'selected_pois': ['C']}}]
    result = paired_diagnostics(rows, cases)
    assert result['intent_groups'] == 2 and result['group_mean_precision_delta'] == 0
    assert len(result['improvements']) == len(result['regressions']) == 1
    assert result['improvements'][0]['pair_has_failure']


def test_explicit_hybrid_error_is_not_disguised_as_keyword(monkeypatch):
    from app.evaluation import rag_retrieval
    monkeypatch.setattr(rag_retrieval, 'search', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('vector missing')))
    _, facts = load_assets()
    with pytest.raises(RuntimeError, match='vector missing'):
        scoped_lookup(facts, mode='hybrid')('南京', [{'name': '南京博物院'}], '江苏古代文明')


def test_production_tail_runs_joint_planner_to_final_json_offline(monkeypatch):
    calls = []
    case, _ = case_and_state(['A'])
    case.update(query='历史', destination='南京', max_per_day=2,
                pois=[{'name': 'A', 'open_time': '09:00-18:00', 'location': {'lng': 118.8, 'lat': 32.0}}])
    def factory(schema, **kw):
        class Client:
            def invoke(self, messages):
                calls.append(kw['task_type'])
                if schema is TravelRoute:
                    return TravelRoute(reasoning='test', days=[{'day': 1, 'theme': '历史', 'spots': [
                        {'name': 'A', 'period': 'morning', 'start_time': '10:00', 'end_time': '11:00'}]}])
                if schema is RouteReview: return RouteReview(reasoning='test', approved=True, score=90)
                if schema is SpotTipsResult: return SpotTipsResult(tips=[])
                if schema is TimeCheckResult: return TimeCheckResult(violations=[])
                raise AssertionError(schema)
        return Client()
    # Any live external query would fail; fixed upstream and replay are required.
    state, diagnostics = run_final_route(case, 'fake', lambda *a: [], factory, {})
    assert grade_final(state, case)['selected_pois'] == ['A']
    assert 'joint_planner' in calls and 'planner' not in calls
    assert not diagnostics['retrieval']['failed']
