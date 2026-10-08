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


def test_paid_ledger_rejects_concurrent_process_and_releases_after_error(tmp_path):
    ledger = tmp_path/'ledger.json'
    with pytest.raises(ValueError):
        with exclusive_ledger(ledger):
            with pytest.raises(RuntimeError, match='locked'):
                with exclusive_ledger(ledger): pass
            raise ValueError('simulated failure')
    with exclusive_ledger(ledger): pass


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
