"""Final-route interest diagnostics. Draft labels never become formal gold."""
from __future__ import annotations

from contextlib import ExitStack
import json
from collections import Counter
from unittest.mock import patch

from langgraph.graph import END, START, StateGraph
from app.evaluation.rag_protocol import fingerprint
from app.planning import nodes
from app.planning.schemas import TravelPlanState
from app.planning.enrichment import bounded_spot_tips
from tests.eval.graders.code_graders import grade_code


def validate_assets(cases, facts, *, formal=False):
    errors = []
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)): errors.append('duplicate case IDs')
    sources = {f['source']: f for f in facts}
    if len(sources) != len(facts): errors.append('duplicate source IDs')
    groups = {}
    for c in cases:
        groups.setdefault(c['intent_group'], set()).add(c['split'])
        names = [p['name'] for p in c['pois']]
        if len(names) < 4 or len(set(names)) != len(names): errors.append(c['id'] + ': invalid pool')
        if not set(c['acceptable_pois'] + c['forbidden_pois']) <= set(names): errors.append(c['id'] + ': unknown label')
        if not set(c['required_evidence']) <= sources.keys(): errors.append(c['id'] + ': unknown evidence')
        for source in c['required_evidence']:
            fact = sources.get(source)
            if fact and (fact['city'] != c['destination'] or not set(fact['entities']).intersection(c['acceptable_pois'])):
                errors.append(c['id'] + ': evidence does not support labeled city/entity')
        if c['split'] not in ('dev', 'test') or not c.get('intent_group'): errors.append(c['id'] + ': invalid split/group')
        if formal and c['answerable'] and not c['required_evidence']: errors.append(c['id'] + ': answerable without evidence')
        if formal and (c.get('annotation_status') != 'human_reviewed' or not all(c.get(k) for k in ('annotator', 'reviewed_at', 'change_log'))):
            errors.append(c['id'] + ': human confirmation required')
    if any(len(v) > 1 for v in groups.values()): errors.append('intent group crosses splits')
    if Counter(c['split'] for c in cases) != Counter(dev=15, test=15): errors.append('expected dev15/test15')
    if Counter(c['destination'] for c in cases) != Counter({'南京': 10, '上海': 10, '三亚': 10}): errors.append('expected ten cases per city')
    if formal and any(f.get('review_status') != 'human_reviewed' or not all(f.get(k) for k in ('annotator', 'reviewed_at', 'change_log')) for f in facts): errors.append('signed source review required')
    if formal:
        from datetime import datetime
        for record in [*cases, *facts]:
            try:
                datetime.fromisoformat(record.get('reviewed_at') or '')
            except (TypeError, ValueError):
                errors.append('invalid human review date')
        for city in ('南京', '上海', '三亚'):
            pool = next(c['pois'] for c in cases if c['destination'] == city)
            if sum(any(p['name'] in f.get('entities', []) and f.get('city') == city for f in facts) for p in pool) < 6:
                errors.append(city + ': knowledge coverage below six candidates')
    return errors


def freeze_assets(cases, facts, config):
    errors = validate_assets(cases, facts, formal=True)
    if errors: raise ValueError('; '.join(errors))
    return {'cases': fingerprint(cases), 'facts': fingerprint(facts), 'config': fingerprint(config)}


def scoped_lookup(facts, limit=3, mode='keyword'):
    """Explicit experimental corpus; no production index or silent fallback."""
    from app.core.travel_knowledge import _keyword_score
    def lookup(city, pois, query):
        names = {p['name'] for p in pois}
        rows = [{**f, 'chunk_id': f['source']+'-0', 'score': _keyword_score(query, f['text'])}
                for f in facts if f['city'] == city and names.intersection(f['entities'])]
        if not rows: return []
        if mode == 'keyword':
            return sorted([r for r in rows if r['score'] >= .25], key=lambda r: (-r['score'], r['source']))[:limit]
        from app.evaluation.rag_retrieval import search
        ranked = search(query, mode, limit, rows=rows)
        return [{**next(r for r in rows if r['source'] == x['source']), **x} for x in ranked]
    return lookup


def summarize(rows, expected):
    """Failed trials remain in denominator; unexecuted trials stay separate."""
    import statistics
    completed = [r for r in rows if not r.get('error') and r.get('metrics')]
    valid = [r for r in completed if r['metrics']['interest_precision'] is not None]
    answerable_attempts = [r for r in rows if r.get('answerable')]
    measured = [r for r in rows if r.get('actual_usage_complete')]
    return {'planned_trials': expected, 'attempted_trials': len(rows), 'completed_trials': len(completed),
            'availability_attempted': len(completed)/len(rows) if rows else None,
            'execution_coverage': len(rows)/expected if expected else None,
            'interest_precision_including_failures': sum(r['metrics']['interest_precision'] for r in valid)/len(answerable_attempts) if answerable_attempts else None,
            'interest_precision_success_only': statistics.mean(r['metrics']['interest_precision'] for r in valid) if valid else None,
            'hard_constraint_pass_rate': sum(r['metrics']['hard_constraint_pass'] for r in completed)/len(rows) if rows else None,
            'exclusion_violation_rate': sum(bool(r['metrics']['forbidden_selected']) for r in completed)/len(completed) if completed else None,
            'mean_rework_completed': statistics.mean(r['metrics']['rework_count'] for r in completed) if completed else None,
            'mean_elapsed_ms_completed': statistics.mean(r['elapsed_ms'] for r in completed) if completed else None,
            'failure_rate_attempted': (len(rows)-len(completed))/len(rows) if rows else None,
            'retrieval_failure_rate': sum(bool(r.get('diagnostics', {}).get('retrieval', {}).get('failed')) for r in rows)/len(rows) if rows else None,
            'actual_usage_measured_trials': len(measured),
            'actual_tokens_known_usage_only': {k: sum(r['actual_usage'][k] for r in measured) for k in ('input_tokens', 'output_tokens')},
            'unknown_interest_scoring': 'manual refusal/evidence check required; excluded from numeric interest scoring'}


def paired_diagnostics(rows, cases, seed=20261008):
    """Bootstrap paired interest differences by intent group, not repetitions."""
    import random
    import statistics
    by_case = {c['id']: c for c in cases}
    pairs = {}
    for r in rows: pairs.setdefault((r['case_id'], r['repetition']), {})[r['arm']] = r
    groups, changes = {}, []
    for (case_id, repetition), pair in pairs.items():
        if len(pair) != 2 or not by_case[case_id]['answerable']: continue
        def score(arm): return (pair[arm].get('metrics') or {}).get('interest_precision') or 0
        delta = score('with_knowledge')-score('without_knowledge')
        groups.setdefault(by_case[case_id]['intent_group'], []).append(delta)
        changes.append({'case_id': case_id, 'repetition': repetition, 'interest_precision_delta': delta,
                        'without_selected': (pair['without_knowledge'].get('metrics') or {}).get('selected_pois', []),
                        'with_selected': (pair['with_knowledge'].get('metrics') or {}).get('selected_pois', []),
                        'pair_has_failure': any(r.get('error') for r in pair.values())})
    values = [statistics.mean(v) for v in groups.values()]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(1000)) if values else []
    return {'paired_answerable_trials': len(changes), 'intent_groups': len(values),
            'group_mean_precision_delta': statistics.mean(values) if values else None,
            'bootstrap_95_percent_interval': [draws[24], draws[974]] if len(values) > 1 else None,
            'improvements': [x for x in changes if x['interest_precision_delta'] > 0],
            'regressions': [x for x in changes if x['interest_precision_delta'] < 0],
            'no_numeric_improvement': [x for x in changes if x['interest_precision_delta'] == 0]}


def grade_final(state, case):
    # Read finalized timeline, not Planner notes or draft route.
    names = [item['name'] for day in (state.final_plan or {}).get('days', [])
             for item in day.get('timeline', []) if item.get('type') == 'attraction']
    unique = set(names)
    expected = set(case['acceptable_pois'])
    final_route = [{'day': d.get('day'), 'spots': [i for i in d.get('timeline', []) if i.get('type') == 'attraction']}
                   for d in (state.final_plan or {}).get('days', [])]
    scored = state.model_copy(update={'route': final_route})
    code = grade_code(scored, case)
    hard_flags = nodes._route_risk_flags(scored)
    return {'selected_pois': names,
            'interest_recall': len(unique & expected) / len(expected) if expected else None,
            'interest_precision': len(unique & expected) / len(unique) if expected and unique else (0 if expected else None),
            'forbidden_selected': sorted(unique & set(case['forbidden_pois'])),
            'hard_constraint_pass': bool(names) and code['objective_pass'] and len(names) == len(unique),
            'hard_constraint_verification_complete': bool(names) and not hard_flags,
            'hard_flags': hard_flags, 'code_graders': code,
            'rework_count': max(0, state.review_round - 1),
            'unknown_interest_requires_manual_refusal_check': not case['answerable'],
            'selection_reasons': (state.final_plan or {}).get('selection_reasons', []),
            'fact_support_status': 'manual_check_required',
            'annotation_status': case['annotation_status']}


def build_final_route_graph(model, lookup, tips_done=None):
    """Production joint-planning tail, with upstream inputs fixed for pairing.

    Meals/tips branches remain production nodes; their external providers are
    replayed or explicitly unavailable by run_final_route's isolated context.
    """
    tips_node = nodes.make_spot_tips_node(model)
    def measured_tips(state):
        try:
            return tips_node(state)
        finally:
            if tips_done is not None: tips_done.set()
    g = StateGraph(TravelPlanState)
    for name, node in [
        ('planner', nodes.make_joint_planner_node(model, knowledge_lookup=lookup)),
        ('route_distance_check', nodes.route_distance_check_node),
        ('risk_gate', nodes.route_risk_gate_node),
        ('reviewer', nodes.make_reviewer_node(model)),
        ('time_check', nodes.make_time_check_node(model)),
        ('main_meal_output', nodes.main_meal_output_node),
        ('spot_tips', bounded_spot_tips(measured_tips)),
        ('finalize', nodes.make_finalize_node()),
    ]: g.add_node(name, node)
    g.add_edge(START, 'planner')
    g.add_conditional_edges('planner', nodes.route_after_planner,
                           {'reviewer': 'route_distance_check', 'time_check': 'route_distance_check'})
    g.add_edge('route_distance_check', 'risk_gate')
    targets = {'reviewer': 'reviewer', 'time_check': 'time_check', 'meal_search': 'main_meal_output', 'spot_tips': 'spot_tips'}
    g.add_conditional_edges('risk_gate', nodes.route_after_risk_gate, targets)
    g.add_conditional_edges('reviewer', nodes.route_after_review, {'planner': 'planner', 'time_check': 'time_check'})
    g.add_conditional_edges('time_check', nodes.route_after_time_check,
                           {**targets, 'planner': 'planner'})
    g.add_edge(['main_meal_output', 'spot_tips'], 'finalize')
    g.add_edge('finalize', END)
    return g.compile()


def route_key(origin, destination, mode):
    return json.dumps([origin, destination, mode], sort_keys=True)


def run_final_route(case, model, lookup, factory, route_replay):
    """Sequential-only isolation: patch external provider seams, never live maps."""
    from threading import Event
    tips_done = Event()
    misses = []
    retrieval = {'failed': False, 'attempts': 0, 'sources': []}
    def recorded_lookup(*args):
        retrieval['attempts'] += 1
        try:
            result = lookup(*args)
            retrieval['sources'] = [r['source'] for r in result]
            return result
        except Exception:
            retrieval['failed'] = True
            raise
    def distance(origin, destination, api_key, mode='drive'):
        key = route_key(origin, destination, mode)
        if key not in route_replay: misses.append(key)
        return route_replay.get(key)
    with ExitStack() as stack:
        for name, replacement in {
            'build_structured_llm': factory,
            'amap_key': lambda: 'replay-only',
            'plan_route_distance': distance,
            'plan_transport_leg': lambda *a, **kw: distance(kw.get('origin', a[0] if a else None), kw.get('destination', a[1] if len(a)>1 else None), 'replay-only'),
            'recommend_chain_hotel': lambda *a, **kw: None,
            'lookup_live_ticket_prices': lambda *a, **kw: {},
        }.items(): stack.enter_context(patch.object(nodes, name, replacement))
        state = TravelPlanState(**{k: v for k, v in case.items() if k in TravelPlanState.model_fields},
                                max_review_rounds=3, max_time_check_rounds=3)
        output = build_final_route_graph(model, recorded_lookup, tips_done).invoke(state, {'recursion_limit': 60})
        # Production still discards late optional output, but experiments must
        # settle its bill before leaving replay patches or starting another arm.
        if not tips_done.wait(timeout=150): raise RuntimeError('optional worker accounting did not settle')
        final = TravelPlanState(**output)
    return final, {'map_misses': misses, 'retrieval': retrieval}
