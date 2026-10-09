"""Paired current joint-Planner -> final itinerary evaluation. Dry-run by default."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from threading import RLock
from contextlib import contextmanager
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.evaluation.rag_protocol import fingerprint
from app.evaluation.route_interest import validate_assets, freeze_assets, scoped_lookup, run_final_route, grade_final, summarize, paired_diagnostics
from scripts.evaluate_rag_generation import CallBudget
from app.planning.schemas import TravelPlanState


@contextmanager
def exclusive_ledger(path):
    """Prevent two experiment processes from spending the same ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix('.lock')
    try:
        handle = lock_path.open('x', encoding='utf-8')
    except FileExistsError as exc:
        raise RuntimeError('experiment ledger locked; reconcile interrupted runs before removing lock') from exc
    try:
        handle.write(str(os.getpid())); handle.flush()
        yield
    finally:
        handle.close()
        lock_path.unlink()


def implementation_fingerprint():
    files = ['app/planning/nodes.py', 'app/planning/schemas.py', 'app/planning/helpers.py',
             'app/planning/enrichment.py', 'app/evaluation/route_interest.py',
             'app/evaluation/rag_retrieval.py', 'scripts/compare_route_interest.py',
             'app/planning/graph.py', 'app/llm/grok.py', 'app/llm/factory.py']
    return fingerprint({p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})


def binding(cases, facts, maps, config):
    for c in cases:
        for origin in c['pois']:
            if not origin.get('location'): raise ValueError('frozen candidates require verified coordinates')
            for destination in c['pois']:
                if origin['name'] == destination['name']: continue
                from app.evaluation.route_interest import route_key
                if route_key(origin['location'], destination.get('location'), 'drive') not in maps.get('routes', {}):
                    raise ValueError('frozen comparison requires complete directed road replay')
    return {**freeze_assets(cases, facts, config), 'maps': fingerprint(maps),
            'implementation': implementation_fingerprint()}


def verify_billing(path, model, url):
    if not path or not path.exists(): raise ValueError('verified account-specific CNY billing rates required; multiplier unknown')
    b = json.loads(path.read_text(encoding='utf-8'))
    if b.get('model') != model or b.get('url') != url or b.get('currency') != 'CNY' or not b.get('verified_at') or not b.get('evidence'):
        raise ValueError('billing evidence does not bind selected provider/model')
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(b['verified_at'])).total_seconds()
    if not 0 <= age <= 86400: raise ValueError('billing verification must be from the last 24 hours')
    if b.get('key_fingerprint') != hashlib.sha256(os.getenv('GROK_API_KEY', '').encode()).hexdigest():
        raise ValueError('billing evidence belongs to another API account/key')
    for k in ('input_per_million', 'output_per_million', 'account_multiplier', 'quota_units_per_CNY'):
        if type(b.get(k)) not in (float, int) or not math.isfinite(b[k]) or b[k] <= 0:
            raise ValueError('invalid billing rate or account multiplier')
    return b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=ROOT/'knowledge/route_interest_v1')
    p.add_argument('--split', choices=('dev', 'test'), default='dev')
    p.add_argument('--limit', type=int, default=5)
    p.add_argument('--repetitions', type=int, default=1)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--retriever', choices=('keyword', 'bm25', 'vector', 'hybrid', 'keyword_rrf'), default='hybrid')
    p.add_argument('--retrieval-only', action='store_true')
    p.add_argument('--model', default='grok-4.6')
    p.add_argument('--billing', type=Path)
    p.add_argument('--freeze', type=Path)
    p.add_argument('--write-freeze', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--out', type=Path, default=ROOT/'data/route_interest/pilot.json')
    p.add_argument('--ledger', type=Path, default=ROOT/'data/route_interest/budget_ledger.json')
    a = p.parse_args(); load_local_env()
    if a.execute and a.out.exists():
        previous = json.loads(a.out.read_text(encoding='utf-8'))
        if previous.get('calls') or previous.get('results'): p.error('existing paid report must not be overwritten; use another output and the same ledger')
    if not 1 <= a.limit <= 15 or not 1 <= a.repetitions <= 3 or not 1 <= a.k <= 5: p.error('limit 1..15, repeats 1..3, k 1..5')
    cases = json.loads((a.assets/'cases.json').read_text(encoding='utf-8'))
    facts = json.loads((a.assets/'facts.json').read_text(encoding='utf-8'))
    errors = validate_assets(cases, facts)
    if errors: p.error('; '.join(errors))
    maps = json.loads((a.assets/'maps.json').read_text(encoding='utf-8')) if (a.assets/'maps.json').exists() else {'routes': {}}
    url = os.getenv('GROK_BASE_URL', 'https://api.x.ai/v1').rstrip('/')
    from app.evaluation.rag_retrieval import embedding_config, lexical_config
    config = {'model': a.model, 'url': url, 'k': a.k, 'retriever': a.retriever, 'embedding': embedding_config(), 'lexical': lexical_config(),
              'max_review_rounds': 3, 'max_time_check_rounds': 3, 'max_output_tokens': 2048,
              'temperature': .3, 'reasoning_effort': os.getenv('GROK_REASONING_EFFORT', 'low'),
              'sdk_max_retries': 0,
              'responses_api': os.getenv('GROK_USE_RESPONSES_API', '1'),
              'upstream': 'fixed_verified_candidates_weather_and_drive_replay'}
    if a.write_freeze:
        if not a.freeze: p.error('--freeze path required')
        from scripts.review_route_interest import validate_snapshots, validate_manifest
        evidence_errors = validate_snapshots(a.assets, facts)+validate_manifest(a.assets, cases, facts)
        if evidence_errors: p.error('; '.join(evidence_errors))
        value = binding(cases, facts, maps, config)
        a.freeze.parent.mkdir(parents=True, exist_ok=True)
        a.freeze.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8'); return
    if a.split == 'test':
        if not a.freeze: p.error('test split requires human-reviewed frozen assets/configuration')
        if json.loads(a.freeze.read_text(encoding='utf-8')) != binding(cases, facts, maps, config): p.error('freeze mismatch')
    # Round-robin cities: the first five requests must not all be Nanjing.
    groups = [[c for c in cases if c['split'] == a.split and c['destination'] == city] for city in ('南京', '上海', '三亚')]
    selected = [c for row in zip(*groups) for c in row][:a.limit]
    expected = len(selected)*2*a.repetitions
    report = {'status': 'dry_run', 'configuration': config, 'cases': fingerprint(cases), 'facts': fingerprint(facts),
              'maps': fingerprint(maps), 'implementation': implementation_fingerprint(),
              'preflight': {'requests': len(selected), 'initial_plans': expected, 'model_call_limit': 540,
                            'CNY_limit': 10, 'cost_estimate': None, 'billing_status': 'unverified'},
              'results': [], 'calls': [], 'label_status': 'exploratory_only' if a.split == 'dev' else 'human_frozen',
              'production_default_changed': False}
    def save():
        report['summary'] = {arm: summarize([r for r in report['results'] if r['arm'] == arm], expected//2)
                             for arm in ('without_knowledge', 'with_knowledge')}
        report['paired_diagnostics'] = paired_diagnostics(report['results'], cases)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    save()
    if a.retrieval_only:
        report['retrieval_diagnostics'] = []
        lookup = scoped_lookup(facts, a.k, a.retriever)
        for case in selected:
            row = {'case_id': case['id']}
            try:
                rows = lookup(case['destination'], case['pois'], case['query'])
                row['sources'] = [r['source'] for r in rows]
                pool_names = {p['name'] for p in case['pois']}
                # Facts may carry aliases (for example, a scenic area's short
                # name). Score only canonical names present in this request's
                # candidate pool; aliases must not inflate retrieval quality.
                row['candidate_entities'] = sorted({n for r in rows for n in r['entities'] if n in pool_names})
                expected_pois = set(case['acceptable_pois'])
                selected_pois = set(row['candidate_entities'])
                row['candidate_precision'] = len(selected_pois & expected_pois) / len(selected_pois) if selected_pois else 0.0
                row['candidate_recall'] = len(selected_pois & expected_pois) / len(expected_pois) if expected_pois else None
                row['candidate_false_positive'] = sorted(selected_pois - expected_pois)
                row['candidate_missed'] = sorted(expected_pois - selected_pois)
            except Exception as exc: row['error'] = type(exc).__name__+': '+str(exc)[:240]
            report['retrieval_diagnostics'].append(row)
        by_id = {c['id']: c for c in selected}
        answerable = [r for r in report['retrieval_diagnostics'] if by_id[r['case_id']]['answerable']]
        negatives = [r for r in report['retrieval_diagnostics'] if not by_id[r['case_id']]['answerable']]
        report['retrieval_summary'] = {
            'status': 'exploratory_retrieval_only_not_final_route_quality',
            'answerable_cases': len(answerable),
            'answerable_candidate_hit': sum(bool(set(r.get('candidate_entities', [])) & set(by_id[r['case_id']]['acceptable_pois'])) for r in answerable),
            'answerable_candidate_hit_rate': (
                sum(bool(set(r.get('candidate_entities', [])) & set(by_id[r['case_id']]['acceptable_pois'])) for r in answerable) / len(answerable)
                if answerable else None
            ),
            'answerable_candidate_precision_mean': (
                sum(r.get('candidate_precision', 0.0) for r in answerable) / len(answerable)
                if answerable else None
            ),
            'answerable_candidate_recall_mean': (
                sum(r.get('candidate_recall', 0.0) for r in answerable) / len(answerable)
                if answerable else None
            ),
            'answerable_missed_cases': [r['case_id'] for r in answerable if r.get('candidate_missed')],
            'answerable_false_positive_cases': [r['case_id'] for r in answerable if r.get('candidate_false_positive')],
            'unanswerable_cases': len(negatives),
            'unanswerable_nonempty_rate': sum(bool(r.get('candidate_entities')) for r in negatives) / len(negatives) if negatives else None,
            'errors': sum(bool(r.get('error')) for r in report['retrieval_diagnostics']),
        }
        report['status'] = 'retrieval_diagnostics_only_not_final_route_evidence'; save(); return
    if not a.execute: print(json.dumps(report['preflight'])); return
    try:
        billing = verify_billing(a.billing, a.model, url)
        # Gateway must expose functioning key-wide usage monitoring before spending.
        def monitor():
            import httpx
            if url != 'https://www.micuapi.ai/v1': raise ValueError('provider billing monitor not implemented for this endpoint')
            response = httpx.get(url.removesuffix('/v1')+'/api/usage/token/', headers={'Authorization': 'Bearer '+os.environ['GROK_API_KEY']}, timeout=15)
            response.raise_for_status()
            value = response.json().get('data', {}).get('total_used')
            if type(value) is not int: raise ValueError('budget monitor missing usage')
            return value
        initial_quota = monitor()
    except Exception as exc:
        report['status'] = 'blocked_before_paid_calls'; report['stop_reason'] = type(exc).__name__+': '+str(exc)[:240]
        save(); print(report['status']); return
    input_price = billing['input_per_million']*billing['account_multiplier']
    output_price = billing['output_per_million']*billing['account_multiplier']
    report['preflight'].update(billing_status='verified', billing=billing,
                              cost_estimate={'worst_output_CNY': 540*2048*output_price/1e6,
                                             'input': 'UTF-8 byte reservation per actual prompt', 'ceiling_CNY': 10})
    budget = CallBudget(540, 10, input_price, output_price, 2048)
    ledger = {'initial_plans': 0, 'budget': budget.snapshot(), 'in_flight': False, 'trials': {},
              'billing_fingerprint': fingerprint(billing)}
    if a.ledger.exists():
        ledger = json.loads(a.ledger.read_text(encoding='utf-8'))
        if ledger['billing_fingerprint'] != fingerprint(billing) or ledger['in_flight']:
            raise RuntimeError('ledger billing changed or unresolved paid call; reconcile before continuing')
        old = ledger['budget']
        budget.calls = old['attempted_calls']; budget.committed_cost = old['committed_cost']
        budget.measured_cost = old['measured_cost_known_usage_only']; budget.unknown_usage_calls = old['unknown_usage_calls']
        if budget.unknown_usage_calls: raise RuntimeError('previous call usage unknown; budget monitoring unresolved')
    lock = RLock()
    halted = []
    def save_ledger():
        ledger['budget'] = budget.snapshot()
        a.ledger.parent.mkdir(parents=True, exist_ok=True)
        temporary = a.ledger.with_suffix('.tmp')
        temporary.write_text(json.dumps(ledger, indent=2)+'\n', encoding='utf-8')
        temporary.replace(a.ledger)
    from app.llm.grok import build_chat_grok, use_grok_responses_api
    # Factory preserves production schema, Responses transport and temperature.
    # All node attempts and helper retries must pass the same durable budget.
    def factory(schema, *, model=None, temperature=0, trial=None, **kw):
        client = build_chat_grok(model=a.model, temperature=temperature)
        client.max_tokens = 2048
        # SDK retries would bypass per-attempt ledger reservations.
        client.max_retries = 0
        method = 'json_schema' if use_grok_responses_api() else 'function_calling'
        structured = client.with_structured_output(schema, method=method, strict=True, include_raw=True)
        class Metered:
            def invoke(self, messages):
                with lock:
                    return self._invoke(messages)
            def _invoke(self, messages):
                if halted: raise RuntimeError('budget monitor halted: '+halted[0])
                try:
                    used = monitor()-initial_quota
                    margin = (sum(len(t.encode('utf-8')) for _, t in messages)+256)*input_price/1e6 + 2048*output_price/1e6
                    if used/billing['quota_units_per_CNY'] + margin > 10: raise RuntimeError('key-wide budget headroom exhausted')
                except Exception:
                    halted.append('pre-call monitor failure'); raise
                reservation = budget.reserve(messages)
                entry = {'task': kw.get('task_type'), 'trial': trial, 'status': 'in_flight', 'usage': None}
                ledger['in_flight'] = True; save_ledger()
                report['calls'].append(entry); report['budget'] = budget.snapshot(); save()
                started = time.perf_counter()
                try:
                    response = structured.invoke(messages)
                    usage = response['raw'].usage_metadata
                    budget.reconcile(reservation, usage)
                    entry.update(status='completed', usage=usage, elapsed_ms=(time.perf_counter()-started)*1000)
                    if budget.unknown_usage_calls: raise RuntimeError('missing usage; further calls prohibited')
                    after = monitor()-initial_quota
                    if after/billing['quota_units_per_CNY'] > 10: raise RuntimeError('key-wide cost limit exceeded')
                    if response.get('parsing_error'): raise ValueError('structured output parse failure')
                    return response['parsed']
                except Exception as exc:
                    entry.update(status='failed', error=type(exc).__name__)
                    halted.append(type(exc).__name__)
                    if not entry.get('usage'): budget.unknown_usage_calls += 1
                    raise
                finally:
                    ledger['in_flight'] = False; save_ledger()
                    report['budget'] = budget.snapshot(); save()
        return Metered()
    if (a.limit > 5 or a.repetitions > 1 or a.split == 'test') and ledger.get('verified_pilot_config') != fingerprint(config):
        report['status'] = 'blocked_before_paid_calls'
        report['stop_reason'] = 'complete the five-request dev pilot with this configuration first'
        save(); return
    report['status'] = 'running'; save()
    for repetition in range(a.repetitions):
        for index, case in enumerate(selected):
            arms = ['without_knowledge', 'with_knowledge']
            if (index+repetition)%2: arms.reverse()
            for arm in arms:
                trial = f"{case['id']}/{arm}/{repetition}"
                execution_key = fingerprint({'case_input': {k: v for k, v in case.items() if k in TravelPlanState.model_fields},
                    'arm': arm, 'repetition': repetition, 'facts': [{k: v for k, v in f.items() if k not in ('review_status', 'annotator', 'reviewed_at', 'change_log')} for f in facts],
                    'config': config, 'maps': fingerprint(maps), 'implementation': report['implementation']})
                if execution_key in ledger.get('trials', {}):
                    cached = ledger['trials'][execution_key]
                    row = {**cached, 'reused_paid_trial': True}
                    if cached.get('final_state'):
                        row['metrics'] = grade_final(TravelPlanState(**cached['final_state']), case)
                    report['results'].append(row); save(); continue
                if ledger['initial_plans'] >= 180 or halted:
                    report['status'] = 'budget_stopped'; save(); return
                ledger['initial_plans'] += 1; save_ledger()
                row = {'case_id': case['id'], 'arm': arm, 'repetition': repetition, 'answerable': case['answerable'], 'human_review': 'pending'}
                started = time.perf_counter()
                try:
                    lookup = scoped_lookup(facts, a.k, a.retriever) if arm == 'with_knowledge' else lambda *a: []
                    from functools import partial
                    state, diagnostics = run_final_route(case, a.model, lookup, partial(factory, trial=trial), maps['routes'])
                    row['planning_elapsed_ms'] = (time.perf_counter()-started)*1000
                    # A timed-out optional worker may still hold a paid request.
                    # Drain its metering before another trial to avoid budget races.
                    with lock: pass
                    actual_calls = [c for c in report['calls'] if c['trial'] == trial]
                    row['actual_model_calls'] = len(actual_calls)
                    row['actual_usage'] = {k: sum((c.get('usage') or {}).get(k, 0) for c in actual_calls) for k in ('input_tokens', 'output_tokens')}
                    row['actual_usage_complete'] = all(c.get('usage') for c in actual_calls)
                    row.update(metrics=grade_final(state, case), final_plan=state.final_plan, diagnostics=diagnostics)
                    row['final_state'] = state.model_dump(mode='json')
                except Exception as exc:
                    row['error'] = type(exc).__name__+': '+str(exc)[:240]
                actual_calls = [c for c in report['calls'] if c['trial'] == trial]
                row['actual_model_calls'] = len(actual_calls)
                row['actual_usage'] = {k: sum((c.get('usage') or {}).get(k, 0) for c in actual_calls) for k in ('input_tokens', 'output_tokens')}
                row['actual_usage_complete'] = bool(actual_calls) and all(c.get('usage') for c in actual_calls)
                row['elapsed_ms'] = (time.perf_counter()-started)*1000
                ledger.setdefault('trials', {})[execution_key] = row; save_ledger()
                report['results'].append(row); save()
                if row.get('error') or budget.unknown_usage_calls or halted:
                    report['status'] = 'stopped_on_failure'; save(); return
    report['status'] = 'exploratory_complete' if a.split == 'dev' else 'frozen_test_complete'; save()
    if a.split == 'dev' and a.limit == 5 and a.repetitions == 1:
        ledger['verified_pilot_config'] = fingerprint(config); save_ledger()


if __name__ == '__main__':
    if '--execute' in sys.argv:
        lock_parser = argparse.ArgumentParser(add_help=False)
        lock_parser.add_argument('--ledger', type=Path, default=ROOT/'data/route_interest/budget_ledger.json')
        lock_args, _ = lock_parser.parse_known_args()
        ledger_path = lock_args.ledger
        with exclusive_ledger(ledger_path): main()
    else:
        main()
