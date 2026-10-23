"""Paired real-model Reviewer ablation with shared Planner drafts and bounded repair."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.evaluation.task_robustness import build_cases, load_assets, fault_distance, grade
from app.evaluation.rag_protocol import fingerprint
from app.planning import nodes
from app.planning.schemas import TravelPlanState
from scripts.compare_route_interest import exclusive_ledger, implementation_fingerprint

ARMS = ('planner_only', 'rules', 'rules_reviewer')
ADVISORY = {'opening_time_unknown', 'user_modification', 'long_road_leg'}


def cases_v2():
    assets = build_cases()
    cases = assets['tasks'] + assets['robustness']
    for case in cases:
        case['ablation_asset_version'] = 'v2_corrected_before_execution'
        rubric = case['rubric']
        if case['category'] == 'map':
            case['state']['query'] += '必须安排至少两个不同景点，保留足够通勤时间。'
        if rubric.get('acceptable') and set(rubric['acceptable']) <= set(rubric.get('excluded', [])):
            alternative = next(p['name'] for p in case['state']['pois'] if p['name'] not in rubric['excluded'])
            old = rubric['acceptable'][0]
            case['state']['query'] = case['state']['query'].replace('想参观' + old, '想参观' + alternative)
            case['state']['attraction_preference'] = case['state']['query']
            rubric['acceptable'] = [alternative]
    return cases


def rule_faults(state):
    # Application rules only: never read the evaluator's required/acceptable sets.
    return sorted(set(nodes._route_risk_flags(state)) - ADVISORY)


def run_arm(initial, arm, planner, reviewer, max_repairs=3, trace=None):
    state = TravelPlanState(**initial)
    repairs = 0
    while True:
        state = state.model_copy(update=nodes.route_distance_check_node(state))
        faults = rule_faults(state)
        if trace is not None:
            with patch.object(nodes, 'record_risk_gate'):
                gate = nodes.route_risk_gate_node(state)
            trace.append({'repair_index': repairs, 'state_before_review': state.model_dump(mode='json'),
                          'rule_faults': faults, 'current_gate': gate})
        if arm == 'planner_only':
            state.approved = True
            break
        state.approved = not faults
        if arm == 'rules_reviewer':
            state = state.model_copy(update=reviewer(state))
            state.approved = state.approved and not faults
        if state.approved or repairs >= max_repairs or not state.route:
            break
        feedback = []
        if faults: feedback.append('确定性规则未通过：' + ', '.join(faults))
        if arm == 'rules_reviewer' and state.route_modify_opinion: feedback.append(state.route_modify_opinion)
        state = state.model_copy(update={'route_modify_opinion': '\n'.join(feedback),
            'approved': False, 'reviewer_issues': [], 'time_violations': [],
            'route_distance_legs': [], 'time_check_done': False})
        state = state.model_copy(update=planner(state))
        repairs += 1
    if arm == 'planner_only':
        # Score the unaudited draft itself. Production Finalize intentionally
        # suppresses rejected itineraries and must not alter this baseline.
        state.final_plan = {
            'approved': True, 'route_issues': [],
            'days': [{'day': day.get('day'), 'timeline': [
                {**copy.deepcopy(spot), 'type': 'attraction'}
                for spot in day.get('spots', [])]} for day in state.route],
        }
    else:
        state.final_plan = nodes.finalize_node(state)['final_plan']
    state.final_plan['ablation_arm'] = arm
    return state, repairs


def summaries(report):
    output = {}
    for arm in ARMS:
        output[arm] = {}
        for suite in ('task', 'robustness'):
            planned = [c for c in report['cases'] if c['suite'] == suite]
            rows = [r for r in report['rows'] if r['arm'] == arm and r['suite'] == suite]
            output[arm][suite] = {'planned': len(planned), 'recorded': len(rows),
                'success': sum(r.get('grade', {}).get('task_success', False) for r in rows),
                'safe_rejections': sum(r.get('grade', {}).get('safe_explained_rejection', False) for r in rows),
                'resilient': sum(r.get('grade', {}).get('robustness_ok', False) for r in rows),
                'unsafe_release': sum(r.get('grade', {}).get('unsafe_release', False) for r in rows),
                'errors': sum(bool(r.get('error')) for r in rows)}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--max-calls', type=int, default=1000)
    p.add_argument('--out', type=Path, default=ROOT/'data/eval/reviewer_ablation_real_v2.json')
    args = p.parse_args()
    load_local_env()
    cases = cases_v2()
    if args.limit: cases = cases[:args.limit]
    _, maps = load_assets()
    model = os.getenv('GROK_MODEL', 'grok-4.6')
    binding = fingerprint({'cases': cases, 'maps': maps, 'implementation': implementation_fingerprint(),
                           'script': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'model': model})
    report = {'binding': binding, 'model': model, 'cases': cases, 'rows': [], 'calls': [],
              'seeds': {}, 'status': 'dry_run', 'cost_status': 'unpriced', 'max_calls': args.max_calls,
              'method': 'shared_joint_planner_draft_paired_audit_tail_max_three_repairs',
              'limits': 'Fixed candidates/maps; no TimeCheck in any arm; author-draft soft rubrics. Not full production graph. Planner-only intentionally bypasses final approval guard. Weather/modification share a neutral initial checkpoint and changed first draft. Shared seed usage charged to each arm logically, paid once.'}
    def save():
        report['summary'] = summaries(report)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report_text = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
        temp = args.out.with_suffix('.tmp')
        temp.write_text(report_text, encoding='utf-8')
        temp.replace(args.out)
    print(f'{len(cases)} cases x 3 arms; max {args.max_calls} paid attempts; cost=unpriced; no hidden retries', flush=True)
    if not args.execute:
        save(); return
    with exclusive_ledger(args.out):
        if args.out.exists():
            old = json.loads(args.out.read_text(encoding='utf-8'))
            if old['binding'] != binding: raise ValueError('binding changed: use a new report')
            if any(c['status'] == 'in_flight' for c in old['calls']): raise RuntimeError('unsettled paid call: reconcile before resuming')
            report = old
        from scripts.compare_route_interest import read_gateway_usage
        read_gateway_usage(os.environ['GROK_BASE_URL'].rstrip('/'), os.environ['GROK_API_KEY'])
        current = {'case': '', 'arm': ''}
        def factory(schema, **kw):
            from app.llm.grok import build_chat_grok, use_grok_responses_api, parse_structured_response
            client = build_chat_grok(model=model, temperature=kw.get('temperature', 0))
            client.max_retries = 0
            client.max_tokens = 4096
            runnable = client.with_structured_output(schema, method='json_schema' if use_grok_responses_api() else 'function_calling', strict=True, include_raw=True)
            class Metered:
                def invoke(self, messages):
                    if len(report['calls']) >= args.max_calls: raise RuntimeError('call_cap')
                    call = {**current, 'task': kw.get('task_type'), 'status': 'in_flight'}
                    report['calls'].append(call); save()
                    started = time.perf_counter()
                    try:
                        raw = runnable.invoke(messages)
                        usage = getattr(raw.get('raw'), 'usage_metadata', None)
                        call.update(status='completed', usage=usage)
                        if not usage: raise RuntimeError('usage_missing')
                        return parse_structured_response(raw, schema)
                    except Exception as exc:
                        call.update(status='failed', error=type(exc).__name__)
                        raise
                    finally:
                        call['elapsed_ms'] = (time.perf_counter()-started)*1000
                        save()
            return Metered()
        with ExitStack() as stack:
            for key, value in {'build_structured_llm': factory,
                'invoke_structured': lambda llm, messages, **kw: llm.invoke(messages),
                'amap_key': lambda: 'replay', 'plan_transport_leg': lambda *a, **kw: None,
                'recommend_chain_hotel': lambda *a, **kw: None, 'lookup_live_ticket_prices': lambda *a, **kw: {}}.items():
                stack.enter_context(patch.object(nodes, key, value))
            planner = nodes.make_joint_planner_node(model, knowledge_lookup=lambda *a: [])
            reviewer = nodes.make_reviewer_node(model)
            report['status'] = 'running'; save()
            for case in cases:
                current.update(case=case['id'], arm='shared_seed')
                if case['id'] not in report['seeds']:
                    before = len(report['calls'])
                    started = time.perf_counter()
                    try:
                        state = TravelPlanState(**case['state'])
                        state = state.model_copy(update=planner(state))
                        if case['category'] in ('weather', 'modification'):
                            state = state.model_copy(update={'modification_notes': case['modification'],
                                'route_modify_opinion': '【用户修改意见】'+case['modification'], 'approved': False,
                                'reviewer_issues': [], 'review_round': 0, 'route_distance_legs': []})
                            if case['category'] == 'weather':
                                state.weather_forecast = case['weather_after']
                                state.weather_note = '天气暂不可用，请复核' if not case['weather_after'] else None
                            state = state.model_copy(update=planner(state))
                        report['seeds'][case['id']] = {'state': state.model_dump(mode='json'),
                            'elapsed_ms': (time.perf_counter()-started)*1000, 'calls': list(range(before, len(report['calls'])))}
                    except Exception as exc:
                        report['seeds'][case['id']] = {'error': type(exc).__name__}
                    save()
                seed = report['seeds'][case['id']]
                # Rotate order to avoid always giving the Reviewer arm later gateway traffic.
                offset = cases.index(case) % 3
                arms = ARMS[offset:] + ARMS[:offset]
                for arm in arms:
                    if any(r['id'] == case['id'] and r['arm'] == arm for r in report['rows']): continue
                    current['arm'] = arm
                    row = {'id': case['id'], 'arm': arm, 'suite': case['suite'], 'category': case['category']}
                    before = len(report['calls']); started = time.perf_counter()
                    try:
                        if seed.get('error'): raise ValueError('shared_seed_failed')
                        counter = {}
                        with patch.object(nodes, 'plan_route_distance', side_effect=fault_distance(maps, case.get('fault') if case['category'] == 'map' else None, counter)):
                            state, repairs = run_arm(seed['state'], arm, planner, reviewer)
                        row.update(grade=grade(state, case, maps), repairs=repairs, final_plan=state.final_plan)
                    except Exception as exc:
                        row['error'] = seed.get('error') or type(exc).__name__
                    row.update(tail_ms=(time.perf_counter()-started)*1000,
                               shared_seed_ms=seed.get('elapsed_ms'), paid_call_indices=list(range(before, len(report['calls']))),
                               shared_call_indices=seed.get('calls', []))
                    report['rows'].append(row); save()
                    print(case['id'], arm, row.get('grade', {}).get('task_success'), row.get('error', ''), flush=True)
                if len(report['calls']) >= args.max_calls or any(c.get('error') == 'RuntimeError' for c in report['calls']):
                    report['status'] = 'stopped_budget_or_usage'; save(); return
            report['status'] = 'complete_pending_human_review'; save()
        print(json.dumps(report['summary'], ensure_ascii=True), flush=True)


if __name__ == '__main__': main()
