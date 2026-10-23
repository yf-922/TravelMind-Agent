"""Bounded real-model soft-constraint challenges; old natural results stay immutable."""
from __future__ import annotations

import argparse
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
from app.evaluation.task_robustness import (build_soft_review_cases, load_assets, fault_distance,
                                           semantic_route_violations, grade_soft_review)
from app.planning import nodes
from app.planning.schemas import TravelPlanState
from scripts.evaluate_reviewer_ablation import ARMS, run_arm, rule_faults
from scripts.compare_route_interest import exclusive_ledger, read_gateway_usage
from app.evaluation.rag_protocol import fingerprint


def preflight(cases, maps):
    output = []
    for case in cases:
        state = TravelPlanState(**case['state']).model_copy(update={'route': case['injected_route']})
        with patch.object(nodes, 'amap_key', return_value='replay'), patch.object(
            nodes, 'plan_route_distance', side_effect=fault_distance(maps, None, {})
        ):
            state = state.model_copy(update=nodes.route_distance_check_node(state))
        faults = rule_faults(state)
        soft = semantic_route_violations(state.route, case, maps)
        expected_bad = case['category'] != 'clean'
        if faults or bool(soft) != expected_bad:
            raise ValueError(f"invalid soft-only challenge {case['id']}: hard={faults}, soft={soft}")
        output.append({'id': case['id'], 'hard_violations': faults, 'soft_violations': soft})
    return output


def summarize(report, legacy):
    replaced = {c['replaces'] for c in report['cases'] if c.get('replaces')}
    new_ids = {c['id'] for c in report['cases'] if c.get('replaces')}
    summary = {}
    for arm in ARMS:
        old = [r for r in legacy['rows'] if r['suite'] == 'task' and r['arm'] == arm and r['id'] not in replaced]
        new = [r for r in report['rows'] if r['arm'] == arm and r['id'] in new_ids]
        combined = old + new
        feasible = [r for r in new if r['id'] != 'soft-review-infeasible']
        summary[arm] = {'legacy_reused': len(old), 'new_recorded': len(new), 'planned_composite': 30,
                        'composite_task_success': sum(r.get('grade', {}).get('task_success', False) for r in combined),
                        'new_feasible_success': sum(r.get('grade', {}).get('task_success', False) for r in feasible),
                        'new_feasible_count': 4,
                        'new_expected_behavior_pass': sum(r.get('grade', {}).get('expected_behavior_pass', False) for r in new),
                        'new_errors': sum(bool(r.get('error')) for r in new)}
    return summary


def review_packet(report):
    lines = ['# Soft Reviewer Challenge: Review Packet', '',
             f"Status: {report['status']}; labels pending human review.",
             'Four feasible semantic defects, one infeasible control, one clean control. '
             'Drafts were deliberately constructed, not naturally generated Planner failures.', '',
             '## Historical + challenge composite (exploratory only)', '',
             '| Arm | Reused natural tasks | New tasks recorded | Success / 30 | Feasible semantic repairs / 4 |',
             '|---|---:|---:|---:|---:|']
    for arm, s in report['summary'].items():
        lines.append(f"| {arm} | {s['legacy_reused']} | {s['new_recorded']} | {s['composite_task_success']} | {s['new_feasible_success']} |")
    lines += ['', 'The composite contains 25 historical natural drafts + 5 controlled challenges; '
              'one challenge is infeasible, so task success and safe refusal are reported separately. '
              'It is not the original 30-request natural success rate. No target percentage is enforced.', '']
    for case in report['cases']:
        lines += [f"## {case['id']}", '', case['state']['query'], '',
                  f"Expected: {case['expected']}; label: {case['label_status']}",
                  'Initial route: '+json.dumps(case['injected_route'], ensure_ascii=False),
                  'Independent challenge rubric: '+json.dumps(case['semantic_rubric'], ensure_ascii=False), '',
                  '| Arm | Final POIs and times | Approved | Hard violations | Soft violations | Repairs | Error |',
                  '|---|---|---|---|---|---:|---|']
        for row in report['rows']:
            if row['id'] != case['id']:
                continue
            selected = [(s.get('name'), s.get('start_time'), s.get('end_time'))
                        for d in (row.get('final_plan') or {}).get('days', []) for s in d.get('timeline', [])
                        if s.get('type') == 'attraction']
            g = row.get('grade', {})
            lines.append(f"| {row['arm']} | {json.dumps(selected, ensure_ascii=False)} | {g.get('approved')} | {g.get('hard_violations')} | {g.get('soft_violations')} | {row.get('repairs')} | {row.get('error', '')} |")
            if row.get('final_plan', {}).get('route_issues'):
                lines.append('\nUser message: '+json.dumps(row['final_plan']['route_issues'], ensure_ascii=False)+'\n')
    usage = {key: sum((c.get('usage') or {}).get(key, 0) for c in report['calls'])
             for key in ('input_tokens', 'output_tokens', 'total_tokens')}
    lines += ['', '## Usage', '', f"Paid attempts: {len(report['calls'])} / {report['max_calls']}; provider usage: {usage}; cost unpriced.",
              '', 'Confirm each initial semantic defect, the acceptability of repaired POIs, '
              'and the infeasible/clean controls before marking verified_v2. '
              'This packet is not a human-verification record.']
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--max-calls', type=int, default=42)
    p.add_argument('--legacy', type=Path, default=ROOT/'data/eval/reviewer_ablation_real_v2.json')
    p.add_argument('--out', type=Path, default=ROOT/'data/eval/soft_review_v3.json')
    args = p.parse_args()
    if not 1 <= args.max_calls <= 42:
        p.error('approved cap is 42 attempts')
    load_local_env()
    legacy = json.loads(args.legacy.read_text(encoding='utf-8'))
    cases = build_soft_review_cases()
    _, maps = load_assets()
    checks = preflight(cases, maps)
    model = legacy['model']
    binding = fingerprint({'cases': cases, 'maps': maps, 'model': model,
                            'nodes': hashlib.sha256(Path(nodes.__file__).read_bytes()).hexdigest(),
                            'script': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    report = {'binding': binding, 'model': model, 'cases': cases, 'rows': [], 'calls': [],
              'max_calls': args.max_calls, 'preflight': checks, 'status': 'dry_run',
              'label_status': 'pending_human_review', 'legacy_sha256': hashlib.sha256(args.legacy.read_bytes()).hexdigest(),
              'cost_status': 'unpriced', 'draft_origin': 'author_controlled_no_model_seed_call',
              'max_repairs': 3, 'output_token_cap': 4096}
    def save():
        report['summary'] = summarize(report, legacy)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(args.out)
        (args.out.parent/'soft_review_v3_review.md').write_text(review_packet(report), encoding='utf-8')
    print('6 cases; 18 arm outcomes; <=42 paid attempts; max output 4096/call; cost unpriced; no paid retries', flush=True)
    if not args.execute:
        save()
        return
    with exclusive_ledger(args.out):
        if args.out.exists():
            old = json.loads(args.out.read_text(encoding='utf-8'))
            if old['binding'] != binding:
                raise ValueError('binding changed: use a new output ledger')
            if any(c['status'] == 'in_flight' for c in old['calls']):
                raise RuntimeError('unsettled call: reconcile before resume')
            report = old
        report['usage_monitor_before'] = read_gateway_usage(os.environ['GROK_BASE_URL'].rstrip('/'), os.environ['GROK_API_KEY'])
        current = {}
        def factory(schema, **kw):
            from app.llm.grok import build_chat_grok, use_grok_responses_api, parse_structured_response
            client = build_chat_grok(model=model, temperature=kw.get('temperature', 0))
            client.max_retries, client.max_tokens = 0, 4096
            runnable = client.with_structured_output(schema, method='json_schema' if use_grok_responses_api() else 'function_calling', strict=True, include_raw=True)
            class Metered:
                def invoke(self, messages):
                    if len(report['calls']) >= args.max_calls:
                        raise RuntimeError('call_cap')
                    call = {**current, 'task': kw.get('task_type'), 'status': 'in_flight',
                            'messages': messages}
                    report['calls'].append(call)
                    save()
                    started = time.perf_counter()
                    try:
                        raw = runnable.invoke(messages)
                        call['usage'] = getattr(raw.get('raw'), 'usage_metadata', None)
                        if not call['usage']:
                            raise RuntimeError('usage_missing')
                        parsed = parse_structured_response(raw, schema)
                        call.update(status='completed', output=parsed.model_dump(mode='json'))
                        return parsed
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
                               'recommend_chain_hotel': lambda *a, **kw: None,
                               'lookup_live_ticket_prices': lambda *a, **kw: {}}.items():
                stack.enter_context(patch.object(nodes, key, value))
            planner = nodes.make_joint_planner_node(model, knowledge_lookup=lambda *a: [])
            reviewer = nodes.make_reviewer_node(model)
            report['status'] = 'running'
            save()
            for case in cases:
                for arm in ARMS:
                    if any(r['id'] == case['id'] and r['arm'] == arm for r in report['rows']):
                        continue
                    current.update(case=case['id'], arm=arm)
                    row = {'id': case['id'], 'arm': arm, 'trace': []}
                    before = len(report['calls'])
                    started = time.perf_counter()
                    try:
                        initial = {**copy.deepcopy(case['state']), 'route': copy.deepcopy(case['injected_route'])}
                        with patch.object(nodes, 'plan_route_distance', side_effect=fault_distance(maps, None, {})):
                            state, repairs = run_arm(initial, arm, planner, reviewer, trace=row['trace'])
                        row.update(final_plan=state.final_plan, repairs=repairs,
                                   grade=grade_soft_review(state, case, maps))
                    except Exception as exc:
                        row['error'] = type(exc).__name__
                    row.update(call_indices=list(range(before, len(report['calls']))),
                               elapsed_ms=(time.perf_counter()-started)*1000)
                    report['rows'].append(row)
                    save()
                    print(case['id'], arm, row.get('grade', {}).get('task_success'), row.get('error', ''), flush=True)
                    if any(c.get('error') in ('RuntimeError',) for c in report['calls']):
                        report['status'] = 'stopped_budget_or_usage'
                        save()
                        return
                report['usage_monitor_latest'] = read_gateway_usage(os.environ['GROK_BASE_URL'].rstrip('/'), os.environ['GROK_API_KEY'])
            report['status'] = 'complete_pending_human_review'
            save()


if __name__ == '__main__':
    main()
