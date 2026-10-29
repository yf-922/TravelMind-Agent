"""Bounded hard-case synthesis with per-case rubric audit and durable API ledger."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.evaluation.task_robustness import load_assets, grade
from scripts.compare_route_interest import exclusive_ledger
from scripts.post_train_runtime import Meter, append_json, capture_messages, replay_check, quality_faults
from scripts.curate_post_train_data import counterfactuals


def audit_plan(plan, case, maps):
    errors, checked = replay_check(plan, case['state'], maps)
    score = grade(checked, case, maps)
    errors = sorted(set(errors + quality_faults(checked) + score['hard_violations'] + score['soft_violations']))
    if case.get('id', '').startswith('newcity-'):
        from scripts.prepare_targeted_city_sft import additional_faults
        errors = sorted(set(errors + additional_faults(plan, case)))
    return errors, score, checked


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=ROOT/'data/post_train/hard_supplement_cases.json')
    p.add_argument('--output', type=Path, default=ROOT/'data/post_train/hard_synthesis_ledger.jsonl')
    p.add_argument('--max-calls', type=int, default=90)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--maps', type=Path, help='Saved real directed road responses; never fetch live roads')
    p.add_argument('--limit', type=int, help='Run only this many pending cases, then resume the same immutable run')
    a = p.parse_args()
    load_local_env()
    cases = json.loads(a.input.read_text(encoding='utf-8'))
    if a.maps:
        maps = json.loads(a.maps.read_text(encoding='utf-8'))['routes']
    else:
        _, maps = load_assets()
    # A witness is audited offline first; never supplied as the API answer.
    for case in cases:
        errors, _, _ = audit_plan(case['feasibility_witness'], case, maps)
        if errors:
            raise RuntimeError(f'input feasibility audit failed: {case["id"]}: {errors}')
    if a.dry_run:
        print(json.dumps({'audited_inputs': len(cases), 'api_calls': 0, 'status': 'feasibility_passed'}))
        return
    with exclusive_ledger(a.output):
        binding = hashlib.sha256(a.input.read_bytes() + Path(__file__).read_bytes()
            + (ROOT/'scripts/post_train_runtime.py').read_bytes()
            + (a.maps.read_bytes() if a.maps else b'')
            + ((ROOT/'scripts/prepare_targeted_city_sft.py').read_bytes() if a.maps else b'')).hexdigest()
        meta = a.output.with_suffix('.meta.json')
        if meta.exists() and json.loads(meta.read_text(encoding='utf-8'))['binding'] != binding:
            raise RuntimeError('run binding changed')
        meta.write_text(json.dumps({'binding': binding, 'cases': len(cases), 'budget': a.max_calls,
            'human_review': 'pending', 'global_max_calls': 1000}), encoding='utf-8')
        # Existing runs, interrupted starts and preflight all count toward global cap.
        existing = 0
        own_calls = a.output.with_suffix('.calls.jsonl')
        for path in (ROOT/'data/post_train').rglob('*.calls.jsonl'):
            if path.resolve() == own_calls.resolve():
                continue
            existing += sum(json.loads(l).get('event') == 'started' for l in path.read_text(encoding='utf-8').splitlines())
        # Reserve all sixty calls of the concurrently running API baseline.
        eval_calls = ROOT/'data/post_train/eval/api_planner-only.calls.jsonl'
        eval_started = sum(json.loads(l).get('event') == 'started' for l in eval_calls.read_text(encoding='utf-8').splitlines()) if eval_calls.exists() else 0
        if existing + max(0, 60-eval_started) + a.max_calls > 1000:
            raise RuntimeError('global budget cannot reserve requested hard run')
        meter = Meter(own_calls, 'gpt-6.1-sol', a.max_calls, 90)
        done = {json.loads(l)['id'] for l in a.output.read_text(encoding='utf-8').splitlines()} if a.output.exists() else set()
        repair_remaining = max(0, a.max_calls-len(cases))
        if own_calls.exists():
            repair_remaining -= sum(json.loads(l).get('event') == 'started' and json.loads(l).get('identity', {}).get('round') == 1 for l in own_calls.read_text(encoding='utf-8').splitlines())
        lock = threading.Lock()
        stop = threading.Event()
        def worker(case):
            nonlocal repair_remaining
            if stop.is_set():
                return None
            state = copy.deepcopy(case['state'])
            row = {'id': case['id'], 'parent_seed': case['source_id'], 'state': state,
                'rubric': case['rubric'], 'mock': False, 'model': 'gpt-6.1-sol', 'status': 'failed',
                'rounds': [], 'chosen': None, 'rejected': None, 'human_review_status': 'pending'}
            state = copy.deepcopy(state)
            for index in range(2):
                messages = capture_messages(state)
                if not index:
                    row['messages'] = messages
                try:
                    plan, call = meter.call(messages, {'sample': case['id'], 'round': index, 'run': 'hard'})
                    errors, score, checked = audit_plan(plan, case, maps)
                    row['rounds'].append({'round': index, 'messages': messages, 'output': plan,
                        'rule_errors': errors, 'grade': score, 'route_distance_legs': checked.route_distance_legs,
                        'call_id': call['call_id'], 'usage': call.get('usage'), 'elapsed_s': call['elapsed_s']})
                    if not errors:
                        row.update(status='accepted', chosen=plan)
                        variants = [('natural_repair', row['rounds'][0]['output'])] if index else []
                        variants += list(counterfactuals(plan, case['state']))
                        for kind, negative in variants:
                            faults, _, _ = audit_plan(negative, case, maps)
                            if faults:
                                row.update(rejected=negative, rejected_rule_errors=faults, preference_source=kind)
                                break
                        break
                    if index == 0:
                        with lock:
                            if repair_remaining <= 0:
                                row['status'] = 'rule_rejected_no_repair_budget'
                                break
                            repair_remaining -= 1
                        state.update(route=plan['days'], meal_slots=plan.get('meal_slots', []), review_round=1,
                            route_modify_opinion='确定性规则未通过：'+', '.join(errors))
                except Exception as exc:
                    row['error'] = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
                    if any(f'status={code}' in row['error'] for code in (400, 401, 403, 404, 429)):
                        stop.set()
                    break
            return row
        with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            pending = [c for c in cases if c['id'] not in done]
            # Stratified pilot covers both cities, multiple days and weather before release.
            if a.limit and all(c['id'].startswith('newcity-') for c in cases):
                pilot_ids = {cases[i]['id'] for i in [0,8,20,32,44,48,68,92] if i < len(cases)}
                pending.sort(key=lambda c: c['id'] not in pilot_ids)
            futures = [pool.submit(worker, c) for c in pending[:a.limit] if c['id'] not in done]
            for future in as_completed(futures):
                row = future.result()
                if row is not None:
                    append_json(a.output, row)
                    print(json.dumps({'id': row['id'], 'status': row['status'], 'calls': meter.count, 'error': row.get('error')}), flush=True)
        rows = [json.loads(l) for l in a.output.read_text(encoding='utf-8').splitlines()] if a.output.exists() else []
        accepted = [r for r in rows if r['status'] == 'accepted']
        from collections import Counter
        unique = {json.dumps(r['messages'], sort_keys=True, ensure_ascii=False) for r in accepted}
        recheck_failures = []
        case_by_id = {c['id']: c for c in cases}
        for row in accepted:
            faults, _, _ = audit_plan(row['chosen'], case_by_id[row['id']], maps)
            if faults:
                recheck_failures.append({'id': row['id'], 'faults': faults})
        summary = {'recorded': len(rows), 'accepted': len(accepted), 'calls': meter.count,
            'first_round_passed': sum(bool(r['rounds']) and not r['rounds'][0]['rule_errors'] for r in rows),
            'natural_pairs': sum(r.get('preference_source') == 'natural_repair' for r in rows),
            'unique_messages': len(unique), 'duplicate_messages': len(accepted)-len(unique),
            'cities': dict(Counter(r['state']['destination'] for r in accepted)),
            'first_round_faults': dict(Counter(f for r in rows for f in (r['rounds'][0]['rule_errors'] if r['rounds'] else []))),
            'chosen_recheck_failures': recheck_failures,
            'human_review': 'pending', 'training_ready': False, 'finished_at': time.time()}
        # Preserve a review packet; producing a packet is not a human pass.
        a.output.with_suffix('.review.json').write_text(json.dumps(accepted[:30], ensure_ascii=False, indent=2), encoding='utf-8')
        tokenizer_path = Path.home()/'.cache/travelmind-qwen-tokenizer'
        if (tokenizer_path/'tokenizer.json').exists():
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
                lengths = [len(tokenizer.apply_chat_template(r['messages'] + [{'role': 'assistant', 'content': json.dumps(r[side], ensure_ascii=False)}], tokenize=True, add_generation_prompt=False))
                    for r in accepted for side in ['chosen', 'rejected'] if r.get(side)]
                summary['qwen_token_audit'] = {'answers': len(lengths), 'max': max(lengths, default=0),
                    'over_3584': sum(n > 3584 for n in lengths), 'silent_truncation': False}
            except Exception as exc:
                summary['qwen_token_audit_error'] = type(exc).__name__
        a.output.with_suffix('.audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
