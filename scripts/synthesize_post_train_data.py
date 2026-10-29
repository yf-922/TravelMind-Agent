"""Real, resumable Joint Planner synthesis from the 60 fixed replay cases."""
from __future__ import annotations
import argparse
import copy
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.planning.schemas import TravelPlanState, TravelRoute
from app.evaluation.task_robustness import load_assets
from scripts.evaluate_reviewer_ablation import cases_v2
from scripts.compare_route_interest import exclusive_ledger
from scripts.post_train_runtime import Meter, append_json, capture_messages, replay_check, quality_faults

OUT = ROOT / 'data/post_train/raw_synthesis_ledger.jsonl'


def mutate_state(seed, i, rng):
    s = copy.deepcopy(seed)
    # Clear every previous answer/repair field by taking only initial case inputs.
    s['days'] = rng.randint(1, min(3, max(1, len(s['pois']))))
    start = date.fromisoformat(s['travel_start_date'])
    s['travel_end_date'] = (start + timedelta(days=s['days'] - 1)).isoformat()
    prefs = ['慢节奏，每天最多两个景点，每个至少90分钟', '亲子轻松游，每个至少90分钟', '带老人，每天最多两个景点，每个至少90分钟', '睡到自然醒，10:00后开始', '必去', '排除']
    pref = rng.choice(prefs)
    s['habit_preference'] = pref
    s['pois'] = rng.sample(s['pois'], min(len(s['pois']), max(s['days'], int(len(s['pois']) * rng.uniform(.6, .8)))))
    if not s['pois']:
        pref = '慢节奏，每天最多两个景点'
    if pref == '必去':
        pref = f'必须参观「{rng.choice(s["pois"])["name"]}」'
    elif pref == '排除':
        pref = f'不要参观「{rng.choice(s["pois"])["name"]}」'
    weather = rng.choice(['晴', '暴雨', '高温'])
    s['weather_forecast'] = [{'date': (start + timedelta(days=d)).isoformat(), 'day_weather': weather,
        'night_weather': weather, 'day_temp': '38' if weather == '高温' else '24', 'night_temp': '22', 'is_bad': weather == '暴雨'} for d in range(s['days'])]
    s['rain_indoor_priority'] = weather == '暴雨'
    s['query'] = f"{s['destination']}{start.isoformat()}起{s['days']}日游；{pref}。" + ('暴雨只安排室内。' if weather == '暴雨' else '38°C高温，正午避免室外暴晒。' if weather == '高温' else '')
    s['attraction_preference'] = s['query']
    s['habit_preference'] = pref
    s['max_review_rounds'] = 1
    return TravelPlanState(**s).model_dump(mode='json')


def inject_plan_defects(plan, state, kind=0):
    x = copy.deepcopy(plan)
    spots = [s for d in x.get('days', []) for s in d.get('spots', [])]
    if not spots:
        raise ValueError('no nonempty chosen plan')
    if kind % 2 == 0:
        spots[0]['name'] = '__候选池外虚构景点__'
    else:
        spots[0]['start_time'], spots[0]['end_time'] = '23:30', '09:00'
    TravelRoute.model_validate(x)
    return x


def mock_plan(state):
    return {'reasoning': 'mock fixture only', 'days': [{'day': d + 1, 'theme': 'fixture', 'spots': [
        {'name': state['pois'][d]['name'], 'period': 'morning', 'start_time': '10:00', 'end_time': '12:00'}]} for d in range(state['days'])], 'meal_slots': [], 'notes': 'mock'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--samples', type=int, default=800)
    p.add_argument('--max-calls', type=int, default=int(os.getenv('MAX_LLM_CALLS', '1000')))
    p.add_argument('--concurrency', type=int, default=int(os.getenv('CONCURRENCY', '8')))
    p.add_argument('--timeout', type=float, default=90)
    p.add_argument('--model-name', default='gpt-6.1-sol')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--mock', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    load_local_env()
    os.environ['TRAVEL_INTEREST_KNOWLEDGE_ENABLED'] = '0'
    mock = args.mock or args.dry_run
    cases, (_, maps) = cases_v2(), load_assets()
    rng = random.Random(args.seed)
    inputs = [(i, cases[i % len(cases)]['id'], mutate_state(cases[i % len(cases)]['state'], i, rng)) for i in range(args.samples)]
    binding = hashlib.sha256(json.dumps({'inputs': inputs, 'maps': maps, 'model': args.model_name,
        'mock': mock, 'script': Path(__file__).read_text(encoding='utf-8'), 'runtime': (ROOT/'scripts/post_train_runtime.py').read_text(encoding='utf-8')}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    with exclusive_ledger(args.output):
        meta = args.output.with_suffix('.meta.json')
        if meta.exists() and json.loads(meta.read_text(encoding='utf-8'))['binding'] != binding:
            raise RuntimeError('run binding changed: choose a new output')
        meta.write_text(json.dumps({'binding': binding, 'model': args.model_name, 'samples': args.samples,
            'mock': mock, 'budget': args.max_calls, 'evaluation_warning': 'same-seed replay, not held-out generalization'}, indent=2), encoding='utf-8')
        done = {json.loads(l)['id'] for l in args.output.read_text(encoding='utf-8').splitlines()} if args.output.exists() else set()
        meter = None if mock else Meter(args.output.with_suffix('.calls.jsonl'), args.model_name, args.max_calls, args.timeout)
        # Reserve a first attempt for every state. Only the remaining budget can repair.
        previous_repairs = 0
        if not mock and meter.path.exists():
            previous_repairs = sum(json.loads(l).get('event') == 'started' and json.loads(l).get('identity', {}).get('round') == 1
                for l in meter.path.read_text(encoding='utf-8').splitlines())
        repair_slots = max(0, args.max_calls - args.samples - previous_repairs)
        import threading
        repair_lock = threading.Lock()
        def worker(item):
            nonlocal repair_slots
            i, parent, initial = item
            identity = f'synth-{i:04d}'
            row = {'id': identity, 'parent_seed': parent, 'state': initial, 'model': args.model_name,
                'mock': mock, 'status': 'failed', 'rounds': [], 'chosen': None, 'rejected': None}
            if not initial['pois']:
                row['status'] = 'infeasible_empty_candidates'
                return row
            state = copy.deepcopy(initial)
            for round_no in range(2):
                messages = capture_messages(state)
                if round_no == 0:
                    row['messages'] = messages
                try:
                    plan, call = (mock_plan(state), {'usage': None, 'elapsed_s': 0}) if mock else meter.call(messages, {'sample': identity, 'round': round_no})
                    errors, checked = replay_check(plan, initial, maps)
                    soft_errors = quality_faults(checked)
                    errors = sorted(set(errors + soft_errors))
                    row['rounds'].append({'round': round_no, 'messages': messages, 'output': plan,
                        'rule_errors': errors, 'soft_errors': soft_errors, 'usage': call.get('usage'), 'elapsed_s': call['elapsed_s'], 'call_id': call.get('call_id'),
                        'route_distance_legs': checked.route_distance_legs})
                    if not errors:
                        row['chosen'] = plan
                        row['rejected'] = row['rounds'][0]['output'] if round_no else inject_plan_defects(plan, initial, i)
                        rejected_errors, rejected_state = replay_check(row['rejected'], initial, maps)
                        rejected_errors = sorted(set(rejected_errors + quality_faults(rejected_state)))
                        if not rejected_errors:
                            raise RuntimeError('rejected plan did not fail validation')
                        row.update(status='accepted', rejected_rule_errors=rejected_errors,
                            preference_source='natural_repair' if round_no else 'injected_verified',
                            sft_messages=messages)
                        break
                    if round_no == 0:
                        with repair_lock:
                            if repair_slots <= 0:
                                row['status'] = 'rule_rejected_no_repair_budget'
                                break
                            repair_slots -= 1
                        state.update(route=[d for d in plan['days']], meal_slots=plan.get('meal_slots', []),
                            route_modify_opinion='确定性规则未通过：' + ', '.join(errors), review_round=1)
                except Exception as exc:
                    row['error'] = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
                    break
            return row
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(worker, item) for item in inputs if f'synth-{item[0]:04d}' not in done]
            recorded = len(done)
            for future in as_completed(futures):
                row = future.result()
                append_json(args.output, row)
                recorded += 1
                print(json.dumps({'id': row['id'], 'status': row['status'], 'error': row.get('error'),
                    'recorded': recorded, 'calls': meter.count if meter else 0, 'time': time.time()}), flush=True)
                # Fail closed on auth/endpoint/config errors, before spending hundreds of calls.
                if row.get('error') and ('status=401' in row['error'] or 'status=404' in row['error']):
                    for f in futures:
                        f.cancel()
                    raise RuntimeError('API authentication/model endpoint failed; stopping')
    print('synthesis complete', flush=True)


if __name__ == '__main__':
    main()
