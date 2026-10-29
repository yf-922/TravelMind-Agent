"""Expand training inputs; teacher never receives a historical target or witness.

Default operations are offline. --capture retrieves map evidence; --pilot makes
at most twenty metered teacher calls. No training or DPO is launched here.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.planning.schemas import TravelPlanState
from app.evaluation.route_interest import route_key
from scripts import build_representative_eval50 as assets
from scripts.compare_route_interest import exclusive_ledger
from scripts.post_train_runtime import Meter, append_json, capture_messages, quality_faults, replay_check
from scripts.prepare_heldout_evidence import opening_evidence_faults
from scripts.prepare_sft_v2_release import explanation_faults, read_rows
from scripts.audit_sft_contract_targets import aggregate_duration_faults
from scripts.run_hard_post_train_synthesis import audit_plan
from scripts.prepare_quality_sft_v4 import dump, frozen, input_collision_check, strict_start_faults

OUT = ROOT / 'data/post_train/v2_large_pool_teacher_v1'
SOURCE = ROOT / 'data/post_train/release_v2/train.ledger.jsonl'
CITIES = ['南京', '上海', '三亚', '成都', '苏州']
# Only pre-planning facts, never history, retrieval traces, route or old answers.
INPUT_FIELDS = {'query', 'destination', 'travel_start_date', 'travel_end_date',
    'attraction_preference', 'food_preference', 'habit_preference', 'max_walking_km',
    'rain_indoor_priority', 'days', 'max_per_day', 'min_rating', 'max_spots',
    'pois', 'weather_forecast', 'weather_note', 'meal_duration_min'}


def clean_input(raw):
    state = TravelPlanState(**{k: copy.deepcopy(v) for k, v in raw.items() if k in INPUT_FIELDS})
    return state.model_dump(mode='json')


def input_rows():
    # Deliberately discard every historical target/message before processing.
    return [dict(id=r['id'], state=clean_input(r['state'])) for r in read_rows(SOURCE)]


def source_cases():
    result = {}
    for file in ['new_city_sft_v1/cases.json', 'hard_supplement_cases.json']:
        for c in json.loads((ROOT/'data/post_train'/file).read_text(encoding='utf-8')):
            result[c['id']] = {k: copy.deepcopy(c[k]) for k in ['id', 'source_id', 'suite', 'category', 'rubric', 'spots_per_day'] if k in c}
    return result


def capture_training_pools():
    saved_out, saved_cities = assets.OUT, assets.CITIES
    try:
        assets.OUT, assets.CITIES = OUT/'retrieval', ['上海', '三亚']
        assets.OUT.mkdir(parents=True, exist_ok=True)
        fresh = assets.prune_reviewed_pools(assets.capture_pools())
    finally:
        assets.OUT, assets.CITIES = saved_out, saved_cities
    pools = json.loads((ROOT/'data/post_train/quality_sft_v4/training_pools.json').read_text(encoding='utf-8'))['pools']
    pools.update(fresh)
    frozen(OUT/'training_pools.json', dict(pools=pools,
        origin='training-only saved retrieval for three cities plus fresh Shanghai/Sanya retrieval'))
    return pools


def expanded_cases(rows, pools):
    refs = source_cases()
    evidence = {r['name']: r for r in json.loads((ROOT/'data/post_train/new_city_sft_v1/indoor_evidence.json').read_text(encoding='utf-8'))}
    cases, isolated = [], []
    for row in rows:
        state = clean_input(row['state'])
        original = copy.deepcopy(state['pois'])
        for poi in original:
            fact = evidence.get(poi['name'])
            # The old snapshot omitted provenance fields, but the separately
            # saved official evidence is reusable for explicitly scoped visits.
            scoped = fact and f"「{poi['name']}」仅参观馆内展厅" in state['query']
            if scoped:
                poi.update(indoor=True, indoor_source=fact['url'], indoor_scope=fact['scope'],
                    indoor_evidence_sha256=fact['sha256'])
            # Old category inference is not evidence of a fully indoor visit.
            if poi.get('indoor') is True and (not poi.get('indoor_scope') or not poi.get('indoor_source') or poi.get('indoor_source') == 'amap_category_inference'):
                poi.update(indoor=None, indoor_source='unknown', indoor_scope='历史标签缺少范围证据，不作为室内保证')
        names = {p['name'] for p in original}
        extra = [copy.deepcopy(p) for p in pools[state['destination']] if p['name'] not in names]
        random.Random(row['id']).shuffle(extra)
        state['pois'] = original + extra[:max(0, 15-len(original))]
        if len(state['pois']) < 15:
            raise RuntimeError('Not enough genuine candidates: '+row['id'])
        state['candidate_pool_status'] = 'sufficient'  # cardinality only, not feasibility certification
        blocked = []
        if state.get('max_walking_km') is not None:
            blocked.append('walking_evidence_pending_not_replaced_by_driving')
        if (state.get('rain_indoor_priority') or '只安排室内' in state['query']) and not any(p.get('indoor') is True for p in state['pois']):
            blocked.append('no_scoped_verified_indoor_candidate')
        case = dict(id=row['id'], source_id=row['id'], state=state,
            source_case=refs.get(row['id']), original_candidate_count=len(original),
            target_origin='fresh_teacher_only_no_historical_answer', isolation_reasons=blocked)
        if blocked:
            isolated.append(case)
        else:
            cases.append(case)
    input_collision_check(cases)
    return cases, isolated


def pilot_cases(cases):
    result = []
    for city in CITIES:
        local = [c for c in cases if c['state']['destination'] == city]
        # One usual multi-day case and one weather/boundary case per city.
        ordinary = next(c for c in local if c['state']['days'] >= 2 and not c['state'].get('rain_indoor_priority') and not any(float(w.get('day_temp', 0)) >= 38 for w in c['state']['weather_forecast']))
        challenge = next(c for c in sorted(local, key=lambda c: not c['state'].get('rain_indoor_priority')) if c['id'] != ordinary['id'] and
            (c['state'].get('rain_indoor_priority') or any(float(w.get('day_temp', 0)) >= 38 for w in c['state']['weather_forecast'])))
        result.extend([ordinary, challenge])
    return result


def reason_duration_faults(plan):
    """Check explicit per-POI minute claims, including selection_reasons."""
    durations = {s['name']: assets.minutes(s['end_time'])-assets.minutes(s['start_time'])
        for d in plan['days'] for s in d['spots']}
    errors = []
    for reason in plan.get('selection_reasons', []):
        if reason['name'] not in durations:
            errors.append('selection_reason_not_selected')
            continue
        if reason.get('source_ids'):
            errors.append('unsupported_source_reference')
        # Only literal scheduling claims, not optional advice or opening clocks.
        text = reason.get('reason', '')
        for m in re.finditer(r'(?:安排|停留|参观|游览|游玩)(?:时长)?(?:约|为|了)?\s*(\d+)\s*分钟', text):
            if int(m[1]) != durations[reason['name']]:
                errors.append('selection_reason_duration_mismatch')
    return sorted(set(errors))


def explanation_count_faults(plan):
    text = plan.get('reasoning', '')+' '+plan.get('notes', '')
    counts = {d['day']: len(d['spots']) for d in plan['days']}
    errors = []
    for day, n in re.findall(r'第\s*(\d+)\s*天(?:安排|参观|游览|游玩)?\s*(\d+)\s*个景点', text):
        if counts.get(int(day)) != int(n):
            errors.append('explanation_day_count_mismatch')
    for n in re.findall(r'(?:全程|总共|合计)\s*(\d+)\s*个景点', text):
        if sum(counts.values()) != int(n):
            errors.append('explanation_total_count_mismatch')
    return sorted(set(errors))


def audit(plan, case, maps):
    errors, checked = replay_check(plan, case['state'], maps)
    errors += quality_faults(checked)
    if case.get('source_case'):
        reference = dict(case['source_case'], state=case['state'])
        errors += audit_plan(plan, reference, maps)[0]
    errors += opening_evidence_faults(plan, case)
    errors += explanation_faults(plan) + aggregate_duration_faults(plan)
    errors += reason_duration_faults(plan) + explanation_count_faults(plan) + strict_start_faults(plan, case['state']['query'])
    if sorted(d['day'] for d in plan['days']) != list(range(1, case['state']['days']+1)):
        errors.append('day_structure')
    if any(not d['spots'] for d in plan['days']):
        errors.append('empty_day_without_independent_infeasibility_proof')
    if plan.get('meal_slots'):
        errors.append('unrequested_meal_slots')
    text = json.dumps({k: plan.get(k) for k in ['reasoning', 'notes', 'selection_reasons']}, ensure_ascii=False)
    if any(t in text for t in ['上一版', '上轮', '本轮修正', '审核反馈', '原方案']):
        errors.append('repair_narrative_in_round_zero_target')
    return sorted(set(errors)), checked


def prepare(capture=False):
    OUT.mkdir(parents=True, exist_ok=True)
    pools = capture_training_pools() if capture else json.loads((OUT/'training_pools.json').read_text(encoding='utf-8'))['pools']
    cases, isolated = expanded_cases(input_rows(), pools)
    frozen(OUT/'cases.json', cases)
    frozen(OUT/'isolated.json', isolated)
    # One union per city; never fetch roads during teacher inference.
    unions = {city: {} for city in CITIES}
    for c in cases:
        for p in c['state']['pois']:
            key = (p['name'], json.dumps(p['location'], sort_keys=True))
            unions[c['state']['destination']][key] = p
    if capture:
        old = assets.OUT
        try:
            assets.OUT = OUT/'roads'
            assets.OUT.mkdir(parents=True, exist_ok=True)
            maps = assets.capture_roads({city: list(p.values()) for city, p in unions.items()})
        finally:
            assets.OUT = old
        frozen(OUT/'maps.json', dict(routes=maps))
    else:
        maps = json.loads((OUT/'maps.json').read_text(encoding='utf-8'))['routes']
    for c in cases:
        ps = c['state']['pois']
        if any(route_key(a['location'], b['location'], 'drive') not in maps for a in ps for b in ps if a['name'] != b['name']):
            raise RuntimeError('Incomplete directed road coverage')
    frozen(OUT/'pilot_ids.json', [c['id'] for c in pilot_cases(cases)])
    from collections import Counter
    summary = dict(source_inputs=len(cases)+len(isolated), callable=len(cases), isolated=len(isolated),
        isolation_reasons=dict(Counter(e for c in isolated for e in c['isolation_reasons'])),
        city_counts=dict(Counter(c['state']['destination'] for c in cases)),
        pool_min=min(len(c['state']['pois']) for c in cases), pool_max=max(len(c['state']['pois']) for c in cases),
        road_edges=len(maps), teacher_calls=0, historical_answers_supplied=False)
    dump(OUT/'input_audit.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return cases, maps


def run_pilot(cases, maps):
    load_local_env()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path.home()/'.cache/travelmind-qwen-tokenizer'), local_files_only=True)
    ledger = OUT/'pilot.ledger.jsonl'
    own_calls = ledger.with_suffix('.calls.jsonl')
    with exclusive_ledger(ROOT/'data/post_train/teacher_global_budget.jsonl'), exclusive_ledger(ledger):
        existing = sum(json.loads(l).get('event') == 'started' for p in (ROOT/'data/post_train').rglob('*.calls.jsonl')
            if p.resolve() != own_calls.resolve() for l in p.read_text(encoding='utf-8').splitlines())
        if existing + 20 > 1000:
            raise RuntimeError('Insufficient global call reservation')
        selected = pilot_cases(cases)
        binding = hashlib.sha256(json.dumps(selected, ensure_ascii=False, sort_keys=True).encode()+Path(__file__).read_bytes()+(OUT/'maps.json').read_bytes()).hexdigest()
        frozen(OUT/'pilot.meta.json', dict(binding=binding, max_calls=20, global_calls_before=existing,
            model='gpt-6.1-sol', target_policy='fresh_teacher_no_historical_answer', cutoff=3584))
        meter = Meter(own_calls, 'gpt-6.1-sol', 20, 90)
        done = {r['id'] for r in read_rows(ledger)} if ledger.exists() else set()
        stop = threading.Event()
        def worker(case):
            if stop.is_set():
                return None
            state = copy.deepcopy(case['state'])
            row = dict(id=case['id'], state=case['state'], messages=capture_messages(state),
                mock=False, model='gpt-6.1-sol', status='failed', chosen=None, rejected=None,
                target_origin='fresh_teacher_only', rounds=[], human_review_status='pending')
            for index in range(2):
                messages = capture_messages(state)
                try:
                    plan, call = meter.call(messages, dict(sample=case['id'], round=index, run='v2_large_pool_teacher_pilot'))
                    errors, checked = audit(plan, case, maps)
                    length = len(tokenizer.apply_chat_template(row['messages']+[dict(role='assistant', content=json.dumps(plan, ensure_ascii=False))], tokenize=True, add_generation_prompt=False))
                    if length > 3584:
                        errors.append('training_sequence_over_3584')
                    row['rounds'].append(dict(round=index, messages=messages, output=plan,
                        rule_errors=errors, qwen_tokens=length, route_distance_legs=checked.route_distance_legs,
                        call_id=call['call_id'], usage=call.get('usage'), elapsed_s=call['elapsed_s']))
                    if not errors:
                        row.update(status='accepted', chosen=plan)
                        break
                    # A long prompt cannot be repaired by cutting off an answer.
                    if index == 0 and 'training_sequence_over_3584' not in errors:
                        state.update(route=plan['days'], meal_slots=plan.get('meal_slots', []), review_round=1,
                            route_modify_opinion='确定性校验未通过：'+', '.join(errors)+'。重新输出自洽完整方案，解释只描述最终路线，不提返工或旧版。')
                    else:
                        row['status'] = 'quarantined'
                        break
                except Exception as exc:
                    row['error'] = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
                    if any(f'status={code}' in row['error'] for code in (400, 401, 403, 404, 429)):
                        stop.set()
                    break
            return row
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, c) for c in selected if c['id'] not in done]
            for future in as_completed(futures):
                row = future.result()
                if row is None:
                    continue
                append_json(ledger, row)
                print(json.dumps(dict(id=row['id'], status=row['status'], calls=meter.count,
                    faults=row['rounds'][-1]['rule_errors'] if row['rounds'] else [], error=row.get('error')), ensure_ascii=False), flush=True)
        rows = read_rows(ledger)
        calls = read_rows(own_calls)
        usage = [r.get('usage') or {} for r in calls if r['event'] in ['completed', 'failed']]
        summary = dict(records=len(rows), accepted=sum(r['status']=='accepted' for r in rows),
            first_round_passed=sum(bool(r['rounds']) and not r['rounds'][0]['rule_errors'] for r in rows),
            calls=meter.count, global_calls=existing+meter.count,
            input_tokens=sum(u.get('prompt_tokens', 0) for u in usage), output_tokens=sum(u.get('completion_tokens', 0) for u in usage),
            total_tokens=sum(u.get('total_tokens', 0) for u in usage),
            training_started=False, full_synthesis_started=False, human_review='pending',
            semantic_limit='Deterministic checks are not exhaustive semantic certification')
        dump(OUT/'pilot.audit.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', action='store_true')
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    cases, maps = prepare(args.capture)
    if args.pilot:
        run_pilot(cases, maps)


if __name__ == '__main__':
    main()
