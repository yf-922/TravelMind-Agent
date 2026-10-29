"""Capture bounded real roads and construct independently checked test witnesses.

Only --capture contacts AMap, never an LLM. Raw replies and failed attempts are
durable; frozen v1 inputs are not edited. Evaluation uses saved responses only.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import time
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.planning.nodes import amap_key
from app.evaluation.task_robustness import grade
from app.evaluation.route_interest import route_key
from scripts.post_train_runtime import append_json, replay_check, quality_faults


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize_batch(raw, origins, destination, evidence_id):
    """Only provider-measured driving values are admissible, never air distance."""
    if raw.get('status') != '1':
        return {}
    routes = {}
    used = set()
    for result in raw.get('results', []):
        try:
            index = int(result['origin_id']) - 1
            if index < 0 or index >= len(origins) or index in used:
                continue
            distance, duration = float(result['distance']), float(result['duration'])
            if (not math.isfinite(distance) or not math.isfinite(duration)
                    or distance <= 0 or duration <= 0 or str(result.get('info', '')) not in ('', 'OK')):
                continue
        except (KeyError, ValueError, TypeError):
            continue
        used.add(index)
        origin = origins[index]
        routes[route_key(origin['location'], destination['location'], 'drive')] = {
            'mode': 'taxi_or_car', 'distance_km': round(distance/1000, 3),
            'duration_min': math.ceil(duration/60), 'source': 'amap_distance_v3_driving',
            'estimate': False, 'evidence_id': evidence_id,
            'scope': 'captured driving time only; no parking, queue, ferry or future traffic guarantee'}
    return routes


def opening_window(poi):
    text = poi.get('open_time') or ''
    matches = re.findall(r'(\d{1,2}):(\d{2})\s*[-~至]\s*(\d{1,2}):(\d{2})', text)
    if not matches:
        return None
    windows = [(int(h1)*60+int(m1), int(h2)*60+int(m2)) for h1,m1,h2,m2 in matches]
    # Use the intersection of every stated interval, including holiday/season
    # alternatives. A witness inside all windows needs no guessed season/date.
    # Disjoint sessions have no intersection and stay unresolved.
    start, end = max(w[0] for w in windows), min(w[1] for w in windows)
    if not 0 <= start < end <= 1440:
        return None
    admission = re.findall(r'最晚(?:进入|入园|入场)\s*(\d{1,2}):(\d{2})', text)
    last = min(int(h)*60+int(m) for h,m in admission) if admission else end
    return start, end, last


def clock(minutes):
    return f'{minutes//60:02d}:{minutes%60:02d}'


def opening_evidence_faults(plan, case):
    """Strict heldout scoring: unparsed opening metadata is not a pass."""
    from app.planning.nodes import _clock_minutes
    by_name = {p['name']: p for p in case['state']['pois']}
    errors = set()
    for day in plan['days']:
        for spot in day['spots']:
            window = opening_window(by_name.get(spot['name'], {}))
            if not window:
                errors.add('heldout_opening_unverified')
                continue
            start, end = _clock_minutes(spot['start_time']), _clock_minutes(spot['end_time'])
            if start < window[0] or end > window[1]:
                errors.add('heldout_opening_window')
            if start > window[2]:
                errors.add('heldout_last_admission')
    return sorted(errors)


def witness(case, maps):
    """Enumerate two-stop schedules, with 15 minutes above captured road time."""
    s, r = case['state'], case['rubric']
    candidates = [p for p in s['pois'] if p['name'] not in r.get('excluded', [])]
    for a, b in itertools.permutations(candidates, 2):
        if not set(r.get('required', [])).issubset({a['name'], b['name']}):
            continue
        aw, bw = opening_window(a), opening_window(b)
        leg = maps.get(route_key(a['location'], b['location'], 'drive'))
        if not aw or not bw or not leg or leg.get('duration_min') is None:
            continue
        duration = r['min_visit']
        start = max(r['earliest'], aw[0])
        end = start+duration
        second = max(end+int(leg['duration_min'])+15, bw[0])
        if start > aw[2] or end > aw[1] or second > bw[2] or second+duration > min(bw[1], r['latest']):
            continue
        plan = {'reasoning': '离线可行解见时间表；道路按固定快照并额外预留15分钟。',
            'days': [{'day': 1, 'theme': '离线可行性证据', 'spots': [
                {'name': a['name'], 'period': 'morning' if start < 720 else 'afternoon',
                 'start_time': clock(start), 'end_time': clock(end)},
                {'name': b['name'], 'period': 'morning' if second < 720 else 'afternoon',
                 'start_time': clock(second), 'end_time': clock(second+duration)}]}],
            'meal_slots': [], 'selection_reasons': [], 'notes': '仅证明快照约束可行；不证明现场运营。', 'modification_concern': ''}
        errors, checked = replay_check(plan, s, maps)
        scored = grade(checked, case, maps)
        if errors or quality_faults(checked) or opening_evidence_faults(plan, case) or not scored['task_success']:
            continue
        return {'plan': plan, 'rubric_grade': scored, 'road_legs': checked.route_distance_legs,
                'opening_evidence': [{'name': p['name'], 'open_time': p.get('open_time'),
                    'source': 'frozen original AMap POI fixture',
                    'method': 'intersection_of_all_stated_windows_and_earliest_last_admission'} for p in [a, b]],
                'traffic_buffer_min': 15, 'review_status': 'independent_arithmetic_and_production_rules_not_human_certified'}
    return None


def required_edges(cases):
    return {route_key(a['location'], b['location'], 'drive')
        for c in cases for a in c['state']['pois'] for b in c['state']['pois'] if a['name'] != b['name']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=ROOT/'data/post_train/heldout_city_cases_v1.json')
    p.add_argument('--output', type=Path, default=ROOT/'data/post_train/heldout_evidence_v1')
    p.add_argument('--capture', action='store_true')
    p.add_argument('--max-requests', type=int, default=40)
    a = p.parse_args()
    cases = json.loads(a.input.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output/'binding.json'
    binding = {'frozen_cases_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest()}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != binding:
        raise RuntimeError('Frozen cases changed; use a new evidence version')
    manifest_path.write_text(json.dumps(binding, indent=2))
    ledger = a.output/'amap_capture.jsonl'
    records = [json.loads(l) for l in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    maps, completed = {}, set()
    for item in records:
        if item.get('event') == 'completed':
            maps.update(normalize_batch(item['response'], item['origins'], item['destination'], item['id']))
            if item['response'].get('status') == '1':
                completed.add(item['id'])
    request_count = sum(r.get('event') == 'started' for r in records)
    if a.capture:
        load_local_env()
        key = amap_key()
        if not key:
            raise RuntimeError('AMAP_API_KEY missing')
        pools = {c['state']['destination']: c['state']['pois'] for c in cases}
        stopped = False
        for city, pool in pools.items():
            for destination in pool:
                origins = [x for x in pool if x['name'] != destination['name']]
                identity = fingerprint([city, origins, destination])
                if identity in completed or request_count >= a.max_requests:
                    continue
                time.sleep(1.05)  # public key QPS is limited; no automatic rapid retries
                request_count += 1
                event = {'id': identity, 'city': city, 'origins': origins, 'destination': destination,
                    'endpoint': 'https://restapi.amap.com/v3/distance', 'type': 1,
                    'captured_at': datetime.now(timezone.utc).isoformat()}
                append_json(ledger, {**event, 'event': 'started'})
                point = lambda v: f'{v["location"]["lng"]},{v["location"]["lat"]}'
                params = urllib.parse.urlencode({'key': key, 'origins': '|'.join(map(point, origins)),
                    'destination': point(destination), 'type': 1, 'output': 'json'})
                try:
                    req = urllib.request.Request(event['endpoint']+'?'+params)
                    with urllib.request.urlopen(req, timeout=20) as reply:
                        raw = json.load(reply)
                    append_json(ledger, {**event, 'event': 'completed', 'response': raw})
                    routes = normalize_batch(raw, origins, destination, identity)
                    maps.update(routes)
                    print(json.dumps({'city': city, 'captured_batches': request_count,
                        'valid_legs': len(routes), 'provider_status': raw.get('info')}), flush=True)
                    if raw.get('status') != '1':
                        stopped = True
                        break  # invalid key/quota: do not consume remaining requests
                except Exception as exc:
                    # Never persist exception text that could contain a key-bearing URL.
                    append_json(ledger, {**event, 'event': 'failed', 'error_type': type(exc).__name__})
                    stopped = True
                    break
            if stopped:
                break
    expected = required_edges(cases)
    (a.output/'maps.json').write_text(json.dumps({'routes': maps, 'source': 'amap_distance_v3_type1',
        'requests': request_count, 'raw_ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None},
        ensure_ascii=False, indent=2), encoding='utf-8')
    train_cities, train_names = set(), set()
    for path in (ROOT/'data/post_train/release_v1').glob('*.ledger.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            train_cities.add(row['state']['destination'])
            train_names.update(x['name'] for x in row['state']['pois'])
    ready, audit = [], []
    for case in cases:
        evidence = witness(case, maps)
        edges = required_edges([case])
        overlap = case['state']['destination'] in train_cities or bool(train_names & {p['name'] for p in case['state']['pois']})
        status = bool(evidence and edges <= maps.keys() and not overlap)
        audit.append({'id': case['id'], 'ready': status, 'witness_found': evidence is not None,
            'missing_directed_roads': len(edges-maps.keys()), 'posttrain_city_or_poi_overlap': overlap})
        if status:
            out = copy.deepcopy(case)
            out['readiness'] = 'offline_replay_ready_not_realworld_certified'
            out['feasibility_evidence'] = evidence
            ready.append(out)
    (a.output/'cases.json').write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {**binding, 'cases': len(cases), 'ready': len(ready), 'captured_roads': len(maps),
        'expected_directed_roads': len(expected), 'missing_roads': len(expected-maps.keys()),
        'requests': request_count, 'llm_calls': 0, 'per_case': audit,
        'opening_source': 'original AMap POI snapshot, not live official operating guarantee',
        'claims': 'city-held-out relative to this posttraining release only; base pretraining exposure unknown',
        'human_review': 'not_performed', 'test_training_use': 'forbidden'}
    (a.output/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='per_case'}, ensure_ascii=True))


if __name__ == '__main__':
    main()
