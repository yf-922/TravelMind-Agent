"""Versioned new-city training inputs with saved provider evidence, not test answers."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.planning.nodes import amap_key, _clock_minutes
from app.evaluation.route_interest import route_key
from scripts.post_train_runtime import append_json
from scripts.prepare_heldout_evidence import opening_window, normalize_batch, clock, fingerprint, opening_evidence_faults

FOLDER = ROOT/'data/post_train/new_city_sft_v1'
POOLS = {
    '成都': ['文殊院', '成都杜甫草堂博物馆', '四川博物院', '成都博物馆', '立巢航空博物馆',
           '三和老爷车博物馆', '明蜀王陵博物馆', '成都永陵博物馆'],
    '苏州': ['寒山寺', '拙政园', '苏州博物馆(本馆)', '西园戒幢律寺', '苏州革命博物馆',
           '苏州园林博物馆', '苏州丝绸博物馆', '沧浪亭'],
}
INDOOR = {
    '成都博物馆': ('https://www.cdmuseum.com', '2F'),
    '苏州博物馆(本馆)': ('https://www.szmuseum.com/News/Index/OpenGuide', '展厅内'),
}


def conservative_day_window(text):
    normalized = text.replace('–', '-').replace('—', '-')
    # Explicit separate ticketed night sessions are not day opening alternatives.
    daytime = ';'.join(part for part in re.split('[；;]', normalized) if '夜游' not in part)
    window = opening_window({'open_time': daytime})
    if window:
        stops = re.findall(r'(\d{1,2}):(\d{2})\s*(?:停止(?:售票|售检票|入馆)|止入)', daytime)
        last = min([window[2]]+[int(h)*60+int(m) for h, m in stops])
        return window[0], window[1], last
    return None


def pools(folder):
    result, provenance = {}, []
    for city, file in [('成都', 'poi_probe_chengdu.json'), ('苏州', 'poi_probe_suzhou.json')]:
        path = folder/file
        raw = json.loads(path.read_text(encoding='utf-8'))
        if raw.get('status') != '1':
            raise RuntimeError('invalid saved POI response')
        by_name = {p['name']: p for p in raw['pois']}
        result[city] = []
        for name in POOLS[city]:
            p = by_name[name]
            window = conservative_day_window(p['biz_ext']['opentime2'])
            if not window or '不对外开放' in name:
                raise RuntimeError('unverified opening: '+name)
            lng, lat = map(float, p['location'].split(','))
            result[city].append({'name': name, 'location': {'lng': lng, 'lat': lat},
                'address': p.get('address') if isinstance(p.get('address'), str) else '',
                'open_time': f'{clock(window[0])}-{clock(window[1])} 最晚进入{clock(window[2])}',
                # Museum names alone do not establish an entirely indoor visit.
                'indoor': None, 'type': p.get('type', '')})
            provenance.append({'city': city, 'name': name, 'poi_id': p['id'], 'raw_opening': p['biz_ext']['opentime2'],
                'derived_window': window, 'method': 'intersection of all stated windows; earliest last admission',
                'source_file': file, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'source_file_mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                'calendar_scope': '2026-11-03 to 2026-11-05, Tue-Thu; snapshot only, not an operating guarantee'})
    return result, provenance


def capture_indoor(folder):
    from bs4 import BeautifulSoup
    evidence = []
    for index, (name, (url, required)) in enumerate(INDOOR.items()):
        path = folder/f'official_indoor_{index}.html'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        body = BeautifulSoup(path.read_bytes(), 'html.parser').get_text(' ', strip=True)
        if required not in body:
            raise RuntimeError('official indoor evidence no longer matches: '+name)
        pos = body.index(required)
        evidence.append({'name': name, 'url': url, 'quote': body[max(0, pos-80):pos+160],
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope': 'Only museum interior exhibition rooms; excludes courtyard, access, outdoor queue',
            'captured_file_mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
    (folder/'indoor_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')


def capture_roads(folder, city_pools, capture):
    ledger = folder/'amap_roads.jsonl'
    records = [json.loads(l) for l in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    maps, done = {}, set()
    for row in records:
        if row['event'] == 'completed' and row['response'].get('status') == '1':
            maps.update(normalize_batch(row['response'], row['origins'], row['destination'], row['id']))
            done.add(row['id'])
    attempts = sum(r['event'] == 'started' for r in records)
    if capture:
        load_local_env()
        key = amap_key()
        if not key:
            raise RuntimeError('AMap key missing')
        for city, pool in city_pools.items():
            for destination in pool:
                origins = [p for p in pool if p['name'] != destination['name']]
                identity = fingerprint([city, origins, destination])
                if identity in done:
                    continue
                if attempts >= 20:
                    raise RuntimeError('road request budget reached (20, plus 2 existing POI probes)')
                time.sleep(1.1)
                attempts += 1
                event = {'id': identity, 'city': city, 'origins': origins, 'destination': destination,
                    'captured_at': datetime.now(timezone.utc).isoformat(), 'type': 1}
                append_json(ledger, dict(event, event='started'))
                point = lambda p: f'{p["location"]["lng"]},{p["location"]["lat"]}'
                params = urllib.parse.urlencode({'key': key, 'origins': '|'.join(map(point, origins)),
                    'destination': point(destination), 'type': 1, 'output': 'json'})
                try:
                    with urllib.request.urlopen('https://restapi.amap.com/v3/distance?'+params, timeout=25) as response:
                        raw = json.load(response)
                except Exception as exc:
                    append_json(ledger, dict(event, event='failed', error_type=type(exc).__name__))
                    raise RuntimeError('road capture failed: '+type(exc).__name__) from None
                append_json(ledger, dict(event, event='completed', response=raw))
                if raw.get('status') != '1':
                    raise RuntimeError('provider rejected road batch: '+str(raw.get('infocode')))
                maps.update(normalize_batch(raw, origins, destination, identity))
                print(json.dumps({'road_batches': attempts, 'saved_roads': len(maps)}), flush=True)
    expected = {route_key(a['location'], b['location'], 'drive') for pool in city_pools.values()
                for a in pool for b in pool if a['name'] != b['name']}
    if expected - maps.keys():
        raise RuntimeError(f'missing {len(expected-maps.keys())} real directed roads; use --capture')
    (folder/'maps.json').write_text(json.dumps({'routes': maps, 'requests': attempts,
        'existing_poi_search_requests': 2}, ensure_ascii=False, indent=2), encoding='utf-8')
    return maps


def additional_faults(plan, case):
    errors = opening_evidence_faults(plan, case)
    days = plan['days']
    state = case['state']
    if sorted(d['day'] for d in days) != list(range(1, state['days']+1)):
        errors.append('missing_or_duplicate_day')
    names = [p['name'] for d in days for p in d['spots']]
    if len(names) != len(set(names)):
        errors.append('global_duplicate_poi')
    if any(len(d['spots']) != case['spots_per_day'] for d in days):
        errors.append('per_day_exact_count')
    if plan.get('meal_slots'):
        errors.append('unrequested_meal_slots')
    selected = set(names)
    if any(r['name'] not in selected or r.get('source_ids') for r in plan.get('selection_reasons', [])):
        errors.append('unsupported_selection_reference')
    explanation = plan.get('reasoning', '')+' '+plan.get('notes', '')+' '+json.dumps(plan.get('selection_reasons', []), ensure_ascii=False)
    if any(t in explanation for t in ['上一版', '上轮', '本轮修正', '审核反馈', '原方案']):
        errors.append('repair_context_in_round_zero_answer')
    # Conservative rejection of unsupported specific facts, not a semantic proof.
    if any(t in explanation for t in ['步行可达', '无需预约', '无需排队', '全年开放', '同属一个区域', '室内景点密集']):
        errors.append('unsupported_explanation_fact')
    for match in re.findall(r'(?:共|合计|全程)([一二三四五六七八九\d]+)个景点', explanation):
        count = int(match) if match.isdigit() else '一二三四五六七八九'.find(match)+1
        if count != len(names):
            errors.append('explanation_count_mismatch')
    return sorted(set(errors))


def feasible_plan(case, maps):
    state, r = case['state'], case['rubric']
    pool = [p for p in state['pois'] if p['name'] not in r.get('excluded', [])
            and (not r.get('indoor_only') or p.get('indoor') is True)]
    n = state['days']*case['spots_per_day']
    for chosen in itertools.permutations(pool, n):
        if not set(r['required']).issubset({p['name'] for p in chosen}):
            continue
        days = []
        for day in range(state['days']):
            events, earliest, previous = [], r['earliest'], None
            for poi in chosen[day*case['spots_per_day']:(day+1)*case['spots_per_day']]:
                window = opening_window(poi)
                start = max(earliest, window[0])
                if previous:
                    leg = maps.get(route_key(previous['location'], poi['location'], 'drive'))
                    if not leg:
                        break
                    start = max(start, earliest+leg['duration_min']+15)
                if case['category'] == 'heat' and poi.get('indoor') is not True and start+r['min_visit'] > 660:
                    start = max(start, 900)
                end = start+r['min_visit']
                if start > window[2] or end > min(window[1], r['latest']):
                    break
                events.append({'name': poi['name'], 'period': 'morning' if start < 720 else 'afternoon',
                    'start_time': clock(start), 'end_time': clock(end)})
                earliest, previous = end, poi
            if len(events) != case['spots_per_day']:
                break
            days.append({'day': day+1, 'theme': '候选内慢节奏参观', 'spots': events})
        if len(days) == state['days']:
            return {'reasoning': '按最终时间表参观所选景点；不同景点不重复，交通留有余量。',
                'days': days, 'meal_slots': [], 'selection_reasons': [], 'notes': '开放与道路按输入快照，实际出行前需核实。',
                'modification_concern': ''}
    raise RuntimeError('no feasible witness: '+case['id'])


def build_cases(city_pools, maps):
    cases = []
    rng = random.Random(20261024)
    for city, full in city_pools.items():
        for i in range(48):
            category = ['ordinary']*8+['multi_day']*12+['heat']*12+['window']*12+['rain']*4
            category = category[i]
            days = 2+(i%2) if category == 'multi_day' else 1
            pool = copy.deepcopy(rng.sample(full, 7+(i%2)))
            per_day = 1 if category == 'rain' else 2
            indoor = next(name for name in INDOOR if name in POOLS[city])
            if category == 'rain':
                if indoor not in {p['name'] for p in pool}:
                    pool[-1] = copy.deepcopy(next(p for p in full if p['name'] == indoor))
                for p in pool:
                    if p['name'] == indoor:
                        p['indoor'] = True  # Explicit visit scope supported by saved official evidence.
            required = indoor if category == 'rain' else pool[0]['name']
            excluded = next(p['name'] for p in reversed(pool) if p['name'] != required)
            earliest = 600+(i%3)*15 if category in ('ordinary', 'window', 'rain') else 540
            latest = 1020 if category == 'heat' else 1050
            duration = 60+(i%3)*15 if category == 'window' else 90
            query = (f'{city}{days}日游，2026年11月3日开始。每天恰好{per_day}个景点，每个至少{duration}分钟；'
                f'全程景点不重复。每天{clock(earliest)}后开始，{clock(latest)}前结束。'
                f'必须参观「{required}」；不要参观「{excluded}」。仅规划景点，不安排餐馆或用餐时间块。')
            if category == 'heat':
                query += '模拟38°C高温，11:00-15:00不安排室外或室内属性未知的景点。'
            if category == 'rain':
                query += f'模拟暴雨，只安排有证据的室内参观；「{indoor}」仅参观馆内展厅，不含庭院和室外活动，其他候选室内属性未知。'
            query += '解释须与最终天数、景点和时间一致；不推测闭馆、预约、阴凉、登阶、邻近等未提供事实。开放时间按候选保守窗口，道路为快照，实际出行前需核实。'
            start = datetime(2026, 11, 3)
            state = {'destination': city, 'days': days, 'max_per_day': per_day, 'pois': pool,
                'main_meal_candidates': [], 'habit_preference': '慢节奏', 'query': query,
                'attraction_preference': query, 'travel_start_date': start.date().isoformat(),
                'travel_end_date': (start+timedelta(days=days-1)).date().isoformat(),
                'candidate_pool_status': 'sufficient', 'rain_indoor_priority': category == 'rain',
                'weather_forecast': [{'date': (start+timedelta(days=d)).date().isoformat(),
                    'day_weather': '暴雨' if category == 'rain' else '晴', 'night_weather': '晴',
                    'day_temp': '38' if category == 'heat' else '24', 'night_temp': '20',
                    'is_bad': category == 'rain'} for d in range(days)]}
            case = {'id': f'newcity-{city}-{i:03}', 'source_id': f'newcity-{city}', 'suite': 'task',
                'training_use': 'allowed', 'category': category, 'spots_per_day': per_day, 'state': state,
                'rubric': {'required': [required], 'excluded': [excluded], 'min_count': days*per_day,
                    'max_count': days*per_day, 'min_visit': duration, 'earliest': earliest, 'latest': latest,
                    **({'indoor_only': True} if category == 'rain' else {})}}
            case['feasibility_witness'] = feasible_plan(case, maps)
            cases.append(case)
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', action='store_true')
    a = p.parse_args()
    city_pools, provenance = pools(FOLDER)
    if a.capture:
        capture_indoor(FOLDER)
    if not (FOLDER/'indoor_evidence.json').exists():
        raise RuntimeError('missing saved indoor evidence; use --capture')
    maps = capture_roads(FOLDER, city_pools, a.capture)
    cases = build_cases(city_pools, maps)
    from scripts.run_hard_post_train_synthesis import audit_plan
    for case in cases:
        errors = audit_plan(case['feasibility_witness'], case, maps)[0]+additional_faults(case['feasibility_witness'], case)
        if errors:
            raise RuntimeError(f'witness rejected {case["id"]}: {errors}')
    text = json.dumps(cases, ensure_ascii=False, indent=2)
    path = FOLDER/'cases.json'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise RuntimeError('frozen new-city cases changed; create a new version')
    path.write_text(text, encoding='utf-8')
    (FOLDER/'opening_evidence.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    (FOLDER/'pilot.json').write_text(json.dumps([cases[i] for i in [0,8,20,32,44,48,68,92]], ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'feasible_cases': len(cases), 'directed_roads': len(maps), 'paid_llm_calls': 0,
        'independent_test_cities': 'untouched', 'semantic_explanation_audit': 'limited checks plus review required'}))


if __name__ == '__main__':
    main()
