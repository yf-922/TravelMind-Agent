"""Capture new-city evidence and freeze a 10-city, 50-task planner holdout.

Only --capture contacts AMap. No teacher or evaluated model is used in design.
Scenario weather/user constraints are synthetic; POIs/openings/roads are sourced.
"""
from __future__ import annotations
import argparse
import copy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_representative_eval50 import admissible, make_witness, score_case
from scripts.prepare_heldout_evidence import fingerprint, normalize_batch, opening_window, clock
from scripts.post_train_runtime import append_json, capture_messages
from app.evaluation.route_interest import route_key
from app.planning.schemas import TravelPlanState

OUT = ROOT / 'data/post_train/online_holdout50_20261027_v2'
CITIES = ['北京', '天津', '杭州', '绍兴', '无锡', '合肥', '武汉', '长沙', '洛阳', '福州']
START = date(2026, 11, 11)  # Wednesday; evidence filter resolves Wed-Fri only.
EXCLUSIONS = {
    '北京': {'故宫博物院-中和殿', '骊画廊'},
    '天津': {'美好生活研究院'},
    '杭州': {'胡庆余堂(总店)'},
    '绍兴': {'鲁迅故里-三味书屋', '鲁迅故里-百草园'},
    '无锡': {'无锡国家数字电影产业园'},
    '合肥': {'包公祠'},
    '武汉': {'武汉江滩', '辛亥革命博物院北区'},
    '洛阳': {'斑马咖啡Zebra craft', '老君山金顶道观群'},
    '福州': {'山海恒达标本观光工厂'},
}
SOURCE_FILES = ['app/planning/graph.py', 'app/planning/nodes.py', 'app/planning/prompts.py',
                'app/planning/schemas.py', 'app/planning/constraints.py',
                'scripts/build_online_holdout50.py', 'scripts/evaluate_online_holdout50.py',
                'app/evaluation/online_replay.py', 'scripts/build_representative_eval50.py',
                'scripts/prepare_sft_v2_release.py', 'scripts/post_train_runtime.py', 'app/llm/openai.py',
                'scripts/run_online_holdout_matrix.py', 'scripts/report_online_holdout50.py']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def capture_pools(out):
    from app.core.env import load_local_env
    from app.planning.helpers import amap_key
    from app.providers.amap.poi import search_city_pois
    load_local_env()
    budget = dict(remaining=180, used=0)
    pools = {}
    queries = [('', '110000'), ('博物馆', '140100'), ('公园', '110000'),
               ('美术馆', '140100'), ('纪念馆', '140100'), ('古迹', '110000'),
               ('风景名胜', '110000'), ('科技馆', '140100'), ('历史文化', '110000'),
               ('城市公园', '110000'), ('纪念馆', '140100'), ('旅游景点', '110000')]
    for city in CITIES:
        pool = []
        for i, (keyword, types) in enumerate(queries):
            p = out / 'raw' / f'poi-{city}-{i}.json'
            if p.exists():
                rows = json.loads(p.read_text(encoding='utf-8'))['rows']
            else:
                append_json(out/'capture_calls.jsonl', dict(event='started', kind='poi', city=city, keyword=keyword))
                rows = search_city_pois(city, amap_key(), keywords=keyword, types=types, offset=25, request_budget=budget)
                dump(p, dict(city=city, keyword=keyword, types=types, rows=rows,
                             captured_at=datetime.now(timezone.utc).isoformat()))
                append_json(out/'capture_calls.jsonl', dict(event='completed', kind='poi', city=city, count=len(rows)))
            kept = 0
            for raw in rows:
                if str(raw.get('cityname', '')).removesuffix('市') != city:
                    continue
                if raw.get('name') in EXCLUSIONS.get(city,set()) or any(t in str(raw.get('name','')) for t in ['暂停开放','暂停营业','暂不开放','咖啡','灯具']):
                    append_json(out/'candidate_filters.jsonl',dict(city=city,name=raw.get('name'),reason='manual_subsite_ambiguity_commercial_or_closed_listing'))
                    continue
                poi, reason = admissible(raw, pool)
                if reason:
                    append_json(out/'candidate_filters.jsonl', dict(city=city, name=raw.get('name'), reason=reason))
                    continue
                poi['provenance'] = dict(raw_file=str(p.relative_to(out)), raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    method='Provider opening intervals conservatively normalized, not future-operation guarantee; indoor unknown.')
                pool.append(poi)
                kept += 1
                if len(pool) >= 20 or kept >= (7 if i == 0 else 5):
                    break
            print(json.dumps(dict(stage='poi', city=city, candidates=len(pool), channel=keyword), ensure_ascii=False), flush=True)
            if len(pool) >= 20:
                break
        if len(pool) < 18:
            raise RuntimeError(f'{city}: only {len(pool)} verified candidates; no fabricated fill')
        pools[city] = pool
        dump(out/'pools.partial.json', dict(pools=pools))
    dump(out/'pools.json', dict(pools=pools, scope='new evidence snapshot, not observed production traffic distribution',
        reviewed_exclusions={c:sorted(v) for c,v in EXCLUSIONS.items()},review_scope='name/category/closure/parent-site screening, not independent physical verification'))
    return pools


def capture_roads(out, pools):
    from app.core.env import load_local_env
    from app.planning.helpers import amap_key
    load_local_env()
    ledger = out/'road_calls.jsonl'
    rows = [json.loads(s) for s in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    maps, done = {}, set()
    for r in rows:
        if r['event'] == 'completed':
            maps.update(normalize_batch(r['response'], r['origins'], r['destination'], r['id']))
            done.add(r['id'])
    attempts = sum(r['event'] == 'started' for r in rows)
    for city, pool in pools.items():
        for dest in pool:
            origins = [p for p in pool if p['name'] != dest['name']]
            ident = fingerprint([city, origins, dest])
            if ident in done:
                continue
            params = dict(key=amap_key(), origins='|'.join(f"{p['location']['lng']},{p['location']['lat']}" for p in origins),
                          destination=f"{dest['location']['lng']},{dest['location']['lat']}", type=1, output='json')
            for retry in range(3):
                if attempts >= 600:
                    raise RuntimeError('Road capture hard budget exhausted')
                event = dict(id=ident, city=city, origins=origins, destination=dest)
                append_json(ledger, dict(event, event='started')); attempts += 1
                try:
                    with urllib.request.urlopen('https://restapi.amap.com/v3/distance?' + urllib.parse.urlencode(params), timeout=25) as f:
                        raw = json.load(f)
                    if raw.get('status') != '1':
                        raise RuntimeError('amap_status_' + str(raw.get('infocode')))
                    accepted = normalize_batch(raw, origins, dest, ident)
                    if len(accepted) != len(origins):
                        raise RuntimeError(f'Incomplete directed road evidence {len(accepted)}/{len(origins)}')
                except Exception as exc:
                    append_json(ledger, dict(event, event='failed', error_type=type(exc).__name__,
                                             code=str(exc) if isinstance(exc, RuntimeError) else 'transport_error'))
                    if retry == 2:
                        raise RuntimeError(f'Road capture failed for {city}; see durable ledger') from None
                    time.sleep(1.5 * (retry + 1)); continue
                append_json(ledger, dict(event, event='completed', response=raw))
                maps.update(accepted)
                break
            print(json.dumps(dict(stage='roads', city=city, calls=attempts, edges=len(maps))), flush=True)
            time.sleep(.35)
    required = {route_key(a['location'], b['location'], 'drive') for pool in pools.values() for a in pool for b in pool if a != b}
    if required - maps.keys():
        raise RuntimeError('Incomplete directed road matrix')
    dump(out/'maps.json', dict(routes=maps, evidence='AMap measured driving snapshots, no straight-line substitution'))
    return maps


def soft_names(pool, city_index):
    # Use explicit provider category, not inferred stairs/indoor/popularity labels.
    groups = [('博物馆', '博物馆'), ('公园', '公园'), ('美术馆', '美术馆'), ('纪念馆', '纪念馆')]
    for offset in range(len(groups)):
        token, label = groups[(city_index + offset) % len(groups)]
        names = [p['name'] for p in pool if token in p.get('category', '')]
        if names:
            return names, label
    raise RuntimeError('No provider-category evidence for soft preference')


def build_cases(pools, maps):
    cases = []
    for ci, city in enumerate(CITIES):
        pool = pools[city]
        anchors = [p for p in pool if opening_window(p)[0] <= 660 and opening_window(p)[1] >= 960 and opening_window(p)[2] >= 780]
        if len(anchors) < 3:
            raise RuntimeError('Too few broad-window anchors: ' + city)
        preferred, label = soft_names(pool, ci)
        for i in range(5):
            local = copy.deepcopy(pool)
            random.Random(271000 + ci*5+i).shuffle(local)
            rule = dict(required=[], excluded=[], counts=[[2, 2]], earliest=540, latest=1080,
                        min_visit=75, forbidden_windows=[])
            state = dict(destination=city, days=1, max_per_day=4, pois=local, query='',
                         travel_start_date=START.isoformat(), main_meal_candidates=[], main_meal_status='not_requested',
                         candidate_pool_status='ready', candidate_missing_coverage=[],
                         max_review_rounds=3, max_time_check_rounds=3)
            a, b = anchors[ci % len(anchors)], anchors[(ci+1) % len(anchors)]
            category, tags, soft, proof = '', [], {}, None
            if i == 0:
                category = 'family_pace_and_preference'
                query = f'带老人和孩子在{city}玩1天，节奏放松，每天最多2个景点，今天安排2个不同景点，每处至少75分钟，09:00至18:00内游玩。优先选择{label}类候选，但硬约束优先；不要为了凑满候选而加点。'
                soft = dict(preferred_names=preferred, minimum_hits=1, source='provider_category', preference=label)
                tags = ['soft_interest', 'pace', 'upper_bound', 'explanation_grounding']
            elif i == 1:
                category = 'current_request_over_history'
                query = f'在{city}玩1天，原先想每天恰好3个景点，每处至少75分钟，09:00至18:00。'
                state['modification_notes'] = f'以这次修改为准：今天恰好2个不同景点，每天最多2个；必须参观「{a["name"]}」，不要参观「{b["name"]}」。所有参观严格晚于10:00开始，18:00前结束。'
                rule.update(required=[a['name']], excluded=[b['name']], earliest=601)
                tags = ['current_instruction_priority', 'required_vs_excluded', 'strict_start', 'last_entry']
            elif i == 2:
                category = 'multiday_variable_allocation'
                counts = [[1,1], [2,2], [1,1]] if ci % 2 else [[2,2], [1,1]]
                state['days'] = len(counts); rule.update(counts=counts, min_visit=90, forbidden_windows=[[720, 810]])
                allocation = '、'.join(f'第{d}天恰好{x[0]}个' for d,x in enumerate(counts,1))
                query = f'在{city}玩{len(counts)}天，{allocation}，每天最多2个，所有天景点全程不重复。每处至少90分钟，每天09:00至18:00，12:00至13:30休息，不能与参观重叠；不要求把空余时间填满。'
                tags = ['unequal_counts', 'global_unique', 'break_window', 'visit_duration', 'commute']
            elif i == 3:
                category = ['heat_exact_boundary', 'below_heat_threshold', 'rain_unknown_evidence', 'tight_evening_window'][ci % 4]
                if ci % 4 == 0:
                    query = f'在{city}玩1天，最高38°C；09:00至18:00安排恰好2个不同景点，每天最多2个，每处至少75分钟。11:00至15:00不能安排室外或室内未知景点，11:00结束或15:00开始允许；不能根据名称推断室内。'
                    tags = ['38_boundary', 'unknown_not_indoor', 'avoid_heat']
                elif ci % 4 == 1:
                    query = f'在{city}玩1天，最高37.9°C，12:00至17:00安排恰好2个不同景点，每天最多2个，每处至少75分钟。没有额外禁游时段，不要擅自增加限制或拒绝。'
                    rule.update(earliest=720, latest=1020); tags = ['37.9_boundary', 'avoid_false_refusal']
                elif ci % 4 == 2:
                    query = f'在{city}玩1天，预报有阵雨，每天最多2个，今天安排2个不同景点，每处至少75分钟，09:00至18:00；如果候选有明确室内证据则优先室内，否则仍接受现有候选，但请提醒携带雨具、复核天气，不能把未知说成已核实室内。'
                    state['rain_indoor_priority'] = True
                    soft = dict(explanation_any=['雨', '伞', '天气'], preference='雨天提醒与证据边界')
                    tags = ['rain_soft_priority', 'unknown_evidence', 'no_false_indoor_claim']
                else:
                    query = f'在{city}玩1天，只能在13:30至17:00游玩，恰好安排2个不同景点，每天最多2个，每处至少60分钟。优先减少跨区折返，但不能压缩通勤或越过停止入场时间。'
                    rule.update(earliest=810, latest=1020, min_visit=60)
                    tags = ['tight_window', 'commute_vs_opening', 'route_efficiency']
            else:
                if ci % 2 == 0:
                    category = 'sparse_but_feasible'; state.update(days=3, pois=copy.deepcopy(anchors[:3]), candidate_pool_status='insufficient', candidate_missing_coverage=['count'])
                    rule.update(counts=[[1,1]]*3, min_visit=90)
                    query = f'在{city}玩3天，目前只检索到3个候选。我接受每天恰好1个不同景点，3天全程不重复，每处至少90分钟，每天09:00至18:00。请用现有候选完成，不因候选少就全部拒绝，也不要重复填充。'
                    tags = ['sparse_feasible', 'no_overrefusal', 'global_unique']
                else:
                    category = 'dense_required_window_infeasible'
                    target = min(pool, key=lambda p: opening_window(p)[0])
                    close = min(1080, opening_window(target)[1])
                    # End at closure, but leave less than the required 90 minutes.
                    earliest = max(opening_window(target)[0], close - 60)
                    rule.update(counts=[[0,0]], min_visit=90, earliest=earliest, latest=close,
                                reason_terms=['时间', '时段', '90', '分钟', '开放'])
                    query = f'在{city}玩1天，候选数量充足，但必须参观「{target["name"]}」且连续游览至少90分钟。今天只允许{clock(earliest)}至{clock(close)}游玩，不能越过时间窗口，禁止拆成多段或改成其他景点冒充完成。若无法同时满足，请保留第1天但景点为空，在notes说明原因及可放宽条件。'
                    proof = dict(kind='required_duration_exceeds_user_window', required_poi=target['name'],
                                 available_minutes=close-earliest, required_minutes=90,
                                 explanation=f'允许游玩时段只有{close-earliest}分钟，无法满足必去景点连续90分钟；需延长时间或放宽要求。')
                    tags = ['dense_but_infeasible', 'time_capacity_proof', 'no_unsafe_partial_route']
            state['query'] = query + ' 不安排餐馆，用餐自行解决；只从给定候选中选择。'
            temp = '38' if i == 3 and ci % 4 == 0 else ('37.9' if i == 3 and ci % 4 == 1 else '26')
            state['weather_forecast'] = [dict(date=(START+timedelta(days=d)).isoformat(), day_temp=temp, night_temp='20',
                day_weather='阵雨' if category=='rain_unknown_evidence' else '晴', night_weather='晴', is_bad=category=='rain_unknown_evidence') for d in range(state['days'])]
            state['travel_end_date'] = (START+timedelta(days=state['days']-1)).isoformat()
            TravelPlanState(**state)
            case = dict(id=f'online-holdout-{ci+1:02d}-{i+1:02d}', city=city, category=category, tags=tags,
                        stratum='normal' if i < 4 else 'hard', expected='explained_infeasible' if proof else 'feasible',
                        state=state, rubric=rule, soft_rubric=soft, input_provenance='new AMap snapshots + explicit synthetic user/weather scenario')
            if proof:
                case['infeasibility_proof'] = proof
            witness_case = copy.deepcopy(case)
            # A feasible witness for the soft target; target is NOT a hard requirement in final scoring.
            if soft.get('preferred_names'):
                witness_case['rubric']['required'] = [soft['preferred_names'][0]]
            case['feasibility_witness'] = make_witness(witness_case, maps)
            if soft.get('explanation_any'):
                case['feasibility_witness']['notes'] += ' 阵雨请带伞，并复核天气；室内属性未知不能当成明确室内。'
            case['messages'] = capture_messages(state)
            cases.append(case)
    return cases


def score_soft(plan, case):
    rule = case.get('soft_rubric', {})
    names = {s['name'] for d in plan.get('days', []) for s in d.get('spots', [])}
    text = str(plan.get('notes', '')) + ' ' + str(plan.get('reasoning', ''))
    faults = []
    if rule.get('preferred_names') and len(names & set(rule['preferred_names'])) < rule['minimum_hits']:
        faults.append('preferred_category_not_selected')
    if rule.get('explanation_any') and not any(w in text for w in rule['explanation_any']):
        faults.append('rain_advice_not_expressed')
    return faults


def freeze(out, pools, maps):
    if (out/'FROZEN.json').exists():
        raise RuntimeError('Immutable suite already frozen; create a new version')
    cases = build_cases(pools, maps)
    controls = []
    for c in cases:
        assert not score_case(c['feasibility_witness'], c, maps), c['id']
        assert not score_soft(c['feasibility_witness'], c), c['id']
        bad = copy.deepcopy(c['feasibility_witness'])
        if c['expected'] == 'feasible':
            bad['days'][0]['spots'][0]['name'] = '__not_a_candidate__'
        else:
            bad['notes'] = bad['reasoning'] = '完成'
        faults = score_case(bad, c, maps)
        assert faults, c['id']
        controls.append(dict(id=c['id'], witness_pass=True, negative_faults=faults,
                             pool_size=len(c['state']['pois'])))
    prior_cities = {'南京','上海','三亚','成都','苏州','丽江','景德镇'}
    assert not set(CITIES) & prior_cities
    # Audit exact message overlap against known released train/validation ledgers, never train on new suite.
    forbidden = set()
    release_dirs = [ROOT/'data/post_train/release_current_protocol_v1_v2', ROOT/'data/post_train/count_infeasible_ab_20261025',
                    ROOT/'data/post_train/shared_dpo_20261026/release']
    for folder in release_dirs:
        for p in folder.rglob('*.json*'):
            if 'evaluation' in str(p): continue
            try:
                text=p.read_text(encoding='utf-8'); obj=json.loads(text) if text.lstrip().startswith('[') else None
                rows=obj if isinstance(obj,list) else [json.loads(x) for x in text.splitlines() if x.strip()]
            except (ValueError, UnicodeError): continue
            for row in rows:
                if isinstance(row,dict) and isinstance(row.get('messages'),list): forbidden.add(fingerprint(row['messages']))
    assert not any(fingerprint(c['messages']) in forbidden for c in cases)
    dump(out/'cases.json', cases); dump(out/'audit.json', controls)
    dump(out/'case_catalog.json', [dict(id=c['id'], city=c['city'], category=c['category'], stratum=c['stratum'],
        expected=c['expected'], query=c['state']['query'], modification=c['state'].get('modification_notes'),
        tags=c['tags'], pool_size=len(c['state']['pois']), soft_rubric=c['soft_rubric']) for c in cases])
    names = ['pools.json', 'maps.json', 'cases.json', 'audit.json', 'case_catalog.json']
    dump(out/'FROZEN.json', dict(created_at=datetime.now(timezone.utc).isoformat(), cities=CITIES, cases=50,
        normal=40, hard=10, feasible=45, infeasible=5, training_use='forbidden', seeds=[42,43,44],
        label='prospective heldout, designed from general requirements/prior failure taxonomy; not fully blind human-authored test',
        known_train_message_overlap=0, scanned_prior_message_count=len(forbidden),
        scope='planner + production LangGraph audit/repair suffix; upstream retrieval/weather frozen, restaurant/hotel/ticket/tips excluded',
        hashes={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in names},
        source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCE_FILES}))
    print(json.dumps(dict(stage='frozen', cases=50, cities=CITIES), ensure_ascii=False))


def verify(out=OUT):
    manifest=json.loads((out/'FROZEN.json').read_text(encoding='utf-8'))
    for name,digest in manifest['hashes'].items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=digest: raise RuntimeError('Frozen asset changed: '+name)
    for name,digest in manifest['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise RuntimeError('Protocol changed: '+name)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',action='store_true');p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'FROZEN.json').exists():
        print(json.dumps(verify(out),ensure_ascii=False));return
    pools=capture_pools(out) if args.capture else json.loads((out/'pools.json').read_text(encoding='utf-8'))['pools']
    maps=capture_roads(out,pools) if args.capture else json.loads((out/'maps.json').read_text(encoding='utf-8'))['routes']
    freeze(out,pools,maps)


if __name__=='__main__': main()
