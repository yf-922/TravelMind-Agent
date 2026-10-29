"""Paired development diagnostic: cropped pools versus saved full city pools.

No paid API, live map calls, training, or independent-test access. Full means
six saved candidates, NOT a measured production retrieval distribution.
"""
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_robustness import load_assets
from app.evaluation.route_interest import route_key
from app.planning.schemas import TravelRoute, TravelPlanState
from app.planning.candidate_react import coverage
from scripts.diagnose_sft_transfer import MODELS, switch, rows
from scripts.evaluate_sft_validation import score
from scripts.post_train_runtime import append_json, capture_messages
from scripts.prepare_sft_v2_release import explanation_faults

MODELS['v3'] = ('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v3', 'TravelMind-Planner-7B-SFT-v3')
OUT = ROOT/'data/post_train/candidate_pool_live_v1'


def expanded_state(original, city_pool):
    """Keep original POI facts and order; append only saved same-city POIs."""
    state = copy.deepcopy(original)
    names = {p['name'] for p in state['pois']}
    state['pois'] += [copy.deepcopy(p) for p in city_pool if p['name'] not in names]
    # Only change candidate facts. Preserve even historical coverage wording,
    # so this experiment does not also introduce a new sufficiency instruction.
    return state


def status(stage, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/'status.tmp'
    p.write_text(json.dumps(dict(stage=stage, updated_at=time.time(), paid_calls=0, **extra)), encoding='utf-8')
    p.replace(OUT/'status.json')
    print(json.dumps(dict(stage=stage, **extra)), flush=True)


def build_cases():
    _, maps = load_assets()
    pools = json.loads((OUT/'pools.json').read_text(encoding='utf-8'))['pools']
    if set(pools) != {'南京','上海','三亚'} or any(len(p)<15 for p in pools.values()):
        raise RuntimeError('Need at least 15 saved genuine candidates per city')
    cases = []
    for row in rows(ROOT/'data/post_train/release_v1/validation.ledger.jsonl'):
        state = expanded_state(row['state'], pools[row['state']['destination']])
        missing = []
        for a in state['pois']:
            for b in state['pois']:
                if a['name'] == b['name']:
                    continue
                mode = 'walk' if state.get('max_walking_km') is not None else 'drive'
                if route_key(a['location'], b['location'], mode) not in maps:
                    missing.append([a['name'], b['name'], mode])
        stats, unmet = coverage(TravelPlanState(**state))
        cases.append(dict(id=row['id'], state=state, messages=capture_messages(state),
            old_count=len(row['state']['pois']), full_count=len(state['pois']),
            production_coverage=stats, production_missing=unmet,
            missing_directed_roads=missing))
    return cases, maps


def summarize(group):
    n = len(group)
    plans = [r['plan'] for r in group if r.get('parse_success') and r.get('plan')]
    days = [d for p in plans for d in p['days']]
    visits = [s for d in days for s in d['spots']]
    return dict(n=n, parsed=sum(r['parse_success'] for r in group),
        passed=sum(r['passed'] for r in group),
        joint_success=sum(r['passed'] and not r['explanation_faults'] for r in group),
        duplicate_cases=sum('duplicate_poi' in r['faults'] for r in group),
        faults=dict(Counter(f for r in group for f in r['faults'])),
        mean_spots_per_parsed_plan=len(visits)/len(plans) if plans else None,
        nonempty_day_fraction=sum(bool(d['spots']) for d in days)/len(days) if days else None,
        token_total=sum((r.get('usage') or {}).get('total_tokens', 0) for r in group),
        mean_elapsed_s=sum(r.get('elapsed_s', 0) for r in group)/n if n else None)


def report(cases, results, baselines):
    summary = {}
    lines = ['# 候选池配对诊断', '',
        '开发集30例×3个种子；新增高德真实检索候选，原有候选及属性保持不变。',
        '裁剪池3–4个；扩充池每城20个。检索动作包含风景名胜、博物馆和公园，不能代表所有生产用户的检索分布。',
        '室内标签仍为历史输入快照声明，未独立认证。未修改生产逻辑、训练集或冻结评分器。', '',
        '|模型|候选池|规则通过 /90|重复案例 /90|规则与解释均通过 /90|',
        '|---|---|---:|---:|---:|']
    for tag in ['v2', 'v3']:
        old = [r for r in baselines if r['tag'] == tag]
        new = [r for r in results if r['tag'] == tag]
        for arm, group in [('cropped', old), ('full_saved', new)]:
            summary[f'{tag}_{arm}'] = s = summarize(group)
            lines.append(f"|{tag}|{arm}|{s['passed']}|{s['duplicate_cases']}|{s['joint_success']}|")
        before = {(r['id'], r['seed']):r for r in old}
        summary[f'{tag}_paired'] = dict(
            improved=[(r['id'], r['seed']) for r in new if r['passed'] and not before[(r['id'],r['seed'])]['passed']],
            regressed=[(r['id'], r['seed']) for r in new if not r['passed'] and before[(r['id'],r['seed'])]['passed']])
    lines += ['', '路线充实度（防止少排、空日造成表面改善）：', '',
        '|模型|候选池|每条路线平均景点数|非空天比例|',
        '|---|---|---:|---:|']
    for tag in ['v2','v3']:
        for arm in ['cropped','full_saved']:
            s = summary[f'{tag}_{arm}']
            lines.append(f"|{tag}|{arm}|{s['mean_spots_per_parsed_plan']}|{s['nonempty_day_fraction']}|")
    lines += ['', '道路证据：缺失有向道路仍计road_unverified，不用直线距离冒充道路。',
        '新请求保持原始模型采样参数与2048输出上限；服务上下文改为8192以容纳真实20候选输入，未截断提示词。原服务上下文为4096。',
        '新增候选也会改变地点类型、属性未知率和上下文长度，因此不能把差异全部归因为候选数量。',
        '三种子属于同题重复生成，不能当作90个独立问题。不用此诊断宣称独立泛化能力。',
        '后续真正的生产分布测试需冻结检索输出：原始/过滤后候选、查询动作、去重、室内证据与未知率、开放时间及道路覆盖；不能编造候选或属性补齐规模。', '']
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT/'docs/post_train_candidate_pool_diagnostic.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases, maps = build_cases()
    baselines = rows(ROOT/'data/post_train/sft_contract_eval_v1/results.jsonl') + rows(ROOT/'data/post_train/sft_contract_eval_v3/results.jsonl')
    baseline = {(r['tag'],r['id'],r['seed']):r for r in baselines if r['tag'] in ['v2','v3']}
    assert len(baseline) == 180
    hard = {c['id']:c for c in json.loads((ROOT/'data/post_train/hard_supplement_cases.json').read_text(encoding='utf-8'))}
    for row in cases:
        if row['id'] in hard:
            hard[row['id']] = dict(hard[row['id']], state=row['state'])
    payload = json.dumps(cases, ensure_ascii=False, sort_keys=True)
    binding = dict(case_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        maps_sha256=hashlib.sha256(json.dumps(maps,sort_keys=True).encode()).hexdigest(),
        scope='development_live_amap_expansion_not_independent_test', paid_calls=0, server_context=8192,
        seeds=[42,43,44], temperature=.3, max_tokens=2048, models=['v2','v3'])
    meta = OUT/'binding.json'
    if meta.exists() and json.loads(meta.read_text(encoding='utf-8')) != binding:
        raise RuntimeError('Immutable experiment binding changed')
    meta.write_text(json.dumps(binding, indent=2), encoding='utf-8')
    (OUT/'cases.json').write_text(payload, encoding='utf-8')
    if '--dry-run' in sys.argv:
        print(json.dumps(dict(cases=len(cases), requests=180, paid_calls=0,
            original_counts=dict(Counter(c['old_count'] for c in cases)),
            expanded_counts=dict(Counter(c['full_count'] for c in cases)),
            cases_missing_roads=sum(bool(c['missing_directed_roads']) for c in cases),
            cases_below_production_coverage=sum(bool(c['production_missing']) for c in cases))))
        return
    output = OUT/'results.jsonl'
    done = {(r['tag'],r['id'],r['seed']) for r in rows(output)} if output.exists() else set()
    def worker(tag, row, seed):
        original = baseline[tag,row['id'],seed]
        request = dict(original['request'], messages=row['messages'])
        assert request['model'] == MODELS[tag][1]
        result = dict(tag=tag,id=row['id'],seed=seed,request=request,
            parse_success=False,passed=False,faults=[],explanation_faults=[])
        started = time.perf_counter()
        try:
            req = urllib.request.Request('http://10.14.3.82:8000/v1/chat/completions',
                data=json.dumps(request,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=90) as reply:
                raw = json.load(reply)
            result.update(response=raw,usage=raw.get('usage'))
            if raw['choices'][0]['finish_reason'] != 'stop':
                raise ValueError('incomplete_generation')
            plan = TravelRoute.model_validate_json(raw['choices'][0]['message']['content']).model_dump(mode='json')
            result.update(parse_success=True,plan=plan)
            result['faults'] = score(plan,row,maps,hard)
            result['explanation_faults'] = explanation_faults(plan)
            result['passed'] = not result['faults']
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc)[:300])
        result['elapsed_s'] = time.perf_counter()-started
        return result
    try:
        for tag in ['v2','v3']:
            pending = [(r,s) for s in [42,43,44] for r in cases if (tag,r['id'],s) not in done]
            if not pending:
                continue
            status('loading_model',model=tag)
            switch(tag,max_model_len=8192)
            status('evaluating',model=tag,pending=len(pending))
            with ThreadPoolExecutor(max_workers=2) as pool:
                for future in as_completed([pool.submit(worker,tag,r,s) for r,s in pending]):
                    r = future.result()
                    append_json(output,r)
                    print(json.dumps({k:r[k] for k in ['tag','id','seed','passed','faults']}),flush=True)
    finally:
        status('restoring_base')
        switch('base')
    result = rows(output)
    if len(result) != 180:
        raise RuntimeError('Incomplete run')
    status('capturing_selected_roads')
    maps = capture_selected_roads(cases,result,maps)
    rescored = []
    by_id = {r['id']:r for r in cases}
    for r in result:
        r = copy.deepcopy(r)
        if r.get('parse_success'):
            r['faults'] = score(r['plan'],by_id[r['id']],maps,hard)
            r['passed'] = not r['faults']
        rescored.append(r)
    (OUT/'rescored_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rescored),encoding='utf-8')
    summary = report(cases,rescored,baselines)
    status('completed',total=len(result),summary=summary)


def capture_selected_roads(cases, results, maps):
    """Fetch only model-selected missing legs, once; retain failures explicitly."""
    from app.core.env import load_local_env
    from app.planning.helpers import amap_key
    from app.providers.amap.direction import plan_route_distance
    load_local_env()
    saved = OUT/'maps.json'
    if saved.exists():
        maps.update(json.loads(saved.read_text(encoding='utf-8'))['routes'])
    requested = {}
    by_id = {r['id']:r['state'] for r in cases}
    for r in results:
        if not r.get('parse_success'):
            continue
        state = by_id[r['id']]
        locations = {p['name']:p['location'] for p in state['pois']+state.get('main_meal_candidates',[])}
        for day in r['plan']['days']:
            events = list(day['spots']) + [dict(s,name=s['restaurant_name']) for s in r['plan'].get('meal_slots',[]) if s['day']==day['day']]
            events.sort(key=lambda s:s['start_time'])
            for a,b in zip(events,events[1:]):
                if a['name'] not in locations or b['name'] not in locations:
                    continue
                origin,dest = locations[a['name']],locations[b['name']]
                mode = 'walk' if state.get('max_walking_km') is not None else 'drive'
                k = route_key(origin,dest,mode)
                if k not in maps:
                    requested[k]=(origin,dest,mode)
    calls = OUT/'road_calls.jsonl'
    attempted = {r['route_key'] for r in rows(calls) if r['event']=='started'} if calls.exists() else set()
    for k,(origin,dest,mode) in requested.items():
        if k in attempted:
            continue
        if len(attempted)>=400:
            break
        append_json(calls,dict(event='started',route_key=k,time=time.time()))
        attempted.add(k)
        try:
            leg = plan_route_distance(origin,dest,amap_key(),mode=mode)
            if not leg or leg.get('duration_min') is None:
                raise ValueError('no_valid_amap_road_response')
            maps[k]=leg
            saved.write_text(json.dumps(dict(routes=maps,captured_at=time.time()),ensure_ascii=False,indent=2),encoding='utf-8')
            append_json(calls,dict(event='completed',route_key=k))
        except Exception as exc:
            append_json(calls,dict(event='failed',route_key=k,error_type=type(exc).__name__))
        time.sleep(.15)
    return maps


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc)[:500])
        raise
