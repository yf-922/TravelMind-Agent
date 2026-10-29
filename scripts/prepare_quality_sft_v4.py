"""Quality-first next release. Historical sources and frozen evaluations stay intact.

Historical schedules are rechecked; prose is replaced with transparent, mechanically
grounded target text (NOT represented as a new teacher answer). Fresh training
requirements use separately captured POI/road assets, never evaluation answers.
No training starts until the generated candidates have received a separate review.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.evaluation.task_robustness import load_assets
from app.planning.schemas import TravelRoute
from scripts import build_representative_eval50 as assets
from scripts.audit_sft_contract_targets import aggregate_duration_faults
from scripts.export_llamafactory_dataset import export
from scripts.post_train_runtime import Meter, append_json, capture_messages, quality_faults, replay_check
from scripts.prepare_sft_v2_release import explanation_faults, read_rows
from scripts.prepare_targeted_city_sft import additional_faults
from scripts.run_hard_post_train_synthesis import audit_plan

OUT = ROOT / 'data/post_train/quality_sft_v4'
CITIES = ['南京', '成都', '苏州']
CALL_CAP = 72


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def frozen(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise RuntimeError('Immutable artifact changed: ' + path.name)
    else:
        dump(path, value)


def status(stage, **values):
    dump(OUT / 'status.json', dict(stage=stage, updated_at=time.time(),
        training_started=False, dpo_auto_start=False, **values))
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=False), flush=True)


def schedule_facts(plan):
    return [dict(day=d['day'], visits=[dict(name=s['name'], start=s['start_time'],
        end=s['end_time'], minutes=assets.minutes(s['end_time'])-assets.minutes(s['start_time']))
        for s in d['spots']]) for d in plan['days']]


def grounded_target(plan, state, infeasible_reason=None):
    """Change prose only; preserve every schedule/name/time/meal/source-independent fact.

    This is deterministic target authorship, with explicit provenance at export.
    Avoid indoor, geography, booking, preference or transfer claims absent evidence.
    """
    result = copy.deepcopy(plan)
    facts = schedule_facts(result)
    if infeasible_reason is not None:
        if any(d['visits'] for d in facts) or result.get('meal_slots'):
            raise ValueError('Never turn a partial route into a certified refusal')
        result['reasoning'] = infeasible_reason
        result['notes'] = '不能满足当前全部约束，因此各天不填充景点；不虚构地点或重复凑数。'
    else:
        result['reasoning'] = '；'.join(f"第{d['day']}天安排{len(d['visits'])}个景点，参观时长合计{sum(s['minutes'] for s in d['visits'])}分钟" for d in facts) + '。具体名称、顺序和时段见逐日安排。'
        result['notes'] = '开放及交通依据为输入快照，不代表实时运营保证。未提供的预约、客流及室内范围不作推断。'
    for d in result['days']:
        d['theme'] = f"第{d['day']}天：{len(d['spots'])}个景点"
    result['selection_reasons'] = [dict(name=s['name'], reason=f"候选池内地点；第{d['day']}天{s['start']}–{s['end']}参观，安排{s['minutes']}分钟。", source_ids=[])
        for d in facts for s in d['visits']]
    # A concern is not needed when no known constraint has been waived.
    result['modification_concern'] = ''
    TravelRoute.model_validate(result)
    return result


def strict_start_faults(plan, query):
    boundaries = [assets.minutes(t) for t in re.findall(r'(\d{1,2}:\d{2})后开始', query)]
    return ['strict_start_endpoint'] if any(assets.minutes(s['start_time']) <= t
        for t in boundaries for d in plan['days'] for s in d['spots']) else []


def legacy_indoor_faults(plan, state):
    """Quarantine weather-sensitive reliance on an unscoped legacy true label."""
    by = {p['name']: p for p in state['pois']}
    weather = state.get('weather_forecast', [])
    errors = []
    for day in plan['days']:
        forecast = weather[day['day']-1] if day['day'] <= len(weather) else {}
        try:
            hot = float(forecast.get('day_temp', 0)) >= 38
        except (ValueError, TypeError):
            hot = False
        indoor_only = state.get('rain_indoor_priority') or '只安排室内' in state.get('query', '')
        for s in day['spots']:
            p = by.get(s['name'], {})
            overlaps = assets.minutes(s['start_time']) < 900 and assets.minutes(s['end_time']) > 660
            if p.get('indoor') is True and (indoor_only or hot and overlaps):
                # New-city scoped declarations are distinguished from old category inference.
                if not p.get('indoor_scope') or not p.get('indoor_source') or p.get('indoor_source') == 'amap_category_inference':
                    errors.append('weather_relies_on_unscoped_legacy_indoor_label')
    return sorted(set(errors))


def audit_history():
    folder = ROOT / 'data/post_train'
    source = folder / 'release_v2/train.ledger.jsonl'
    original = read_rows(source)
    _, maps_old = load_assets()
    maps_new = json.loads((folder/'new_city_sft_v1/maps.json').read_text(encoding='utf-8'))['routes']
    cases_new = {c['id']: c for c in json.loads((folder/'new_city_sft_v1/cases.json').read_text(encoding='utf-8'))}
    cases_hard = {c['id']: c for c in json.loads((folder/'hard_supplement_cases.json').read_text(encoding='utf-8'))}
    kept, decisions = [], []
    for source_row in original:
        row = copy.deepcopy(source_row)
        plan, state = row['chosen'], row['state']
        errors = []
        if row['id'] == 'synth-0127':
            errors.append('known_repair_narrative_leakage')
        if row['id'] in cases_new:
            errors += audit_plan(plan, cases_new[row['id']], maps_new)[0]
        elif row['id'] in cases_hard:
            errors += audit_plan(plan, cases_hard[row['id']], maps_old)[0]
        else:
            checks, checked = replay_check(plan, state, maps_old)
            errors += checks + quality_faults(checked)
        errors += strict_start_faults(plan, state['query']) + legacy_indoor_faults(plan, state)
        # Prose defects are logged separately. Reauthorship never fixes bad routes.
        old_prose_faults = explanation_faults(plan) + aggregate_duration_faults(plan)
        target = grounded_target(plan, state)
        errors += explanation_faults(target) + aggregate_duration_faults(target)
        decision = dict(id=row['id'], decision='quarantine' if errors else 'keep_reauthored_prose',
            route_faults=sorted(set(errors)), old_narrow_prose_faults=old_prose_faults,
            original_target_sha256=hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            preserved_schedule=[[(s['name'], s['start_time'], s['end_time'], s['period']) for s in d['spots']] for d in plan['days']] ==
                [[(s['name'], s['start_time'], s['end_time'], s['period']) for s in d['spots']] for d in target['days']])
        decisions.append(decision)
        if errors:
            continue
        row.update(chosen=target, messages=capture_messages(state), original_messages=row['messages'],
            original_teacher_target=plan, rejected=None, rejected_rule_errors=[],
            prompt_origin='current_production_capture_of_historical_state',
            target_origin='deterministically_reauthored_grounded_prose_with_unchanged_teacher_schedule',
            semantic_review_scope='rule/rubric replay and grounded prose construction; not exhaustive human certification')
        kept.append(row)
    payload = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in kept)
    p = OUT/'historical_cleaned.ledger.jsonl'
    if p.exists() and p.read_text(encoding='utf-8') != payload:
        raise RuntimeError('Cleaned history changed; version the run')
    p.write_text(payload, encoding='utf-8')
    summary = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original=len(original), kept=len(kept), quarantined=len(original)-len(kept), decisions=decisions,
        prose_change='all explanation/theme/selection text reauthored from final schedule; original teacher target retained',
        limitations=['No new teacher judgment or human semantic certification',
            'Legacy candidate labels are not upgraded; weather-sensitive unscoped true labels quarantined',
            'Non-weather POI factual labels retain historical provenance; not independently field verified'])
    frozen(OUT/'historical_audit.json', summary)
    print(json.dumps(dict(history_read=len(original), kept=len(kept), quarantined=len(original)-len(kept))), flush=True)
    return kept


def capture_training_assets():
    """Reuse retrieval code, with a NEW output namespace, not benchmark pools/cases."""
    saved_out, saved_cities = assets.OUT, assets.CITIES
    try:
        assets.OUT = OUT/'retrieval'
        assets.CITIES = CITIES
        assets.OUT.mkdir(parents=True, exist_ok=True)
        pools = assets.prune_reviewed_pools(assets.capture_pools())
        maps = assets.capture_roads(pools)
        frozen(OUT/'training_pools.json', dict(pools=pools, origin='separate fresh Amap training retrieval; no evaluation case or witness copied'))
        return pools, maps
    finally:
        assets.OUT, assets.CITIES = saved_out, saved_cities


def input_collision_check(cases):
    keys = [json.dumps(capture_messages(c['state']), ensure_ascii=False, sort_keys=True) for c in cases]
    if len(keys) != len(set(keys)):
        raise RuntimeError('Duplicate new training messages')
    forbidden = set()
    for name in ['representative_eval50_v1/cases.json', 'heldout_evidence_v1/cases.json']:
        for c in json.loads((ROOT/'data/post_train'/name).read_text(encoding='utf-8')):
            forbidden.add(json.dumps(capture_messages(c['state']), ensure_ascii=False, sort_keys=True))
    for r in read_rows(ROOT/'data/post_train/release_v1/validation.ledger.jsonl'):
        forbidden.add(json.dumps(capture_messages(r['state']), ensure_ascii=False, sort_keys=True))
    if set(keys) & forbidden or any(c['state']['destination'] in ['丽江', '景德镇'] for c in cases):
        raise RuntimeError('Evaluation/validation leakage')


def build_cases(pools, maps):
    cases = []
    for ci, city in enumerate(CITIES):
        full = pools[city]
        for i in range(16):
            rng = random.Random(704001+ci*100+i)
            local = copy.deepcopy(rng.sample(full, 15+(i%3)))
            # Distinct requirements, not evaluation cases with changed city/name.
            rule = dict(required=[], excluded=[], counts=[[2, 2]], earliest=525, latest=1065,
                min_visit=[45, 75, 105][i%3], forbidden_windows=[])
            temp = '27'
            category = 'heat_opening' if i<4 else 'multi_day_joint' if i<8 else 'current_priority' if i<10 else 'canonical_reference' if i<12 else 'count_exact_pair' if i<14 else 'count_upper_pair'
            if i<4:
                temp = ['38.2', '39', '37.7', '38.5'][i]
                rule.update(forbidden_windows=[[690, 795]] if i%2 else [], required=[local[0]['name']])
            elif i<8:
                rule.update(counts=[[2, 2], [1, 1]] if i%2 else [[1, 1], [1, 1], [2, 2]], forbidden_windows=[[735, 810]])
                temp = '38.2' if i in [5, 6] else '27'
            elif i<10:
                rule.update(required=[local[1]['name']], excluded=[local[0]['name']], earliest=616+i)
            elif i<12:
                named = [p for p in local if '(' in p['name'] or '（' in p['name']]
                rule.update(required=[(named or local)[0]['name']], earliest=555+i)
            elif i<14:
                local = copy.deepcopy(cases[-1]['state']['pois']) if i==13 else local[:3]
                rule.update(counts=[[2, 2], [1, 1]] if i==12 else [[2, 2], [2, 2]], min_visit=45)
            else:
                local = copy.deepcopy(cases[-1]['state']['pois']) if i==15 else local[:3]
                rule.update(counts=[[2, 2], [1, 1]] if i==14 else [[2, 2], [2, 2]], min_visit=45)
                # Upper bounds do not require filling every day's cap.
                if i==15:
                    rule['counts']=[[1, 2], [1, 2]]
            days = len(rule['counts'])
            count_text = '；'.join(f'第{d}天恰好{lo}个景点' for d,(lo,hi) in enumerate(rule['counts'],1))
            if i==15:
                count_text = '每天至少1个、最多2个景点，不要求填满上限'
            query = (f'为{city}安排{days}天行程，2026年11月4日出发。{count_text}。'
                f"各景点至少参观{rule['min_visit']}分钟，所有天之间也不能重复景点。"
                f"每天不早于{assets.clock(rule['earliest'])}开始，须在{assets.clock(rule['latest'])}前结束（允许等于端点）。"
                '只安排景点，不填写餐馆。候选地点是可选集合，不要求全部使用。'
                '若不存在同时满足所有当前要求的路线，保持各天景点为空并具体解释原因；不擅自少排、不重复凑数。'
                '解释必须依据最终时刻表与输入事实，不推断室内、预约或排队信息。')
            if rule['required']:
                query += f"必须参观「{rule['required'][0]}」。"
            if rule['forbidden_windows']:
                a,b = rule['forbidden_windows'][0]
                query += f'{assets.clock(a)}至{assets.clock(b)}留空休息，不与任何景点参观重叠。'
            query += f'最高温{temp}°C；按输入协议处理高温，候选室内未知保持未知。'
            state = dict(destination=city, query=query, days=days, max_per_day=3, pois=local,
                main_meal_candidates=[], route_distance_mode='drive', travel_start_date='2026-11-04',
                travel_end_date=(date(2026,11,4)+timedelta(days=days-1)).isoformat(),
                candidate_pool_status='ready' if len(local)>=15 else 'insufficient',
                weather_forecast=[dict(date=(date(2026,11,4)+timedelta(days=d)).isoformat(), day_temp=temp,
                    night_temp='21', day_weather='晴', night_weather='晴', is_bad=False) for d in range(days)])
            if i in [8, 9]:
                state['modification_notes'] = f"本次要求优先：不要参观「{local[0]['name']}」，改为必须参观「{local[1]['name']}」；每天只安排2个。"
                state['query'] = state['query'].replace(f"必须参观「{local[1]['name']}」。", f"先前希望参观「{local[0]['name']}」，先前每日3个的想法由本次修改覆盖。")
            infeasible = i==13
            case = dict(id=f'quality4-{ci+1:02d}-{i+1:02d}', category=category,
                training_use='allowed', state=state, rubric=rule,
                expected='explained_infeasible' if infeasible else 'feasible')
            if infeasible:
                reason = '只有3个可选景点，2天每天恰好2个且全程不重复共需要4个不同景点，候选数量不足，无法满足全部要求。'
                case['rubric']['reason_terms']=['候选', '数量', '3个', '三个']
                case['infeasibility_proof'] = dict(kind='pigeonhole', available=3, required=4, explanation=reason)
            try:
                witness = assets.make_witness(case, maps)
            except RuntimeError:
                # A mandatory candidate may have incompatible opening hours. Select
                # a different required POI BEFORE freezing or contacting the teacher.
                if not rule['required'] or i in [8, 9]:
                    raise
                for p in local[1:]:
                    old = rule['required'][0]
                    rule['required']=[p['name']]
                    state['query']=state['query'].replace(f'必须参观「{old}」', f'必须参观「{p["name"]}」')
                    try:
                        witness=assets.make_witness(case,maps)
                        break
                    except RuntimeError:
                        continue
                else:
                    raise RuntimeError('No independently feasible training requirement')
            case['feasibility_witness'] = witness
            cases.append(case)
    input_collision_check(cases)
    frozen(OUT/'cases.json', cases)
    frozen(OUT/'maps.json', dict(routes=maps))
    frozen(OUT/'preflight.json', dict(cases=len(cases), feasible=sum(c['expected']=='feasible' for c in cases),
        infeasible=sum(c['expected']!='feasible' for c in cases), candidate_sizes=sorted(set(len(c['state']['pois']) for c in cases)),
        witness_check_passed=len(cases), call_cap=CALL_CAP, reviewer_calls=0,
        data_origin='fresh training retrieval and separately constructed requirements; no evaluation answers',
        leakage_scope='exact-message collision check plus independent requirement construction; shared real venues are possible'))
    return cases


def check_candidate(plan, case, maps):
    try:
        errors=assets.score_case(plan,case,maps)
        if case['expected']=='feasible':
            errors+=additional_references(plan)
        return sorted(set(errors))
    except (ValueError, TypeError, KeyError):
        return ['invalid_time_or_structure']


def additional_references(plan):
    selected={s['name'] for d in plan['days'] for s in d['spots']}
    return ['unsupported_selection_reference'] if any(r['name'] not in selected or r.get('source_ids') for r in plan.get('selection_reasons',[])) else []


def run_synthesis(cases, maps, concurrency=4, limit=None):
    load_local_env()
    own=OUT/'synthesis.calls.jsonl'
    external=sum(json.loads(l).get('event')=='started' for p in (ROOT/'data/post_train').rglob('*.calls.jsonl')
        if p.resolve()!=own.resolve() for l in p.read_text(encoding='utf-8').splitlines() if l.strip())
    if external+CALL_CAP>1000:
        raise RuntimeError('Global teacher budget cannot reserve this run')
    model=os.environ.get('POST_TRAIN_TEACHER_MODEL','gpt-6.1-sol')
    meter=Meter(own,model,CALL_CAP,90)
    ledger=OUT/'synthesis.jsonl'
    done={r['id'] for r in read_rows(ledger)} if ledger.exists() else set()
    pending=[c for c in cases if c['id'] not in done]
    # Pilot covers all cities and most important feasible/infeasible mechanisms.
    pending.sort(key=lambda c:c['id'] not in {'quality4-01-01','quality4-02-06','quality4-03-09','quality4-01-12','quality4-02-14','quality4-03-15'})
    if limit:
        pending=pending[:limit]
    stop=threading.Event()
    repair_lock=threading.Lock()
    # Reserve first-round calls for every remaining case, including across resume.
    repairs_used=sum(json.loads(l).get('event')=='started' and json.loads(l).get('identity',{}).get('round')==1
        for l in own.read_text(encoding='utf-8').splitlines()) if own.exists() else 0
    repairs_left=max(0,CALL_CAP-len(cases)-repairs_used)
    def worker(case):
        nonlocal repairs_left
        if stop.is_set():
            return None
        state=copy.deepcopy(case['state'])
        row=dict(id=case['id'], state=copy.deepcopy(state), category=case['category'], messages=capture_messages(state),
            mock=False, model=model, status='failed', chosen=None, rejected=None, rounds=[],
            target_origin='teacher_schedule_with_deterministically_reauthored_grounded_prose',
            semantic_review_status='pending_agent_review')
        for rnd in range(2):
            try:
                messages=capture_messages(state)
                plan,call=meter.call(messages,dict(sample=case['id'],round=rnd,run='quality_sft_v4'))
                errors=check_candidate(plan,case,maps)
                row['rounds'].append(dict(round=rnd,messages=messages,output=plan,rule_errors=errors,
                    call_id=call['call_id'],usage=call.get('usage'),elapsed_s=call['elapsed_s'],
                    raw_narrow_explanation_faults=explanation_faults(plan)+aggregate_duration_faults(plan)))
                if not errors:
                    reason=case.get('infeasibility_proof',{}).get('explanation')
                    target=grounded_target(plan,case['state'],reason)
                    final_errors=check_candidate(target,case,maps)+explanation_faults(target)+aggregate_duration_faults(target)
                    if final_errors:
                        raise RuntimeError('Grounded target failed recheck')
                    row.update(status='candidate_accepted',chosen=target,original_teacher_target=plan)
                    break
                if rnd==0:
                    with repair_lock:
                        if repairs_left<=0:
                            break
                        repairs_left-=1
                    state=copy.deepcopy(case['state'])
                    state.update(route=plan['days'], route_modify_opinion='本地确定性检查未通过：'+','.join(errors)+'。重新输出满足原始当前要求的完整最终方案，不提返工过程。', review_round=1)
            except Exception as exc:
                row['error_type']=type(exc).__name__
                safe=str(exc) if isinstance(exc,RuntimeError) and str(exc).startswith('API call ') else ''
                row['api_error']=safe
                if any(f'status={s}' in safe for s in [400,401,403,404,429]):
                    stop.set()
                break
        return row
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures=[pool.submit(worker,c) for c in pending]
        for f in as_completed(futures):
            row=f.result()
            if row is not None:
                append_json(ledger,row)
                status('synthesizing',last_id=row['id'],last_status=row['status'],calls=meter.count,
                    completed=len(read_rows(ledger)),target=len(cases),global_started=external+meter.count)
    records=read_rows(ledger) if ledger.exists() else []
    summary=dict(completed=len(records),target=len(cases),accepted=sum(r['status']=='candidate_accepted' for r in records),
        calls=meter.count,global_started=external+meter.count,
        tokens=sum((rd.get('usage') or {}).get('total_tokens',0) for r in records for rd in r['rounds']),
        failed=[r['id'] for r in records if r['status']!='candidate_accepted'],
        agent_review='pending; no training release automatically certified', paid_reviewer_calls=0)
    dump(OUT/'synthesis_summary.json',summary)
    status('synthesis_complete_pending_quality_review' if len(records)==len(cases) else 'pilot_complete',**summary)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',action='store_true')
    p.add_argument('--assets',action='store_true')
    p.add_argument('--limit',type=int)
    p.add_argument('--concurrency',type=int,default=4)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ['TRAVEL_INTEREST_KNOWLEDGE_ENABLED']='0'
    status('historical_audit')
    audit_history()
    if args.assets:
        status('capturing_separate_training_assets')
        pools,maps=capture_training_assets()
        cases=build_cases(pools,maps)
    elif (OUT/'cases.json').exists():
        cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
        maps=json.loads((OUT/'maps.json').read_text(encoding='utf-8'))['routes']
    else:
        status('history_audit_complete_pending_training_assets')
        return
    if args.run:
        run_synthesis(cases,maps,args.concurrency,args.limit)
    else:
        status('preflight_passed',cases=len(cases),paid_llm_calls=0)


if __name__=='__main__':
    main()
