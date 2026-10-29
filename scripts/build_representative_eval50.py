"""Build a frozen 40-normal/10-hard development suite from genuine Amap facts.

No teacher calls or model generation. All witnesses and adversarial controls
are scored before freezing. Candidate indoor attributes are never guessed.
"""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.core.env import load_local_env
from app.planning.helpers import amap_key
from app.providers.amap.poi import search_city_pois, poi_to_spot
from app.planning.schemas import TravelPlanState, TravelRoute
from app.evaluation.route_interest import route_key
from scripts.post_train_runtime import append_json, capture_messages, replay_check
from scripts.prepare_heldout_evidence import normalize_batch, clock, fingerprint, opening_window
from scripts.prepare_targeted_city_sft import conservative_day_window
from scripts.prepare_sft_v2_release import explanation_faults

OUT=ROOT/'data/post_train/representative_eval50_v1'
CITIES=['南京','上海','三亚','成都','苏州']
MANUAL_EXCLUSIONS={
    '南京':{'钟山风景名胜区','南京城墙景区(神策门至太平门)'},
    '上海':{'人民公园-相亲角','万国建筑博览群'},
    '三亚':{'蜈支洲岛旅游风景区-情人岛','三亚西岛海洋文化旅游区','西岛渔村',
          '亚龙湾风景区-天下第一湾','三亚·中交凤凰岛'},
    '成都':{'Paulmann柏曼灯具(成都旗舰店)','锦城湖2号湖区','新川之心公园东区'},
    '苏州':{'樱花园'},
}


def prune_reviewed_pools(pools):
    """Explicit review decisions, recorded rather than silently dropping points."""
    result={}
    for city,pool in pools.items():
        result[city]=[p for p in pool if p['name'] not in MANUAL_EXCLUSIONS[city]]
        if len(result[city])<18:
            raise RuntimeError('Manual review leaves too few candidates: '+city)
    dump(OUT/'manual_candidate_review.json',dict(excluded={c:sorted(v) for c,v in MANUAL_EXCLUSIONS.items()},
        rationale='commercial shop misretrieval; parent/subsite overlap; ambiguous local name; island/ferry/access not covered by driving snapshots',
        retained={c:len(p) for c,p in result.items()},scope='name/address/category review, not independent field verification'))
    return result


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def normalized_name(name):
    text=re.sub(r'[\s()（）·\-]','',name)
    return re.sub(r'(?:旅游风景区|风景区|旅游区|景区)$','',text)


def admissible(raw,pool):
    p=poi_to_spot(raw)
    if not p or not p.get('name'):
        return None,'invalid'
    if p.get('rating') is not None and p['rating']<4.5:
        return None,'production_rating_filter'
    if any(t in p['name'] for t in ['停车','售票','游客中心','游客服务','出入口','检票','洗手间','码头','塑像','祭堂','情侣园']):
        return None,'facility_or_known_subsite'
    if p['name']=='牛王岭' and '西岛' in p.get('address',''):
        return None,'west_island_subsite'
    if '博物院' in p['name'] and any(t in p['name'] for t in ['展厅','馆藏','特展']):
        return None,'museum_subsite'
    for q in pool:
        a,b=normalized_name(p['name']),normalized_name(q['name'])
        if a==b or min(len(a),len(b))>=3 and (a in b or b in a):
            return None,'duplicate_or_parent_child_name'
        if math.hypot(p['location']['lat']-q['location']['lat'],.85*(p['location']['lng']-q['location']['lng']))<.0015:
            return None,'near_coincident_150m'
    text=(p.get('open_time') or '').replace('–','-').replace('—','-')
    # Snapshot rules apply Wed-Fri. Do not silently assume a day-specific closure
    # is open. Ambiguous listings are kept in raw responses, not in this suite.
    if re.search(r'周[三四五].{0,8}(?:闭馆|关闭|不开放)',text) or re.search(r'周[三四五][、至和-]周?[一二三四五六日天].{0,8}(?:闭馆|关闭|不开放)',text):
        return None,'weekday_closure_unresolved'
    window=conservative_day_window(text)
    if not window or window[1]-window[0]<180 or window[2]-window[0]<90:
        return None,'opening_missing_or_ambiguous'
    p.update(raw_open_time=p['open_time'],open_time=f'{clock(window[0])}-{clock(window[1])} 最晚进入{clock(window[2])}',
        indoor=None,indoor_source='unknown',indoor_scope='无独立室内证据，不根据名称推断',
        category=str(raw.get('type') or ''),source='amap',amap_id=raw.get('id'))
    return p,None


def capture_pools():
    cached=OUT/'pools.json'
    if cached.exists():
        return json.loads(cached.read_text(encoding='utf-8'))['pools']
    load_local_env()
    budget={'remaining':90,'used':0}
    pools={}
    queries=[('', '110000'),('公园','110000'),('博物馆','140100'),('风景名胜','110000'),
             ('美术馆','140100'),('纪念馆','140100'),('古迹','110000'),('景点','110000')]
    for city in CITIES:
        pools[city]=[]
        for index,(query,types) in enumerate(queries):
            raw_path=OUT/f'poi_{city}_{index}.json'
            if raw_path.exists():
                raw=json.loads(raw_path.read_text(encoding='utf-8'))['rows']
            else:
                append_json(OUT/'poi_calls.jsonl',dict(event='started',city=city,query=query,time=time.time()))
                raw=search_city_pois(city,amap_key(),keywords=query,types=types,offset=25,request_budget=budget)
                dump(raw_path,dict(city=city,query=query,types=types,rows=raw,captured_at=datetime.now(timezone.utc).isoformat()))
                append_json(OUT/'poi_calls.jsonl',dict(event='completed',city=city,query=query,count=len(raw)))
            accepted=0
            for row in raw:
                if str(row.get('cityname','')).removesuffix('市')!=city:
                    continue
                p,reason=admissible(row,pools[city])
                if reason:
                    append_json(OUT/'poi_filters.jsonl',dict(city=city,name=row.get('name'),reason=reason))
                    continue
                p['provenance']=dict(raw_file=raw_path.name,raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    opening_method='conservative intersection of stated clock intervals; Wed-Fri scenario; not future operating guarantee')
                pools[city].append(p)
                accepted+=1
                # A mix of query channels rather than filling the pool entirely
                # with the first category. Later channels can fill remaining slots.
                if len(pools[city])>=25 or accepted>=8:
                    break
            print(json.dumps(dict(stage='poi',city=city,query=query,count=len(pools[city])),ensure_ascii=False),flush=True)
            if len(pools[city])>=25:
                break
        if len(pools[city])<18:
            raise RuntimeError(f'{city}: only {len(pools[city])} usable genuine candidates; refuse fabricated fill')
    dump(cached,dict(pools=pools,search_http_this_run=budget['used'],captured_at=datetime.now(timezone.utc).isoformat(),
        scope='purpose-designed development retrieval snapshot, not measured production traffic distribution'))
    return pools


def capture_roads(pools):
    path=OUT/'road_calls.jsonl'
    ledger=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    maps={}; done=set()
    for r in ledger:
        if r['event']=='completed' and r['response'].get('status')=='1':
            maps.update(normalize_batch(r['response'],r['origins'],r['destination'],r['id']))
            done.add(r['id'])
    attempts=sum(r['event']=='started' for r in ledger)
    load_local_env()
    for city,pool in pools.items():
        for dest in pool:
            origins=[p for p in pool if p['name']!=dest['name']]
            ident=fingerprint([city,origins,dest])
            if ident in done or all(route_key(p['location'],dest['location'],'drive') in maps for p in origins):
                continue
            if attempts>=160:
                raise RuntimeError('Road batch call budget 160 exhausted')
            event=dict(id=ident,city=city,origins=origins,destination=dest,time=time.time())
            append_json(path,dict(event,event='started'));attempts+=1
            point=lambda p:f"{p['location']['lng']},{p['location']['lat']}"
            params=urllib.parse.urlencode(dict(key=amap_key(),origins='|'.join(map(point,origins)),destination=point(dest),type=1,output='json'))
            try:
                with urllib.request.urlopen('https://restapi.amap.com/v3/distance?'+params,timeout=25) as reply:
                    raw=json.load(reply)
            except Exception as exc:
                append_json(path,dict(event,event='failed',error_type=type(exc).__name__))
                raise RuntimeError('Road batch failed: '+type(exc).__name__) from None
            append_json(path,dict(event,event='completed',response=raw))
            if raw.get('status')!='1':
                raise RuntimeError('Amap road rejected: '+str(raw.get('infocode')))
            maps.update(normalize_batch(raw,origins,dest,ident))
            print(json.dumps(dict(stage='roads',city=city,batches=attempts,edges=len(maps))),flush=True)
            time.sleep(1.05)
    required={route_key(a['location'],b['location'],'drive') for pool in pools.values() for a in pool for b in pool if a['name']!=b['name']}
    missing=required-maps.keys()
    dump(OUT/'maps.json',dict(routes=maps,batches=attempts,missing=len(missing)))
    if missing:
        raise RuntimeError(f'{len(missing)} directed roads unavailable; cannot certify all-candidate evaluation')
    return maps


def minutes(t):
    h,m=map(int,t.split(':'))
    return 60*h+m


def score_case(plan,case,maps):
    """Explicit rubric + production raw-output checks; never an LLM judge."""
    try:
        plan=TravelRoute.model_validate(plan).model_dump(mode='json')
    except Exception:
        return ['schema_parse']
    rule=case['rubric']; state=case['state']
    days=plan['days']; spots=[s for d in days for s in d['spots']]
    names=[s['name'] for s in spots]
    errors=[]
    if case['expected']=='explained_infeasible':
        if spots or plan.get('meal_slots'):
            errors.append('infeasible_unsafe_schedule')
        explanation=plan.get('notes','')+' '+plan.get('reasoning','')
        if not re.search(r'无法|不足|不可行|不能满足|不够|冲突',explanation):
            errors.append('infeasible_not_explained')
        if not any(t in explanation for t in rule['reason_terms']):
            errors.append('infeasible_wrong_reason')
        if sorted(d['day'] for d in days)!=list(range(1,state['days']+1)):
            errors.append('day_structure')
        return sorted(set(errors))
    checks,_=replay_check(plan,state,maps)
    errors.extend(checks)
    if sorted(d['day'] for d in days)!=list(range(1,state['days']+1)):
        errors.append('day_structure')
    if len(names)!=len(set(names)):
        errors.append('global_duplicate')
    if not set(rule['required']).issubset(names):
        errors.append('required_missing')
    if set(names)&set(rule['excluded']):
        errors.append('excluded_selected')
    by_name={p['name']:p for p in state['pois']}
    for day in days:
        d=day['day']
        if not 1<=d<=len(rule['counts']):
            continue
        low,high=rule['counts'][d-1]
        if not low<=len(day['spots'])<=high:
            errors.append('day_count')
        for s in day['spots']:
            start,end=minutes(s['start_time']),minutes(s['end_time'])
            if start<rule['earliest'] or end>rule['latest']:
                errors.append('user_time_window')
            if end-start<rule['min_visit']:
                errors.append('minimum_visit')
            if s['name'] in by_name:
                window=opening_window(by_name[s['name']])
                if not window or start<window[0] or end>window[1] or start>window[2]:
                    errors.append('opening_evidence')
            for a,b in rule.get('forbidden_windows',[]):
                if start<b and end>a:
                    errors.append('user_break_overlap')
    if plan.get('meal_slots'):
        errors.append('unrequested_meals')
    return sorted(set(errors))


def make_witness(case,maps):
    state=case['state']; rule=case['rubric']
    if case['expected']=='explained_infeasible':
        return dict(reasoning=case['infeasibility_proof']['explanation'],notes=case['infeasibility_proof']['explanation'],
            days=[dict(day=d,theme='条件不足',spots=[]) for d in range(1,state['days']+1)],
            meal_slots=[],selection_reasons=[],modification_concern='')
    required=set(rule['required']); used=set(); result=[]
    for d,(low,high) in enumerate(rule['counts'],1):
        count=low
        eligible=[p for p in state['pois'] if p['name'] not in rule['excluded'] and p['name'] not in used]
        eligible.sort(key=lambda p:p['name'] not in required)
        found=None
        for group in itertools.permutations(eligible,count):
            events=[]; current=rule['earliest']; previous=None
            for p in group:
                w=opening_window(p);start=max(current,w[0])
                if previous:
                    leg=maps.get(route_key(previous['location'],p['location'],'drive'))
                    if not leg:
                        break
                    start=max(start,current+leg['duration_min']+15)
                intervals=list(rule.get('forbidden_windows',[]))
                if float(state['weather_forecast'][d-1]['day_temp'])>=38 and p.get('indoor') is not True:
                    intervals.append([660,900])
                for a,b in sorted(intervals):
                    if start<b and start+rule['min_visit']>a:
                        start=b
                end=start+rule['min_visit']
                if end>min(w[1],rule['latest']) or start>w[2]:
                    break
                events.append(dict(name=p['name'],period='morning' if start<720 else 'afternoon',start_time=clock(start),end_time=clock(end)))
                previous=p;current=end
            if len(events)==count:
                selected={s['name'] for s in events}
                # Ensure all remaining mandatory POIs fit by the last day.
                missing=required-used-selected
                if d==len(rule['counts']) and missing:
                    continue
                found=events;break
        if found is None:
            raise RuntimeError('No witness: '+case['id'])
        used.update(s['name'] for s in found)
        result.append(dict(day=d,theme='候选内参观',spots=found))
    plan=dict(reasoning='按用户当前数量与时间要求安排；景点全程不重复。',notes='按给定开放快照与道路证据安排；不推断未知室内属性。',
        days=result,meal_slots=[],selection_reasons=[],modification_concern='')
    errors=score_case(plan,case,maps)
    if errors:
        raise RuntimeError(f'Witness rejected {case["id"]}: {errors}')
    return plan


def build_cases(pools,maps):
    cases=[]
    for ci,city in enumerate(CITIES):
        pool=pools[city]
        # Anchors must fit the late-start variant and have a usable day window.
        anchors=[p for p in pool if opening_window(p)[0]<=660 and opening_window(p)[1]>=960 and opening_window(p)[2]>=780]
        if len(anchors)<3:
            raise RuntimeError('Insufficient flexible anchors: '+city)
        for i in range(10):
            local=copy.deepcopy(pool)
            random.Random(5000+ci*10+i).shuffle(local)
            a,b=anchors[(i+ci)%len(anchors)],anchors[(i+ci+1)%len(anchors)]
            rule=dict(required=[],excluded=[],counts=[[2,2]],earliest=540,latest=1080,min_visit=60,forbidden_windows=[])
            query=f'在{city}游玩1天，候选仅代表可选地点，不必全部安排。每天最多2个景点，本日安排2个不同景点，每个至少60分钟，09:00至18:00之间参观；不安排餐馆。'
            state=dict(destination=city,days=1,max_per_day=3,pois=local,query=query,rewritten_query=None,
                travel_start_date='2026-11-04',main_meal_candidates=[],route_distance_mode='drive',
                candidate_pool_status='ready',candidate_missing_coverage=[],
                weather_forecast=[dict(date='2026-11-04',is_bad=False,day_weather='晴',night_weather='晴',day_temp='26',night_temp='20')])
            category=['upper_limit','required_excluded','latest_instruction','multi_day_allocation',
                'below_heat_threshold','heat_boundary','strict_start','midday_break','sparse_feasible','infeasible'][i]
            tags=[]
            if i==0:
                tags=['upper_bound_not_quota','default_vs_user_limit','opening_and_roads']
            elif i==1:
                query+=f' 必须参观「{a["name"]}」，不要参观「{b["name"]}」。'
                rule.update(required=[a['name']],excluded=[b['name']]);tags=['must_vs_excluded','canonical_identity']
            elif i==2:
                query=f'在{city}玩1天，原来想安排每天恰好3个景点，每个至少60分钟，09:00至18:00之间参观，不安排餐馆。'
                state['modification_notes']=f'以本次修改为准：每天最多2个景点，本日安排2个；不要参观「{a["name"]}」，必须参观「{b["name"]}」。'
                rule.update(required=[b['name']],excluded=[a['name']]);tags=['current_modification_priority','old_exact_count_is_stale']
            elif i==3:
                state['days']=2
                rule['counts']=[[1,1],[2,2]]
                query=f'在{city}玩2天，每天最多2个景点，第一天恰好1个、第二天恰好2个，全程3个不同景点；每个至少60分钟，每天09:00至18:00，不安排餐馆。'
                tags=['unequal_daily_allocation','global_uniqueness','cap_not_daily_quota']
            elif i==4:
                rule.update(earliest=720,latest=1020)
                query=f'在{city}玩1天，预报最高37.9°C；只在12:00至17:00之间安排2个不同景点，每个至少60分钟，每天最多2个，不安排餐馆。不得自行把未触发的高温阈值规则当成禁游要求。'
                state['weather_forecast'][0]['day_temp']='37.9';tags=['37.9_not_38','do_not_invent_restrictions']
            elif i==5:
                query=f'在{city}玩1天，当天最高38°C。09:00至18:00内安排2个不同景点，每个至少60分钟，每天最多2个；11:00至15:00不安排室外或室内未知景点，11:00结束或15:00开始允许。不安排餐馆。'
                state['weather_forecast'][0]['day_temp']='38';tags=['38_heat_trigger','half_open_boundary','unknown_not_indoor']
            elif i==6:
                rule.update(earliest=601,required=[a['name']])
                query=f'在{city}玩1天，所有景点必须严格晚于10:00开始（按分钟计最早10:01），18:00前结束；恰好2个不同景点，每天最多2个，每个至少60分钟，必须参观「{a["name"]}」，不安排餐馆。'
                tags=['strict_after_not_at','last_admission','required_poi']
            elif i==7:
                rule.update(forbidden_windows=[[720,840]],required=[b['name']])
                query=f'在{city}玩1天，09:00至18:00安排2个不同景点，每天最多2个，每个至少60分钟；12:00至14:00留作休息，不得与参观重叠，12:00结束或14:00开始允许。必须参观「{b["name"]}」。休息不需要填写餐馆，不安排餐馆。'
                tags=['user_break','no_unrequested_meals','explanation_consistency']
            elif i==8:
                state['days']=3;state['pois']=copy.deepcopy(anchors[:3]);state['candidate_pool_status']='insufficient';state['candidate_missing_coverage']=['count']
                rule['counts']=[[1,1]]*3
                query=f'在{city}玩3天。检索暂时只有这3个候选，接受每天仅1个；每天恰好1个不同景点，全程不重复，每个至少60分钟，每天09:00至18:00。不补造景点，不安排餐馆。'
                tags=['sparse_but_feasible','no_duplicate_fill','explain_shortage']
            else:
                rule.update(counts=[[0,0]],reason_terms=['候选','景点','数量','高温','室内'])
                tags=['detect_infeasibility','no_fabrication','explain_not_unsafe_release']
                if ci%2==0:
                    rule['reason_terms']=['候选','数量','2个','两个']
                    state['pois']=copy.deepcopy(anchors[:2]);state['candidate_pool_status']='insufficient';state['candidate_missing_coverage']=['count']
                    query=f'在{city}玩1天，必须安排恰好3个不同景点，但允许选择的候选只有这2个，禁止重复、禁止新增景点。每个至少60分钟。若无法同时满足，请保持该天景点列表为空并在notes说明冲突，不得输出部分路线冒充满足。'
                    proof=dict(kind='pigeonhole',required_distinct=3,available_distinct=2,explanation='候选只有2个不同景点，无法满足恰好3个且不重复的要求；需要补候选或放宽数量。')
                else:
                    rule['reason_terms']=['高温','室内','未知','时段']
                    state['weather_forecast'][0]['day_temp']='38'
                    query=f'在{city}玩1天，最高38°C；只能在11:00至15:00安排至少1个景点，且该时段只接受明确室内候选。当前所有候选的室内属性均未知，不允许自行推断或扩充候选。若无解，保持景点列表为空并在notes解释，不得安排未知地点冒充室内。'
                    proof=dict(kind='eligible_set_empty',known_indoor=0,allowed_window=[660,900],explanation='高温时段只允许明确室内候选，当前候选室内属性全部未知，无法满足；需核实室内证据或调整时段。')
            state['query']=query
            state['weather_forecast']=[dict(state['weather_forecast'][0],date=(datetime(2026,11,4)+timedelta(days=d)).date().isoformat()) for d in range(state['days'])]
            TravelPlanState(**state)
            case=dict(id=f'representative50-{ci+1}-{i+1:02d}',city=city,stratum='normal' if i<8 else 'hard',category=category,tags=tags,
                expected='feasible' if i<9 else 'explained_infeasible',state=state,rubric=rule)
            if i==9:
                case['infeasibility_proof']=proof
            case['feasibility_witness']=make_witness(case,maps)
            case['messages']=capture_messages(state)
            cases.append(case)
    return cases


def audit(cases,maps):
    results=[]
    for case in cases:
        witness=case['feasibility_witness']
        errors=score_case(witness,case,maps)
        if errors:
            raise RuntimeError(f'Invalid witness {case["id"]}: {errors}')
        if case['messages']!=capture_messages(case['state']):
            raise RuntimeError('Production message mismatch')
        negative=copy.deepcopy(witness)
        if case['expected']=='feasible':
            # Deliberately out-of-pool candidate must never be accepted, even
            # when all the original timestamps and counts still look right.
            negative['days'][0]['spots'][0]['name']='__审计用不存在候选__'
        else:
            negative['notes']='安排完成';negative['reasoning']='安排完成'
        faults=score_case(negative,case,maps)
        if not faults:
            raise RuntimeError('Negative control unexpectedly passed')
        controls=[dict(kind='out_of_pool_or_unexplained_rejection',faults=faults)]
        if case['expected']=='feasible':
            duplicate=copy.deepcopy(witness)
            flat=[s for d in duplicate['days'] for s in d['spots']]
            if len(flat)>1:
                flat[1]['name']=flat[0]['name']
                f=score_case(duplicate,case,maps)
                if 'global_duplicate' not in f:
                    raise RuntimeError('Duplicate control not detected')
                controls.append(dict(kind='global_duplicate',faults=f))
            too_short=copy.deepcopy(witness)
            spot=too_short['days'][0]['spots'][0]
            spot['end_time']=clock(minutes(spot['start_time'])+30)
            f=score_case(too_short,case,maps)
            if 'minimum_visit' not in f:
                raise RuntimeError('Duration control not detected')
            controls.append(dict(kind='minimum_visit',faults=f))
            if case['category']=='strict_start':
                boundary=copy.deepcopy(witness)
                spot=boundary['days'][0]['spots'][0]
                spot['start_time']='10:00';spot['end_time']='11:00'
                f=score_case(boundary,case,maps)
                if 'user_time_window' not in f:
                    raise RuntimeError('Strict start equality escaped')
                controls.append(dict(kind='strict_start_equality',faults=f))
            if case['category'] in ['midday_break','heat_boundary']:
                boundary=copy.deepcopy(witness)
                spot=boundary['days'][0]['spots'][0]
                spot['start_time']='12:00';spot['end_time']='13:00'
                f=score_case(boundary,case,maps)
                expected='user_break_overlap' if case['category']=='midday_break' else 'heat_exposure_or_unknown_indoor'
                # Production risk gate may use a broader code; shared heat
                # helper provides the exact protocol-specific witness check.
                if case['category']=='heat_boundary':
                    from app.planning.constraints import heat_violations
                    if not heat_violations(TravelPlanState(**case['state']),boundary['days']):
                        raise RuntimeError('High temperature control escaped')
                elif expected not in f:
                    raise RuntimeError('Midday break control escaped')
                if not f:
                    raise RuntimeError('Boundary control not rejected')
                controls.append(dict(kind=case['category'],faults=f))
        results.append(dict(id=case['id'],witness_passed=True,negative_rejected=faults,controls=controls,
            pool_size=len(case['state']['pois']),expected=case['expected']))
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'FROZEN.json').exists():
        print('Already frozen; create a new version for changes');return
    if args.capture:
        pools=prune_reviewed_pools(capture_pools());maps=capture_roads(pools)
    else:
        pools=prune_reviewed_pools(json.loads((OUT/'pools.json').read_text(encoding='utf-8'))['pools'])
        maps=json.loads((OUT/'maps.json').read_text(encoding='utf-8'))['routes']
    cases=build_cases(pools,maps)
    results=audit(cases,maps)
    dump(OUT/'cases.json',cases);dump(OUT/'audit.json',results);dump(OUT/'evaluated_pools.json',pools)
    source_files=['scripts/build_representative_eval50.py','scripts/post_train_runtime.py','app/planning/nodes.py',
        'app/planning/helpers.py','app/planning/constraints.py','scripts/prepare_heldout_evidence.py',
        'scripts/prepare_sft_v2_release.py']
    dump(OUT/'FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),cases=50,normal=40,hard=10,
        scope='new development benchmark, authoring protocol informed by prior model errors; not blind independent generalization',
        training_use='forbidden',paid_llm_calls=0,manual_semantic_review='not full human certification',
        hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['cases.json','pools.json','evaluated_pools.json','maps.json','audit.json','manual_candidate_review.json']},
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}))
    print(json.dumps(dict(stage='frozen',cases=len(cases),normal=40,hard=10,pools={c:len(p) for c,p in pools.items()},roads=len(maps)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
