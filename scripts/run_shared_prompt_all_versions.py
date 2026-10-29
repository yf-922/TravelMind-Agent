"""Evaluate every saved checkpoint with current production messages.

Runs paired Planner-Only and Rules+Rework on the frozen 50-case development
suite. Rework allows the initial answer plus at most three rule-only repairs.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import re
import urllib.request
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.planning.schemas import TravelRoute
from app.planning.prompts import JOINT_PLANNER_SYSTEM
from scripts.post_train_runtime import capture_messages
from scripts.build_representative_eval50 import score_case
from scripts.prepare_sft_v2_release import explanation_faults
from scripts.diagnose_sft_transfer import MODELS, switch
from scripts.run_sft_validation_job import ready
from scripts.evaluate_representative50 import SUITE, verify

MODELS.update({
    'v3':('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v3','TravelMind-Planner-7B-SFT-v3'),
    'mc':('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-migration-control','TravelMind-Planner-7B-SFT-migration-control'),
    'v4_clean':('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v4-clean','TravelMind-Planner-7B-SFT-v4-clean'),
    'v4':('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v4','TravelMind-Planner-7B-SFT-v4')})
TAGS=['base','v1','v2','v3','mc','v4_clean','v4','large']
OUT=SUITE/'shared_prompt_current_v2'
LOCK=threading.Lock()
STRICT_CLOCK = re.compile(r'^(?:[01]\d|2[0-3]):[0-5]\d$')

def invalid_clock_fields(plan):
    """Return structured-output time fields that are not strict HH:MM."""
    bad=[]
    for day in plan.get('days', []):
        for spot in day.get('spots', []):
            for key in ('start_time','end_time'):
                value=spot.get(key)
                if not isinstance(value,str) or not STRICT_CLOCK.fullmatch(value):
                    bad.append(f'day{day.get("day")}.spots.{key}')
    for slot in plan.get('meal_slots', []):
        for key in ('start_time','end_time'):
            value=slot.get(key)
            if not isinstance(value,str) or not STRICT_CLOCK.fullmatch(value):
                bad.append(f'meal_slots.{key}')
    return bad

def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []

def run_one(tag,mode,cases,seeds,concurrency):
    name=MODELS[tag][1]
    run=OUT/f'{tag}_{mode}'
    run.mkdir(parents=True,exist_ok=True)
    output=run/'results.jsonl'
    existing=rows(output)
    grouped=defaultdict(list)
    for row in existing: grouped[(row['id'],row['seed'])].append(row)
    # On interruption, keep the final complete record for each paired key.
    # Partial generations never reach JSONL because append occurs once per case.
    dedup={key:vals[-1] for key,vals in grouped.items()}
    if len(dedup)!=len(existing):
        output.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in dedup.values()),encoding='utf-8')
    done=set(dedup)
    messages={c['id']:capture_messages(c['state']) for c in cases}
    expected={c['id']:c['messages'] for c in cases}
    if any(m[0]['content']!=JOINT_PLANNER_SYSTEM for m in messages.values()):
        raise RuntimeError('Current production prompt capture mismatch')
    for c in cases:
        if messages[c['id']][-1] != expected[c['id']][-1]:
            raise RuntimeError('Current production human-message differs from frozen eval case: '+c['id'])
    (run/'protocol.json').write_text(json.dumps(dict(model=name,tag=tag,mode=mode,
        message_policy='same capture_messages(state) used by current production; system hash pinned; same per-case human message; current vLLM JSON schema',
        system_sha256=hashlib.sha256(JOINT_PLANNER_SYSTEM.encode()).hexdigest(),
        schema_sha256=hashlib.sha256(json.dumps(TravelRoute.model_json_schema(),sort_keys=True).encode()).hexdigest(),
        suite_cases=len(cases),seeds=seeds,temperature=.3,max_tokens=2048,
        rounds='Planner-Only=one generation; Rules+Rework=initial + at most 3 repairs; deterministic rule faults only',
        request_messages_sha256={k:hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest() for k,v in messages.items()}),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    byid={c['id']:c for c in cases}
    pending=[(c,s) for s in seeds for c in cases if (c['id'],s) not in done]
    def worker(case,seed):
        t=time.perf_counter(); ident=(case['id'],seed)
        record=dict(id=case['id'],seed=seed,model=name,tag=tag,mode=mode,parse_success=False,passed=False,
                    faults=[],explanation_faults=[],rounds=[],request_messages=messages[case['id']])
        feedback=''
        try:
            plan=None
            n_calls=1 if mode=='planner-only' else 4
            for round_no in range(n_calls):
                msg=copy.deepcopy(messages[case['id']])
                if feedback:
                    msg[-1]['content'] += '\n\n确定性规则返工反馈（仅修正指出的问题，仍须遵守原输入及Schema）：'+feedback
                reqdata=dict(model=name,messages=msg,temperature=.3,seed=seed,max_tokens=2048,
                    response_format={'type':'json_schema','json_schema':dict(name='TravelRoute',strict=False,schema=TravelRoute.model_json_schema())})
                req=urllib.request.Request('http://10.14.3.82:8000/v1/chat/completions',
                    data=json.dumps(reqdata,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=120) as response:
                    raw=json.load(response)
                record['usage_total']=record.get('usage_total',0)+(raw.get('usage') or {}).get('total_tokens',0)
                if raw['choices'][0]['finish_reason']!='stop': raise ValueError('incomplete_generation')
                plan=TravelRoute.model_validate_json(raw['choices'][0]['message']['content']).model_dump(mode='json')
                # Mark schema-valid JSON before deterministic rubric evaluation. A
                # malformed clock must be recorded as a model output fault rather
                # than being misclassified as an evaluator parse failure.
                invalid_clocks=invalid_clock_fields(plan)
                faults=(['invalid_time_format'] if invalid_clocks else [])
                if not invalid_clocks:
                    faults=score_case(plan,case,maps)
                ef=explanation_faults(plan)
                record['rounds'].append(dict(round=round_no,plan=plan,faults=faults,explanation_faults=ef,
                    request_messages=msg,usage=raw.get('usage')))
                record.update(plan=plan,parse_success=True,faults=faults,explanation_faults=ef,rounds_used=round_no+1)
                if not faults or mode=='planner-only': break
                feedback='；'.join(faults)
        except Exception as exc:
            record.update(error_type=type(exc).__name__,error=str(exc)[:500])
        record['passed']=record['parse_success'] and not record['faults']
        record['joint_passed']=record['passed'] and not record['explanation_faults']
        record['elapsed_s']=time.perf_counter()-t
        return record
    completed=len(done)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures=[pool.submit(worker,c,s) for c,s in pending]
        for future in as_completed(futures):
            record=future.result()
            with LOCK:
                with output.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            completed+=1
            if completed%25==0:
                print(json.dumps(dict(tag=tag,mode=mode,done=completed,target=len(cases)*len(seeds)),ensure_ascii=False),flush=True)
    result=rows(output)
    if {(r['id'],r['seed']) for r in result}!={(c['id'],s) for c in cases for s in seeds}:
        raise RuntimeError('Incomplete/duplicate result set')
    return result

def summarize(rs,cases):
    byid={c['id']:c for c in cases}
    out={}
    groups=[('all',rs),('normal',[r for r in rs if byid[r['id']]['stratum']=='normal']),
            ('hard',[r for r in rs if byid[r['id']]['stratum']=='hard']),
            ('feasible',[r for r in rs if byid[r['id']]['expected']=='feasible']),
            ('explained_infeasible',[r for r in rs if byid[r['id']]['expected']=='explained_infeasible'])]
    for k,g in groups:
        n=len(g); out[k]=dict(n=n,parsed=sum(r['parse_success'] for r in g),
            rules_pass=sum(r['passed'] for r in g),joint_pass=sum(r['joint_passed'] for r in g),
            mean_rounds=sum(r.get('rounds_used',0) for r in g)/n if n else None,
            token_total=sum(r.get('usage_total',0) for r in g),
            mean_latency_s=sum(r.get('elapsed_s',0) for r in g)/n if n else None)
    out['faults']=dict(Counter(f for r in rs for f in r['faults']))
    out['explanation_faults']=dict(Counter(f for r in rs for f in r['explanation_faults']))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--concurrency',type=int,default=4)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    ap.add_argument('--tags',nargs='+',choices=TAGS,default=TAGS)
    ap.add_argument('--modes',nargs='+',choices=['planner-only','rules-rework'],default=['planner-only','rules-rework'])
    args=ap.parse_args()
    if not 1<=args.concurrency<=4 or len(set(args.seeds))!=len(args.seeds):raise ValueError('Invalid concurrency/seeds')
    manifest=verify()
    cases=json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))
    global maps
    maps=json.loads((SUITE/'maps.json').read_text(encoding='utf-8'))['routes']
    if len(cases)!=50:raise RuntimeError('Frozen suite is not 50 cases')
    OUT.mkdir(parents=True,exist_ok=True)
    binding=dict(suite_sha256=manifest['hashes']['cases.json'],system=JOINT_PLANNER_SYSTEM,
        system_sha256=hashlib.sha256(JOINT_PLANNER_SYSTEM.encode()).hexdigest(),
        source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['app/planning/prompts.py','scripts/post_train_runtime.py','scripts/build_representative_eval50.py']},
        tags=args.tags,modes=args.modes,seeds=args.seeds,concurrency=args.concurrency,paid_calls=0)
    bp=OUT/'binding.json'
    if bp.exists() and json.loads(bp.read_text(encoding='utf-8'))!=json.loads(json.dumps(binding)):
        raise RuntimeError('New run binding changed; create a fresh directory')
    atomic(bp,binding)
    status=dict(stage='starting',updated_at=time.time(),done=0,target=len(args.tags)*len(args.modes)*len(cases)*len(args.seeds),
        counts={},paid_calls=0,current_model=None)
    atomic(OUT/'status.json',status)
    try:
        for tag in args.tags:
            status.update(stage='loading_model',updated_at=time.time(),current_model=tag);atomic(OUT/'status.json',status)
            switch(tag,max_model_len=8192);ready(MODELS[tag][1])
            for mode in args.modes:
                status.update(stage='evaluating',updated_at=time.time(),current_mode=mode);atomic(OUT/'status.json',status)
                rs=run_one(tag,mode,cases,args.seeds,args.concurrency)
                sm=summarize(rs,cases);atomic(OUT/f'{tag}_{mode}'/'summary.json',sm)
                status['counts'][f'{tag}/{mode}']=len(rs);status['done']=sum(status['counts'].values());status['updated_at']=time.time();atomic(OUT/'status.json',status)
        report=[]
        for tag in args.tags:
            for mode in args.modes:
                sm=json.loads((OUT/f'{tag}_{mode}'/'summary.json').read_text(encoding='utf-8'))
                report.append((tag,mode,sm))
        md=['# 当前生产提示词下各版本统一评测','','50题开发集×3种子；当前生产提示词 `JOINT_PLANNER_SYSTEM` 和 `capture_messages(state)`；Planner-Only 单次生成，Rules+Rework 初稿后最多3轮规则返工。非独立测试集。','','|模型|模式|规则通过|规则+解释|解析|平均调用轮次|Token|平均秒|','|---|---|---:|---:|---:|---:|---:|---:|']
        for tag,mode,sm in report:
            a=sm['all'];md.append(f"|{tag}|{mode}|{a['rules_pass']}/{a['n']}|{a['joint_pass']}/{a['n']}|{a['parsed']}/{a['n']}|{a['mean_rounds']:.2f}|{a['token_total']}|{a['mean_latency_s']:.1f}|")
        md += ['', '## 分层结果','','|模型|模式|常规规则|困难规则|有解规则|无解正确解释|主要剩余规则故障|','|---|---|---:|---:|---:|---:|---|']
        for tag,mode,sm in report:
            md.append(f"|{tag}|{mode}|{sm['normal']['rules_pass']}/{sm['normal']['n']}|{sm['hard']['rules_pass']}/{sm['hard']['n']}|{sm['feasible']['rules_pass']}/{sm['feasible']['n']}|{sm['explained_infeasible']['rules_pass']}/{sm['explained_infeasible']['n']}|{json.dumps(sm['faults'],ensure_ascii=False)}|")
        md+=['','## 口径与边界','','逐题人类可读说明及每轮原始请求、输出、规则故障和Token保存在本报告对应目录下。Rework反馈只传确定性规则故障，不调用LLM Reviewer。各轮尝试最多4次，遇到规则通过立即停止；解释一致性单独报告，不触发返工。不同训练版本的训练提示词不同，这是模型历史训练的一部分；本次评测请求对所有版本统一使用当前生产提示词。','']
        (ROOT/'docs/post_train_shared_prompt_all_versions.md').write_text('\n'.join(md),encoding='utf-8')
        status.update(stage='completed',updated_at=time.time());atomic(OUT/'status.json',status)
    finally:
        if status.get('stage')!='completed':
            try:switch('base',max_model_len=8192)
            except Exception:pass

if __name__=='__main__':main()
