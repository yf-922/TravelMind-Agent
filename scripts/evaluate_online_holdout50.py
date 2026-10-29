"""Evaluate immutable heldout tasks using the production LangGraph repair loop."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.online_replay import HTTPBackend, ProductionAuditBackend, MockBackend, Session, production_replay, run_session
from scripts.build_online_holdout50 import OUT, verify, score_soft
from scripts.build_representative_eval50 import score_case
from scripts.post_train_runtime import append_json
from scripts.prepare_sft_v2_release import explanation_faults


def score_result(result, case, maps):
    """Independent grader runs AFTER the online workflow, never feeds rework."""
    if not result.get('plan'):
        return dict(result, faults=['no_parsed_plan'], passed=False, joint_passed=False,
                    soft_faults=[], explanation_faults=[])
    errors=score_case(result['plan'],case,maps)
    raw_errors=score_case(result['raw_plan'],case,maps)
    ef=explanation_faults(result['plan'])
    sf=score_soft(result['plan'],case)
    feasible=case['expected']=='feasible'
    approved=result['approved']
    # Correct refusal is distinct from delivering a usable itinerary.
    correct=result['parse_success'] and not result.get('error_type') and not errors and (result['mode']=='planner-only' or (approved if feasible else not approved))
    return dict(result, faults=errors, raw_faults=raw_errors, explanation_faults=ef,soft_faults=sf,
                passed=correct,joint_passed=correct and not ef and not sf,
                false_approval=result['mode']=='online-rework' and approved and bool(errors),
                feasible_delivery=feasible and correct and (approved or result['mode']=='planner-only'),
                correct_refusal=not feasible and correct)


def percentile(values,q):
    return sorted(values)[max(0,math.ceil(len(values)*q)-1)] if values else None


def summarize(rows,cases):
    byid={c['id']:c for c in cases}
    groups={'all':rows}
    for field in ['city','category','stratum','expected']:
        for val in sorted({c[field] for c in cases}):groups[field+'/'+val]=[r for r in rows if byid[r['id']][field]==val]
    result={}
    for key,rr in groups.items():
        n=len(rr);success=sum(r['passed'] for r in rr)
        if not n:
            result[key]=dict(n=0)
            continue
        soft_rows=[r for r in rr if byid[r['id']].get('soft_rubric')]
        feasible_rows=[r for r in rr if byid[r['id']]['expected']=='feasible']
        infeasible_rows=[r for r in rr if byid[r['id']]['expected']!='feasible']
        result[key]=dict(n=n,parse_success=sum(r['parse_success'] for r in rr), outcome_correct=success,
            feasible_n=len(feasible_rows), feasible_correct=sum(r['passed'] for r in feasible_rows),
            infeasible_n=len(infeasible_rows), correct_refusals=sum(r['passed'] for r in infeasible_rows),
            feasible_false_refusals=sum(r['parse_success'] and not r['approved'] for r in feasible_rows if r['mode']=='online-rework'),
            soft_applicable_n=len(soft_rows), soft_correct=sum(r['parse_success'] and not r['soft_faults'] for r in soft_rows),
            explanation_faults=dict(Counter(f for r in rr for f in r['explanation_faults'])),
            joint_correct=sum(r['joint_passed'] for r in rr), false_approvals=sum(r.get('false_approval',False) for r in rr),
            raw_rule_correct=sum(not r.get('raw_faults',['no_plan']) for r in rr),
            token_total=sum(r['usage']['total_tokens'] for r in rr),
            tokens_per_correct_task=sum(r['usage']['total_tokens'] for r in rr)/success if success else None,
            mean_reworks=sum(r['reworks'] for r in rr)/n, calls_by_role=dict(sum((Counter(r['calls_by_role']) for r in rr),Counter())),
            latency_mean_s=sum(r['elapsed_s'] for r in rr)/n,
            mean_output_chars=sum(sum(len(c.get('raw_content','')) for c in r['calls']) for r in rr)/n,
            tokens_by_role={role:{k:sum(c.get('usage',{}).get(k,0) or 0 for r in rr for c in r['calls'] if c['task']==role)
                for k in ['prompt_tokens','completion_tokens','total_tokens']} for role in ['joint_planner','reviewer','time_check']},
            latency_p50_s=percentile([r['elapsed_s'] for r in rr],.5),latency_p95_s=percentile([r['elapsed_s'] for r in rr],.95),
            faults=dict(Counter(f for r in rr for f in r['faults'])),soft_faults=dict(Counter(f for r in rr for f in r['soft_faults'])),
            errors=dict(Counter(r['error_type'] for r in rr if r.get('error_type'))),
            call_errors=dict(Counter(c['error_type'] for r in rr for c in r['calls'] if c.get('error_type'))))
    result['per_case_seed_successes']={c['id']:sum(r['passed'] for r in rows if r['id']==c['id']) for c in cases}
    result['stability']=dict(all_seeds_pass=sum(v==3 for v in result['per_case_seed_successes'].values()),
                            no_seed_pass=sum(v==0 for v in result['per_case_seed_successes'].values()),
                            note='3 decoding seeds, not 3 training seeds; uncertainty must cluster by case/city')
    return result


def run(args):
    manifest=verify(args.suite)
    cases=json.loads((args.suite/'cases.json').read_text(encoding='utf-8'))
    maps=json.loads((args.suite/'maps.json').read_text(encoding='utf-8'))['routes']
    if args.limit: cases=cases[:args.limit]
    out=args.output/args.tag;out.mkdir(parents=True,exist_ok=True)
    backend_class=ProductionAuditBackend if args.audit_production_adapter else HTTPBackend
    backend=MockBackend() if args.mock else backend_class(base_url=args.base_url,planner_model=args.model_name,
        reviewer_model=args.reviewer_model,time_model=args.time_model,api_key=os.getenv('EVAL_PLANNER_API_KEY','local'),
        audit_url=args.audit_url,audit_api_key=os.getenv('EVAL_AUDIT_API_KEY'),max_tokens=args.max_tokens)
    binding=dict(suite_sha256=manifest['hashes']['cases.json'],source_hashes=manifest['source_hashes'],
        model=args.model_name,reviewer_model=args.reviewer_model,time_model=args.time_model or args.reviewer_model,
        audit_production_adapter=args.audit_production_adapter,
        base_url=args.base_url,audit_url=args.audit_url or args.base_url,mock=args.mock,seeds=args.seeds,modes=args.modes,
        max_tokens=args.max_tokens,concurrency=args.concurrency,case_ids=[c['id'] for c in cases],
        engine='production LangGraph; frozen upstream facts; real planner/audit/repair/finalize; no rubric feedback',
        label='MOCK PIPELINE CHECK, NOT MODEL PERFORMANCE' if args.mock else 'prospective new-city holdout')
    bp=out/'binding.json'
    if bp.exists() and json.loads(bp.read_text(encoding='utf-8'))!=binding:raise RuntimeError('Binding changed; choose new tag')
    bp.write_text(json.dumps(binding,ensure_ascii=False,indent=2),encoding='utf-8')
    path=out/'results.jsonl'
    old=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()] if path.exists() else []
    identities=[(r['id'],r['seed'],r['mode']) for r in old]
    if len(identities)!=len(set(identities)):raise RuntimeError('Duplicate completed task; audit before resume')
    done=set(identities);lock=threading.Lock()
    def ledger(event):
        with lock:append_json(out/'calls.jsonl',event)
    tasks=[(c,s,m) for c in cases for s in args.seeds for m in args.modes if (c['id'],s,m) not in done]
    random.Random(271027).shuffle(tasks)
    def worker(compiled,c,seed,mode):
        session=Session(c,seed,backend,maps,tag=args.tag,ledger=ledger)
        result=score_result(run_session(compiled,session,mode),c,maps)
        result['mock']=args.mock
        with lock:
            append_json(path,result)
            old.append(result)
            (out/'status.json').write_text(json.dumps(dict(completed=len(old),target=len(cases)*len(args.seeds)*len(args.modes),
                last_case=c['id'],last_mode=mode,updated_at=time.time()),indent=2),encoding='utf-8')
        return result
    with production_replay(maps) as compiled:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for f in as_completed([pool.submit(worker,compiled,*t) for t in tasks]):
                r=f.result()
                if r.get('error_type') in {'HTTPError','URLError','TimeoutError'}:
                    # Persist errors and avoid treating an unavailable service as low capability.
                    print(json.dumps(dict(service_error=r['error_type'],case=r['id'])),flush=True)
    report={mode:summarize([r for r in old if r['mode']==mode],cases) for mode in args.modes}
    report['binding']=binding
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({m:report[m]['all'] for m in args.modes},ensure_ascii=False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,default=OUT);p.add_argument('--output',type=Path,default=OUT/'runs')
    p.add_argument('--tag',required=True);p.add_argument('--mock',action='store_true');p.add_argument('--model-name')
    p.add_argument('--reviewer-model');p.add_argument('--time-model');p.add_argument('--base-url',default='http://10.14.3.82:8000/v1')
    p.add_argument('--audit-url');p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--audit-production-adapter',action='store_true',help='Use existing production OpenAI adapter for fixed API audit roles')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--modes',nargs='+',choices=['planner-only','online-rework'],default=['planner-only','online-rework'])
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if not args.mock and (not args.model_name or not args.reviewer_model):p.error('Specify planner and fixed reviewer model explicitly')
    if len(set(args.seeds))!=3 or len(args.seeds)!=3:p.error('Exactly 3 distinct decoding seeds required')
    if not 1<=args.concurrency<=4:p.error('concurrency must be 1..4')
    if not args.tag.replace('-','').replace('_','').isalnum():p.error('Invalid tag')
    run(args)


if __name__=='__main__':main()
