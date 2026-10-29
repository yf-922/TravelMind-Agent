"""Run only frozen messages against a local OpenAI-compatible model service."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_representative_eval50 import OUT as SUITE, score_case
from scripts.post_train_runtime import append_json
from scripts.prepare_sft_v2_release import explanation_faults
from app.planning.schemas import TravelRoute


def verify():
    manifest=json.loads((SUITE/'FROZEN.json').read_text(encoding='utf-8'))
    for name,digest in manifest['hashes'].items():
        if hashlib.sha256((SUITE/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen asset changed: '+name)
    for name,digest in manifest['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen protocol changed; version the benchmark: '+name)
    return manifest


def summarize(results,cases):
    by_id={c['id']:c for c in cases}
    summary={}
    for label,group in [('all',results)]+[(key,[r for r in results if by_id[r['id']]['stratum']==key]) for key in ['normal','hard']]+[
        (key,[r for r in results if by_id[r['id']]['expected']==key]) for key in ['feasible','explained_infeasible']]:
        n=len(group)
        feasible=[r for r in group if by_id[r['id']]['expected']=='feasible' and r.get('parse_success')]
        days=[d for r in feasible for d in r['plan']['days']]
        summary[label]=dict(n=n,parsed=sum(r['parse_success'] for r in group),passed=sum(r['passed'] for r in group),
            joint_passed=sum(r['passed'] and not r['explanation_faults'] for r in group),
            explanation_fault_cases=sum(bool(r['explanation_faults']) for r in group),
            mean_spots_per_parsed_feasible_plan=sum(len(d['spots']) for d in days)/len(feasible) if feasible else None,
            feasible_nonempty_day_fraction=sum(bool(d['spots']) for d in days)/len(days) if days else None,
            token_total=sum((r.get('usage') or {}).get('total_tokens',0) for r in group),
            mean_latency_s=sum(r['elapsed_s'] for r in group)/n if n else None)
    from collections import Counter
    summary['faults']=dict(Counter(f for r in results for f in r['faults']))
    summary['errors']=dict(Counter(r['error_type'] for r in results if r.get('error_type')))
    summary['by_city']={city:dict(n=sum(by_id[r['id']]['city']==city for r in results),
        passed=sum(r['passed'] and by_id[r['id']]['city']==city for r in results)) for city in sorted({c['city'] for c in cases})}
    summary['by_category']={category:dict(n=sum(by_id[r['id']]['category']==category for r in results),
        passed=sum(r['passed'] and by_id[r['id']]['category']==category for r in results))
        for category in sorted({c['category'] for c in cases})}
    summary['per_case_pass_counts']={c['id']:sum(r['passed'] for r in results if r['id']==c['id']) for c in cases}
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mock',action='store_true',help='Score stored witnesses only; NOT model performance')
    p.add_argument('--model-name');p.add_argument('--tag',default='local')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--mode',choices=['planner-only','rules-rework'],default='planner-only')
    args=p.parse_args()
    if not args.mock and not args.model_name:
        p.error('--model-name required for real evaluation')
    if not 1<=args.concurrency<=4 or len(args.seeds)!=len(set(args.seeds)):
        p.error('Concurrency must be 1..4, seeds must be unique')
    if not args.tag.replace('-','').replace('_','').isalnum():
        p.error('Tag must contain letters, numbers, hyphens or underscores')
    manifest=verify()
    cases=json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))
    maps=json.loads((SUITE/'maps.json').read_text(encoding='utf-8'))['routes']
    out=SUITE/('mock_smoke' if args.mock else 'runs')/args.tag
    out.mkdir(parents=True,exist_ok=True)
    binding=dict(suite_sha256=manifest['hashes']['cases.json'],model=args.model_name,mock=args.mock,
        seeds=args.seeds,temperature=.3,max_tokens=2048,mode=args.mode,paid_calls=0,
        endpoint='fixed local server; no paid fallback',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    meta=out/'binding.json'
    if meta.exists() and json.loads(meta.read_text())!=binding:
        raise RuntimeError('Run binding changed; use a new tag')
    meta.write_text(json.dumps(binding,indent=2),encoding='utf-8')
    output=out/'results.jsonl'
    results=[json.loads(l) for l in output.read_text(encoding='utf-8').splitlines()] if output.exists() else []
    done={(r['id'],r['seed']) for r in results}
    def worker(case,seed):
        start=time.perf_counter()
        r=dict(id=case['id'],seed=seed,tag=args.tag,mock=args.mock,parse_success=False,passed=False,faults=[],explanation_faults=[],rounds=[])
        try:
            feedback=''
            for round_no in range(4 if args.mode=='rules-rework' else 1):
                if args.mock:
                    plan=case['feasibility_witness']
                else:
                    messages=case['messages']
                    if feedback:
                        messages=messages[:-1]+[dict(role='user',content=messages[-1]['content']+'\n\n确定性规则返工反馈：'+feedback)]
                    request=dict(model=args.model_name,messages=messages,temperature=.3,seed=seed,max_tokens=2048,
                        response_format={'type':'json_schema','json_schema':dict(name='TravelRoute',strict=False,schema=TravelRoute.model_json_schema())})
                    req=urllib.request.Request('http://10.14.3.82:8000/v1/chat/completions',data=json.dumps(request,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=90) as reply:
                        raw=json.load(reply)
                    r['usage']=raw.get('usage'); r['total_tokens']=r.get('total_tokens',0)+(raw.get('usage') or {}).get('total_tokens',0)
                    if raw['choices'][0]['finish_reason']!='stop': raise ValueError('incomplete_generation')
                    plan=TravelRoute.model_validate_json(raw['choices'][0]['message']['content']).model_dump(mode='json')
                faults=score_case(plan,case,maps); ef=explanation_faults(plan)
                r['rounds'].append(dict(round=round_no,plan=plan,faults=faults,explanation_faults=ef))
                r.update(plan=plan,parse_success=True,faults=faults,passed=not faults,explanation_faults=ef,rounds_used=round_no+1)
                if not faults or args.mode!='rules-rework': break
                feedback='；'.join(faults)
        except Exception as exc:
            r.update(error_type=type(exc).__name__)
        r['elapsed_s']=time.perf_counter()-start
        return r
    pending=[(c,s) for s in args.seeds for c in cases if (c['id'],s) not in done]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for f in as_completed([pool.submit(worker,c,s) for c,s in pending]):
            r=f.result();append_json(output,r);results.append(r)
    result=summarize(results,cases)
    result['label']='MOCK PIPELINE VALIDATION; NOT MODEL PERFORMANCE' if args.mock else 'development benchmark, not independent test'
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
