"""Full frozen 30-case dev comparison, three paired seeds, local models only."""
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_sft_transfer import MODELS, switch, rows
from scripts.post_train_runtime import capture_messages, append_json
from scripts.evaluate_sft_validation import score
from scripts.prepare_sft_v2_release import explanation_faults
from app.evaluation.task_robustness import load_assets
from app.planning.schemas import TravelRoute
MODELS['mc']=('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-migration-control','TravelMind-Planner-7B-SFT-migration-control')
OUT = ROOT/'data/post_train/sft_contract_eval_migration_control'

def status(stage, **extra):
    tmp = OUT/'status.tmp'
    tmp.write_text(json.dumps(dict(stage=stage, updated_at=time.time(), paid_calls=0, **extra)),encoding='utf-8')
    tmp.replace(OUT/'status.json')
    print(json.dumps(dict(stage=stage,**extra)),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    from scripts.run_sft_validation_job import remote
    for _ in range(180):
        progress=json.loads(remote("from pathlib import Path\nprint(Path('/home/jiale/TravelMind-posttrain/logs/sft-migration-control-status.json').read_text())"))
        if progress['stage']=='sft_failed':raise RuntimeError('Migration control failed')
        if progress['stage']=='sft_complete_pending_validation':break
        status('waiting_for_training',training_stage=progress['stage'])
        time.sleep(15)
    else:raise RuntimeError('Migration control wait exceeded 45 minutes')
    selected = rows(ROOT/'data/post_train/release_v1/validation.ledger.jsonl')
    assert len(selected)==30 and len({r['id'] for r in selected})==30
    cases = [dict(id=r['id'],state=r['state'],messages=capture_messages(r['state'])) for r in selected]
    payload=json.dumps(cases,ensure_ascii=False,sort_keys=True)
    _,maps=load_assets()
    hard={c['id']:c for c in json.loads((ROOT/'data/post_train/hard_supplement_cases.json').read_text(encoding='utf-8'))}
    sources=['app/planning/nodes.py','app/planning/helpers.py','app/planning/constraints.py','scripts/post_train_runtime.py','scripts/evaluate_sft_validation.py','scripts/run_hard_post_train_synthesis.py','scripts/prepare_sft_v2_release.py','scripts/evaluate_migration_control.py']
    hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
    meta=dict(protocol='planner-contract-v1',seeds=[42,43,44],temperature=.3,max_tokens=2048,n=30,models=['mc'],baseline='frozen sft_contract_eval_v1 v1/v2 requests; equality checked below',paid_calls=0,
        source_hashes=hashes,input_sha256=hashlib.sha256(payload.encode()).hexdigest(),maps_sha256=hashlib.sha256(json.dumps(maps,sort_keys=True).encode()).hexdigest(),
        schema=TravelRoute.model_json_schema(),scope='development, not independent test; repeated generations are correlated')
    path=OUT/'binding.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=meta:raise RuntimeError('Protocol changed; create new version')
    path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'cases.json').write_text(payload,encoding='utf-8')
    output=OUT/'results.jsonl'
    done={(r['tag'],r['seed'],r['id']) for r in rows(output)} if output.exists() else set()
    def worker(tag,row,seed):
        request=dict(model=MODELS[tag][1],messages=row['messages'],temperature=.3,seed=seed,max_tokens=2048,
          response_format={'type':'json_schema','json_schema':{'name':'TravelRoute','strict':False,'schema':TravelRoute.model_json_schema()}})
        result=dict(tag=tag,id=row['id'],seed=seed,request=request,parse_success=False,passed=False,faults=[],explanation_faults=[])
        start=time.perf_counter()
        try:
            req=urllib.request.Request('http://10.14.3.82:8000/v1/chat/completions',data=json.dumps(request,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=90) as reply:raw=json.load(reply)
            result.update(response=raw,usage=raw.get('usage'))
            if raw['choices'][0]['finish_reason']!='stop':raise ValueError('incomplete_generation')
            plan=TravelRoute.model_validate_json(raw['choices'][0]['message']['content']).model_dump(mode='json')
            result.update(parse_success=True,plan=plan)
            result['faults']=score(plan,row,maps,hard)
            result['explanation_faults']=explanation_faults(plan)
            result['passed']=not result['faults']
        except Exception as exc:result.update(error_type=type(exc).__name__)
        result['elapsed_s']=time.perf_counter()-start
        return result
    try:
        for tag in ['mc']:
            pending=[(r,s) for s in [42,43,44] for r in cases if (tag,s,r['id']) not in done]
            if not pending:continue
            status('loading_model',model=tag)
            switch(tag)
            status('evaluating',model=tag,pending=len(pending))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(worker,tag,r,s) for r,s in pending]
                for future in as_completed(futures):
                    result=future.result();append_json(output,result)
                    print(json.dumps({k:result[k] for k in ['tag','id','seed','passed','faults']}),flush=True)
    finally:
        status('restoring_base');switch('base')
    results=rows(output)
    baseline=rows(ROOT/'data/post_train/sft_contract_eval_v1/results.jsonl')
    expected={(r['seed'],r['id']):{k:v for k,v in r['request'].items() if k!='model'} for r in baseline if r['tag']=='v2'}
    for r in results:
        if {k:v for k,v in r['request'].items() if k!='model'}!=expected[(r['seed'],r['id'])]:raise RuntimeError('Baseline protocol mismatch')
    baseline+=rows(ROOT/'data/post_train/sft_contract_eval_v3/results.jsonl')
    results=baseline+results
    summary={}
    for tag in ['v1','v2','v3','mc']:
        group=[r for r in results if r['tag']==tag]
        summary[tag]=dict(n=len(group),passed=sum(r['passed'] for r in group),parsed=sum(r['parse_success'] for r in group),
            by_seed={str(s):sum(r['passed'] for r in group if r['seed']==s) for s in [42,43,44]},
            all_three_pass=sum(all(r['passed'] for r in group if r['id']==c['id']) for c in cases),
            faults=dict(Counter(f for r in group for f in r['faults'])),explanation_fault_cases=sum(bool(r['explanation_faults']) for r in group),
            joint_success=sum(r['passed'] and not r['explanation_faults'] for r in group),total_tokens=sum((r.get('usage') or {}).get('total_tokens',0) for r in group))
    counts={tag:{c['id']:sum(r['passed'] for r in results if r['tag']==tag and r['id']==c['id']) for c in cases} for tag in ['v1','v2','v3','mc']}
    summary['paired_case_success_counts']=counts
    summary['improved']=[k for k in counts['v1'] if counts['mc'][k]>counts['v3'][k]]
    summary['regressed']=[k for k in counts['v1'] if counts['mc'][k]<counts['v3'][k]]
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 仅迁移提示词对照实验','','同30例、三种子、temperature0.3、输出2048；v1/v2复用逐字一致的原始请求记录，v3新运行。未使用独立测试集。','',
      '|模型|各seed通过 /30|规则通过 /90|三次全过 /30|规则+解释 /90|','|---|---|---:|---:|---:|']
    for tag in ['v1','v2','v3','mc']:
        t=summary[tag];lines.append(f"|{tag}|{list(t['by_seed'].values())}|{t['passed']}|{t['all_three_pass']}|{t['joint_success']}|")
    lines+=['','mc使用v3中完全相同的249条历史输入与答案，不加27条补数，固定48步。保留4条既有开始端点措辞问题作为诊断控制，不能视为部署数据认证。固定步数下曝光频率和epoch不同；不能仅据本实验断言补数本身有害。','','mc错误：'+json.dumps(summary['mc']['faults'],ensure_ascii=False),'','没有自动启动DPO；先检查逐例错误和解释再决定。']
    (ROOT/'docs/post_train_migration_control_result.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    a,b=summary['mc'],summary['v3']
    if a['joint_success']>b['joint_success'] and a['all_three_pass']>=b['all_three_pass']:
        decision='Without-supplement control improves observed joint quality; inspect supplemental mixing/coverage before further synthesis. Exposure is a confound; do not claim intrinsic data harm.'
    elif a['joint_success']<=b['joint_success']:
        decision='Removing supplement did not improve observed joint quality; inspect migration/legacy target consistency and optimization exposure before more data.'
    else:decision='Mixed indicators; inspect paired errors, do not select solely by aggregate pass rate.'
    (OUT/'next_step_decision.json').write_text(json.dumps(dict(decision=decision,independent_test_used=False,new_training_auto_start=False,dpo_auto_start=False),indent=2),encoding='utf-8')
    status('completed',total=90,summary=summary)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc));raise
