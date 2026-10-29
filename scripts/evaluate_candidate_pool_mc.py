"""Append MC to the frozen 20-candidate development diagnostic."""
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import evaluate_candidate_pool_diagnostic as shared
from scripts.diagnose_sft_transfer import MODELS, switch, rows
from scripts.post_train_runtime import append_json
from scripts.evaluate_sft_validation import score
from scripts.prepare_sft_v2_release import explanation_faults
from app.planning.schemas import TravelRoute
from app.evaluation.task_robustness import load_assets

SOURCE = shared.OUT
OUT = ROOT/'data/post_train/candidate_pool_mc_v1'
MODELS['mc'] = ('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-migration-control',
                'TravelMind-Planner-7B-SFT-migration-control')


def status(stage, **extra):
    OUT.mkdir(parents=True,exist_ok=True)
    temp=OUT/'status.tmp'
    temp.write_text(json.dumps(dict(stage=stage,updated_at=time.time(),paid_calls=0,**extra)),encoding='utf-8')
    temp.replace(OUT/'status.json')
    print(json.dumps(dict(stage=stage,**extra)),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert json.loads((SOURCE/'status.json').read_text())['stage']=='completed'
    cases=json.loads((SOURCE/'cases.json').read_text(encoding='utf-8'))
    originals=rows(SOURCE/'results.jsonl')
    templates={(r['id'],r['seed']):r['request'] for r in originals if r['tag']=='v2'}
    assert len(cases)==30 and len(templates)==90
    binding=dict(source_hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
        for name in ['cases.json','binding.json','results.jsonl','maps.json']},
        script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seeds=[42,43,44],model=MODELS['mc'][1],context=8192,paid_calls=0,
        scope='development diagnostic; frozen inputs; no training or independent test')
    meta=OUT/'binding.json'
    if meta.exists() and json.loads(meta.read_text())!=binding:
        raise RuntimeError('Run binding changed')
    meta.write_text(json.dumps(binding,indent=2),encoding='utf-8')
    _,maps=load_assets()
    maps.update(json.loads((SOURCE/'maps.json').read_text(encoding='utf-8'))['routes'])
    hard={c['id']:c for c in json.loads((ROOT/'data/post_train/hard_supplement_cases.json').read_text(encoding='utf-8'))}
    for row in cases:
        if row['id'] in hard:
            hard[row['id']]=dict(hard[row['id']],state=row['state'])
    output=OUT/'results.jsonl'
    done={(r['id'],r['seed']) for r in rows(output)} if output.exists() else set()
    def worker(row,seed):
        request=dict(templates[row['id'],seed],model=MODELS['mc'][1])
        assert request['messages']==row['messages']
        result=dict(tag='mc',id=row['id'],seed=seed,request=request,parse_success=False,
            passed=False,faults=[],explanation_faults=[])
        started=time.perf_counter()
        try:
            req=urllib.request.Request('http://10.14.3.82:8000/v1/chat/completions',
                data=json.dumps(request,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=90) as reply:
                raw=json.load(reply)
            result.update(response=raw,usage=raw.get('usage'))
            if raw['choices'][0]['finish_reason']!='stop':
                raise ValueError('incomplete_generation')
            plan=TravelRoute.model_validate_json(raw['choices'][0]['message']['content']).model_dump(mode='json')
            result.update(parse_success=True,plan=plan)
            result['faults']=score(plan,row,maps,hard)
            result['explanation_faults']=explanation_faults(plan)
            result['passed']=not result['faults']
        except Exception as exc:
            result.update(error_type=type(exc).__name__,error=str(exc)[:300])
        result['elapsed_s']=time.perf_counter()-started
        return result
    pending=[(r,s) for s in [42,43,44] for r in cases if (r['id'],s) not in done]
    if pending:
        try:
            status('loading_mc',pending=len(pending))
            switch('mc',max_model_len=8192)
            status('evaluating',pending=len(pending))
            with ThreadPoolExecutor(max_workers=2) as pool:
                for future in as_completed([pool.submit(worker,r,s) for r,s in pending]):
                    result=future.result()
                    append_json(output,result)
                    print(json.dumps({k:result[k] for k in ['id','seed','parse_success','faults']}),flush=True)
        finally:
            status('restoring_base')
            switch('base')
    results=rows(output)
    assert len(results)==90
    status('capturing_selected_roads')
    shared.OUT=OUT
    maps=shared.capture_selected_roads(cases,results,maps)
    by_id={r['id']:r for r in cases}
    combined=[]
    for r in originals+results:
        r=copy.deepcopy(r)
        if r.get('parse_success'):
            r['faults']=score(r['plan'],by_id[r['id']],maps,hard)
            r['passed']=not r['faults']
        combined.append(r)
    (OUT/'rescored_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined),encoding='utf-8')
    cropped=(rows(ROOT/'data/post_train/sft_contract_eval_v1/results.jsonl')+
        rows(ROOT/'data/post_train/sft_contract_eval_v3/results.jsonl')+
        rows(ROOT/'data/post_train/sft_contract_eval_migration_control/results.jsonl'))
    summary={}
    lines=['# 20候选补测：v2、v3、MC','','同30个开发案例、三种子；MC请求除模型名外逐字段复用v2冻结请求。三个扩池模型以共同道路回放重新评分。','',
        '|模型|候选池|规则通过 /90|重复 /90|规则与解释通过 /90|平均景点数|非空天比例|',
        '|---|---|---:|---:|---:|---:|---:|']
    for tag in ['v2','v3','mc']:
        for arm,source in [('cropped',cropped),('20_candidates',combined)]:
            group=[r for r in source if r['tag']==tag]
            assert len(group)==90
            s=shared.summarize(group)
            summary[tag+'_'+arm]=s
            lines.append(f"|{tag}|{arm}|{s['passed']}|{s['duplicate_cases']}|{s['joint_success']}|{s['mean_spots_per_parsed_plan']:.2f}|{s['nonempty_day_fraction']:.1%}|")
    lines+=['','这是开发诊断，未使用独立测试；新增候选会同时改变类型、属性未知率和上下文长度，不能把变化全归因于数量。',
        '旧候选室内标记未独立核实，新候选室内未知；不凭名称认证室内。道路缺失仍记失败。',
        '原v2/v3结果及源资产保持不变；本报告使用共同道路资产复评分，需与原报告版本区分。',
        '未新增训练、DPO或付费大模型调用。']
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'docs/post_train_candidate_pool_mc_result.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status('completed',total=90,summary=summary)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc)[:500])
        raise
