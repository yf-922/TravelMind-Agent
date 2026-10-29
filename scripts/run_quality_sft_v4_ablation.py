"""After authorized server SFT, evaluate both arms and restore Base; no paid calls."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_sft_validation_job import remote, ready
from scripts.diagnose_sft_transfer import MODELS, switch, rows
from scripts.evaluate_representative50 import SUITE, verify, summarize
from scripts.run_representative50_all_versions import cluster_interval
from scripts.diagnose_representative50_results import normalize_format
from scripts.build_representative_eval50 import score_case
from scripts.prepare_sft_v2_release import explanation_faults
from scripts.prepare_quality_sft_v4 import OUT, dump

ARMS=['v4_clean','v4']
for arm,suffix in [('v4_clean','v4-clean'),('v4','v4')]:
    MODELS[arm]=('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-'+suffix,
        'TravelMind-Planner-7B-SFT-'+suffix)


def status(stage,**kw):
    dump(OUT/'ablation_job_status.json',dict(stage=stage,updated_at=time.time(),paid_calls=0,dpo_auto_start=False,**kw))
    print(json.dumps(dict(stage=stage,**kw)),flush=True)


def analyze():
    cases=json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))
    by={c['id']:c for c in cases}
    maps=json.loads((SUITE/'maps.json').read_text(encoding='utf-8'))['routes']
    records={'v2':rows(SUITE/'runs/all_versions_v1_v2/results.jsonl')}
    records.update({arm:rows(SUITE/'runs'/('quality4_'+arm)/'results.jsonl') for arm in ARMS})
    expected={(c['id'],s) for c in cases for s in [42,43,44]}
    for group in records.values():
        if len(group)!=150 or {(r['id'],r['seed']) for r in group}!=expected:
            raise RuntimeError('Incomplete or duplicated benchmark')
        for r in group:
            if r['request']['messages']!=by[r['id']]['messages']:
                raise RuntimeError('Frozen request mismatch')
    summaries={t:summarize(g,cases) for t,g in records.items()}
    diagnostic={}
    for tag,group in records.items():
        passed=joint=0
        for r in group:
            if not r.get('parse_success') or not r.get('plan'):
                continue
            p,_=normalize_format(r['plan'],by[r['id']]['state'])
            try:
                ok=not score_case(p,by[r['id']],maps)
                passed+=ok; joint+=ok and not explanation_faults(p)
            except (ValueError,TypeError,KeyError):
                pass
        diagnostic[tag]=dict(rules=passed,joint=joint,label='post-hoc representation-only diagnostic, not primary')
    comparisons={t:{'rules':cluster_interval(records[t],records['v2'],cases),
        'joint':cluster_interval(records[t],records['v2'],cases,'joint')} for t in ARMS}
    comparisons['augmented_vs_clean']={'rules':cluster_interval(records['v4'],records['v4_clean'],cases),
        'joint':cluster_interval(records['v4'],records['v4_clean'],cases,'joint')}
    actual=json.loads(remote("import json\nfrom pathlib import Path\nr=Path('/home/jiale/TravelMind-posttrain/logs')\nprint(json.dumps({a:json.loads((r/('quality-'+a+'-token-summary.json')).read_text()) for a in ['v4_clean','v4']}))"))
    result=dict(summaries=summaries,paired=comparisons,actual_training_tokens=actual,format_diagnostic=diagnostic,
        limitations=['development only; independent test not used','fixed48 optimizer updates do not equalize exposure or actual token budget',
            'all target prose mechanically grounded, not original teacher prose','one training seed per arm; inference seeds are not training repeats'],paid_calls=0)
    dump(OUT/'ablation_summary.json',result)
    lines=['# 清洗与补数SFT对照','','冻结五城市50题×3种子，Planner-Only，不使用独立测试。v2复用同协议已保存结果。',
        '训练从原始Qwen基座分别开始，48步、同LoRA/LR/种子；固定更新次数不代表Token或曝光完全相等。','',
        '|版本|规则 /150|规则+解释 /150|常规 /120|困难 /30|无解 /15|',
        '|---|---:|---:|---:|---:|---:|']
    for tag,s in summaries.items():
        lines.append(f"|{tag}|{s['all']['passed']}|{s['all']['joint_passed']}|{s['normal']['passed']}|{s['hard']['passed']}|{s['explained_infeasible']['passed']}|")
    lines+=['','## 配对差值','',json.dumps(comparisons,ensure_ascii=False,indent=2),'',
        '## 实际训练Token','',json.dumps(actual,ensure_ascii=False,indent=2),'',
        '## 仅格式修复诊断（非主成绩）','',json.dumps(diagnostic,ensure_ascii=False,indent=2),'',
        '## 解释边界','',
        '共同通过使用冻结的窄解释检查，不等于全面事实真实性。仍需人工复核时长、天数、数量、室内事实与可行题误拒绝。',
        '本轮仅比较两组SFT候选与v2；未新增付费API基线、未训练DPO。数据清洗含隔离、提示词迁移与解释改写，不能把相对v2的变化归因于单一因素。']
    (ROOT/'docs/post_train_quality_v4_ablation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {tag:dict(rule=s['all']['passed'],joint=s['all']['joint_passed']) for tag,s in summaries.items()}


def main():
    verify()
    status('waiting_for_two_sft_arms')
    for _ in range(180):
        state=json.loads(remote("from pathlib import Path\nprint(Path('/home/jiale/TravelMind-posttrain/logs/quality-sft-v4-status.json').read_text())"))
        if state['stage']=='training_failed':
            raise RuntimeError('SFT failed; evaluation not started')
        if state['stage']=='training_complete':
            break
        time.sleep(30)
    else:
        raise RuntimeError('Two-arm SFT exceeded90min; inspect server logs')
    ready('Qwen2.5-7B-Instruct')
    try:
        for arm in ARMS:
            path=SUITE/'runs'/('quality4_'+arm)/'results.jsonl'
            if path.exists() and len(rows(path))==150:
                continue
            status('loading_model',arm=arm)
            switch(arm,max_model_len=8192)
            status('evaluating',arm=arm)
            with (OUT/(arm+'-eval.stdout.log')).open('ab') as out,(OUT/(arm+'-eval.stderr.log')).open('ab') as err:
                subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'scripts/evaluate_representative50.py'),
                    '--model-name',MODELS[arm][1],'--tag','quality4_'+arm,'--concurrency','4'],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,check=True)
    finally:
        status('restoring_base')
        switch('base',max_model_len=8192)
    status('completed',result=analyze())


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc)[:500])
        raise
