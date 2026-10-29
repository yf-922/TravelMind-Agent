"""Generate the contract experiment report from completed, versioned ledgers."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.supplement_planning_contract import OUT, SOURCE, check
from scripts.prepare_sft_v2_release import read_rows

def report():
    eval_dir=ROOT/'data/post_train/sft_contract_eval_v1'
    summary=json.loads((eval_dir/'summary.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    cases={c['id']:c for c in json.loads((OUT/'cases.json').read_text(encoding='utf-8'))}
    maps=json.loads((OUT/'maps.json').read_text(encoding='utf-8'))['routes']
    records=read_rows(OUT/'synthesis.jsonl')
    rechecked={r['id']:check(r['chosen'],cases[r['id']],maps) for r in records if r['chosen']}
    if any(rechecked.values()):raise RuntimeError('Accepted sample failed final replay')
    usages=[json.loads(l) for p in [ROOT/'data/post_train/contract_supplement_v1/synthesis.calls.jsonl',ROOT/'data/post_train/contract_supplement_v2/synthesis.calls.jsonl',OUT/'synthesis.calls.jsonl'] if p.exists() for l in p.read_text(encoding='utf-8').splitlines()]
    quality=dict(deterministic_rechecked=rechecked,total_started_including_stopped_pilot=sum(r['event']=='started' for r in usages),
        total_recorded_tokens_including_pilot=sum((r.get('usage') or {}).get('total_tokens',0) for r in usages),
        semantic_scope='Explicit counts/durations, references, repair leakage only; not an exhaustive semantic certification',
        old_leak_quarantined=['synth-0127'],old_releases_modified=False,test_cities_untouched=True)
    (OUT/'final_quality_audit.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 统一约束协议：完整开发集重复比较与定向补数', '',
      '生成时间：'+datetime.now().isoformat(timespec='seconds'),'',
      '30个固定开发状态，每模型三次生成，temperature=0.3，输出上限2048，种子42/43/44。输入为重新捕获的新生产协议，不能与旧单次通过数直接作同协议比较。未使用独立测试集；重复生成并非90个独立问题。','',
      '| 模型 | seed42 /30 | seed43 /30 | seed44 /30 | 合计 /90 | 三次全过 /30 | 解析 /90 | 规则+解释检查 /90 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for tag in ['v1','v2']:
        s=summary[tag]; b=s['by_seed']
        lines.append(f"| {tag} | {b['42']} | {b['43']} | {b['44']} | {s['passed']} | {s['all_three_pass']} | {s['parsed']} | {s['joint_success']} |")
    lines+=['',f"按每例三次成功数比较：v2改善{len(summary['improved'])}例，退步{len(summary['regressed'])}例。仅代表本开发集，不能据此宣称独立泛化或因果归因。",'', '## 错误分布','']
    for tag in ['v1','v2']:lines.append(tag+'：`'+json.dumps(summary[tag]['faults'],ensure_ascii=False)+'`')
    lines+=['','## 补数与清理','',f"32个输入完成{audit['completed']}个，合格{audit['accepted']}个，失败/隔离{len(audit['failed'])}个。正式调用{audit['calls']}次，包含停止的pilot累计started {quality['total_started_including_stopped_pilot']}次。已记账token {quality['total_recorded_tokens_including_pilot']}，中断未返usage部分不能假定为零。",'',
      f"合格样本训练token最大{audit['max_tokens']}，未允许截断。已逐条重放确定性审计；这些检查不等于完整语义认证。",'',
      '已确认的旧返工叙述synth-0127从新清理副本隔离。旧release_v1/v2及其原始messages、权重保留。未启动新SFT或DPO。新补数按新协议单独导出；历史数据若迁移新Prompt需单独审计并注明，不可静默改写。','',
      '## 下一步决策','',
      '根据三次全过数和同例成败波动优先判断稳定性，再检查重复、数量、高温和道路的成对变化。先做合格补数的语义审阅及新协议训练集版本冻结，再决定是否开展新SFT；DPO应在SFT稳定提升后另行决定。最终四组含API比较仍在封存独立集上统一记录。']
    (ROOT/'docs/post_train_contract_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(stage='reported',eval_generations=sum(summary[t]['n'] for t in ['v1','v2']),supplement_accepted=audit['accepted'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
    start=time.time()
    while True:
        eval_path=ROOT/'data/post_train/sft_contract_eval_v1/status.json'
        state=json.loads(eval_path.read_text()) if eval_path.exists() else {}
        audit_path=OUT/'audit.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        if state.get('stage')=='failed':raise RuntimeError('Evaluation failed; inspect preserved logs')
        if state.get('stage')=='completed' and audit.get('stage')=='completed':break
        if not args.wait or time.time()-start>7200:raise RuntimeError('Both jobs must complete first')
        time.sleep(10)
    print(json.dumps(report()),flush=True)
if __name__=='__main__':main()
