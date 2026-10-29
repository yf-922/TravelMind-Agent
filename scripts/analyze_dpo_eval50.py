"""Compare completed DPO runs with matched historical SFT development runs."""
import json
from collections import Counter
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/post_train'
SUITE=DATA/'representative_eval50_v1'
OLD=DATA/'count_infeasible_ab_20261025/evaluation_total96/regression'
NEW=SUITE/'shared_prompt_current_v2'

def main():
    current='--current-schema' in sys.argv
    cases=json.loads((SUITE/'cases.json').read_text(encoding='utf-8')); byid={c['id']:c for c in cases}
    results={}; summaries={}; canonical=None
    models=[('base','base'),('v2','v2_sft'),('A','A_sft'),('B','B_sft')] if current else [('v2','v2-current'),('A','A'),('B','B')]
    for tag,parent in models:
        for stage in (['sft'] if tag=='base' else ['sft','dpo']):
            for mode in ['planner-only','rules-rework']:
                folder=(OLD/f'{parent}_{mode}') if stage=='sft' else NEW/f'{tag}_dpo_{mode}'
                if current and stage=='sft': folder=SUITE/f'sft_schema_control_20261027/{parent}_{mode}'
                protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
                signature={k:protocol[k] for k in ['system_sha256','schema_sha256','seeds','temperature','max_tokens','request_messages_sha256']}
                if canonical is None: canonical=signature
                if current: assert canonical==signature, 'Schema/protocol mismatch '+str(folder)
                assert {k:v for k,v in canonical.items() if k!='schema_sha256'}=={k:v for k,v in signature.items() if k!='schema_sha256'}, 'Protocol mismatch '+str(folder)
                rows=[json.loads(l) for l in (folder/'results.jsonl').read_text(encoding='utf-8').splitlines()]
                assert len(rows)==150 and {(r['id'],r['seed']) for r in rows}=={(c['id'],s) for c in cases for s in [42,43,44]}
                key=f'{tag}/{stage}/{mode}'; results[key]=rows
                def passes(rs): return sum(bool(r['passed']) and not r.get('error_type') for r in rs)
                groups={g:[r for r in rows if byid[r['id']][field]==value] for g,field,value in [
                    ('normal','stratum','normal'),('hard','stratum','hard'),('feasible','expected','feasible'),('infeasible','expected','explained_infeasible')]}
                summaries[key]=dict(n=150,passed=passes(rows),joint=sum(r['joint_passed'] and not r.get('error_type') for r in rows),
                    parsed=sum(r['parse_success'] for r in rows),errors=dict(Counter(r['error_type'] for r in rows if r.get('error_type'))),
                    groups={g:dict(n=len(rs),passed=passes(rs)) for g,rs in groups.items()},
                    faults=dict(Counter(f for r in rows for f in r['faults'])),
                    tokens=sum(r.get('usage_total',0) for r in rows),mean_s=sum(r['elapsed_s'] for r in rows)/150,
                    reworks=sum(max(0,len(r['rounds'])-1) for r in rows)/150,
                    source=str(folder.relative_to(ROOT)))
    paired={}
    for tag in ['v2','A','B']:
        for mode in ['planner-only','rules-rework']:
            a={(r['id'],r['seed']):r for r in results[f'{tag}/sft/{mode}']}
            b={(r['id'],r['seed']):r for r in results[f'{tag}/dpo/{mode}']}
            gain=[k for k in a if not a[k]['passed'] and b[k]['passed']]; loss=[k for k in a if a[k]['passed'] and not b[k]['passed']]
            delta={c['id']:sum(int(b[c['id'],s]['passed'])-int(a[c['id'],s]['passed']) for s in [42,43,44])/3 for c in cases}
            strata=[[c['id'] for c in cases if c['stratum']==st] for st in ['normal','hard']]
            rng=random.Random(20261027)
            samples=sorted(sum(delta[rng.choice(group)] for group in strata for _ in group)/50*100 for _ in range(4000))
            paired[f'{tag}/{mode}']=dict(improved=len(gain),regressed=len(loss),delta_pp=(len(gain)-len(loss))/150*100,
                case_bootstrap_95pct=[samples[100],samples[3899]],regressed_cases=sorted({k[0] for k in loss}))
    out=SUITE/('sft_schema_control_20261027/comparison.json' if current else 'dpo_eval_50x3_20261027/comparison.json')
    out.write_text(json.dumps(dict(summary=summaries,paired=paired,protocol=canonical),ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# DPO 与对应 SFT：50×3 对照','','50题为已多次用于诊断的开发集，不是全新独立测试。三种子为同题重复，不能视为150道独立题。SFT使用历史已完成记录，DPO使用本次记录；system、逐题首轮messages哈希、seed、温度、输出上限一致。服务上下文8192。未重新调用教师API。',
        '重要混杂：DPO使用带严格HH:MM pattern的Schema（ae669174…），历史SFT没有该pattern（8e12d928…）。已验证当前Schema仅删除pattern即可复现旧哈希。时间格式及整体增益不能全部归因于DPO；需要SFT在同一Schema下重测。当前仅为描述性比较。','',
        '“规则通过”是硬约束评分；本数据没有独立的常识评分器，因此不把它另命名为Commonsense。酒店/航班/工具幻觉也没有对应测试字段。',
        '|版本|模式|规则 /150|规则+解释 /150|常规 /120|困难 /30|有解 /135|无解 /15|Token|平均秒|平均返工|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,s in summaries.items():
        tag,stage,mode=key.split('/'); g=s['groups']
        lines.append(f"|{tag}-{stage}|{mode}|{s['passed']}|{s['joint']}|{g['normal']['passed']}|{g['hard']['passed']}|{g['feasible']['passed']}|{g['infeasible']['passed']}|{s['tokens']}|{s['mean_s']:.2f}|{s['reworks']:.2f}|")
    if current:
        lines[0]='# 当前生产 Schema：Base、SFT、DPO 50×3 对照'
        lines[2]='50题开发集×seed42/43/44；全部已完成，共2100条。逐题messages、当前Schema、system、温度0.3、max_tokens2048均核验一致；上下文8192、并发4。无付费教师API。'
        lines[3]='三组DPO来自此前完整运行，Base和三组SFT来自本次补测。各组解析150/150且无异常；时间格式受结构化生成约束，不能据此证明模型自行学会合法时间。该50题已参与多轮诊断，不是全新独立测试集。'
    lines+=['','配对改善/退步及按案例分层bootstrap区间：','','```json',json.dumps(paired,ensure_ascii=False,indent=2),'```',
        '','错误分类与源文件见 comparison.json。时延为不同时间段、并发4的客户端响应时间，不能当作严格硬件隔离性能实验。偏好验证准确率不等于路线任务通过率。']
    report='post_train_current_schema_comparison.md' if current else 'post_train_dpo_eval50_comparison.md'
    (ROOT/'docs'/report).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=summaries,paired=paired),ensure_ascii=False))

if __name__=='__main__': main()
