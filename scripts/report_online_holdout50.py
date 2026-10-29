"""Report frozen holdout arms without mixing in historical protocols or mock scores."""
from __future__ import annotations
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_online_holdout50 import OUT,verify

TAGS=['base','v2_sft','A_sft','B_sft','v2_dpo','A_dpo','B_dpo']

def paired_city_interval(left,right,cases,repeats=3000):
    """Paired city-cluster bootstrap; keep all five cases and three seeds together."""
    def keyed(rows):return {(r['id'],r['seed'],r['mode']):r for r in rows}
    a,b=keyed(left),keyed(right)
    if set(a)!=set(b):return None
    cities={c['id']:c['city'] for c in cases}
    groups={city:[] for city in sorted(set(cities.values()))}
    for k in a:groups[cities[k[0]]].append(int(b[k]['passed'])-int(a[k]['passed']))
    if not all(groups.values()):return None
    rng=random.Random(271027);values=[];names=list(groups)
    for _ in range(repeats):
        sample=[x for name in rng.choices(names,k=len(names)) for x in groups[name]]
        values.append(100*sum(sample)/len(sample))
    values.sort()
    return [values[int(.025*repeats)],values[int(.975*repeats)]]

def main():
    verify();cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
    lines=['# 新城市线上闭环评测（冻结协议）','',
        '50题 × 3生成种子 × 2模式 × 7模型；三种子不是三次独立训练。未完成的组不作最终排名。', '',
        '|模型|模式|完成数/150|结果正确率|可行任务成功率|正确拒绝率|JSON成功率|软偏好满足率|平均返工|平均耗时秒|总Token|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    arms={}
    def pct(a,b):return f'{100*a/b:.1f}%' if b else '—'
    for tag in TAGS:
        folder=OUT/'matrix'/tag
        if not (folder/'summary.json').exists():
            lines.append(f'|{tag}|待完成|—|—|—|—|—|—|—|—|—|');continue
        report=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if report['binding']['mock']:raise RuntimeError('Mock results must not enter formal report')
        arms[tag]=[json.loads(x) for x in (folder/'results.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        for mode in ['planner-only','online-rework']:
            s=report[mode]['all'];n=s['n']
            lines.append(f"|{tag}|{mode}|{n}/150|{pct(s['outcome_correct'],n)}|{pct(s['feasible_correct'],s['feasible_n'])}|{pct(s['correct_refusals'],s['infeasible_n'])}|{pct(s['parse_success'],n)}|{pct(s['soft_correct'],s['soft_applicable_n'])}|{s['mean_reworks']:.2f}|{s['latency_mean_s']:.1f}|{s['token_total']:,}|")
            lines.extend([])
    lines+=['','## 配对差异与不确定性','',
        '以Base为对照，按城市成簇配对Bootstrap（3000次）；同题三种子一起重采样。仅10个城市，区间仍不稳定，不宣称统计显著。','',
        '|模型|模式|相对Base正确率差95%区间（百分点）|','|---|---|---:|']
    for tag,rows in arms.items():
        if tag=='base' or 'base' not in arms:continue
        for mode in ['planner-only','online-rework']:
            a=[r for r in arms['base'] if r['mode']==mode];b=[r for r in rows if r['mode']==mode]
            if len(a)!=150 or len(b)!=150:continue
            interval=paired_city_interval(a,b,cases)
            if interval:lines.append(f'|{tag}|{mode}|[{interval[0]:.1f}, {interval[1]:.1f}]|')
    lines+=['','## 口径与诊断','',
        '- 可行任务135次/模式，无法满足硬约束的任务15次/模式；正确拒绝单列，不作为路线交付。',
        '- 软偏好只在带soft_rubric的任务中计分；自动代理评分不能代替人工体验评审。',
        '- 每组summary.json另有normal/hard、城市、约束类型、硬错误分布、解释错误、误放行/误拒绝、角色Token、P50/P95耗时和三种子稳定性。',
        '- 成本使用真实usage；没有配置价格的API不估算金额。calls.jsonl保留失败输出、实际消息和返工证据。',
        '- Reviewer和time_check固定线上API；上游检索、天气和道路事实冻结，不覆盖在线检索质量。',
        '- 测试集冻结后禁止用于训练或调参；如据此修改模型，此套结果须降为开发集结果。','']
    path=ROOT/'docs/post_train_online_holdout50_report.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(path)

if __name__=='__main__':main()
