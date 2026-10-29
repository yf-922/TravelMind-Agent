"""Resumable serialized GPU0 comparison, followed by data-adjustment diagnosis.

All real available checkpoints are included; the environment smoke checkpoint
is labeled separately. No training, paid LLM calls, or test-target export.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_sft_transfer import MODELS, switch, rows
from scripts.evaluate_representative50 import SUITE, verify, summarize

MODELS.update({
    'v3':('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v3','TravelMind-Planner-7B-SFT-v3'),
    'mc':('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-migration-control','TravelMind-Planner-7B-SFT-migration-control'),
    'v4_clean':('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v4-clean','TravelMind-Planner-7B-SFT-v4-clean'),
    'v4':('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v4','TravelMind-Planner-7B-SFT-v4'),
    'large':('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v2-large','TravelMind-Planner-SFT-v2-large')})
TAGS=['base','v1','v2','v3','mc','v4_clean','v4','large']
OFFICIAL=TAGS
OUT=SUITE/'all_versions_rework_v1'


def write(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def run_tag(tag):
    return 'all_versions_rework_v1_'+tag


def get_results(tag):
    path=SUITE/'runs'/run_tag(tag)/'results.jsonl'
    return rows(path) if path.exists() else []


def status(stage,**extra):
    counts={tag:len(get_results(tag)) for tag in TAGS}
    write(OUT/'status.json',dict(stage=stage,updated_at=time.time(),started_at=START,
        completed_requests=sum(counts.values()),target_requests=len(TAGS)*150,counts=counts,paid_calls=0,**extra))
    print(json.dumps(dict(stage=stage,counts=counts,**extra),ensure_ascii=False),flush=True)


def cluster_interval(a,b,cases,key='passed'):
    """Paired, stratified case bootstrap; seeds stay together within each case."""
    aa={(r['id'],r['seed']):r for r in a};bb={(r['id'],r['seed']):r for r in b}
    deltas={}
    def success(r):
        return r['passed'] and (not r['explanation_faults'] if key=='joint' else True)
    for c in cases:
        deltas[c['id']]=sum(int(success(aa[c['id'],s]))-int(success(bb[c['id'],s])) for s in [42,43,44])/3
    groups=[[c['id'] for c in cases if c['stratum']==st] for st in ['normal','hard']]
    rng=random.Random(20261024);samples=[]
    for _ in range(4000):
        samples.append(sum(deltas[rng.choice(group)] for group in groups for _ in group)/50)
    samples.sort()
    return dict(delta_percentage_points=100*sum(deltas.values())/50,
        paired_case_bootstrap_95pct=[100*samples[100],100*samples[3899]],
        note='stratified case bootstrap, correlated seeds kept together; conditional on this designed development suite')


def analyze():
    verify()
    cases=json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))
    by_id={c['id']:c for c in cases}
    results={t:get_results(t) for t in TAGS}
    expected={(c['id'],s) for c in cases for s in [42,43,44]}
    canonical=None
    summaries={}
    for tag,group in results.items():
        if {(r['id'],r['seed']) for r in group}!=expected or len(group)!=150:
            raise RuntimeError('Incomplete/duplicate evaluations: '+tag)
        actual={}
        for r in group:
            q={k:v for k,v in r['request'].items() if k!='model'}
            if q['messages']!=by_id[r['id']]['messages']:
                raise RuntimeError('Frozen message mismatch')
            actual[r['id'],r['seed']]=q
        if canonical is None:
            canonical=actual
        elif actual!=canonical:
            raise RuntimeError('Non-model request parameters differ')
        summaries[tag]=summarize(group,cases)
    write(OUT/'summary.json',summaries)
    comparisons={t:dict(rule=cluster_interval(results[t],results['base'],cases),
        joint=cluster_interval(results[t],results['base'],cases,'joint')) for t in OFFICIAL if t!='base'}
    write(OUT/'paired_comparison.json',comparisons)
    # A development reference, not a deployment or independent-test selection.
    best=max(OFFICIAL,key=lambda t:(summaries[t]['all']['joint_passed'],summaries[t]['all']['passed']))
    best_sft=max(['v1','v2','v3','mc','v4_clean','v4','large'],key=lambda t:(summaries[t]['all']['joint_passed'],summaries[t]['all']['passed']))
    lines=['# 五城市50例：全部现有版本对照','',
        '同一冻结50题×3种子（42/43/44），temperature=0.3，输出2048，服务上下文8192；Planner-Only，无返工。所有请求除模型名外逐字段相同。',
        '40例常规前置、10例困难；正常120次、困难30次；45例有解与5例无解分别评分。Smoke仅供环境验证参考。','',
        '|模型|常规规则 /120|困难规则 /30|总规则 /150|规则+解释 /150|解析 /150|',
        '|---|---:|---:|---:|---:|---:|']
    for t in TAGS:
        s=summaries[t]
        lines.append(f"|{t}|{s['normal']['passed']}|{s['hard']['passed']}|{s['all']['passed']}|{s['all']['joint_passed']}|{s['all']['parsed']}|")
    lines+=['','当前开发集共同通过最多的正式版本：'+best+'；SFT版本中为：'+best_sft+'。排名不构成部署认证。','',
        '|模型|可行路线 /135|正确处理无解 /15|解释问题|平均景点数（有解）|有解非空天|输入输出总Token|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for t in TAGS:
        s=summaries[t];a=s['all']
        lines.append(f"|{t}|{s['feasible']['passed']}|{s['explained_infeasible']['passed']}|{a['explanation_fault_cases']}|{a['mean_spots_per_parsed_feasible_plan']}|{a['feasible_nonempty_day_fraction']}|{a['token_total']}|")
    lines+=['','## 约束类型','', '|类型|'+'|'.join(TAGS)+'|','|---|'+'|'.join(['---:']*len(TAGS))+'|']
    categories=sorted({c['category'] for c in cases})
    for category in categories:
        lines.append('|'+category+'|'+'|'.join(str(summaries[t]['by_category'][category]['passed'])+'/15' for t in TAGS)+'|')
    lines+=['','## 城市','', '|城市|'+'|'.join(TAGS)+'|','|---|'+'|'.join(['---:']*len(TAGS))+'|']
    for city in ['南京','上海','三亚','成都','苏州']:
        lines.append('|'+city+'|'+'|'.join(str(summaries[t]['by_city'][city]['passed'])+'/30' for t in TAGS)+'|')
    lines+=['','## 与Base的配对比较','','按案例分层重采样的95%区间，三个种子绑定在同题内；不把150次当作150道独立题。','',
        '|模型|规则差值（百分点）|规则区间|规则+解释差值（百分点）|共同通过区间|','|---|---:|---|---:|---|']
    for t,c in comparisons.items():
        lines.append(f"|{t}|{c['rule']['delta_percentage_points']:.1f}|{c['rule']['paired_case_bootstrap_95pct']}|{c['joint']['delta_percentage_points']:.1f}|{c['joint']['paired_case_bootstrap_95pct']}|")
    # Priorities use paired failures rather than aggregate pass rate alone.
    base={(r['id'],r['seed']):r for r in results['base']}
    chosen={(r['id'],r['seed']):r for r in results[best_sft]}
    priorities=[]
    for category in categories:
        selected=[r for r in results[best_sft] if by_id[r['id']]['category']==category]
        fail=[r for r in selected if not r['passed']]
        regress=[r for r in fail if base[r['id'],r['seed']]['passed']]
        common=[r for r in fail if not base[r['id'],r['seed']]['passed']]
        priorities.append(dict(category=category,n=15,failures=len(fail),regressions_vs_base=len(regress),
            shared_failures_with_base=len(common),explanation_faults=sum(bool(r['explanation_faults']) for r in selected),
            fault_counts=dict(Counter(f for r in fail for f in r['faults'])),
            failed_case_ids=sorted({r['id'] for r in fail})))
    priorities.sort(key=lambda p:(p['regressions_vs_base'],p['failures'],p['explanation_faults']),reverse=True)
    actions={
        'upper_limit':'检查旧答案是否把默认上限当配额。补少量有充分候选的上限/恰好数量成对样本，不能只加稀疏候选。',
        'required_excluded':'补必去与排除同时存在、地点排列顺序变化的样本；审计历史必去/排除与最终选点是否一致。',
        'latest_instruction':'优先审查旧目标是否遵循最新修改；补“旧恰好数量→新上限/新禁去”的反事实对，输入和最终答案必须同版。',
        'multi_day_allocation':'补非均匀每日数量、多日全程去重与充足候选共同出现的例子，不能每天固定同样点数。',
        'below_heat_threshold':'补同一路线条件下37.9/38阈值对照；先检查温度解析是否错误触发，不把规则解析缺陷当作训练问题。',
        'heat_boundary':'补高温11点结束/15点开始与跨入禁限区间的边界对；未知室内保持未知，先补可靠属性证据而非名称猜测。',
        'strict_start':'清理历史“之后”却恰好在端点开始的目标；补严格之后与含端点开始的最小对照，联合检查开放和最晚入场。',
        'midday_break':'补用户休息窗口、真实通勤与不额外生成餐馆的组合；核查解释是否与时间表相符。',
        'sparse_feasible':'保留少量有解稀疏样本，示范少排而不重复；不得把所有稀疏状态标成拒绝。',
        'infeasible':'单独补客观无解、正确说明缺口与可行近邻对；控制拒绝样本占比，避免可行题被保守拒绝。'}
    lines+=['','## 数据调整诊断','',
        '以下以SFT中共同通过最多的 '+best_sft+' 为诊断参考。优先级依据相对Base的新增失败，其次剩余失败；先人工复核失败事实，再决定是否补数。',
        '现有评测答案、路线见证与同题改写均禁止进入训练。新增例子需用不同需求组合与候选子集另建训练母本，保留城市与案例来源分组记录。','']
    for item in priorities:
        lines+=['### '+item['category'],'',
            f"15次中失败{item['failures']}次，其中相对Base新增失败{item['regressions_vs_base']}次、与Base共同失败{item['shared_failures_with_base']}次；解释问题{item['explanation_faults']}次。",
            '错误：'+json.dumps(item['fault_counts'],ensure_ascii=False),
            '案例：'+', '.join(item['failed_case_ids']),
            '建议：'+actions[item['category']], '']
    top=priorities[:3]
    lines+=['优先复核的三个类别：'+', '.join(p['category'] for p in top)+'。若这些类别没有剩余错误，则不为凑量补数据。',
        '如果SFT整体未超过Base，优先排查历史目标和协议迁移一致性，暂缓DPO；如果仅解释退步，优先清洗叙述与时长，而非扩大路线样本。',
        '候选分布建议：以有证据的大候选池样本为主，保留少量稀疏有解、边界及无解；训练占比不能机械照抄评测80/20。',
        '补数需先做小规模受控增量对照（不改评测题、不自动续训），保留原最佳模型；训练步数、曝光频率与样本混合应一并记录。',
        '', '## 证据边界','',
        '本报告自动统计和建议只依据已保存的确定性评分，不代表完成了人工逐句语义复核。室内未知、复杂开放时间、评分器覆盖和驾驶快照时效均需保留原始证据核查。',
        '有解题不能用空排得分；无解题解释使用关键词核对，不能替代完整论证审查。解释共同通过也不是开放域真实性证明。',
        '该集为有意识设计的开发基准，城市和题型存在相关性；Bootstrap仅反映该集上的抽样变化，未覆盖训练随机性或真实流量偏差。',
        '无API教师新调用、无DPO模型训练、无独立测试集使用。原始请求与响应、Token、延迟均保留。']
    write(OUT/'data_adjustment_diagnosis.json',dict(reference=best_sft,priorities=priorities,
        overall_reference=best,automatic_training=False,paid_calls=0,training_data_created=False))
    (ROOT/'docs/post_train_representative50_all_versions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(reference=best,sft_reference=best_sft,
        summary={t:dict(passed=summaries[t]['all']['passed'],joint=summaries[t]['all']['joint_passed']) for t in TAGS})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    verify()
    meta=OUT/'binding.json'
    binding=dict(suite_sha256=hashlib.sha256((SUITE/'FROZEN.json').read_bytes()).hexdigest(),
        models={t:MODELS[t] for t in TAGS},seeds=[42,43,44],concurrency=4,context=8192,
        runner_sha256=hashlib.sha256((ROOT/'scripts/evaluate_representative50.py').read_bytes()).hexdigest(),
        orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),paid_calls=0)
    if meta.exists() and json.loads(meta.read_text())!=json.loads(json.dumps(binding)):
        raise RuntimeError('Batch binding changed; create a new run version')
    write(meta,binding)
    try:
        for tag in TAGS:
            if len(get_results(tag))==150:
                continue
            status('loading_model',model=tag)
            switch(tag,max_model_len=8192)
            status('evaluating',model=tag)
            with (OUT/(tag+'.stdout.log')).open('ab') as stdout,(OUT/(tag+'.stderr.log')).open('ab') as stderr:
                proc=subprocess.Popen([sys.executable,'-X','utf8','-u',str(ROOT/'scripts/evaluate_representative50.py'),
                    '--model-name',MODELS[tag][1],'--tag',run_tag(tag),'--concurrency','4','--mode','rules-rework'],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
                while True:
                    try:
                        code=proc.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        status('evaluating',model=tag)
                if code:
                    raise RuntimeError(f'{tag} runner failed with code {code}; inspect logs')
            status('model_complete',model=tag)
    finally:
        status('restoring_base')
        switch('base')
    status('analyzing')
    result=analyze()
    status('completed',result=result)


START=time.time()
if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        status('failed',error_type=type(exc).__name__,error=str(exc)[:700])
        raise
