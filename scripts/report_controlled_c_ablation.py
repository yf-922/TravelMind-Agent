"""Render the seven-arm baseline plus C-SFT/C-DPO incremental comparison."""
from __future__ import annotations
import argparse,json
import math
from pathlib import Path
TAGS=['base','v2_sft','A_sft','B_sft','v2_dpo','A_dpo','B_dpo','C_sft','C_dpo']
MODES=['planner-only','online-rework']

def qtile(values,q):
    values=sorted(values)
    return values[max(0,math.ceil(q*len(values))-1)] if values else None

def latency_by_role(run_dir,tag,mode):
    path=run_dir/tag/'results.jsonl'
    if not path.exists():return None
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows=[row for row in rows if row.get('mode')==mode]
    grouped={role:[] for role in ['joint_planner','reviewer','time_check']}
    for row in rows:
        calls=[call for call in row.get('calls',[]) if call.get('status')=='completed']
        for role in grouped:
            elapsed=[call.get('elapsed_s') for call in calls
                     if call.get('task')==role and call.get('elapsed_s') is not None]
            if elapsed:grouped[role].append(sum(elapsed))
    return {role:{'p50':qtile(values,.5),'p95':qtile(values,.95),'n':len(values)}
            for role,values in grouped.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);root=p.parse_args().run_dir
    rows={};missing=[]
    for tag in TAGS:
        f=root/tag/'summary.json'
        if not f.exists():missing.append(tag);continue
        rows[tag]=json.loads(f.read_text(encoding='utf-8'))
    out=['# 九模型受控消融对比报告','',
      '每模型 50 场景 × 3 解码 seed × 2 模式；3 个 seed 是重复生成配置，不是独立训练。Planner 使用生产 Joint Planner 与生产提示词；Reviewer/time_check 不调用教师 API，改用冻结 rubric 的确定性本地规则反馈，最多额外返工 3 次。模型统一完整 BF16、流式 vLLM、关闭 Prefix Cache、并发 1。Planner 为线上 temperature=0.3、top_p=0.95；所有权重组使用共同 seed 1001/1002/1003，返工 Planner seed 加 10000。','',
      '|模型|模式|N|Feasible Delivery %|Feasible All-Rule %|Overall Delivery %|Overall All-Rule %|本地规则通过率 %|Overall任务正确率 %|Pool实体忠实度 %|Opening %|Timeline/Transit %|No-Duplicate %|Pass@1 Mean±Std %|All-3-Pass %|Prompt tok/task|Completion tok/task|Planner tok/call|Planner chars/call|Feasible Planner/E2E P50/P95 s|TTFT P50 s|TPOT P50 ms/token|Total tok|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x,d=1):return '—' if x is None else f'{x:.{d}f}'
    def pct(x):return '—' if x is None else f'{x:.1f}%'
    for tag in TAGS:
      if tag not in rows:
        for mode in MODES:out.append(f'|{tag}|{mode}|待完成|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|')
        continue
      sm=rows[tag]
      for mode in MODES:
        if mode not in sm:out.append(f'|{tag}|{mode}|待完成|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|—|');continue
        x=sm[mode]
        out.append('|'+ '|'.join([tag,mode,str(x['n']),pct(x['delivery_rate']),pct(x['all_rule_pass_rate']),
          pct(x['overall_delivery_rate']),pct(x['overall_all_rule_pass_rate']),pct(x['local_rule_acceptance_rate']),pct(x['correct_task_rate']),
          pct(x['in_pool_entity_faithfulness']),pct(x['subrule_pass_rate']['opening']),pct(x['subrule_pass_rate']['timeline_transit']),
          pct(x['subrule_pass_rate']['no_duplicate']),
          fmt(x['pass_at_1_mean_pct'])+'±'+fmt(x['pass_at_1_seed_std_pp']),pct(x['all_3_pass_rate_pct']),
          fmt(x['avg_prompt_tokens']),fmt(x['avg_completion_tokens']),
          fmt(x['avg_planner_completion_tokens']),fmt(x['avg_planner_output_chars']),
          fmt(x['feasible_e2e_latency_p50_s'])+'/'+fmt(x['feasible_e2e_latency_p95_s']),fmt(x['ttft_p50_s'],2),fmt(x['normalized_tpot_p50_ms']),
          f"{x['total_tokens']:,}"])+'|')
    out+=['','## 返工效率','',
      '|模型|样本集|真实返工触发率 %|首轮规则失败代理 %|平均返工轮次|返工挽回率 %|Planner总 Token|','|---|---|---:|---:|---:|---:|---:|']
    for tag in TAGS:
      x=rows.get(tag,{}).get('rework_efficiency')
      for group in ['feasible','infeasible']:
        if not x or group not in x:out.append(f'|{tag}|{group}|待完成|—|—|—|—|')
        else:
          total=rows[tag]['online-rework'][group]['role_token_totals']['joint_planner']['total_tokens']
          y=x[group]
          out.append(f"|{tag}|{group}|{fmt(y['rework_trigger_rate_pct'])}|{fmt(y['rule_fail_trigger_proxy_pct'])}|{fmt(y['avg_rework_turns'],2)}|{fmt(y['rework_recovery_rate_pct'])}|{total:,}|")
    out+=['','## 分角色延迟（排除外部审核的模型性能口径）','',
      '|模型|模式|Planner 累计 P50/P95 秒|Reviewer API 累计 P50/P95 秒|time_check API 累计 P50/P95 秒|样本数|',
      '|---|---|---:|---:|---:|---:|']
    for tag in TAGS:
      for mode in MODES:
        values=latency_by_role(root,tag,mode)
        if not values:
          out.append(f'|{tag}|{mode}|待完成|待完成|待完成|—|');continue
        def role_fmt(role):
          x=values[role]
          return '—' if x['p50'] is None else f"{x['p50']:.2f}/{x['p95']:.2f}"
        n=values['joint_planner']['n']
        out.append(f"|{tag}|{mode}|{role_fmt('joint_planner')}|{role_fmt('reviewer')}|{role_fmt('time_check')}|{n}|")
    out+=['','## 无解场景（每模型 15 次生成）','',
      '|模型|模式|不可行识别率 %|拒答合规率 %|路线幻觉率 %|Infeasible E2E P50/P95 s|','|---|---|---:|---:|---:|---:|']
    for tag in TAGS:
      for mode in MODES:
        x=rows.get(tag,{}).get(mode,{}).get('infeasible')
        if not x:out.append(f'|{tag}|{mode}|待完成|—|—|—|')
        else:out.append(f"|{tag}|{mode}|{pct(x['infeasible_recognition_rate'])}|{pct(x['refusal_compliance_rate'])}|{pct(x['infeasible_hallucination_rate'])}|{fmt(x['e2e_latency_p50_s'])}/{fmt(x['e2e_latency_p95_s'])}|")
    out+=['','## SFT 与 DPO 配对比较','',
      '|配对|Planner-Only 输出 Token 差（DPO-SFT）|Rules+Rework 总 Token 节省率|','|---|---:|---:|']
    for sft,dpo,label in [('v2_sft','v2_dpo','v2'),('A_sft','A_dpo','A'),('B_sft','B_dpo','B'),('C_sft','C_dpo','C')]:
      if sft not in rows or dpo not in rows:out.append(f'|{label}|待完成|待完成|');continue
      a=rows[sft]['planner-only']['feasible']['avg_planner_completion_tokens'];b=rows[dpo]['planner-only']['feasible']['avg_planner_completion_tokens']
      st=rows[sft]['online-rework']['total_tokens'];dt=rows[dpo]['online-rework']['total_tokens']
      save=(100*(st-dt)/st) if st else None
      out.append(f'|{label}|{fmt(b-a)} tokens ({fmt(100*(b-a)/a) if a else "—"}%)|{pct(save)}|')
    out+=['','## 归因明细与解释模板','',
      '每个模型目录的 `results.jsonl` 保存任务终态、原始 Planner 回复、每轮请求/usage、规则故障和实际重规划次数；`calls.jsonl` 保存调用开始与结束记录。`summary.json` 含分城市、错误分布及角色 Token。`warmup.json` 记录丢弃预热样本的解析数和额外 Token。',
      '', '建议按以下顺序写结论，避免将相关性说成因果：',
      '', '> 在固定候选池、生产提示词与 Schema、相同 LangGraph 审核/返工路径、统一 BF16 vLLM、无 Prefix Cache 和相同并发下，DPO 相比其对应 SFT 的 Planner-Only All-Rule Pass 从 X% 变为 Y%，其中 Timeline/Transit 子项从 A% 变为 B%；输出长度变化为 C%（结合 temperature=0.3 下三个共同 seed 的 Pass@1 均值、标准差和 All-3-Pass 描述采样稳定性）。Rules+Rework 的平均返工从 R1 变为 R2，累计 Token 从 T1 变为 T2（差异 Z%），返工挽回率为 Q%。E2E P50/P95 与 TTFT/TPOT 分别为……。这些结果支持/不支持“首轮质量提高并降低闭环成本”的结论；每模型只有 50 个场景、来自 10 个城市，且模型按单卡顺序加载，延迟差异不能单独归因于权重，结论限于当前冻结集。',
      '', '## 重要口径','',
      '- 主表 Feasible Delivery、Feasible All-Rule 与路线子规则以每模型 45 个可行题为分母，未交付按失败计；Overall Delivery 和 Overall All-Rule 另以 50 题为分母。无解题从可行子项与可行延迟分位数排除。无解单表每模型 15 次生成，正确拒答要求空路线、完整天数及解释与证明的冲突类型一致；任务正确率把正确拒答计成功。',
      '- Rework Recovery 按同一场景、同一 seed 配对：首轮未通过、闭环最终通过的占比。另报告规则首轮失败代理与生产图实际触发率，两者不必相等。',
      '- Prompt/Completion Token 含生产 Reviewer 与 time_check 的角色汇总，按任务求均值；Planner Completion Token/字符均值按实际 Planner 请求数求值，结合 `planner_calls` 解读返工成本。',
      '- Planner-Only 与 Rules+Rework 的 Planner 累计 P50/P95 按每条轨迹独立汇总 Planner 调用耗时；Reviewer 与 time_check 外部 API 延迟另列，不用于归因模型快慢。闭环 E2E P50/P95 仍表示用户端完整等待时间，会包含审核服务及其网络耗时。',
      '- 本评测没有线上 Reviewer/time_check；本地反馈只来自 score_case 冻结确定性 rubric 的错误码，不具备语义 Reviewer 的理解能力。质量结论对应“规则反馈返工”，不能解释为线上语义审核闭环效果。',
      '- TPOT 由 Planner 流式首个有文本 chunk 到流结束的时间差除以 completion tokens−1 得出；包含本地请求往返、vLLM 排队和服务端生成，不等同于隔离 GPU kernel 的纯推理速度。',
      '- 当前单张 24GB GPU 不能同时驻留七个 BF16 权重；为保证形态统一采用随机组序的顺序加载，不能宣称做到了跨模型同一时刻交替。每组自己的 P50/P95 适用于服务级观察；权重间延迟差仍受时间漂移影响。',
      '- Planner 使用线上 temperature=0.3、top_p=0.95 和三个共同 seed（1001/1002/1003）；每个额外 Planner 轮次将 seed 加 10000。All-3-Pass 是 50 个基础场景三次都通过的比例；三个重复采样不是独立训练，也不构成广泛线上流量的随机性估计。',
      '- API 金额按供应商最终账单计算；Token usage 仅统计返回的 usage，SDK 内部失败重试若未返回 usage 可能缺失。',
      '']
    out += ['', '## C 版靶向指标', '',
      '|模型|Planner-Only 可行集通过率|Rules+Rework 可行集通过率|首轮 user_time_window 错误 / 135|可行集返工挽回率|无解识别率（Planner / Rework）|合规拒答率（Planner / Rework）|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for tag in ['B_sft', 'B_dpo', 'C_sft', 'C_dpo']:
      sm = rows.get(tag, {})
      planner = sm.get('planner-only', {}).get('feasible', {})
      rework = sm.get('online-rework', {}).get('feasible', {})
      failures = (planner.get('hard_faults') or {}).get('user_time_window', 0)
      recovery = (sm.get('rework_efficiency') or {}).get('feasible', {}).get('rework_recovery_rate_pct')
      p_inf = sm.get('planner-only', {}).get('infeasible', {})
      r_inf = sm.get('online-rework', {}).get('infeasible', {})
      out.append('|'+ '|'.join([tag, pct(planner.get('all_rule_pass_rate')), pct(rework.get('all_rule_pass_rate')),
        f'{failures}/135', pct(recovery),
        pct(p_inf.get('infeasible_recognition_rate'))+' / '+pct(r_inf.get('infeasible_recognition_rate')),
        pct(p_inf.get('refusal_compliance_rate'))+' / '+pct(r_inf.get('refusal_compliance_rate'))])+'|')
    out += ['', '解释：user_time_window 错误数取 Planner-Only 可行集首轮，分母为 45 个可行场景 × 3 个共同 seed；不可行题单独统计，每个模型每种模式 15 次生成。C 版与 B 版使用同一冻结评测集，不代表独立盲测。', '']
    path=root/'controlled_9_model_ablation_report.md';path.write_text('\n'.join(out),encoding='utf-8')
    print(json.dumps(dict(report=str(path),completed=sorted(rows),waiting=missing)))
if __name__=='__main__':main()
