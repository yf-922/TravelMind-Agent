"""Publish reviewable outcomes without exposing prompts or changing paid ledgers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_reviewer_ablation import ARMS
from scripts.evaluate_soft_review import summarize


def public_report(report, legacy):
    expected = {(case['id'], arm) for case in report['cases'] for arm in ARMS}
    actual = [(row['id'], row['arm']) for row in report['rows']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('one recorded outcome per case and arm is required')
    if any(call['status'] == 'in_flight' for call in report['calls']):
        raise ValueError('unsettled paid attempt')
    outcomes = []
    for row in report['rows']:
        calls = [report['calls'][i] for i in row['call_indices']]
        failures = [{key: call.get(key) for key in ('task', 'status', 'error', 'usage')}
                    for call in calls if call['status'] != 'completed']
        grade = row.get('grade', {})
        outcomes.append({
            'id': row['id'], 'arm': row['arm'], 'final_plan': row.get('final_plan'),
            'grade': grade, 'repairs': row.get('repairs'), 'error': row.get('error'),
            'elapsed_ms': row.get('elapsed_ms'), 'failed_calls': failures,
            'execution_failure_observed': bool(row.get('error') or failures),
            'semantic_refusal_without_model_failure': bool(
                grade.get('safe_explained_rejection') and not row.get('error') and not failures),
            'reviewer_outputs': [call.get('output') for call in calls
                                 if call.get('task') == 'reviewer' and call['status'] == 'completed'],
        })
    usage = {key: sum((call.get('usage') or {}).get(key, 0) for call in report['calls'])
             for key in ('input_tokens', 'output_tokens', 'total_tokens')}
    return {
        'status': report['status'], 'label_status': report['label_status'],
        'formal_test_report': False,
        'scope': '25 historical natural drafts plus 5 controlled challenges; clean control separate',
        'model': report['model'], 'binding': report['binding'],
        'legacy_sha256': report['legacy_sha256'], 'summary': summarize(report, legacy),
        'cases': [{key: case.get(key) for key in ('id', 'category', 'replaces', 'label_status',
                  'expected', 'semantic_rubric', 'rubric', 'injected_route', 'annotation_reason')}
                  | {'query': case['state']['query']} for case in report['cases']],
        'outcomes': outcomes, 'paid_attempts': len(report['calls']),
        'successful_parsed_calls': sum(c['status'] == 'completed' for c in report['calls']),
        'failed_calls': sum(c['status'] != 'completed' for c in report['calls']),
        'usage': usage, 'usage_complete': all(c.get('usage') for c in report['calls']),
        'cost_status': report['cost_status'], 'max_calls': report['max_calls'],
        'requested_max_output_tokens': report['output_token_cap'],
        'observed_max_output_tokens': max(((c.get('usage') or {}).get('output_tokens', 0)
                                          for c in report['calls']), default=0),
        'limitations': [
            ('Controlled case labels are verified_v2; historical labels retain original provenance.'
             if report.get('label_status') == 'verified_v2' else
             'Challenge labels await explicit human confirmation; historical labels remain author drafts.'),
            'Controlled drafts test audit-tail repair, not natural initial Planner error frequency.',
            'Hard scoring partly reuses application risk rules; this is not independent travel validation.',
            'Semantic scoring uses separate predeclared labels, not Reviewer approval or generated notes.',
            'Historical and new runs are not one simultaneous 30-request end-to-end experiment.',
            'Safe refusal is separate from task completion; swallowed call failures remain visible.',
            'TimeCheck and live upstream retrieval/SSE are absent; no full-service quality or latency claim.',
            'Provider output usage may exceed the requested cap; requested cap is not verified enforcement.',
        ],
    }


def render(report):
    lines = ['# Reviewer 软约束挑战：人工复核材料', '',
             f"状态：`{report['label_status']}`；非正式冻结测试报告。模型：`{report['model']}`。", '',
             '以下组合只复用 25 条历史自然草案，替换加入 5 条人工构造挑战；正常对照单列。'
             '它不是重新运行的 30 条自然请求，也不用于估计自然请求中的缺陷比例。', '',
             '| 组别 | 组合任务成功 / 30 | 新增可满足挑战修复 / 4 |',
             '|---|---:|---:|']
    for arm, s in report['summary'].items():
        lines.append(f"| {arm} | {s['composite_task_success']} | {s['new_feasible_success']} |")
    lines += ['', '成功要求：最终批准、生产硬约束及用例附加要求全部满足、语义标签全部满足。'
              '不可满足场景的安全拒绝不计为完成行程。硬约束评分部分复用生产规则；'
              + ('受控用例标签已确认；新增运行结果仍需单独核对。' if report['label_status'] == 'verified_v2'
                 else '语义标签是作者预设，尚待人工确认。'), '']
    for case in report['cases']:
        lines += [f"## {case['id']}", '', case['query'], '',
                  f"预期：`{case['expected']}`；标签：`{case['label_status']}`。",
                  '初始草案：`'+json.dumps(case['injected_route'], ensure_ascii=False)+'`',
                  '待复核语义判据：`'+json.dumps(case['semantic_rubric'], ensure_ascii=False)+'`', '',
                  '| 组别 | 最终景点与时间 | 批准 | 硬违规 | 软违规 | 返工 | 调用失败 |',
                  '|---|---|---|---|---|---:|---|']
        for row in report['outcomes']:
            if row['id'] != case['id']:
                continue
            final = row.get('final_plan') or {}
            selected = [(s.get('name'), s.get('start_time'), s.get('end_time'))
                        for day in final.get('days', []) for s in day.get('timeline', [])
                        if s.get('type') == 'attraction']
            g = row['grade']
            failures = ', '.join(str(c['task'])+': '+str(c['error']) for c in row['failed_calls'])
            lines.append(f"| {row['arm']} | {json.dumps(selected, ensure_ascii=False)} | {g.get('approved')} | {g.get('hard_violations')} | {g.get('soft_violations')} | {row.get('repairs')} | {failures or row.get('error') or '无'} |")
            if row['arm'] == 'rules_reviewer':
                lines += ['', 'Reviewer 实际结构化输出：',
                          '```json', json.dumps(row['reviewer_outputs'], ensure_ascii=False, indent=2), '```',
                          '面向用户的结果原因：'+json.dumps(final.get('route_issues', []), ensure_ascii=False), '']
                if row['execution_failure_observed']:
                    lines += ['**本条发生过模型调用/解析失败。最终安全拒绝可计入兜底，不能记为全程无异常的语义拒绝。**', '']
        lines += ['请复核：初稿是否确实违背需求？最终选点和时段是否满足？预设判据是否合理？', '']
    lines += ['## 实际用量与复现', '',
              f"新增付费尝试 **{report['paid_attempts']}/{report['max_calls']}**，解析成功 {report['successful_parsed_calls']} 次，失败 {report['failed_calls']} 次；费用 `unpriced`。",
              'Provider 实际 usage：`'+json.dumps(report['usage'])+'`。',
              f"请求的输出上限 {report['requested_max_output_tokens']}；Provider 最大输出 usage {report['observed_max_output_tokens']}。两者不同，不能宣称供应商严格执行上限。", '',
              '只从账本汇总，不重新付费调用：', '', '```powershell',
              'python scripts/summarize_soft_review.py',
              'python scripts/review_soft_review_labels.py export',
              '```', '', '限制：'] + ['- '+x for x in report['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=ROOT/'data/eval/soft_review_v3.json')
    parser.add_argument('--legacy', type=Path, default=ROOT/'data/eval/reviewer_ablation_real_v2.json')
    parser.add_argument('--out', type=Path, default=ROOT/'evaluation/soft_review_v3.json')
    args = parser.parse_args()
    raw, old = args.ledger.read_bytes(), args.legacy.read_bytes()
    report = json.loads(raw)
    if hashlib.sha256(old).hexdigest() != report['legacy_sha256']:
        raise ValueError('historical ledger changed')
    result = public_report(report, json.loads(old))
    result['ledger_sha256'] = hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    args.out.with_suffix('.md').write_text(render(result), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'summary', 'paid_attempts', 'failed_calls', 'usage')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
