"""Four-model paired offline replay evaluation with durable raw API evidence."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.evaluation.task_robustness import load_assets, grade
from scripts.evaluate_reviewer_ablation import cases_v2
from scripts.compare_route_interest import exclusive_ledger
from scripts.post_train_runtime import Meter, append_json, capture_messages, replay_check, quality_faults


def prepare(case):
    s = copy.deepcopy(case['state'])
    if case.get('modification'):
        s['modification_notes'] = case['modification']
        s['query'] += '\n当前修改优先：' + case['modification']
    if case['category'] == 'weather':
        s['weather_forecast'] = case['weather_after']
        s['weather_note'] = '天气暂不可用，请复核' if not case['weather_after'] else None
    return s


def render_report(folder, output, expected_total=60, independent=False):
    rows = []
    for path in folder.glob('*_*.jsonl'):
        if '.calls.' in path.name:
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('kind') == 'post_train_evaluation':
                rows.append(row)
    scope = (f'固定 {expected_total} 场景，后训练未见城市的离线道路回放测试；不证明基座预训练未见或现场运营可执行。'
        if independent else '固定 60 场景，同源回放诊断；不是独立泛化测试。')
    text = ['# 四组后训练对比', '', scope + 'Planner-Only 与 Rules+Rework 分开比较。',
        '未执行项保留待测，失败计入分母。API Token/延迟来自原始调用，不从合成通过样本反推。', '',
        f'|组|模式|完成/{expected_total}|JSON成功率|硬约束通过率|软rubric通过率|任务成功率|越界率|平均返工|平均秒|总Token|',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for tag in ['base', 'sft', 'dpo', 'api']:
        for mode in ['planner-only', 'rules-rework']:
            subset = [r for r in rows if r['tag'] == tag and r['mode'] == mode]
            if not subset:
                text.append(f'|{tag}|{mode}|待测|—|—|—|—|—|—|—|—|')
                continue
            n = len(subset)
            pct = lambda key: f'{100 * sum(bool(r.get("grade", {}).get(key)) for r in subset)/n:.1f}%'
            parse = 100 * sum(bool(r.get('parse_success')) for r in subset)/n
            unknown = 100 * sum('unknown_poi' in r.get('rule_errors', []) for r in subset)/n
            tokens = sum(r.get('total_tokens', 0) for r in subset)
            text.append(f'|{tag}|{mode}|{n}/{expected_total}|{parse:.1f}%|{pct("hard_pass")}|{pct("soft_pass")}|{pct("task_success")}|{unknown:.1f}%|{sum(r["repairs"] for r in subset)/n:.2f}|{sum(r["elapsed_s"] for r in subset)/n:.2f}|{tokens}|')
    text += ['', '范围限制：室内标签来自现有候选元数据推断；登阶强度缺少独立标签，不能声称测得。当前修改直接放入首轮输入，四组统一，不与旧 Reviewer 实验混算。API 代理服务按返回的 model/request id 留档，费用未定价。']
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp')
    temp.write_text('\n'.join(text)+'\n', encoding='utf-8')
    temp.replace(output)


def load_heldout_suite(cases_path, maps_path):
    for path in [cases_path, maps_path]:
        manifest = path.parent/'manifest.json'
        if manifest.exists():
            expected = json.loads(manifest.read_text(encoding='utf-8')).get(path.name)
            if expected != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('heldout frozen artifact hash mismatch')
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    maps = json.loads(maps_path.read_text(encoding='utf-8'))['routes']
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('empty or duplicate heldout test cases')
    from scripts.prepare_heldout_evidence import required_edges, opening_evidence_faults
    for case in cases:
        if (case.get('readiness') != 'offline_replay_ready_not_realworld_certified'
                or case.get('training_use') != 'forbidden'):
            raise ValueError('heldout evidence is not ready or test-only')
        if required_edges([case]) - maps.keys():
            raise ValueError('incomplete heldout directed road matrix')
        # Witness is an evaluator-only feasibility label, never added to state.
        plan = case['feasibility_evidence']['plan']
        errors, checked = replay_check(plan, case['state'], maps)
        if errors or quality_faults(checked) or opening_evidence_faults(plan, case) or not grade(checked, case, maps)['task_success']:
            raise ValueError('heldout feasibility witness failed recheck')
    return cases, maps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-name', required=True)
    p.add_argument('--tag', choices=['base', 'sft', 'dpo', 'api'], required=True)
    p.add_argument('--mode', choices=['planner-only', 'rules-rework'], default='planner-only')
    p.add_argument('--max-calls', type=int, default=240)
    p.add_argument('--limit', type=int, default=60)
    p.add_argument('--cases', type=Path, help='ready heldout cases.json; requires --maps')
    p.add_argument('--maps', type=Path, help='frozen heldout maps.json; no online map calls')
    p.add_argument('--output-dir', type=Path, default=ROOT/'data/post_train/eval')
    p.add_argument('--out', type=Path, default=ROOT/'docs/post_train_ablation_report.md')
    a = p.parse_args()
    load_local_env()
    if bool(a.cases) != bool(a.maps):
        p.error('--cases and --maps must be provided together')
    independent = a.cases is not None
    if independent:
        cases, maps = load_heldout_suite(a.cases, a.maps)
        # Separate ledgers/reports protect the already-recorded 60-case API run.
        a.output_dir = a.output_dir/'city-heldout-v1'
        if a.out == ROOT/'docs/post_train_ablation_report.md':
            a.out = ROOT/'docs/post_train_heldout_report.md'
    else:
        cases = cases_v2()
        _, maps = load_assets()
    expected_total = len(cases)
    cases = cases[:a.limit]
    path = a.output_dir/f'{a.tag}_{a.mode}.jsonl'
    binding = hashlib.sha256(json.dumps({'cases': cases, 'maps': maps, 'model': a.model_name,
        'script': Path(__file__).read_text(encoding='utf-8'), 'runtime': (ROOT/'scripts/post_train_runtime.py').read_text(encoding='utf-8')}, sort_keys=True).encode()).hexdigest()
    with exclusive_ledger(path):
        meta = path.with_suffix('.meta.json')
        if meta.exists() and json.loads(meta.read_text(encoding='utf-8'))['binding'] != binding:
            raise RuntimeError('evaluation binding changed; choose new output directory')
        meta.write_text(json.dumps({'binding': binding, 'model': a.model_name, 'cases': cases}, ensure_ascii=False, indent=2), encoding='utf-8')
        done = {json.loads(l)['id'] for l in path.read_text(encoding='utf-8').splitlines()} if path.exists() else set()
        meter = Meter(path.with_suffix('.calls.jsonl'), a.model_name, a.max_calls)
        for case in cases:
            if case['id'] in done:
                continue
            start = time.perf_counter()
            row = {'kind': 'post_train_evaluation', 'id': case['id'], 'tag': a.tag, 'model': a.model_name,
                'mode': a.mode, 'repairs': 0, 'parse_success': False, 'rounds': [], 'total_tokens': 0}
            initial = prepare(case)
            state = copy.deepcopy(initial)
            try:
                for round_no in range(1 if a.mode == 'planner-only' else 4):
                    messages = capture_messages(state)
                    plan, call = meter.call(messages, {'case': case['id'], 'round': round_no})
                    row['parse_success'] = True
                    row['total_tokens'] += (call.get('usage') or {}).get('total_tokens', 0)
                    errors, checked = replay_check(plan, initial, maps, case.get('fault') if case['category'] == 'map' else None)
                    errors = sorted(set(errors + quality_faults(checked)))
                    row['rounds'].append({'messages': messages, 'output': plan, 'rule_errors': errors, 'call_id': call['call_id']})
                    row['rule_errors'] = errors
                    row['repairs'] = round_no
                    if a.mode != 'planner-only':
                        checked.final_plan['approved'] = not errors
                    checked.review_round = round_no + 1
                    row['grade'] = grade(checked, case, maps)
                    if independent:
                        from scripts.prepare_heldout_evidence import opening_evidence_faults
                        extra = opening_evidence_faults(plan, case)
                        scored = row['grade']
                        errors = sorted(set(errors + extra + scored['hard_violations'] + scored['soft_violations']))
                        scored['hard_violations'] = sorted(set(scored['hard_violations'] + extra))
                        scored['hard_pass'] = not scored['hard_violations']
                        scored['task_success'] = scored['task_success'] and not errors
                        scored['unsafe_release'] = scored['approved'] and bool(errors)
                        row['rule_errors'] = errors
                        row['rounds'][-1]['rule_errors'] = errors
                    if not errors or a.mode == 'planner-only':
                        break
                    state.update(route=plan['days'], meal_slots=plan.get('meal_slots', []),
                        route_modify_opinion='确定性规则未通过：' + ', '.join(errors), review_round=round_no + 1)
            except Exception as exc:
                row['error'] = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
                row['grade'] = {'hard_pass': False, 'soft_pass': False, 'task_success': False}
            row['elapsed_s'] = time.perf_counter() - start
            append_json(path, row)
            render_report(a.output_dir, a.out, expected_total, independent)
            print(json.dumps({'id': row['id'], 'tag': a.tag, 'success': row.get('grade', {}).get('task_success'), 'error': row.get('error'), 'calls': meter.count}), flush=True)
            if row.get('error') and 'MAX_LLM_CALLS' in row['error']:
                break


if __name__ == '__main__':
    main()
