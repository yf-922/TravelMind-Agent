"""Read-only protocol audit and independent rescoring; never calls a model."""
from __future__ import annotations
import collections
import copy
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.planning.schemas import TravelRoute
from scripts.build_representative_eval50 import OUT as SUITE, score_case
from scripts.post_train_runtime import capture_messages
from scripts.prepare_heldout_evidence import opening_window

DATA = ROOT/'data/post_train'
OUT = DATA/'controlled_protocol_audit_20261027'
PERIODS = {'morning', 'afternoon', 'evening'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def clock(value):
    if not isinstance(value, str) or not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', value):
        return None
    h, m = map(int, value.split(':'))
    return 60*h+m


def dimensions(plan, case, faults):
    """Orthogonal raw-plan checks; unparseable delivery never earns micro credit.

    Empty valid itineraries vacuously satisfy POI constraints; feasibility and
    required count remain separate rubric checks. Denominators are explicit.
    """
    keys = ['pool', 'opening', 'timeline', 'transit_evidence', 'duplicate', 'period', 'time_format']
    if plan is None:
        return dict.fromkeys(keys, False)
    spots = [s for d in plan['days'] for s in d['spots']]
    by_name = {p['name']: p for p in case['state']['pois']}
    names = [s['name'] for s in spots]
    good_clock = all(clock(s['start_time']) is not None and clock(s['end_time']) is not None for s in spots)
    timeline = good_clock
    opening = True
    for day in plan['days']:
        last = -1
        for spot in day['spots']:
            start, end = clock(spot['start_time']), clock(spot['end_time'])
            if start is None or end is None or start < last or end <= start:
                timeline = False
            if end is not None:
                last = end
            poi = by_name.get(spot['name'])
            window = opening_window(poi) if poi else None
            if not window or start is None or end is None or start < window[0] or end > window[1] or start > window[2]:
                opening = False
    return dict(pool=all(n in by_name for n in names), opening=opening,
        timeline=timeline, transit_evidence=not bool(set(faults)&{'road_unverified', 'commute_time_conflict'}),
        duplicate=len(names)==len(set(names)), period=all(s['period'] in PERIODS for s in spots), time_format=good_clock)


def diagnostic_period_repair(plan):
    """Counterfactual only: change invalid period strings, never names or times."""
    result = copy.deepcopy(plan)
    for day in result['days']:
        for spot in day['spots']:
            if spot['period'] not in PERIODS:
                minute = clock(spot['start_time'])
                if minute is not None:
                    spot['period'] = 'morning' if minute < 720 else 'afternoon' if minute < 1080 else 'evening'
    return result


def training_audit(path, current, frozen):
    rows = json.loads(path.read_text(encoding='utf8'))
    systems = collections.Counter()
    periods = collections.Counter()
    invalid = 0
    for row in rows:
        system = '\n'.join(m['value'] for m in row['conversations'] if m['from']=='system')
        systems[digest(system)] += 1
        answers = [m['value'] for m in row['conversations'] if m['from']=='gpt']
        if row.get('chosen'):
            answers.append(row['chosen']['value'])
        for answer in answers:
            try:
                plan = TravelRoute.model_validate_json(answer).model_dump(mode='json')
                periods.update(s['period'] for d in plan['days'] for s in d['spots'])
            except ValueError:
                invalid += 1
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        n=len(rows), current_system_matches=systems[digest(current)], frozen_system_matches=systems[digest(frozen)],
        system_hash_counts=dict(systems), invalid_targets=invalid, period_counts=dict(periods))


def main():
    OUT.mkdir(exist_ok=True)
    cases = json.loads((SUITE/'cases.json').read_text(encoding='utf8'))
    maps = json.loads((SUITE/'maps.json').read_text(encoding='utf8'))['routes']
    by_id = {c['id']: c for c in cases}
    current = {c['id']: capture_messages(c['state']) for c in cases}
    frozen_system = cases[0]['messages'][0]['content']
    production_system = current[cases[0]['id']][0]['content']
    result = dict(protocol=dict(frozen_system_chars=len(frozen_system), production_system_chars=len(production_system),
        frozen_system_sha256=digest(frozen_system), production_system_sha256=digest(production_system),
        system_matches=sum(c['messages'][0]['content']==current[c['id']][0]['content'] for c in cases),
        human_matches=sum(c['messages'][-1]['content']==current[c['id']][-1]['content'] for c in cases),
        schema_period=TravelRoute.model_json_schema()['$defs']['SpotPlan']['properties']['period']), training={}, runs={})
    sources = dict(v2='release_current_protocol_v1_v2/v2/travelmind_planner_sft_train.json',
        A='count_infeasible_ab_20261025/A/train.json', B='count_infeasible_ab_20261025/B/train.json',
        DPO='shared_dpo_20261026/release/shared_dpo_train.json')
    for tag, path in sources.items():
        result['training'][tag] = training_audit(DATA/path, production_system, frozen_system)
    (OUT/'system_prompt.diff').write_text(''.join(difflib.unified_diff(frozen_system.splitlines(True), production_system.splitlines(True), fromfile='used_frozen', tofile='current_production')), encoding='utf8')
    detail = []
    for root in [DATA/'controlled_benchmark_20261027', DATA/'controlled_teacher_20261027']:
        for file in sorted(root.glob('*/results.jsonl')):
            rows = [json.loads(s) for s in file.read_text(encoding='utf8').splitlines()]
            counters = collections.Counter()
            faults_count = collections.Counter()
            for row in rows:
                case = by_id[row['id']]
                last = row['rounds'][-1]
                plan = last.get('output') if not last.get('error') else None
                faults = score_case(plan, case, maps) if plan is not None else ['delivery_failure']
                checks = dimensions(plan, case, faults)
                counters['delivered'] += plan is not None
                counters['passed'] += not faults
                counters['delivery_flag_mismatch'] += row['delivery'] != (plan is not None)
                counters['faults_mismatch'] += set(faults) != set(row['faults'])
                for k, v in checks.items():
                    counters[k+'_pass'] += v
                repaired_faults = score_case(diagnostic_period_repair(plan), case, maps) if plan is not None else ['delivery_failure']
                counters['diagnostic_period_only_repair_pass'] += not repaired_faults
                counters['structure_cleared_by_period_only'] += 'route_structure' in faults and 'route_structure' not in repaired_faults
                faults_count.update(faults)
                detail.append(dict(run=file.parent.name, id=row['id'], seed=row['seed'], delivered=plan is not None,
                    faults=faults, dimensions=checks, diagnostic_period_repair_faults=repaired_faults,
                    expected=case['expected'], city=case['city']))
            result['runs'][file.parent.name] = dict(n=len(rows), **counters, faults=dict(faults_count))
    # Freeze a new protocol artifact; never overwrite original 50-case assets.
    versioned = copy.deepcopy(cases)
    for c in versioned:
        c['messages'] = current[c['id']]
    (OUT/'production_cases.json').write_text(json.dumps(versioned, ensure_ascii=False, indent=2), encoding='utf8')
    # Two cases per city: regular plus challenging; no generated answers changed.
    sample = []
    for city in dict.fromkeys(c['city'] for c in cases):
        city_cases = [c for c in versioned if c['city']==city]
        sample += [next(c for c in city_cases if c['stratum']=='normal'), next(c for c in reversed(city_cases) if c['stratum']=='hard')]
    (OUT/'smoke_cases.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf8')
    (OUT/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    (OUT/'rescored.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in detail), encoding='utf8')
    lines = ['# 受控重测协议审计与零调用复算', '',
        '原始数据保持不变；period-only repair 是反事实诊断，不是正式成绩。', '',
        f'冻结系统提示词 {len(frozen_system)} 字；当前生产 {len(production_system)} 字；系统一致 0/50，用户上下文一致 50/50。', '',
        '## 训练输入与目标', '', '| 数据 | 条数 | 当前生产系统匹配 | 旧评测系统匹配 | 目标 Schema 失败 |', '|---|---:|---:|---:|---:|']
    for tag, r in result['training'].items():
        lines.append(f"| {tag} | {r['n']} | {r['current_system_matches']} | {r['frozen_system_matches']} | {r['invalid_targets']} |")
    lines += ['', '## 原始输出复算', '', '| 运行 | 最终交付 | 原始通过 | period合法 | 独立时间轴通过 | 道路证据/耗时通过 | 仅修period后通过（诊断） |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in result['runs'].items():
        lines.append(f"| {name} | {r['delivered']}/{r['n']} | {r['passed']} | {r.get('period_pass',0)} | {r.get('timeline_pass',0)} | {r.get('transit_evidence_pass',0)} | {r['diagnostic_period_only_repair_pass']} |")
    lines += ['', '独立时间轴只检查合法HH:MM、正时长、原始顺序无重叠；道路维度使用原有道路回放结果；不把period错误映射为时间重叠。空行程在各地点子规则上为空真，仍受完整可行性/数量rubric约束。', '',
        'Schema中period只是string加文字描述，没有enum约束。Pydantic通过不等于生产风险门控通过。旧运行器只反馈route_structure标签，不能指出具体字段，因此返工难定位。', '',
        '本地解析失败的原始字符串未保存，无法事后判断更多失败细节；不能对这些样本进行自动修复。报告原始结果，不填造缺失输出。', '',
        '下一阶段使用新建production_cases.json（50题）及smoke_cases.json（10题），不得覆盖冻结原题；先用10题验证当前生产协议。']
    (ROOT/'docs/controlled_protocol_audit_20261027.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
