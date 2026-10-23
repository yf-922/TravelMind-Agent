"""Deterministic impossible-request replay; no model or network calls."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.rag_protocol import fingerprint
from app.evaluation.task_robustness import (
    build_soft_review_cases, fault_distance, grade_soft_review, load_assets,
)
from app.planning import nodes
from scripts.evaluate_reviewer_ablation import run_arm
from scripts.evaluate_soft_review import preflight


def evaluate():
    cases = [c for c in build_soft_review_cases()
             if c['category'] in {'time-deadlock', 'indoor-pool-conflict'}]
    _, maps = load_assets()
    checks = preflight(cases, maps)
    rows = []
    for case in cases:
        for arm in ('planner_only', 'rules', 'rules_reviewer'):
            calls = {'planner': 0, 'reviewer': 0}

            def planner(state):
                calls['planner'] += 1
                return {'route': copy.deepcopy(case['injected_route']),
                        'review_round': state.review_round + 1}

            def reviewer(state):
                calls['reviewer'] += 1
                return {'approved': True, 'route_modify_opinion': '', 'reviewer_issues': []}

            initial = {**copy.deepcopy(case['state']), 'route': copy.deepcopy(case['injected_route'])}
            trace = []
            provider = {}
            with ExitStack() as stack:
                for name, replacement in {
                    'amap_key': lambda: 'replay',
                    'plan_route_distance': fault_distance(maps, None, provider),
                    'plan_transport_leg': lambda *args, **kwargs: None,
                    'recommend_chain_hotel': lambda *args, **kwargs: None,
                    'lookup_live_ticket_prices': lambda *args, **kwargs: {},
                }.items():
                    stack.enter_context(patch.object(nodes, name, replacement))
                state, repairs = run_arm(initial, arm, planner, reviewer, max_repairs=3, trace=trace)
            result = grade_soft_review(state, case, maps)
            row = {'id': case['id'], 'arm': arm, 'repairs': repairs,
                   'callback_calls': calls, 'grade': result,
                   'final_plan': state.final_plan,
                   'trace': [{'repair_index': step['repair_index'], 'rule_faults': step['rule_faults']}
                             for step in trace],
                   'candidate_pool_unchanged': state.pois == initial['pois']}
            if arm != 'planner_only' and not (
                repairs == 3 and len(trace) == 4 and result['safe_explained_rejection']
                and not result['task_success'] and row['candidate_pool_unchanged']
            ):
                raise AssertionError(f"unsafe rejection: {case['id']} / {arm}")
            rows.append(row)
    return {
        'scope': 'deterministic_fault_injection_not_live_model_ablation',
        'model_calls': 0, 'external_calls': 0, 'max_repairs': 3,
        'binding': fingerprint({'cases': cases, 'maps': maps,
                               'implementation': {
                                   str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (Path(__file__), Path(nodes.__file__),
                                                ROOT / 'scripts/evaluate_reviewer_ablation.py',
                                                ROOT / 'app/evaluation/task_robustness.py')
                               }}),
        'label_status': 'verified_v2', 'cases': cases, 'preflight': checks, 'rows': rows,
        'summary': {arm: {
            'safe_rejections': sum(r['grade']['safe_explained_rejection'] for r in rows if r['arm'] == arm),
            'cases': len(cases),
        } for arm in ('planner_only', 'rules', 'rules_reviewer')},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/eval/infeasible_safety_v3.json')
    args = parser.parse_args()
    report = evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'summary': report['summary'],
                      'model_calls': 0, 'external_calls': 0}))


if __name__ == '__main__':
    main()
