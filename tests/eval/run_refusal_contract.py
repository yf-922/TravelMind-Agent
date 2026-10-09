"""Offline fault injection: optimistic LLM verdict must not approve unsafe output."""
import argparse
from contextlib import ExitStack
from datetime import date
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.planning import nodes
from app.planning.schemas import TravelPlanState, RouteReview, TimeCheckResult


def load_cases():
    return json.loads((ROOT/'tests/eval/refusal_cases.json').read_text(encoding='utf-8'))


def evaluate(case):
    state = TravelPlanState(query=case['query'], destination='南京', days=1,
                            travel_start_date=date(2026, 10, 15),
                            travel_end_date=date(2026, 10, 15),
                            pois=case['pois'], route=case['route'],
                            modification_notes=case.get('modification_notes'),
                            max_walking_km=case.get('max_walking_km'),
                            review_round=4, max_review_rounds=3,
                            time_check_round=2, max_time_check_rounds=3)
    # Adversarial verdicts are intentionally wrong; no provider/model called.
    with ExitStack() as stack:
        stack.enter_context(patch.object(nodes, 'build_structured_llm', return_value=object()))
        invoke = stack.enter_context(patch.object(nodes, 'invoke_structured', return_value=RouteReview(
            reasoning='故障注入：错误批准', approved=True, score=100)))
        stack.enter_context(patch.object(nodes, 'amap_key', return_value=None))
        stack.enter_context(patch.object(nodes, 'lookup_live_ticket_prices', return_value={}))
        state = state.model_copy(update=nodes.route_risk_gate_node(state))
        gate_detected = case['expected_flag'] in state.route_risk_flags
        state = state.model_copy(update=nodes.make_reviewer_node('offline')(state))
        reviewer_approved = state.approved
        invoke.return_value = TimeCheckResult(reasoning='故障注入：漏报时间冲突', violations=[])
        state = state.model_copy(update=nodes.make_time_check_node('offline')(state))
        next_step = nodes.route_after_time_check(state)
        plan = nodes.finalize_node(state)['final_plan']
    flags = plan['audit_policy']['risk_flags']
    return {'id': case['id'], 'category': case['category'], 'gate_detected': gate_detected,
            'injected_reviewer_approved': reviewer_approved,
            'final_approved': plan['approved'], 'final_risk_flags': flags,
            'bounded_exit': next_step == ['meal_search', 'spot_tips'],
            'reason_exposed_in_audit': case['expected_flag'] in flags,
            'user_route_issues': plan['route_issues'],
            'passed': gate_detected and not plan['approved'] and case['expected_flag'] in flags
                      and next_step == ['meal_search', 'spot_tips']}


def report():
    rows = [evaluate(case) for case in load_cases()]
    return {'kind': 'offline_rule_contract_fault_injection_not_real_model_quality',
            'external_calls': 0, 'cases': len(rows), 'passed': sum(r['passed'] for r in rows),
            'false_approvals': sum(r['final_approved'] for r in rows), 'results': rows,
            'limits': 'Fixed pool and injected drafts; no actual Planner, retrieval, retry loop or real LLM. Audit flags do not prove user-facing explanation quality. Does not relabel the old 39 cases.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'data/eval/refusal_contract.json')
    a = p.parse_args()
    result = report()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
