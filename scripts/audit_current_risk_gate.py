"""Audit actual ledger usage and current-policy decisions; never invent unseen calls."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.planning import nodes
from app.planning.schemas import TravelPlanState
from app.evaluation.task_robustness import load_assets, fault_distance


def audit(report):
    cases = [c for c in report['cases'] if c['suite'] == 'task']
    _, maps = load_assets()
    rows = []
    observed_call_ids = set()
    initial_reviewer_ids = []
    for case in cases:
        seed = report.get('seeds', {}).get(case['id'], {})
        outcome = next((r for r in report['rows'] if r['id'] == case['id']
                        and r['arm'] == 'rules_reviewer'), None)
        call_ids = (outcome or {}).get('paid_call_indices', [])
        audit_ids = [i for i in call_ids if report['calls'][i].get('task') in ('reviewer', 'time_check')]
        observed_call_ids.update(audit_ids)
        reviewer_ids = [i for i in audit_ids if report['calls'][i].get('task') == 'reviewer']
        if reviewer_ids:
            initial_reviewer_ids.append(reviewer_ids[0])
        row = {'id': case['id'], 'seed_available': bool(seed.get('state')),
               'observed_audit_calls': len(audit_ids)}
        if seed.get('state'):
            state = TravelPlanState(**copy.deepcopy(seed['state']))
            with patch.object(nodes, 'amap_key', return_value='replay'), patch.object(
                nodes, 'plan_route_distance', side_effect=fault_distance(maps, None, {})
            ):
                state = state.model_copy(update=nodes.route_distance_check_node(state))
            # Metrics must not be incremented by an offline counterfactual audit.
            with patch.object(nodes, 'record_risk_gate'):
                decision = nodes.route_risk_gate_node(state)
            row.update(flags=decision['route_risk_flags'],
                       review_required=decision['review_required'],
                       time_check_required=decision['time_check_required'],
                       initial_skip=decision['review_skipped'])
        rows.append(row)
    calls = [report['calls'][i] for i in sorted(observed_call_ids)]
    usage_complete = all(c.get('status') == 'completed' and c.get('usage') for c in calls)
    usage = {key: sum((c.get('usage') or {}).get(key, 0) for c in calls)
             for key in ('input_tokens', 'output_tokens', 'total_tokens')}
    decisions_complete = all(r['seed_available'] for r in rows)
    all_require_review = decisions_complete and all(r['review_required'] for r in rows)
    return {
        'schema_version': 1, 'external_calls': False,
        'method': 'current_policy_on_saved_initial_states_and_observed_provider_usage',
        'source_binding': report.get('binding'), 'requests': len(cases), 'rows': rows,
        'observed_audit_calls_by_task': dict(Counter(c['task'] for c in calls)),
        'observed_audit_usage': usage, 'observed_usage_complete': usage_complete,
        'initial_skip_count': sum(r.get('initial_skip', False) for r in rows),
        'initial_review_required_count': sum(r.get('review_required', False) for r in rows),
        'initial_deterministic_hard_clean_count': sum(
            not (set(r.get('flags', [])) - {'preference_review', 'opening_time_unknown', 'long_road_leg', 'user_modification'})
            for r in rows if r['seed_available']),
        'initial_call_reduction_ratio': (sum(r.get('initial_skip', False) for r in rows) / len(rows))
            if rows and decisions_complete else None,
        'initial_gate_opportunity_status': 'measured_decisions_not_executed_adaptive_arm',
        'initial_reviewer_token_saving_ratio': 0.0 if all_require_review and usage_complete else None,
        'initial_reviewer_tokens_avoided': 0 if all_require_review and usage_complete else None,
        'observed_initial_reviewer_tokens': sum((report['calls'][i].get('usage') or {}).get('total_tokens', 0)
                                               for i in initial_reviewer_ids),
        'historical_reviewer_only_reduction_ratio': 0.0 if all_require_review else None,
        'full_current_policy_audit_calls': None,
        'full_current_policy_audit_tokens': None,
        'full_current_policy_call_reduction_ratio': None,
        'full_current_policy_token_reduction_ratio': None,
        'limitations': [
            'Historical experiment has no TimeCheck calls and no adaptive-policy arm.',
            'Intermediate repair states and per-round gate decisions were not recorded.',
            'Initial decisions are replayed with current code, not live production executions.',
            'Tokenizer counts cannot reconstruct unseen completions or provider billing usage.',
        ],
    }


def render(result):
    usage = result['observed_audit_usage']
    lines = ['# Current Risk Gate: Evidence Audit', '',
             f"- Requests: {result['requests']} saved task requests.",
             f"- Current initial gate: review required {result['initial_review_required_count']}; skipped {result['initial_skip_count']}.",
             f"- Observed historical audit calls: {result['observed_audit_calls_by_task']}.",
             f"- Recorded provider audit Token: input {usage['input_tokens']}, output {usage['output_tokens']}, total {usage['total_tokens']}.",
             f"- Initial review skip opportunity: {result['initial_call_reduction_ratio']:.1%}."
                if result['initial_call_reduction_ratio'] is not None else '- Initial skip opportunity: unavailable.',
             '- Full current Reviewer + TimeCheck call reduction: unavailable.',
             '- Full current audit Token saving: unavailable.',
             f"- Initial Reviewer Token saving opportunity: {result['initial_reviewer_token_saving_ratio']}; avoided Token {result['initial_reviewer_tokens_avoided']}.",
             '', '## Why the old 61.8% is not a replacement metric', '',
             'All current initial states requesting preferences enter semantic review. '
             'The historical ledger has no TimeCheck calls and no recorded adaptive arm. '
             'Do not invent TimeCheck tokens, reconstruct missing completions with a tokenizer, '
             'or present an offline routing decision as measured online savings.',
             '', '## Per request', '', '| Request | Review | TimeCheck | Flags |', '|---|---|---|---|']
    for row in result['rows']:
        lines.append(f"| {row['id']} | {row.get('review_required', 'unknown')} | {row.get('time_check_required', 'unknown')} | {', '.join(row.get('flags', []))} |")
    lines += ['', '## Evidence gaps', ''] + ['- ' + x for x in result['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=ROOT/'data/eval/reviewer_ablation_real_v2.json')
    parser.add_argument('--out', type=Path, default=ROOT/'evaluation/adaptive_routing_replay.json')
    parser.add_argument('--markdown-out', type=Path, default=ROOT/'evaluation/adaptive_routing_replay.md')
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    result = audit(json.loads(raw))
    result['ledger_sha256'] = hashlib.sha256(raw).hexdigest()
    result['policy_sha256'] = hashlib.sha256(Path(nodes.__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    args.markdown_out.write_text(render(result), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows',)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
