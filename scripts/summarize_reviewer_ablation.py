"""Report paired benefits/regressions and shared-seed logical resource costs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from scripts.evaluate_reviewer_ablation import ARMS


def analyze(report):
    by_key = {(row['id'], row['arm']): row for row in report['rows']}
    paired = []
    resources = {}
    for case in report['cases']:
        a = by_key.get((case['id'], 'rules'), {})
        b = by_key.get((case['id'], 'rules_reviewer'), {})
        if not a or not b: continue
        ga, gb = a.get('grade', {}), b.get('grade', {})
        if any(ga.get(k) != gb.get(k) for k in ('task_success', 'robustness_ok', 'unsafe_release', 'safe_explained_rejection')) or a.get('error') != b.get('error'):
            paired.append({'id': case['id'], 'rules': ga, 'reviewer': gb,
                           'rules_error': a.get('error'), 'reviewer_error': b.get('error'),
                           'interpretation': 'execution_failure_difference_not_pure_semantic_gain' if a.get('error') or b.get('error') else 'paired_output_difference'})
    for arm in ARMS:
        rows = [row for row in report['rows'] if row['arm'] == arm]
        indices = [i for row in rows for i in row.get('paid_call_indices', []) + row.get('shared_call_indices', [])]
        tokens = {k: sum((report['calls'][i].get('usage') or {}).get(k, 0) for i in indices)
                  for k in ('input_tokens', 'output_tokens', 'total_tokens')}
        elapsed = [(row['tail_ms'] + row['shared_seed_ms']) / 1000 for row in rows if row.get('shared_seed_ms') is not None]
        resources[arm] = {'recorded_logical_tokens_shared_seed_included': tokens,
            'mean_seconds_complete_seed_rows': statistics.mean(elapsed) if elapsed else None,
            'timed_rows': len(elapsed), 'repairs': sum(row.get('repairs', 0) for row in rows),
            'logical_call_indices': len(indices)}
    return {'binding': report['binding'], 'model': report['model'], 'summary': report['summary'],
            'paired_differences_rules_vs_reviewer': paired, 'resources': resources,
            'paid_attempts': len(report['calls']), 'paid_statuses': Counter(c['status'] for c in report['calls']),
            'recorded_paid_tokens': {k: sum((c.get('usage') or {}).get(k, 0) for c in report['calls'])
                                     for k in ('input_tokens', 'output_tokens', 'total_tokens')},
            'cost_status': 'unpriced', 'usage_complete': all(c.get('usage') for c in report['calls']),
            'notes': ['Author draft labels; fixed candidates and saved roads',
                      'Logical arm costs include the shared seed; do not sum them as paid cost',
                      'Failed shared seed has no recovered elapsed measurement',
                      'No LLM TimeCheck in any arm; audit-tail experiment, not full production graph',
                      'One repetition per case; no stability/online-accuracy claim']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text(encoding='utf-8')))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__': main()
