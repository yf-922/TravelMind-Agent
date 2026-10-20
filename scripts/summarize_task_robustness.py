"""Summarize original real trials and flag rubric defects without rewriting them."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def summarize(report):
    rows = report['rows']
    calls = report['calls']
    cases = {c['id']: c for c in report['cases']}
    diagnostics = []
    for case in cases.values():
        rubric = case['rubric']
        if (rubric.get('min_match', 0) and rubric.get('acceptable') and
                set(rubric['acceptable']) <= set(rubric.get('excluded', []))):
            diagnostics.append({'id': case['id'], 'issue': 'all_acceptable_pois_also_excluded',
                                'action': 'do_not_treat_as_feasible_task_failure'})
    groups = {}
    for group in ('task', 'map', 'weather', 'modification'):
        selected = [r for r in rows if r['suite'] == 'task'] if group == 'task' else [r for r in rows if r['category'] == group and r['suite'] == 'robustness']
        elapsed = sorted(r['elapsed_ms'] / 1000 for r in selected if r.get('elapsed_ms') is not None)
        groups[group] = {'recorded': len(selected),
            'task_success': sum(r.get('grade', {}).get('task_success', False) for r in selected),
            'safe_rejections': sum(r.get('grade', {}).get('safe_explained_rejection', False) for r in selected),
            'raw_unsafe_releases': sum(r.get('grade', {}).get('unsafe_release', False) for r in selected),
            'execution_errors': Counter(r['error'] for r in selected if r.get('error')),
            'latency_seconds': {'mean': statistics.mean(elapsed), 'p50': statistics.median(elapsed),
                                'p95_nearest_rank': elapsed[max(0, __import__('math').ceil(.95 * len(elapsed)) - 1)],
                                'timed_rows': len(elapsed)} if elapsed else None}
    map_rows = [r for r in rows if r.get('category') == 'map']
    unverified = [r['id'] for r in map_rows if 'road_unverified' in r.get('grade', {}).get('hard_violations', []) and r.get('grade', {}).get('approved')]
    hidden_count = [r['id'] for r in map_rows if r.get('grade', {}).get('hard_violations') == ['required_count']]
    for case_id in hidden_count:
        diagnostics.append({'id': case_id, 'issue': 'min_two_rubric_but_request_only_says_max_two',
                            'action': 'not_evidence_of_road_failure_or_user_constraint_violation'})
    usage = {key: sum((c.get('usage') or {}).get(key, 0) for c in calls)
             for key in ('input_tokens', 'output_tokens', 'total_tokens')}
    return {'source_binding': report['binding'], 'model': report['model'],
            'raw_summary': report['summary'], 'groups': groups,
            'call_count': len(calls), 'call_statuses': Counter(c['status'] for c in calls),
            'recorded_usage': usage, 'usage_complete': all(bool(c.get('usage')) for c in calls),
            'cost_status': 'unpriced', 'confirmed_road_unverified_approvals': unverified,
            'rubric_diagnostics': diagnostics,
            'limits': ['Fixed real candidate pools and saved road replay; not live search or online SLA',
                       'Author-draft soft labels, not independently confirmed human quality scores',
                       'Interrupted trial stays a failure; no paid replay',
                       'Original results preserved; rubric diagnostics do not relabel them as successes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text(encoding='utf-8')))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__': main()
