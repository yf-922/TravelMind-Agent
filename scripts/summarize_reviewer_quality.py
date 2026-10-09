"""Report quality outcomes separately from production hard-rule validity."""
import argparse
import json
from pathlib import Path


def analyze(report):
    cases = {c['id']: c for c in report['cases']}
    groups = {}
    for row in report['rows']:
        category = cases[row['id']]['quality']['category']
        group = groups.setdefault(category, {})
        stats = group.setdefault(row['arm'], {'planned': sum(c['quality']['category'] == category for c in cases.values()),
            'recorded': 0, 'quality_passed': 0, 'quality_and_approved': 0, 'errors': 0,
            'bad_drafts_repaired': 0, 'clean_controls_degraded': 0, 'unsafe_quality_approvals': 0})
        stats['recorded'] += 1
        out = row.get('outcome', {})
        good = out.get('quality', {}).get('passed', False)
        clean = cases[row['id']]['quality']['expected_clean']
        stats['quality_passed'] += good
        stats['quality_and_approved'] += good and bool(out.get('final_approved'))
        stats['errors'] += bool(row.get('error'))
        stats['bad_drafts_repaired'] += not clean and good
        stats['clean_controls_degraded'] += clean and not good
        stats['unsafe_quality_approvals'] += not good and bool(out.get('final_approved'))
    return {'execution_status': report['status'], 'by_category': groups,
            'actual_calls': len(report['calls']),
            'actual_tokens_known_only': {key: sum((c.get('usage') or {}).get(key, 0) for c in report['calls'])
                                         for key in ('input_tokens', 'output_tokens')},
            'unknown_usage_calls': sum(not c.get('usage') for c in report['calls']),
            'cost_status': 'unpriced',
            'limits': 'Controlled injected drafts, not natural Planner failure prevalence. No independent Judge or human signed labels. Unexecuted and failed trials cannot be counted as repaired.',
            'rows': report['rows']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = analyze(json.loads(a.report.read_text(encoding='utf-8')))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=True))
