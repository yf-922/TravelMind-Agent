"""Persist a privacy-safe partial/complete pilot summary; no provider calls."""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_planner_rag import summary


def paired_report(report):
    grouped = {}
    for row in report['results']:
        grouped.setdefault((row['case_id'], row['repetition']), {})[row['arm']] = row
    pairs = [v for v in grouped.values() if set(v) == {'with_rag', 'without_rag'} and
             not any(r.get('error') for r in v.values())]
    arms = {}
    for arm in ('without_rag', 'with_rag'):
        rows = [pair[arm] for pair in pairs]
        arms[arm] = {'completed_pairs': len(pairs),
                     'mean_latency_ms': statistics.mean(r['latency_ms'] for r in rows) if rows else None,
                     'input_tokens': sum(r.get('usage', {}).get('input_tokens', 0) for r in rows),
                     'output_tokens': sum(r.get('usage', {}).get('output_tokens', 0) for r in rows),
                     'basic_rule_pass': sum(r['basic_rules']['passed'] for r in rows)}
    return {'status': report['status'], 'stop_reason': report.get('stop_reason'),
            'fingerprint': report['fingerprint'], 'model': report['model'], 'transport': report.get('transport'),
            'preflight': report['preflight'], 'scope': report['scope'], 'budget_proxy': report.get('budget'),
            'gateway_quota': report.get('gateway_quota'),
            'all_recorded_summary': summary(report['results'], report['preflight']['requests'] * report['preflight']['repetitions']),
            'completed_pairs_only': arms,
            'human_review_complete': False,
            'trials': [{k: r[k] for k in ('case_id', 'arm', 'repetition', 'latency_ms', 'usage',
                        'generated', 'basic_rules', 'human_review_status') if k in r} for r in report['results']],
            'limitations': ['Incomplete pairs excluded only from paired comparison, retained in all-recorded summary.',
                            'Unrun trials are incomplete execution, not model failures.',
                            'No human-confirmed semantic scores; controlled single-candidate cases, not production quality.',
                            'Proxy cost does not include gateway group multipliers; quota measurement is key-wide.']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = paired_report(json.loads(a.input.read_text(encoding='utf-8')))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['completed_pairs_only'], ensure_ascii=False))


if __name__ == '__main__': main()
