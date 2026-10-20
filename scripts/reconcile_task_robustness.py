"""Record an interrupted trial without repeating a possibly paid request."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_task_robustness import summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    pending = [c for c in report['calls'] if c['status'] == 'in_flight']
    if not pending:
        print('No unsettled calls.'); return
    backup = args.report.with_suffix('.interrupted-backup.json')
    if backup.exists(): raise RuntimeError('backup exists; inspect before reconciliation')
    shutil.copy2(args.report, backup)
    for call in pending:
        call.update(status='interrupted_unknown_usage', error='InterruptedExecution',
                    reconciliation='No result recovered; no automatic paid replay; usage unknown')
        if not any(r['id'] == call['case'] for r in report['rows']):
            case = next(c for c in report['cases'] if c['id'] == call['case'])
            report['rows'].append({'id': case['id'], 'suite': case['suite'],
                'category': case['category'], 'fault': case.get('fault'),
                'error': 'InterruptedExecution', 'elapsed_ms': None,
                'reconciliation': 'Counted as execution failure, not retried'})
    report['status'] = 'reconciled_pending_resume'
    report['summary'] = summary(report['cases'], report['rows'], True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Reconciled', len(pending), 'unsettled calls; backup:', backup)


if __name__ == '__main__': main()
