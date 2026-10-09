"""Record an abandoned pilot call without guessing usage or replaying it."""
import argparse
import json
import os
from pathlib import Path


def reconcile(ledger_path, report_path):
    lock = ledger_path.with_suffix('.lock')
    if os.name == 'nt' and lock.exists():
        import subprocess
        pid = int(lock.read_text().strip())
        output = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                                capture_output=True, text=True, check=True).stdout
        if str(pid) in output:
            raise RuntimeError('owner process still running; cannot reconcile')
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    report = json.loads(report_path.read_text(encoding='utf-8'))
    pending = [c for c in report['calls'] if c['status'] == 'in_flight']
    if not ledger.get('in_flight') or not pending:
        raise ValueError('no matching interrupted call')
    for call in pending:
        call.update(status='interrupted_usage_unknown', error='process terminated before settlement')
    audit = {'report': str(report_path), 'unknown_usage_calls': len(pending),
             'calls': report['calls'], 'replayed': False,
             'cost_status': 'unpriced', 'usage_status': 'unknown_not_estimated'}
    ledger.setdefault('interruption_audits', []).append(audit)
    ledger['budget']['unknown_usage_calls'] += len(pending)
    ledger['in_flight'] = False
    report.update(status='interrupted', stop_reason='process terminated; incomplete trial not replayed')
    ledger_path.with_suffix('.interrupted-backup.json').write_text(ledger_path.read_text(encoding='utf-8'), encoding='utf-8')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if lock.exists(): lock.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    reconcile(a.ledger, a.report)
