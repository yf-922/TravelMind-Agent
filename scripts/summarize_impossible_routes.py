"""Regrade saved impossible requests against immutable original constraints."""
import argparse
import json
from pathlib import Path


def analyze(reports):
    cases, rows = {}, {}
    for report in reports:
        cases.update({c['id']: c for c in report['cases'] if c['expected'] == 'reject'})
        for row in report['rows']:
            if row['expected'] == 'reject':
                key = (row['id'], row['arm'])
                if key in rows: raise ValueError('duplicate negative result; no last-write wins')
                rows[key] = row
    summary, details = {}, []
    for (case_id, arm), row in rows.items():
        case = cases[case_id]
        out = row.get('outcome', {})
        plan = out.get('final_plan', {})
        spots = [s for d in plan.get('days', []) for s in d.get('timeline', []) if s.get('type') == 'attraction']
        original_failures = []
        if not spots: original_failures.append('required_nonempty_itinerary')
        pool = {p['name']: p for p in case['state']['pois']}
        if any(s['name'] not in pool for s in spots): original_failures.append('outside_pool')
        if '只能安排室内' in case['state']['query'] and any(pool.get(s['name'], {}).get('indoor') is not True for s in spots):
            original_failures.append('indoor_only')
        if case['state'].get('modification_notes') == '改为10:00到11:00游玩':
            if any(not ('10:00' <= s.get('start_time', '') < s.get('end_time', '') <= '11:00') for s in spots):
                original_failures.append('original_required_window')
        messages = plan.get('route_issues', [])
        # Discovery heuristic only. Keep original text for human confirmation.
        reason_hint = any(any(word in str(m) for word in ('无法', '不能', '无可用', '没有可', '没有已核验', '冲突', '不可满足')) for m in messages)
        record = {'id': case_id, 'arm': arm, 'error': row.get('error'),
                  'approved': out.get('final_approved'), 'original_constraint_failures': original_failures,
                  'unsafe_release': bool(out.get('final_approved')),
                  'rejected_with_reason_hint': not row.get('error') and out.get('final_approved') is False and reason_hint,
                  'message_check': 'heuristic_pending_human_review', 'user_messages': messages}
        details.append(record)
        s = summary.setdefault(arm, {'planned': 10, 'recorded': 0, 'unsafe_release': 0, 'errors': 0, 'rejected_with_reason_hint': 0})
        s['recorded'] += 1
        for key in ('unsafe_release', 'rejected_with_reason_hint'): s[key] += record[key]
        s['errors'] += bool(record['error'])
    return {'summary': summary, 'results': details, 'interpretation': 'No rerun; scored original fixed-pool constraints. Any approval of a predeclared impossible request is unsafe. Message heuristics require human confirmation.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = analyze([json.loads(path.read_text(encoding='utf-8')) for path in a.reports])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['summary']))
