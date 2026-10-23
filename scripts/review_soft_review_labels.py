"""Export human review forms and import explicit, case-bound confirmations."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.rag_protocol import fingerprint


def export_form(report):
    return {'report_binding': report['binding'], 'reviewer': '', 'reviewed_at': '',
            'instruction': 'Review request, initial defect, final route and refusal. Confirm only after human review.',
            'decisions': [{'id': c['id'], 'case_fingerprint': fingerprint(c), 'confirm': False,
                           'comment': '', 'query': c['state']['query'],
                           'rubric': c['semantic_rubric'], 'expected': c['expected']}
                          for c in report['cases']]}


def apply_confirmation(report, form):
    if report['status'] != 'complete_pending_human_review':
        raise ValueError('all experiment outcomes must be recorded before review')
    if form.get('report_binding') != report['binding']:
        raise ValueError('review is bound to a different report')
    if not str(form.get('reviewer', '')).strip():
        raise ValueError('human reviewer identity required')
    stamp = datetime.fromisoformat(form.get('reviewed_at', ''))
    if stamp.tzinfo is None:
        raise ValueError('review timestamp requires timezone')
    decisions = {d['id']: d for d in form.get('decisions', [])}
    if len(decisions) != len(report['cases']) or len(form['decisions']) != len(decisions):
        raise ValueError('all cases require exactly one decision')
    for case in report['cases']:
        decision = decisions.get(case['id'], {})
        if decision.get('confirm') is not True or decision.get('case_fingerprint') != fingerprint(case):
            raise ValueError('unconfirmed or changed case; revise assets in a new experiment version')
    # Validate the entire document before mutating any label.
    for case in report['cases']:
        case['label_status'] = 'verified_v2'
        case['human_review'] = {'reviewer': form['reviewer'], 'reviewed_at': form['reviewed_at'],
                                'comment': decisions[case['id']].get('comment', ''),
                                'original_case_fingerprint': decisions[case['id']]['case_fingerprint']}
    report['status'] = 'complete_human_verified'
    report['label_status'] = 'verified_v2'
    report['verification_scope'] = 'new controlled challenges only; historical rows retain original provenance'
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('export', 'import'))
    p.add_argument('--ledger', type=Path, default=ROOT/'data/eval/soft_review_v3.json')
    p.add_argument('--form', type=Path, default=ROOT/'data/eval/soft_review_v3_human_review.json')
    args = p.parse_args()
    report = json.loads(args.ledger.read_text(encoding='utf-8'))
    if args.mode == 'export':
        if args.form.exists():
            raise ValueError('review form already exists; choose a new path')
        args.form.parent.mkdir(parents=True, exist_ok=True)
        args.form.write_text(json.dumps(export_form(report), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    else:
        form = json.loads(args.form.read_text(encoding='utf-8'))
        result = apply_confirmation(report, form)
        # Preserve the raw paid ledger and write a separately bound verified copy.
        verified = args.ledger.with_name(args.ledger.stem+'_verified.json')
        if verified.exists():
            raise ValueError('verified copy already exists')
        verified.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(args.form if args.mode == 'export' else verified)


if __name__ == '__main__':
    main()
