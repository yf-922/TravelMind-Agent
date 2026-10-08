"""Export editable annotations; validate/import actual human review; freeze test."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.rag_protocol import validate_annotations, validate_corpus, freeze_payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('export', 'import', 'import-corpus', 'freeze'))
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cases = json.loads(a.cases.read_text(encoding='utf-8'))
    rows = json.loads(a.corpus.read_text(encoding='utf-8'))
    if a.operation == 'export':
        # Never auto-promote machine drafts to human-reviewed labels.
        payload = {'instructions': 'Edit query, type, facts and relevance. Sign annotator/reviewed_at/change_log per case. Do not mark drafts reviewed without reading sources.',
                   'source_instructions': 'Review each source validity and advice/fact distinction. Sign review_status=human_reviewed, annotator, reviewed_at and change_log. import-corpus exports evidence separately.',
                   'cases': cases, 'evidence': rows}
    elif a.operation == 'import-corpus':
        if not isinstance(cases, dict) or 'evidence' not in cases:
            p.error('import-corpus requires an edited review packet with evidence')
        payload = cases['evidence']
        errors = validate_corpus(payload, True, True)
        if errors:
            p.error('; '.join(errors))
    else:
        if isinstance(cases, dict):
            cases = cases['cases']
        errors = validate_annotations(cases, rows, require_review=True)
        if errors:
            p.error('; '.join(errors))
        if a.operation == 'freeze':
            if not a.config:
                p.error('--config required for freeze')
            try:
                payload = freeze_payload(cases, rows, json.loads(a.config.read_text(encoding='utf-8')))
            except ValueError as exc:
                p.error(str(exc))
        else:
            payload = cases
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
