"""Validate provenance, quotas and annotations without paid API calls."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.rag_protocol import validate_corpus,validate_annotations,fingerprint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--cases',type=Path)
    p.add_argument('--require-quota',action='store_true')
    p.add_argument('--require-review',action='store_true')
    a=p.parse_args()
    rows=json.loads(a.corpus.read_text(encoding='utf-8'))
    errors=validate_corpus(rows,a.require_quota)
    if a.cases:
        cases=json.loads(a.cases.read_text(encoding='utf-8'))
        errors+=validate_annotations(cases,rows,a.require_review,a.require_quota)
    print(json.dumps({'chunks':len(rows),'cities':dict(Counter(r['city'] for r in rows)),
        'fingerprint':fingerprint(rows),'errors':errors},ensure_ascii=True,indent=2))
    return bool(errors)


if __name__=='__main__':raise SystemExit(main())
