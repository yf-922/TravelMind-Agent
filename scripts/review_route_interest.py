"""Export/import signed route-selection labels and validate official snapshots."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.route_interest import validate_assets
from app.evaluation.rag_protocol import fingerprint


def validate_snapshots(directory, facts):
    errors = []
    for f in facts:
        if not f['source'].startswith('interest_'): continue
        path = directory/'snapshots'/(f['source']+'.txt')
        if not path.exists(): errors.append(f['source']+': missing official snapshot'); continue
        text = path.read_text(encoding='utf-8')
        if hashlib.sha256(text.encode()).hexdigest() != f['snapshot_hash']: errors.append(f['source']+': changed snapshot')
        if f['text'] not in text: errors.append(f['source']+': quote absent from snapshot')
        if hashlib.sha256(f['text'].encode()).hexdigest() != f['content_hash']: errors.append(f['source']+': changed fact')
    return errors


def validate_manifest(directory, cases, facts):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    errors = []
    for key, value in [('case_fingerprint', cases), ('fact_fingerprint', facts)]:
        if manifest.get(key) != fingerprint(value): errors.append('asset manifest mismatch: '+key)
    if (directory/'maps.json').exists():
        maps = json.loads((directory/'maps.json').read_text(encoding='utf-8'))
        if manifest.get('map_fingerprint') != fingerprint(maps): errors.append('asset manifest mismatch: maps')
    return errors


def import_reviews(original, edited, *, source=False):
    key = 'source' if source else 'id'
    if {r[key] for r in original} != {r[key] for r in edited} or len(original) != len(edited): raise ValueError('review IDs changed or duplicated')
    allowed = {'review_status', 'annotator', 'reviewed_at', 'change_log'} if source else {
        'annotation_status', 'annotator', 'reviewed_at', 'change_log', 'acceptable_pois', 'forbidden_pois', 'required_evidence', 'reference_note', 'answerable'}
    old = {r[key]: r for r in original}
    for r in edited:
        if {k: v for k, v in r.items() if k not in allowed} != {k: v for k, v in old[r[key]].items() if k not in allowed}:
            raise ValueError('query, pool, provenance or split cannot be changed during review')
        from datetime import datetime
        status = r.get('review_status' if source else 'annotation_status')
        if status == 'human_reviewed':
            if not all(r.get(k) for k in ('annotator', 'reviewed_at', 'change_log')): raise ValueError('signed human review required')
            datetime.fromisoformat(r['reviewed_at'])
    return edited


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('validate', 'export', 'import'))
    p.add_argument('--assets', type=Path, default=ROOT/'knowledge/route_interest_v1')
    p.add_argument('--file', type=Path)
    a = p.parse_args()
    cases = json.loads((a.assets/'cases.json').read_text(encoding='utf-8'))
    facts = json.loads((a.assets/'facts.json').read_text(encoding='utf-8'))
    if a.operation == 'export':
        if not a.file: p.error('--file required')
        a.file.parent.mkdir(parents=True, exist_ok=True)
        a.file.write_text(json.dumps({'cases': cases, 'facts': facts}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8'); return
    if a.operation == 'import':
        if not a.file: p.error('--file required')
        edited = json.loads(a.file.read_text(encoding='utf-8'))
        cases = import_reviews(cases, edited['cases']); facts = import_reviews(facts, edited['facts'], source=True)
    errors = validate_assets(cases, facts)+validate_snapshots(a.assets, facts)
    if a.operation != 'import': errors += validate_manifest(a.assets, cases, facts)
    if errors: p.error('; '.join(errors))
    if a.operation == 'import':
        for name, rows in (('cases', cases), ('facts', facts)):
            (a.assets/(name+'.json')).write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        manifest = json.loads((a.assets/'manifest.json').read_text(encoding='utf-8'))
        manifest.update(case_fingerprint=fingerprint(cases), fact_fingerprint=fingerprint(facts))
        (a.assets/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('30-case assets and snapshots valid; formal gates: '+str(len(validate_assets(cases, facts, formal=True))))


if __name__ == '__main__': main()
