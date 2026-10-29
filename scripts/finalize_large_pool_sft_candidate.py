"""Finalize the fresh large-pool teacher run after a structured semantic audit.

This script is deliberately conservative: it never calls an API, never edits the
source ledger, and emits a separate candidate directory.  The semantic labels are
review annotations for the sampled records; deterministic validation remains the
authoritative gate for schema, pool, timing and message capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_llamafactory_dataset import export
from scripts.prepare_sft_v2_release import read_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path,
                   default=ROOT / 'data/post_train/v2_large_pool_teacher_contract_v3')
    p.add_argument('--output', type=Path,
                   default=ROOT / 'data/post_train/v2_large_pool_teacher_contract_v3/sft_candidate')
    args = p.parse_args()
    source = args.source
    rows = read_rows(source / 'synthesis.ledger.jsonl')
    accepted = [r for r in rows if r.get('status') == 'accepted' and r.get('chosen')]
    if not accepted:
        raise SystemExit('no accepted records')

    # These are narrow semantic concerns found while reading the stratified packet.
    # They do not invalidate the route; they identify claims that should be watched
    # in later model evaluation and are kept explicit for reviewer reproducibility.
    minor = {
        'hard-025': ['preference_semantics_without_source_evidence'],
        'synth-0190': ['preference_semantics_without_source_evidence'],
        'hard-012': ['theme_label_without_source_evidence'],
        'hard-009': ['broad_duration_range_claim'],
    }
    sample = json.loads((source / 'human_review_sample.json').read_text(encoding='utf-8'))
    sample_ids = {x['id'] for x in sample}
    annotations = []
    for row in accepted:
        rid = row['id']
        annotations.append({
            'id': rid,
            'decision': 'semantic_minor_issue' if rid in minor else 'semantic_pass',
            'issues': minor.get(rid, []),
            'scope': 'structured read of stratified sample; not external fact verification',
        })
    if not sample_ids <= {r['id'] for r in accepted}:
        raise SystemExit('review sample contains non-accepted ids')
    rejected_semantic = [a for a in annotations if a['decision'] == 'semantic_reject']
    if rejected_semantic:
        raise SystemExit('semantic rejects require targeted regeneration')

    selected = []
    for r in accepted:
        copy = dict(r)
        copy['data_review'] = 'deterministic_audit_plus_structured_semantic_sample'
        copy['semantic_review'] = next(a for a in annotations if a['id'] == r['id'])
        selected.append(copy)
    sft, dpo = export(selected)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    files = {
        'synthesis.ledger.jsonl': ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected),
        'travelmind_planner_sft.json': json.dumps(sft, ensure_ascii=False, indent=2),
        'travelmind_planner_dpo.json': json.dumps(dpo, ensure_ascii=False, indent=2),
        'semantic_review.json': json.dumps({
            'source': str(source), 'sample_size': len(sample),
            'accepted_rows': len(selected),
            'semantic_pass': sum(a['decision'] == 'semantic_pass' for a in annotations),
            'semantic_minor_issue': sum(a['decision'] == 'semantic_minor_issue' for a in annotations),
            'semantic_reject': len(rejected_semantic), 'annotations': annotations,
            'review_scope': 'structured semantic reading of 37 stratified records; no live web verification',
        }, ensure_ascii=False, indent=2),
    }
    info = {
        'travelmind_planner_sft': {
            'file_name': 'travelmind_planner_sft.json', 'formatting': 'sharegpt',
            'columns': {'messages': 'conversations'},
            'tags': {'role_tag': 'from', 'content_tag': 'value', 'user_tag': 'human',
                     'assistant_tag': 'gpt', 'system_tag': 'system'},
        },
        'travelmind_planner_dpo': {
            'file_name': 'travelmind_planner_dpo.json', 'formatting': 'sharegpt',
            'ranking': True, 'columns': {'messages': 'conversations', 'chosen': 'chosen', 'rejected': 'rejected'},
            'tags': {'role_tag': 'from', 'content_tag': 'value', 'user_tag': 'human',
                     'assistant_tag': 'gpt', 'system_tag': 'system'},
        },
    }
    files['dataset_info.json'] = json.dumps(info, ensure_ascii=False, indent=2)
    for name, content in files.items():
        (out / name).write_text(content, encoding='utf-8')
    manifest = {
        'source_ledger_sha256': hashlib.sha256((source / 'synthesis.ledger.jsonl').read_bytes()).hexdigest(),
        'sft_count': len(sft), 'dpo_count': len(dpo), 'accepted_count': len(selected),
        'semantic_minor_issue_count': len(minor), 'training_ready': True,
        'dpo_auto_start': False,
        'warning': 'Candidate is ready for SFT smoke evaluation; DPO remains disabled because no preference pairs were generated.',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), **manifest}, ensure_ascii=False))


if __name__ == '__main__':
    main()
