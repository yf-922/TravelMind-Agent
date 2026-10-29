"""Export actual round-zero messages without reconstructing prompt text."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.planning.schemas import TravelRoute


def export(rows):
    sft, dpo, seen = [], [], set()
    for r in rows:
        if r.get('status') != 'accepted' or not r.get('chosen'):
            continue
        if r.get('mock') is None:
            raise ValueError('unversioned legacy row: cannot establish real/mock provenance')
        TravelRoute.model_validate(r['chosen'])
        key = json.dumps(r['messages'], sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        conv = [{'from': {'system': 'system', 'user': 'human', 'assistant': 'gpt'}[m['role']], 'value': m['content']} for m in r['messages']]
        chosen = {'from': 'gpt', 'value': json.dumps(r['chosen'], ensure_ascii=False)}
        sft.append({'conversations': conv + [chosen]})
        if r.get('rejected') and r.get('rejected_rule_errors'):
            TravelRoute.model_validate(r['rejected'])
            dpo.append({'conversations': conv, 'chosen': chosen,
                'rejected': {'from': 'gpt', 'value': json.dumps(r['rejected'], ensure_ascii=False)}})
    return sft, dpo


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=ROOT/'data/post_train/curated_synthesis_ledger.jsonl')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    rows = [json.loads(l) for l in a.input.read_text(encoding='utf-8').splitlines() if l.strip()]
    if any(r.get('mock') is not False for r in rows) and not a.dry_run:
        raise ValueError('mock/unversioned data cannot enter a real export; use --dry-run for fixtures')
    if a.output_dir is None:
        a.output_dir = ROOT/'data/post_train'/('mock_export' if a.dry_run else '')
    sft, dpo = export(rows)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [('travelmind_planner_sft', sft), ('travelmind_planner_dpo', dpo)]:
        (a.output_dir/f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    info = {name: {'file_name': f'{name}.json', 'formatting': 'sharegpt',
        'columns': {'messages': 'conversations', **({'chosen': 'chosen', 'rejected': 'rejected'} if ranking else {})},
        'tags': {'role_tag': 'from', 'content_tag': 'value', 'user_tag': 'human', 'assistant_tag': 'gpt', 'system_tag': 'system'},
        **({'ranking': True} if ranking else {})} for name, ranking in [('travelmind_planner_sft', False), ('travelmind_planner_dpo', True)]}
    (a.output_dir/'dataset_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps({'sft': len(sft), 'dpo': len(dpo), 'mock_rows': sum(bool(r.get('mock')) for r in rows)}))


if __name__ == '__main__':
    main()
