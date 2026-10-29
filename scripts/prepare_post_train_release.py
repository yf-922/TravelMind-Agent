"""Recheck candidates, quarantine repair-context leakage, split by candidate pool.

Agent inspection is explicitly distinct from human acceptance. Exports remain
small pilot datasets, not evidence of independent generalization.
"""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_robustness import load_assets
from scripts.post_train_runtime import replay_check, quality_faults, capture_messages
from scripts.run_hard_post_train_synthesis import audit_plan
from scripts.export_llamafactory_dataset import export


def group_key(row):
    state = row['state']
    return json.dumps([state['destination'], sorted(p['name'] for p in state['pois'])], ensure_ascii=False)


def main():
    folder = ROOT/'data/post_train'
    output = folder/'release_v1'
    output.mkdir(parents=True, exist_ok=True)
    base = [json.loads(l) for l in (folder/'quality_selected_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    hard = [json.loads(l) for l in (folder/'hard_synthesis_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    hard_cases = {r['id']: r for r in json.loads((folder/'hard_supplement_cases.json').read_text(encoding='utf-8'))}
    _, maps = load_assets()
    rows, quarantined, seen = [], [], set()
    for original in base+hard:
        if original.get('status') != 'accepted':
            continue
        row = copy.deepcopy(original)
        faults, state = replay_check(row['chosen'], row['state'], maps)
        faults += quality_faults(state)
        if row['id'].startswith('hard-'):
            faults += audit_plan(row['chosen'], hard_cases[row['id']], maps)[0]
        if row['messages'] != capture_messages(row['state']):
            faults.append('captured_messages_changed')
        selected = {s['name'] for d in row['chosen']['days'] for s in d['spots']}
        if any(r['name'] not in selected or r.get('source_ids') for r in row['chosen'].get('selection_reasons', [])):
            faults.append('unsupported_selection_reference')
        explanation = row['chosen']['reasoning'] + row['chosen'].get('notes', '')
        # Do not supervise first-round inputs with explicit references to unseen repairs.
        if any(token in explanation for token in ['上一版', '上轮', '本轮修正', '审核反馈', '原方案']):
            faults.append('repair_context_in_round_zero_answer')
        if row['id'] in {'synth-0097', 'synth-0148'}:
            faults.append('agent_review_unseen_previous_plan_reference')
        key = json.dumps(row['messages'], ensure_ascii=False, sort_keys=True)
        if key in seen:
            faults.append('duplicate_messages')
        seen.add(key)
        if faults:
            quarantined.append({'id': row['id'], 'faults': sorted(set(faults))})
            continue
        row['data_review'] = 'automatic_recheck_and_agent_sample_review_not_human_certified'
        # Synthetic negatives with copied explanations are a confound. Retain
        # them in source ledgers, but only natural preference pairs enter pilot DPO.
        if row.get('preference_source') != 'natural_repair':
            row.update(rejected=None, rejected_rule_errors=[])
        rows.append(row)
    groups = sorted({group_key(r) for r in rows}, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    validation_groups = set(groups[:max(1, len(groups)//5)])
    splits = {'train': [], 'validation': []}
    for row in rows:
        splits['validation' if group_key(row) in validation_groups else 'train'].append(row)
    info = {}
    for split, members in splits.items():
        (output/f'{split}.ledger.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in members), encoding='utf-8')
        sft, dpo = export(members)
        for stage, data in [('sft', sft), ('dpo', dpo)]:
            name = f'travelmind_planner_{stage}_{split}'
            (output/f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            info[name] = {'file_name': f'{name}.json', 'formatting': 'sharegpt',
                'columns': {'messages': 'conversations', **({'chosen': 'chosen', 'rejected': 'rejected'} if stage == 'dpo' else {})},
                'tags': {'role_tag': 'from', 'content_tag': 'value', 'user_tag': 'human', 'assistant_tag': 'gpt', 'system_tag': 'system'},
                **({'ranking': True} if stage == 'dpo' else {})}
    (output/'dataset_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    summary = {'candidates': len(base)+sum(r.get('status') == 'accepted' for r in hard),
        'passed': len(rows), 'quarantined': quarantined, 'split_group': 'city_and_exact_candidate_name_set',
        'train': len(splits['train']), 'validation': len(splits['validation']),
        'group_overlap': len({group_key(r) for r in splits['train']} & {group_key(r) for r in splits['validation']}),
        'train_natural_dpo': sum(bool(r.get('rejected')) for r in splits['train']),
        'validation_natural_dpo': sum(bool(r.get('rejected')) for r in splits['validation']),
        'scope': 'smoke/pilot only; same-city validation; independent test assets still incomplete',
        'synthetic_dpo_policy': 'excluded: explanation-copy shortcut; source records preserved',
        'human_review': 'not_performed', 'full_training_release': False}
    (output/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    # Balanced compact packet for agent inspection. This never marks a human pass.
    sample = []
    buckets = {}
    for row in rows:
        buckets.setdefault((row['state']['destination'], row.get('preference_source')), []).append(row)
    while buckets and len(sample) < 30:
        for key in list(buckets):
            row = buckets[key].pop(0)
            sample.append({'id': row['id'], 'query': row['state']['query'], 'reasoning': row['chosen']['reasoning'],
                'route': row['chosen']['days'], 'notes': row['chosen'].get('notes'), 'kind': row.get('preference_source')})
            if not buckets[key]:
                del buckets[key]
            if len(sample) == 30:
                break
    (output/'agent_review_packet.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf-8')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.json') if p.name != 'manifest.json'}
    (output/'manifest.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == '__main__':
    main()
