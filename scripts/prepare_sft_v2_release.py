"""Recheck a versioned SFT release; preserve frozen validation and teacher text."""
import copy
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_robustness import load_assets
from app.planning.nodes import _clock_minutes
from scripts.post_train_runtime import replay_check, quality_faults, capture_messages
from scripts.run_hard_post_train_synthesis import audit_plan
from scripts.export_llamafactory_dataset import export


def explanation_faults(plan):
    """Check only explicit measurable claims; not a general semantic judge."""
    text = plan['reasoning']+' '+plan.get('notes', '')
    spots = [s for d in plan['days'] for s in d['spots']]
    durations = [_clock_minutes(s['end_time'])-_clock_minutes(s['start_time']) for s in spots]
    faults = set()
    prefix = r'(?:(?:均|各|连续|游览|游玩|参观|安排|停留|时长|为|了|达|不少于|不低于|至少|超过|大于|不超过|至多|最多|约|\s))*'
    pattern = r'(?:每个(?:景点)?|每处|各景点)('+prefix+r')(\d+)(?:\s*[—–~至-]\s*(\d+))?\s*分钟'
    for m in re.finditer(pattern, text):
        prefix, number = m.group(1), int(m.group(2))
        if m.group(3):
            wrong = any(not number <= n <= int(m.group(3)) for n in durations)
        elif any(t in prefix for t in ['至少', '不少于', '不低于']):
            wrong = any(n < number for n in durations)
        elif any(t in prefix for t in ['不超过', '至多', '最多']):
            wrong = any(n > number for n in durations)
        elif any(t in prefix for t in ['超过', '大于']):
            wrong = any(n <= number for n in durations)
        elif '约' in prefix:
            continue  # Approximate natural-language claims need review, not exact equality.
        else:
            wrong = any(n != number for n in durations)
        if wrong:
            faults.add('explanation_visit_duration_mismatch')
    for r in plan.get('selection_reasons', []):
        selected = [s for s in spots if s['name'] == r['name']]
        if not selected or r.get('source_ids'):
            faults.add('unsupported_selection_reference')
    if any(t in text for t in ['上一版', '上轮', '本轮修正', '审核反馈', '原方案', '原先集中', '本轮将原先']):
        faults.add('repair_context_in_round_zero_answer')
    return sorted(faults)


def read_rows(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-draft', action='store_true')
    args = parser.parse_args()
    folder = ROOT/'data/post_train'
    new = folder/'new_city_sft_v1'
    out = folder/'release_v2'
    if (out/'TRAINING_STARTED.json').exists():
        raise RuntimeError('Cannot modify a release already used for training')
    cases = {c['id']: c for c in json.loads((new/'cases.json').read_text(encoding='utf-8'))}
    new_rows = read_rows(new/'synthesis.jsonl')
    if len(new_rows) != len(cases) or {r['id'] for r in new_rows} != set(cases):
        raise RuntimeError('Incomplete or duplicate synthesis records')
    maps_new = json.loads((new/'maps.json').read_text(encoding='utf-8'))['routes']
    _, maps_old = load_assets()
    hard = {c['id']: c for c in json.loads((folder/'hard_supplement_cases.json').read_text(encoding='utf-8'))}
    validation = read_rows(folder/'release_v1/validation.ledger.jsonl')
    forbidden_messages = {json.dumps(r['messages'], ensure_ascii=False, sort_keys=True) for r in validation}
    frozen_cases = json.loads((folder/'heldout_evidence_v1/cases.json').read_text(encoding='utf-8'))
    forbidden_cities = {c['state']['destination'] for c in frozen_cases}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path.home()/'.cache/travelmind-qwen-tokenizer'), local_files_only=True)
    old_rows = read_rows(folder/'release_v1/train.ledger.jsonl')
    selected, quarantined, seen, lengths = [], [], set(), []
    for original in old_rows+new_rows:
        row = copy.deepcopy(original)
        faults = []
        if row.get('status') != 'accepted' or row.get('mock') is not False:
            faults.append('not_real_accepted')
        if not row.get('chosen'):
            quarantined.append({'id': row['id'], 'faults': faults+['missing_answer']})
            continue
        if row['id'] in cases:
            faults += audit_plan(row['chosen'], cases[row['id']], maps_new)[0]
        elif row['id'] in hard:
            faults += audit_plan(row['chosen'], hard[row['id']], maps_old)[0]
        else:
            errors, checked = replay_check(row['chosen'], row['state'], maps_old)
            faults += errors+quality_faults(checked)
        faults += explanation_faults(row['chosen'])
        if row['messages'] != capture_messages(row['state']):
            faults.append('production_messages_changed')
        key = json.dumps(row['messages'], ensure_ascii=False, sort_keys=True)
        if key in seen:
            faults.append('duplicate_messages')
        if key in forbidden_messages or row['state']['destination'] in forbidden_cities:
            faults.append('validation_or_test_leakage')
        seen.add(key)
        length = len(tokenizer.apply_chat_template(row['messages']+[{'role': 'assistant',
            'content': json.dumps(row['chosen'], ensure_ascii=False)}], tokenize=True, add_generation_prompt=False))
        if length > 3584:
            faults.append('over_cutoff_no_truncation_allowed')
        if faults:
            quarantined.append({'id': row['id'], 'faults': sorted(set(faults))})
            continue
        row.update(rejected=None, rejected_rule_errors=[],
            data_review='deterministic_full_recheck_plus_agent_stratified_review_not_human_certification')
        selected.append(row)
        lengths.append(length)
    out.mkdir(parents=True, exist_ok=True)
    payloads = {}
    info = {}
    for split, rows in [('train', selected), ('validation', validation)]:
        payloads[f'{split}.ledger.jsonl'] = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
        sft, _ = export(rows)
        if len(sft) != len(rows):
            raise RuntimeError('Export silently changed record count')
        name = f'travelmind_planner_sft_{split}'
        payloads[name+'.json'] = json.dumps(sft, ensure_ascii=False, indent=2)
        info[name] = {'file_name': name+'.json', 'formatting': 'sharegpt',
            'columns': {'messages': 'conversations'}, 'tags': {'role_tag': 'from', 'content_tag': 'value',
                'user_tag': 'human', 'assistant_tag': 'gpt', 'system_tag': 'system'}}
    payloads['dataset_info.json'] = json.dumps(info, indent=2)
    for name, text in payloads.items():
        path = out/name
        if path.exists() and path.read_text(encoding='utf-8') != text and not args.refresh_draft:
            raise RuntimeError('Immutable release changed; use a new version')
        path.write_text(text, encoding='utf-8')
    summary = {'input_old_train': len(old_rows), 'input_new': len(new_rows), 'train': len(selected),
        'new_kept': sum(r['id'] in cases for r in selected), 'validation': len(validation),
        'validation_ledger_unchanged': (out/'validation.ledger.jsonl').read_bytes() == (folder/'release_v1/validation.ledger.jsonl').read_bytes(),
        'quarantined': quarantined, 'max_train_tokens': max(lengths), 'over_cutoff': 0,
        'cities': dict(Counter(r['state']['destination'] for r in selected)),
        'new_categories': dict(Counter(cases[r['id']]['category'] for r in selected if r['id'] in cases)),
        'explanation_audit_scope': 'explicit duration claims, selection references, repair leakage; stratified agent reading, not exhaustive semantic proof',
        'human_certified': False, 'training_authorization': 'user instructed supplement-audit-SFT in order',
        'dpo_auto_start': False,
        'sha256': {name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in payloads}}
    if not summary['validation_ledger_unchanged']:
        raise RuntimeError('Frozen validation changed')
    (out/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
