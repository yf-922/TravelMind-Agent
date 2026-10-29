"""Summarize paired diagnostic interventions without rewriting raw scores."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_robustness import load_assets
from app.planning.nodes import _clock_minutes
from scripts.evaluate_sft_validation import score
from scripts.diagnose_sft_transfer import OUT, rows


def main():
    results = rows(OUT/'results.jsonl')
    inputs = {r['id']:r for r in json.loads((OUT/'cases.json').read_text(encoding='utf-8'))}
    _, maps = load_assets()
    hard = {c['id']:c for c in json.loads((ROOT/'data/post_train/hard_supplement_cases.json').read_text(encoding='utf-8'))}
    baselines = {'base': ROOT/'data/post_train/sft_validation_v1/base.jsonl',
        'v1': ROOT/'data/post_train/sft_validation_v1/sft.jsonl',
        'v2': ROOT/'data/post_train/sft_validation_v2/sft.jsonl'}
    diagnostic, pairs, repeats = [], [], []
    expected = {(tag,arm,k) for tag in baselines for arm in ['original','rules','metadata'] for k in inputs}
    keys = [(r['tag'],r['arm'],r['id']) for r in results]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError('Incomplete or duplicate diagnostic recordings')
    for tag in baselines:
        original = {r['id']:r for r in results if r['tag']==tag and r['arm']=='original'}
        previous = {r['id']:r for r in rows(baselines[tag])}
        repeats.append({'tag':tag, 'n':len(original),
            'changed_pass_status':[k for k in original if original[k]['passed'] != previous[k]['passed']],
            'same_raw_content':sum(original[k].get('response',{}).get('choices',[{}])[0].get('message',{}).get('content') ==
                previous[k].get('response',{}).get('choices',[{}])[0].get('message',{}).get('content') for k in original)})
        for arm in ['original','rules','metadata']:
            group = [r for r in results if r['tag']==tag and r['arm']==arm]
            normalized, normalization_errors = [], []
            for r in group:
                if not r.get('plan'):
                    normalized.append(False)
                    continue
                plan = copy.deepcopy(r['plan'])
                plan['meal_slots'] = []
                try:
                    for day in plan['days']:
                        for s in day['spots']:
                            minute = _clock_minutes(s['start_time'])
                            s['period'] = 'morning' if minute < 720 else 'afternoon' if minute < 1080 else 'evening'
                    normalized.append(not score(plan,inputs[r['id']],maps,hard))
                except Exception as exc:
                    normalized.append(False)
                    normalization_errors.append({'id':r['id'],'error_type':type(exc).__name__})
            diagnostic.append({'tag':tag,'arm':arm,'n':len(group),'passed':sum(r['passed'] for r in group),
                'parsed':sum(r['parse_success'] for r in group),'format_and_meal_normalized_diagnostic_pass':sum(normalized),
                'normalization_errors':normalization_errors,
                'faults':dict(Counter(f for r in group for f in r['faults'])),
                'explanation_fault_cases':sum(bool(r['explanation_faults']) for r in group),
                'truncated':sum(r.get('response',{}).get('choices',[{}])[0].get('finish_reason')=='length' for r in group)})
            if arm != 'original':
                pairs.append({'tag':tag,'arm':arm,
                    'improved':[r['id'] for r in group if r['passed'] and not original[r['id']]['passed']],
                    'regressed':[r['id'] for r in group if not r['passed'] and original[r['id']]['passed']]})
    summary = {'groups':diagnostic,'paired_changes':pairs,'original_repeat_stability':repeats,
        'scope':'15 diagnosis-selected development cases, not an unbiased benchmark; original results preserved',
        'paid_calls':0,'local_requests':len(results)}
    (OUT/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
