"""Zero-token diversity selection and frozen city-held-out test specification."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.post_train_runtime import replay_check, quality_faults
from app.evaluation.task_robustness import load_assets, grade


def main():
    folder = ROOT/'data/post_train'
    rows = [json.loads(l) for l in (folder/'curated_synthesis_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    # Keep natural correction evidence and actual multi-stop routing first.
    def priority(row):
        return (row.get('preference_source') != 'natural_repair',
            -sum(max(0, len(d['spots'])-1) for d in row['chosen']['days']))
    buckets, selected = Counter(), []
    for row in sorted(rows, key=priority):
        s = row['state']
        weather = tuple(sorted({w['day_weather'] for w in s['weather_forecast']}))
        habit = s['habit_preference']
        profile = 'required' if '必须参观' in habit else 'exclude' if '不要参观' in habit else habit
        key = (s['destination'], s['days'], weather, profile)
        if buckets[key] >= 2 and row.get('preference_source') != 'natural_repair':
            continue
        selected.append(row)
        buckets[key] += 1
    (folder/'quality_selected_ledger.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected), encoding='utf-8')
    _, maps = load_assets()
    hard = []
    for row in sorted(selected, key=priority):
        spots = [s for d in row['chosen']['days'] for s in d['spots']]
        if not any(len(d['spots']) >= 2 for d in row['chosen']['days']):
            continue
        s = copy.deepcopy(row['state'])
        required = spots[0]['name']
        unused = next((p['name'] for p in s['pois'] if p['name'] not in {x['name'] for x in spots}), None)
        additions = [f'必须参观「{required}」', '每天最多两个景点']
        rubric = {'min_count': len(spots), 'required': [required], 'max_count': 2*s['days']}
        if unused:
            additions.append(f'不要参观「{unused}」')
            rubric['excluded'] = [unused]
        s['query'] += '；' + '；'.join(additions) + f'；全程至少{len(spots)}个不同景点。'
        s['attraction_preference'] = s['query']
        faults, checked = replay_check(row['chosen'], s, maps)
        faults += quality_faults(checked)
        case = {'id': f'hard-{len(hard):03d}', 'state': s, 'rubric': rubric,
            'category': 'multi_constraint', 'suite': 'task', 'source_id': row['id']}
        score = grade(checked, case, maps)
        # Witness establishes feasibility under these code checks, not human optimality.
        if faults or not score['task_success']:
            continue
        hard.append({**case, 'feasibility_witness': row['chosen'],
            'witness_status': 'code_verified_not_human_reviewed'})
        if len(hard) >= 60:
            break
    (folder/'hard_supplement_cases.json').write_text(json.dumps(hard, ensure_ascii=False, indent=2), encoding='utf-8')
    heldout = []
    for city, filename in [('丽江', 'lijiang-3d-sunny-nature.json'), ('景德镇', 'jingdezhen-3d-sunny-history.json')]:
        asset = json.loads((ROOT/'tests/eval/fixtures'/filename).read_text(encoding='utf-8'))
        for i in range(12):
            state = {k: copy.deepcopy(asset[k]) for k in ['destination', 'pois', 'travel_start_date', 'travel_end_date']}
            state.update(days=1, travel_end_date=asset['travel_start_date'], max_per_day=2,
                habit_preference='慢节奏', main_meal_candidates=[])
            required = asset['pois'][i % len(asset['pois'])]['name']
            excluded = asset['pois'][(i+1) % len(asset['pois'])]['name']
            state['query'] = f'{city}一日游，10:00以后开始，18:00前结束，恰好两个不同景点，每个至少90分钟。必须参观「{required}」；不要参观「{excluded}」。'
            state['attraction_preference'] = state['query']
            heldout.append({'id': f'city-heldout-{city}-{i:02d}', 'state': state,
                'rubric': {'min_count': 2, 'max_count': 2, 'required': [required], 'excluded': [excluded], 'earliest': 600, 'latest': 1080, 'min_visit': 90},
                'category': 'heldout_city_multi_constraint', 'suite': 'task',
                'training_use': 'forbidden', 'readiness': 'blocked_missing_saved_road_evidence_and_feasibility_review'})
    frozen = folder/'heldout_city_cases_v1.json'
    text = json.dumps(heldout, ensure_ascii=False, indent=2)
    if frozen.exists() and frozen.read_text(encoding='utf-8') != text:
        raise RuntimeError('frozen tests changed; create an explicit new version')
    frozen.write_text(text, encoding='utf-8')
    summary = {'curated_before': len(rows), 'diversity_selected': len(selected), 'feasible_hard_candidates': len(hard),
        'frozen_heldout_cases': len(heldout), 'frozen_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'heldout_status': 'not ready: road evidence and feasibility review required',
        'new_api_calls': 0, 'scope': 'attraction planning only; no restaurants',
        'budget_proposal_within_existing_cap': {'hard_synthesis_max_calls': 90, 'city_heldout_api_max_calls': 24},
        'human_review_status': 'pending', 'generalization_claim': 'not established; holdout only refers to this post-training data, not base-model pretraining'}
    (folder/'quality_pivot_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == '__main__':
    main()
