"""Zero-token curation: verified counterfactuals and a stratified review packet.

Does not authorize training. Human review and Qwen token-length checks remain gates.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.post_train_runtime import replay_check, quality_faults
from scripts.evaluate_reviewer_ablation import cases_v2
from app.evaluation.task_robustness import load_assets
from app.planning.schemas import TravelRoute
from app.planning.nodes import _clock_minutes


def clock(n):
    return f'{n//60:02d}:{n%60:02d}'


def counterfactuals(plan, state):
    """Keep syntax/schema intact, changing only one event or selected name."""
    selected = {s['name'] for d in plan['days'] for s in d['spots']}
    pool = state['pois']
    by_name = {p['name']: p for p in pool}
    for di, day in enumerate(plan['days']):
        for si, spot in enumerate(day['spots']):
            if si:
                x = copy.deepcopy(plan)
                current = x['days'][di]['spots'][si]
                duration = _clock_minutes(current['end_time']) - _clock_minutes(current['start_time'])
                # Five-minute overlap: superficially plausible, unlike backwards clocks.
                start = _clock_minutes(day['spots'][si-1]['end_time']) - 5
                if start >= 0 and start + duration < 1440:
                    current.update(start_time=clock(start), end_time=clock(start+duration))
                    yield 'subtle_time_overlap', x
                x = copy.deepcopy(plan)
                current = x['days'][di]['spots'][si]
                current.update(start_time=day['spots'][si-1]['end_time'], end_time=clock(min(1439, _clock_minutes(day['spots'][si-1]['end_time'])+duration)))
                yield 'zero_commute_gap', x
            if '38°C' in state['query']:
                outside = next((p for p in pool if p.get('indoor') is False and p['name'] not in selected), None)
                if outside:
                    x = copy.deepcopy(plan)
                    x['days'][di]['spots'][si].update(name=outside['name'], period='afternoon', start_time='12:00', end_time='13:30')
                    yield 'heat_exposure_known_outdoor', x
            if f'必须参观「{spot["name"]}」' in state['query']:
                replacement = next((p for p in pool if p['name'] not in selected), None)
                if replacement:
                    x = copy.deepcopy(plan)
                    x['days'][di]['spots'][si]['name'] = replacement['name']
                    yield 'missing_explicit_required_poi', x
            if by_name.get(spot['name'], {}).get('indoor') is True and state.get('rain_indoor_priority'):
                outside = next((p for p in pool if p.get('indoor') is False and p['name'] not in selected), None)
                if outside:
                    x = copy.deepcopy(plan)
                    x['days'][di]['spots'][si]['name'] = outside['name']
                    yield 'rain_known_outdoor', x
            # Same-city real POI absent from the current sampled candidate pool.
            allowed = {p['name'] for p in pool}
            for case in cases_v2():
                if case['state']['destination'] != state['destination']:
                    continue
                other = next((p for p in case['state']['pois'] if p['name'] not in allowed), None)
                if other:
                    x = copy.deepcopy(plan)
                    x['days'][di]['spots'][si]['name'] = other['name']
                    yield 'real_same_city_out_of_pool', x
                    break
            # Shift a plausible visit into a likely closed hour, validate rather than assume.
            x = copy.deepcopy(plan)
            x['days'][di]['spots'][si].update(period='morning', start_time='06:00', end_time='07:30')
            yield 'opening_window_shift', x


def curate(rows, maps):
    output, seen, issues, types = [], set(), Counter(), Counter()
    for row in rows:
        if row.get('mock') is not False or row.get('status') != 'accepted':
            continue
        errors, checked = replay_check(row['chosen'], row['state'], maps)
        errors += quality_faults(checked)
        if errors:
            issues['chosen_failed_recheck'] += 1
            continue
        # Dedup on semantic state, ignoring pool order and sample identifiers.
        state_key = {k: row['state'].get(k) for k in ['query', 'destination', 'days', 'weather_forecast', 'habit_preference']}
        state_key['pois'] = sorted(p['name'] for p in row['state']['pois'])
        key = json.dumps(state_key, sort_keys=True, ensure_ascii=False)
        if key in seen:
            issues['duplicate_semantic_state'] += 1
            continue
        seen.add(key)
        variants = []
        if row.get('preference_source') == 'natural_repair':
            variants.append(('natural_repair', row['rejected']))
        variants.extend(counterfactuals(row['chosen'], row['state']))
        # Balance defect types instead of always selecting the easiest candidate edit.
        variants.sort(key=lambda pair: (pair[0] != 'natural_repair', types[pair[0]]))
        verified = None
        for kind, negative in variants:
            try:
                TravelRoute.model_validate(negative)
                faults, rejected = replay_check(negative, row['state'], maps)
                faults = sorted(set(faults + quality_faults(rejected)))
            except (ValueError, TypeError):
                continue
            expected = {
                'natural_repair': set(faults),
                'subtle_time_overlap': {'route_structure'},
                'zero_commute_gap': {'commute_time_conflict'},
                'heat_exposure_known_outdoor': {'heat_exposure_or_unknown_indoor'},
                'missing_explicit_required_poi': {'required_poi_missing'},
                'rain_known_outdoor': {'weather_outdoor_conflict', 'indoor_constraint'},
                'real_same_city_out_of_pool': {'unknown_poi'},
                'opening_window_shift': {'opening_time_conflict'},
            }[kind]
            if set(faults) & expected:
                verified = kind, negative, faults
                break
        curated = copy.deepcopy(row)
        if verified:
            kind, negative, faults = verified
            curated.update(rejected=negative, rejected_rule_errors=faults, preference_source=kind)
            types[kind] += 1
        else:
            # Preserve SFT material without inventing a useful preference pair.
            curated.update(rejected=None, rejected_rule_errors=[], preference_source=None)
            issues['no_nontrivial_negative_sft_only'] += 1
        curated['human_review_status'] = 'pending'
        output.append(curated)
    return output, dict(issues), dict(types)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=ROOT/'data/post_train/raw_synthesis_ledger.jsonl')
    p.add_argument('--output', type=Path, default=ROOT/'data/post_train/curated_synthesis_ledger.jsonl')
    a = p.parse_args()
    rows = []
    for line in a.input.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    _, maps = load_assets()
    output, issues, types = curate(rows, maps)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in output), encoding='utf-8')
    # Interleave cities and defect types before selecting the review packet.
    buckets = {}
    for row in output:
        buckets.setdefault((row['state']['destination'], row.get('preference_source')), []).append(row)
    packet = []
    while buckets and len(packet) < 30:
        for k in list(buckets):
            packet.append(buckets[k].pop(0))
            if not buckets[k]:
                del buckets[k]
            if len(packet) >= 30:
                break
    a.output.with_suffix('.review.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'input_rows': len(rows), 'curated_sft_candidates': len(output), 'preference_types': types,
        'issues': issues, 'human_review': 'pending', 'qwen_token_lengths': 'pending',
        'training_ready': False, 'independent_test_assets': 'pending'}
    a.output.with_suffix('.audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == '__main__':
    main()
