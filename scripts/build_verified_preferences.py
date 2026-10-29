"""Build auditable, route-grounded preference pairs without model API calls.

SFT answers remain untouched. Both DPO sides receive the same factual rendering
policy; no side retains copied claims that a changed itinerary is valid.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_robustness import load_assets
from app.evaluation.route_interest import route_key
from app.planning.schemas import TravelRoute
from app.planning.nodes import _clock_minutes
from scripts.curate_post_train_data import clock
from scripts.prepare_heldout_evidence import opening_window
from scripts.post_train_runtime import replay_check, quality_faults, capture_messages
from scripts.run_hard_post_train_synthesis import audit_plan
from scripts.export_llamafactory_dataset import export


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def grounded_answer(plan, state):
    """Describe the actual schedule; make no unsupported quality/safety claims.

    This deterministic target transformation is recorded in provenance, and must
    not be confused with a new teacher-generated reasoning trace.
    """
    p = copy.deepcopy(plan)
    schedule = '；'.join(f'第{d["day"]}天：' + '、'.join(
        f'{s["name"]} {s["start_time"]}-{s["end_time"]}' for s in d['spots']) for d in p['days'])
    p['reasoning'] = ('【景点相邻】本版按以下顺序及时间安排：' + schedule + '。'
        '【用户偏好】需求以输入中的本轮约束为准。'
        '【无重复】地点及出现次数见逐日清单。'
        '【天气适配】天气与候选室内标签以输入快照为准。')
    p['selection_reasons'] = []
    p['notes'] = '本版时间与地点详见逐日清单；开放信息和道路耗时以固定快照为准。'
    p['modification_concern'] = ''
    for d in p['days']:
        d['theme'] = f'第{d["day"]}天游览安排'
    TravelRoute.model_validate(p)
    return p


def faults(plan, row, maps, hard_cases):
    if row['id'] in hard_cases:
        errors, _, checked = audit_plan(plan, hard_cases[row['id']], maps)
    else:
        errors, checked = replay_check(plan, row['state'], maps)
        errors = sorted(set(errors + quality_faults(checked)))
    return set(errors), checked


def family(errors):
    aliases = {'commute_gap': 'commute', 'commute_time_conflict': 'commute',
               'route_structure': 'overlap', 'opening_time_conflict': 'opening',
               'required_poi_missing': 'required', 'excluded_poi': 'excluded',
               'requested_window': 'requested_window', 'late_start': 'requested_window',
               'visit_too_short': 'visit_duration', 'pace_count': 'pace'}
    return {aliases.get(e, e) for e in errors}


def subtle_variants(plan, state, maps):
    """Near-boundary defects, rather than 06:00 openings or zero-hour commutes."""
    locations = {p['name']: p for p in state['pois']}
    mode = 'walk' if state.get('max_walking_km') is not None else 'drive'
    for di, day in enumerate(plan['days']):
        for si, spot in enumerate(day['spots']):
            duration = _clock_minutes(spot['end_time'])-_clock_minutes(spot['start_time'])
            window = opening_window(locations[spot['name']])
            if window:
                for start in [window[0]-5, window[1]+5-duration]:
                    if not 0 <= start < start+duration < 1440:
                        continue
                    changed = copy.deepcopy(plan)
                    changed['days'][di]['spots'][si].update(start_time=clock(start), end_time=clock(start+duration),
                        period='morning' if start < 720 else 'afternoon')
                    yield 'opening_boundary_5min', changed
            if si:
                previous = day['spots'][si-1]
                leg = maps.get(route_key(locations[previous['name']]['location'], locations[spot['name']]['location'], mode))
                if leg and leg.get('duration_min', 0) > 5:
                    start = _clock_minutes(previous['end_time'])+int(leg['duration_min'])-5
                    if 0 <= start < start+duration < 1440:
                        changed = copy.deepcopy(plan)
                        changed['days'][di]['spots'][si].update(start_time=clock(start), end_time=clock(start+duration),
                            period='morning' if start < 720 else 'afternoon')
                        yield 'commute_deficit_5min', changed
            query = state['query']
            if f'必须参观「{spot["name"]}」' in query:
                selected = {s['name'] for d in plan['days'] for s in d['spots']}
                for poi in state['pois']:
                    if poi['name'] not in selected and f'不要参观「{poi["name"]}」' not in query:
                        changed = copy.deepcopy(plan)
                        changed['days'][di]['spots'][si]['name'] = poi['name']
                        yield 'missing_explicit_required_poi', changed


def build_pair(row, maps, hard_cases, counts):
    if row['messages'] != capture_messages(row['state']):
        return None, 'message_mismatch'
    chosen = grounded_answer(row['chosen'], row['state'])
    errors, checked = faults(chosen, row, maps, hard_cases)
    if errors:
        return None, 'chosen_failed_recheck'
    variants = list(subtle_variants(chosen, row['state'], maps))
    if row.get('preference_source') == 'natural_repair' and row.get('rejected'):
        if all(d['spots'] for d in row['rejected']['days']):
            variants.insert(0, ('natural_repair', row['rejected']))
    options = []
    for kind, negative in variants:
        # Name replacement can change more than one causal factor; retain only
        # variants with one observable violated rule family and complete roads.
        negative = grounded_answer(negative, row['state'])
        bad, rejected = faults(negative, row, maps, hard_cases)
        if not bad or len(family(bad)) != 1 or 'road_unverified' in bad:
            continue
        if family(bad) & {'heat_exposure_or_unknown_indoor', 'weather_outdoor_conflict',
                          'indoor_constraint', 'unknown_poi', 'required_count'}:
            continue  # heuristic indoor labels / trivial absence are weak evidence
        changed = sum(a != b for a, b in zip(
            [s for d in chosen['days'] for s in d['spots']],
            [s for d in negative['days'] for s in d['spots']]))
        if kind != 'natural_repair' and changed != 1:
            continue
        options.append((kind, negative, bad, rejected))
    if not options:
        return None, 'no_single_proven_defect'
    options.sort(key=lambda item: (counts[next(iter(family(item[2])))], item[0] != 'natural_repair'))
    kind, rejected, bad, rejected_state = options[0]
    category = next(iter(family(bad)))
    counts[category] += 1
    pair = copy.deepcopy(row)
    pair.update(chosen=chosen, rejected=rejected, rejected_rule_errors=sorted(bad),
        preference_source=kind, preference_family=category,
        preference_provenance={'version': 'grounded-single-defect-v2', 'source_id': row['id'],
            'teacher_chosen_sha256': digest(row['chosen']), 'messages_sha256': digest(row['messages']),
            'answer_transformation': 'both sides factual schedule rendering; original SFT preserved',
            'chosen_faults': [], 'rejected_faults': sorted(bad),
            'human_review': 'not_performed', 'is_teacher_authored_reasoning': False,
            'chosen_road_evidence': checked.route_distance_legs,
            'rejected_road_evidence': rejected_state.route_distance_legs})
    return pair, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, default=ROOT/'data/post_train/release_v1')
    p.add_argument('--output', type=Path, default=ROOT/'data/post_train/preferences_v2')
    a = p.parse_args()
    _, maps = load_assets()
    hard_cases = {c['id']: c for c in json.loads((ROOT/'data/post_train/hard_supplement_cases.json').read_text(encoding='utf-8'))}
    a.output.mkdir(parents=True, exist_ok=True)
    summary = {'version': 'grounded-single-defect-v2', 'new_llm_calls': 0,
        'scope': 'controlled rule preferences; no subjective/expert preference claim',
        'maps_sha256': digest(maps), 'full_training_release': False, 'splits': {}}
    split_messages, info = {}, {}
    for split in ['train', 'validation']:
        rows = [json.loads(l) for l in (a.release/f'{split}.ledger.jsonl').read_text(encoding='utf-8').splitlines()]
        pairs, skipped, counts = [], Counter(), Counter()
        for row in rows:
            pair, reason = build_pair(row, maps, hard_cases, counts)
            if pair:
                pairs.append(pair)
            else:
                skipped[reason] += 1
        split_messages[split] = {digest(r['messages']) for r in pairs}
        (a.output/f'{split}.ledger.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in pairs), encoding='utf-8')
        _, data = export(pairs)
        name = f'travelmind_verified_dpo_{split}'
        (a.output/f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        info[name] = {'file_name': name+'.json', 'formatting': 'sharegpt', 'ranking': True,
            'columns': {'messages': 'conversations', 'chosen': 'chosen', 'rejected': 'rejected'},
            'tags': {'role_tag': 'from', 'content_tag': 'value', 'user_tag': 'human', 'assistant_tag': 'gpt', 'system_tag': 'system'}}
        summary['splits'][split] = {'input': len(rows), 'pairs': len(pairs), 'families': dict(counts), 'skipped': dict(skipped)}
    assert not split_messages['train'] & split_messages['validation']
    summary['message_overlap'] = 0
    (a.output/'dataset_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    (a.output/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == '__main__':
    main()
