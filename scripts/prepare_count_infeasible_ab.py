"""Proof-checked count infeasibility SFT pairs; no teacher or map API calls.

Experimental dataset gate only: production risk handling is intentionally not
changed. Every empty answer needs a verified pigeonhole certificate.
"""
import copy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.planning.schemas import TravelPlanState, TravelRoute
from scripts.post_train_runtime import capture_messages, replay_check
from scripts.prepare_heldout_evidence import opening_window, opening_evidence_faults, clock
from scripts.regen_v1_v2_current_protocol import make_tokenizer

OUT = ROOT/'data/post_train/count_infeasible_ab_20261025'
SOURCE = ROOT/'data/post_train/release_current_protocol_v1_v2/v2'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def check(case, plan):
    """Only two supported targets: certified impossible or witnessed feasible."""
    errors = []
    try:
        plan = TravelRoute.model_validate(plan).model_dump(mode='json')
    except Exception:
        return ['schema']
    state = case['state']
    names = [s['name'] for d in plan['days'] for s in d['spots']]
    pool = {p['name'] for p in state['pois']}
    required = case['required_total']
    # Recompute certificate from actual prompt/state; never trust the expected label alone.
    requested = re.search(r'全程必须恰好安排(\d+)个不同景点', state['query'])
    pool_claim = re.search(r'去重后(\d+)个候选', state['query'])
    if (len(pool) != case['pool_unique'] or not requested or int(requested[1]) != required
            or not pool_claim or int(pool_claim[1]) != len(pool)):
        errors.append('certificate_input_mismatch')
    impossible = required > len(pool)
    if impossible != (case['expected'] == 'infeasible'):
        errors.append('certificate_label_mismatch')
    if sorted(d['day'] for d in plan['days']) != list(range(1, state['days']+1)):
        errors.append('days')
    if impossible:
        if names or plan['meal_slots'] or plan['selection_reasons']:
            errors.append('unsafe_partial_schedule')
        explanation = plan['notes']+' '+plan['reasoning']
        if not re.search('无法|不足|不能满足|不可行|冲突', explanation):
            errors.append('not_explained')
        if not any(w in explanation for w in ['候选', '不同景点', '去重']):
            errors.append('wrong_reason')
        if not all(str(n) in explanation for n in [len(pool), required]):
            errors.append('proof_numbers_missing')
    else:
        if len(names) != required or len(set(names)) != required or not set(names) <= pool:
            errors.append('count_or_membership')
        if any(len(d['spots']) != 1 for d in plan['days']):
            errors.append('daily_allocation')
        errors += replay_check(plan, state, {})[0]
        errors += opening_evidence_faults(plan, case)
        for d in plan['days']:
            for s in d['spots']:
                start = int(s['start_time'][:2])*60+int(s['start_time'][3:])
                end = int(s['end_time'][:2])*60+int(s['end_time'][3:])
                if end-start < case['duration'] or start < 540 or end > 1080:
                    errors.append('visit_window')
    return sorted(set(errors))


def pair(source, split, index, tokenizer):
    raw = source['state']
    eligible = [copy.deepcopy(p) for p in raw['pois'] if opening_window(p)
                # A September-only dated schedule is not evidence for November.
                and not re.search(r'(?<!\d)\d{2}-\d{2}至\d{2}-\d{2}', p.get('open_time', ''))]
    eligible = list({p['name']:p for p in eligible}.values())
    random.Random(7000+index).shuffle(eligible)
    count = min(1+index%3, len(eligible))
    if count < 1:
        raise ValueError('no usable candidates')
    duration = [60, 75, 90][(index//3)%3]
    pool = []
    for p in eligible:
        window = opening_window(p)
        if max(540, window[0])+duration <= min(1080, window[1]) and max(540, window[0]) <= window[2]:
            pool.append(p)
        if len(pool) == count:
            break
    if len(pool) != count:
        raise ValueError('insufficient usable windows')
    # Real POI metadata is preserved; weather and request are explicit synthetic controls.
    state = TravelPlanState(query='', destination=raw['destination'], pois=pool, days=count,
        travel_start_date='2026-11-04', max_per_day=3, main_meal_candidates=[],
        candidate_pool_status='insufficient', candidate_missing_coverage=['count'],
        route_distance_mode='drive', weather_forecast=[dict(date=(date(2026,11,4)+timedelta(days=d)).isoformat(),
            day_temp='26', night_temp='20', day_weather='晴', night_weather='晴', is_bad=False) for d in range(count)])
    rows = []
    for impossible in [True, False]:
        required = count+1 if impossible else count
        policies = [
            '先判断条件是否能同时成立。若不可行，保留每一天但spots为空，在notes写明数量冲突和可调整条件，不以部分路线冒充完成。',
            '请核对去重后的候选能否达到指定数量。无法达到时，各天不安排景点并说明缺几个，建议补候选或降低总数；有解时正常给路线。',
            '必须同时满足候选范围、全程去重与总数；如发生数量矛盾，则输出对应天数的空景点列表并解释，不要硬排。',
            '不要把少排视为满足恰好数量。条件不足时明确说明不可行，景点留空；条件足够则按要求安排，不得无故拒绝。',
        ]
        policy = policies[(index+(0 if split=='train' else 1))%4]
        query = (f'{state.destination}{count}日游。全程必须恰好安排{required}个不同景点，每天最多2个，'
            f'每个至少{duration}分钟，每天09:00至18:00；只可使用下列去重后{count}个候选，禁止新增和重复。'
            f'允许名单仅限本次候选快照，不安排餐馆；若有解每天至少安排1个景点。{policy}')
        s = state.model_copy(update={'query':query,
            'candidate_pool_status':'pending',
            'candidate_missing_coverage':[]}).model_dump(mode='json')
        if impossible:
            explanation = f'候选去重后只有{count}个，要求恰好{required}个不同景点且禁止新增和重复，数量不足，无法满足。'
            plan = dict(reasoning=explanation,
                days=[dict(day=d+1, spots=[], theme='数量约束不可行') for d in range(count)],
                meal_slots=[], selection_reasons=[],
                notes=f'需{required}个不同景点，但仅有{count}个候选，缺{required-count}个。请补候选或将总数降至{count}个；暂不提供部分路线。', modification_concern='')
        else:
            days = []
            for d, p in enumerate(pool):
                start = max(540, opening_window(p)[0])
                days.append(dict(day=d+1, theme='单点慢游', spots=[dict(name=p['name'],
                    period='morning' if start<720 else 'afternoon', start_time=clock(start), end_time=clock(start+duration))]))
            plan = dict(reasoning=f'候选有{count}个，恰好安排{required}个不同景点，每天1个；按输入开放窗口参观，不重复。',
                days=days, meal_slots=[], selection_reasons=[dict(name=p['name'], reason='按给定开放窗口单独安排一天，不重复。', source_ids=[]) for p in pool],
                notes=f'全程{count}个不同景点，每处{duration}分钟。无餐馆候选，meal_slots留空；开放和天气按快照，出行前需复核。', modification_concern='')
        case = dict(id=f'count-{split}-{index:03d}-'+('impossible' if impossible else 'feasible'),
            pair_id=f'count-{split}-{index:03d}', source_id=source['id'],
            state=s, expected='infeasible' if impossible else 'feasible',
            required_total=required, pool_unique=count, duration=duration)
        faults = check(case, plan)
        if faults:
            raise ValueError(','.join(faults))
        messages = capture_messages(s)
        tokens = len(tokenizer.apply_chat_template(messages+[dict(role='assistant', content=json.dumps(plan, ensure_ascii=False))], tokenize=True, add_generation_prompt=False))
        if tokens > 3584:
            raise ValueError('length')
        rows.append(dict(case, status='accepted', chosen=plan, messages=messages, mock=False,
            target_origin='deterministic_proof_and_feasible_witness', teacher_api_calls=0,
            proof=dict(kind='distinct_count_pigeonhole', actual_unique=count, required_unique=required,
                       impossible=impossible), qwen_tokens=tokens))
    return rows


def build(sources, split, pairs, tokenizer, forbidden=None):
    groups = {}
    for r in sources:
        groups.setdefault(r['state']['destination'], []).append(r)
    for city, rows in groups.items():
        random.Random('count-ab-'+city+'-'+split).shuffle(rows)
    ordered = []
    while any(groups.values()):
        for city in sorted(groups):
            if groups[city]:
                ordered.append(groups[city].pop())
    output = []
    failures = []
    seen = set(forbidden or ())
    for attempt, source in enumerate(ordered):
        try:
            rows = pair(source, split, attempt, tokenizer)
            keys = {json.dumps(r['messages'], ensure_ascii=False) for r in rows}
            if seen & keys:
                raise ValueError('duplicate message across selected families/splits')
        except ValueError as exc:
            failures.append(dict(id=source['id'], error=str(exc)))
            continue
        output.extend(rows)
        seen.update(keys)
        if len(output) == pairs*2:
            return output, failures
    raise RuntimeError(f'Need {pairs} disjoint source families for {split}, got {len(output)//2}')


def export(rows):
    return [dict(conversations=[dict(**{'from':{'system':'system','user':'human'}[m['role']], 'value':m['content']})
        for m in r['messages']]+[{'from':'gpt','value':json.dumps(r['chosen'],ensure_ascii=False)}]) for r in rows]


def main():
    if (OUT/'manifest.json').exists():
        raise RuntimeError('Release exists; refusing to rewrite experiment')
    tokenizer = make_tokenizer()
    train_source = json.loads((SOURCE/'train.ledger.jsonl').read_text(encoding='utf-8'))
    val_source = json.loads((SOURCE/'validation.ledger.jsonl').read_text(encoding='utf-8'))
    test_source = json.loads((ROOT/'data/post_train/heldout_evidence_v1/cases.json').read_text(encoding='utf-8'))
    train, skipped_train = build(train_source, 'train', 48, tokenizer)
    train_messages = {json.dumps(r['messages'], ensure_ascii=False) for r in train}
    val, skipped_val = build(val_source, 'validation', 12, tokenizer, train_messages)
    val_messages = {json.dumps(r['messages'], ensure_ascii=False) for r in val}
    test, skipped_test = build(test_source, 'test', 20, tokenizer, train_messages | val_messages)
    sets = [{r['source_id'] for r in split} for split in [train,val,test]]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError('Source-family leakage')
    hashes = [{hashlib.sha256(json.dumps(r['messages'],ensure_ascii=False).encode()).hexdigest() for r in split}
              for split in [train,val,test]]
    if any(hashes[i]&hashes[j] for i in range(3) for j in range(i)):
        raise RuntimeError('Message leakage')
    for split, rows in [('train',train),('validation',val),('test',test)]:
        write(OUT/f'{split}.ledger.json', rows)
    old_train = json.loads((SOURCE/'travelmind_planner_sft_train.json').read_text(encoding='utf-8'))
    old_val = json.loads((SOURCE/'travelmind_planner_sft_validation.json').read_text(encoding='utf-8'))
    info = {}
    for name in ['train', 'count_validation', 'legacy_validation']:
        info[name] = dict(file_name=name+'.json', formatting='sharegpt', columns={'messages':'conversations'},
            tags=dict(role_tag='from',content_tag='value',user_tag='human',assistant_tag='gpt',system_tag='system'))
    for arm in ['A','B']:
        folder = OUT/arm
        write(folder/'train.json', export(train) if arm=='A' else old_train+export(train))
        write(folder/'count_validation.json', export(val))
        write(folder/'legacy_validation.json', old_val)
        write(folder/'dataset_info.json', info)
        # Smoke checks start from the real arm's base and use the actual training pool.
    manifest = dict(train_new=96, train_infeasible=48, train_feasible=48, arm_A_train=96,
        arm_B_train=346, count_validation=24, legacy_validation=30, diagnostic_test=40,
        train_cities=dict(__import__('collections').Counter(r['state']['destination'] for r in train)),
        test_cities=sorted({r['state']['destination'] for r in test}), max_tokens=max(r['qwen_tokens'] for r in train+val+test),
        source_families_disjoint=True, actual_messages_disjoint=True, paid_api_calls=0,
        targets='deterministic certified labels; not teacher generations',
        scope='Count-only intervention; hot/unknown-indoor infeasibility is not trained in this experiment.',
        causal_limit='Same added data and 48 optimizer steps, but starts and data mixture differ; A also inherits prior v2 training.',
        production_unchanged=True, skipped=dict(train=skipped_train,validation=skipped_val,test=skipped_test))
    files = [OUT/f'{split}.ledger.json' for split in ['train','validation','test']]
    files += [p for arm in ['A','B'] for p in (OUT/arm).glob('*.json')]
    manifest['sha256'] = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write(OUT/'manifest.json', manifest)
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':
    main()
