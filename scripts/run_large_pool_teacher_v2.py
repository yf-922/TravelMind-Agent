"""Strict, budgeted fresh-teacher synthesis. Never trains or uses old targets.

Preparation reuses saved training POIs and roads only. Run a ten-case canary
before --all; both phases share one immutable ledger and call/repair budgets.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.planning.schemas import TravelRoute
from scripts import prepare_v2_large_pool_teacher as previous
from scripts.prepare_quality_sft_v4 import dump, frozen, input_collision_check
from scripts.post_train_runtime import Meter, append_json, capture_messages
from scripts.prepare_sft_v2_release import read_rows
from scripts.compare_route_interest import exclusive_ledger
from scripts.export_llamafactory_dataset import export

OUT = ROOT/'data/post_train/v2_large_pool_teacher_v2'
OLD = previous.OUT
# A conservative scope restriction, not a claim that these venues are unsafe.
# Driving evidence cannot certify their ferry timetable or return crossing.
FERRY_UNVERIFIED = {'三亚西岛海洋文化旅游区', '蜈支洲岛旅游风景区'}
MICRO_AREAS = ['陆家嘴', '人民广场', '新街口', '夫子庙', '春熙路', '天府广场', '平江路', '观前街', '三亚湾', '海棠湾', '亚龙湾']


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def strict_schema():
    schema = copy.deepcopy(TravelRoute.model_json_schema())
    def visit(node):
        if isinstance(node, dict):
            node.pop('default', None)
            if node.get('type') == 'object':
                node['additionalProperties'] = False
                node['required'] = list(node.get('properties', {}))
            for v in list(node.values()):
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)
    visit(schema)
    return schema


def validate_raw(text):
    """No fence stripping, substring extraction, type coercion or prose rewrite."""
    from jsonschema import Draft202012Validator
    raw = json.loads(text)
    Draft202012Validator(strict_schema()).validate(raw)
    return TravelRoute.model_validate(raw).model_dump(mode='json')


def usage_totals(records):
    totals = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    for r in records:
        if r.get('event') in ['completed', 'failed']:
            u = r.get('usage') or {}
            for k in totals:
                totals[k] += int(u.get(k) or 0)
    return totals


def evidence_faults(plan, state, messages):
    errors = []
    selected = [s['name'] for d in plan['days'] for s in d['spots']]
    if set(selected) & FERRY_UNVERIFIED:
        errors.append('ferry_timetable_unverified')
    explanation = ' '.join([plan.get('reasoning', ''), plan.get('notes', ''),
        *[d.get('theme', '') for d in plan['days']],
        *[r.get('reason', '') for r in plan.get('selection_reasons', [])]])
    source = '\n'.join(m['content'] for m in messages)
    for term in MICRO_AREAS:
        if term in explanation and term not in source:
            errors.append('unsupported_micro_area:'+term)
    # Narrow affirmative claims only; absence of a match is not semantic proof.
    for pattern in [r'无需预约', r'无需排队', r'无须预约', r'保证无障碍', r'全程无台阶', r'适老设施齐全']:
        for match in re.finditer(pattern, explanation):
            prefix = explanation[max(0, match.start()-12):match.start()]
            if re.search(r'不能保证|不保证|未确认|不确定|无法确认|不得声称|不推断|不能假定', prefix):
                continue
            if not re.search(pattern, source):
                errors.append('unsupported_facility_or_access_claim')
    return sorted(set(errors))


def audit(plan, case, maps, tokenizer):
    errors, checked = previous.audit(plan, case, maps)
    messages = capture_messages(case['state'])
    errors += evidence_faults(plan, case['state'], messages)
    tokens = len(tokenizer.apply_chat_template(messages+[dict(role='assistant', content=json.dumps(plan, ensure_ascii=False))],
        tokenize=True, add_generation_prompt=False))
    if tokens > 3584:
        errors.append('training_sequence_over_3584')
    return sorted(set(errors)), checked, tokens


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    pools = json.loads((OLD/'training_pools.json').read_text(encoding='utf-8'))['pools']
    original = json.loads((OLD/'cases.json').read_text(encoding='utf-8'))
    isolated = json.loads((OLD/'isolated.json').read_text(encoding='utf-8'))
    cases, changes = [], []
    for old in original:
        case = copy.deepcopy(old)
        state = case['state']
        removed = [p['name'] for p in state['pois'] if p['name'] in FERRY_UNVERIFIED]
        if removed:
            required = (case.get('source_case') or {}).get('rubric', {}).get('required', [])
            required += re.findall(r'必须参观「([^」]+)」', state['query'])
            if set(required) & set(removed):
                case['isolation_reasons'] = ['required_poi_needs_ferry_evidence']
                isolated.append(case)
                continue
            state['pois'] = [p for p in state['pois'] if p['name'] not in FERRY_UNVERIFIED]
            names = {p['name'] for p in state['pois']}
            extra = [copy.deepcopy(p) for p in pools[state['destination']] if p['name'] not in names and p['name'] not in FERRY_UNVERIFIED]
            state['pois'] += extra[:15-len(state['pois'])]
            changes.append(dict(id=case['id'], removed=removed, reason='missing_ferry_evidence'))
        if len(state['pois']) < 15:
            raise RuntimeError('Refuse candidate pool below 15')
        cases.append(case)
    input_collision_check(cases)
    # Bind to the saved evidence, without copying evaluation assets or answers.
    maps = json.loads((OLD/'maps.json').read_text(encoding='utf-8'))['routes']
    from app.evaluation.route_interest import route_key
    for c in cases:
        ps = c['state']['pois']
        if any(route_key(a['location'], b['location'], 'drive') not in maps for a in ps for b in ps if a['name'] != b['name']):
            raise RuntimeError('Missing real driving evidence after substitution')
    frozen(OUT/'cases.json', cases)
    frozen(OUT/'isolated.json', isolated)
    frozen(OUT/'input_changes.json', changes)
    frozen(OUT/'evidence_binding.json', dict(maps_file=str((OLD/'maps.json').relative_to(ROOT)),
        maps_sha256=hashlib.sha256((OLD/'maps.json').read_bytes()).hexdigest(),
        pools_sha256=hashlib.sha256((OLD/'training_pools.json').read_bytes()).hexdigest()))
    from collections import Counter
    summary = dict(callable=len(cases), isolated=len(isolated), source_inputs=len(cases)+len(isolated),
        minimum_candidates=min(len(c['state']['pois']) for c in cases),
        isolation_reasons=dict(Counter(e for c in isolated for e in c['isolation_reasons'])),
        new_teacher_answer_required=True, historical_answer_used=False)
    dump(OUT/'input_audit.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return cases, maps


class StrictMeter(Meter):
    def __init__(self, path, model, max_calls, stop, max_tokens=2500000):
        super().__init__(path, model, max_calls, 90)
        self.stop = stop
        self.max_tokens = max_tokens
        self.used_tokens = usage_totals(read_rows(self.path) if self.path.exists() else [])['total_tokens']

    def call(self, messages, identity):
        payload = dict(model=self.model, messages=messages, temperature=0.3,
            response_format=dict(type='json_schema', json_schema=dict(name='TravelRoute', strict=True, schema=strict_schema())),
            max_completion_tokens=3072, reasoning_effort='low', store=False)
        with self.lock:
            if self.stop.is_set() or self.count >= self.max_calls or self.used_tokens >= self.max_tokens:
                raise RuntimeError('Budget or stop guard reached')
            self.count += 1
            call_id = self.count
            append_json(self.path, dict(event='started', call_id=call_id, identity=identity, time=time.time(), request=payload))
        start = time.perf_counter()
        raw = None
        try:
            response = self.client.chat.completions.create(**payload)
            raw = response.model_dump(mode='json')
            if raw['choices'][0].get('finish_reason') != 'stop':
                raise ValueError('response_not_complete')
            plan = validate_raw(raw['choices'][0]['message']['content'])
            record = dict(event='completed', parse_success=True)
        except Exception as exc:
            record = dict(event='failed', parse_success=False, error_type=type(exc).__name__,
                http_status=getattr(exc, 'status_code', None))
            # Safely actionable validation paths, never SDK exception bodies.
            if hasattr(exc, 'absolute_path'):
                record['validation_path'] = list(exc.absolute_path)
            if record['http_status'] in [400, 401, 403, 404, 429]:
                self.stop.set()
        record.update(call_id=call_id, identity=identity, elapsed_s=time.perf_counter()-start)
        if raw is not None:
            record.update(response=raw, usage=raw.get('usage'))
        with self.lock:
            self.used_tokens += int((record.get('usage') or {}).get('total_tokens') or 0)
            reported_input = int((record.get('usage') or {}).get('prompt_tokens') or 0)
            if reported_input > 20000 or self.used_tokens >= self.max_tokens:
                record['guard_triggered'] = 'reported_input_over_20000' if reported_input > 20000 else 'total_token_cap'
                self.stop.set()
            append_json(self.path, record)
        if record['event'] == 'failed':
            raise RuntimeError(f"Call {call_id}: {record['error_type']} status={record.get('http_status')}")
        return plan, record


def canary_order(cases):
    # Includes the two previous formatting failures and reviewed problem types.
    ids = ['synth-0162', 'newcity-成都-008', 'synth-0070', 'synth-0085',
        'synth-0126', 'synth-0127', 'synth-0190', 'newcity-成都-044', 'newcity-苏州-044', 'newcity-苏州-009']
    rank = {name: i for i, name in enumerate(ids)}
    return sorted(cases, key=lambda c: (rank.get(c['id'], len(ids)), c['id']))


def summary(cases):
    ledger = OUT/'synthesis.ledger.jsonl'
    calls_path = ledger.with_suffix('.calls.jsonl')
    rows = read_rows(ledger) if ledger.exists() else []
    calls = read_rows(calls_path) if calls_path.exists() else []
    accepted = [r for r in rows if r['status'] == 'accepted']
    from collections import Counter
    tokens = sorted(r['qwen_tokens'] for r in accepted)
    result = dict(updated_at=datetime.now(timezone.utc).isoformat(), input_cases=len(cases), recorded=len(rows),
        accepted=len(accepted), first_round_passed=sum(bool(r['rounds']) and not r['rounds'][0]['rule_errors'] for r in rows),
        statuses=dict(Counter(r['status'] for r in rows)), calls=sum(r['event']=='started' for r in calls),
        api_parse_failed=sum(r['event']=='failed' for r in calls), usage=usage_totals(calls),
        pending=len(cases)-len(rows), guards=[dict(call_id=r['call_id'], reason=r['guard_triggered']) for r in calls if r.get('guard_triggered')],
        city_counts=dict(Counter(r['state']['destination'] for r in accepted)),
        first_round_faults=dict(Counter(e for r in rows for e in (r['rounds'][0]['rule_errors'] if r['rounds'] else []))),
        qwen_tokens=dict(min=min(tokens, default=0), max=max(tokens, default=0), over_3584=sum(t>3584 for t in tokens)),
        training_started=False, dpo_started=False, training_ready=False,
        review_scope='deterministic rules and narrow factual checks; not exhaustive semantic certification')
    dump(OUT/'status.json', result)
    return result


def run(cases, maps, limit=None, full=False, concurrency=2, max_calls=289):
    load_local_env()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path.home()/'.cache/travelmind-qwen-tokenizer'), local_files_only=True)
    ledger = OUT/'synthesis.ledger.jsonl'
    calls_path = ledger.with_suffix('.calls.jsonl')
    with exclusive_ledger(ROOT/'data/post_train/teacher_global_budget.jsonl'), exclusive_ledger(ledger):
        existing = sum(json.loads(l).get('event')=='started' for p in (ROOT/'data/post_train').rglob('*.calls.jsonl')
            if p.resolve()!=calls_path.resolve() for l in p.read_text(encoding='utf-8').splitlines())
        if existing+max_calls > 1000:
            raise RuntimeError('Global 1000-call reservation exceeded')
        bind_files = [Path(__file__), ROOT/'scripts/prepare_v2_large_pool_teacher.py', ROOT/'scripts/post_train_runtime.py',
            ROOT/'app/planning/nodes.py', ROOT/'app/planning/helpers.py', ROOT/'app/planning/constraints.py']
        binding = digest(dict(cases=cases, evidence=json.loads((OUT/'evidence_binding.json').read_text(encoding='utf-8')),
            files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bind_files}))
        seeded=sum(bool(r.get('carried_from')) for r in read_rows(ledger)) if ledger.exists() else 0
        repair_cap=min(50,max(0,max_calls-(len(cases)-seeded)))
        frozen(OUT/'run.meta.json', dict(binding=binding, max_calls=max_calls, max_repairs=repair_cap,
            model='gpt-6.1-sol', max_reported_total_tokens=2500000, cutoff=3584,
            strict_json=True, no_historical_targets=True))
        old_calls = read_rows(calls_path) if calls_path.exists() else []
        if any(r.get('guard_triggered') for r in old_calls):
            raise RuntimeError('Existing usage guard requires investigation; automatic resume disabled')
        if full:
            before = summary(cases)
            if before['recorded'] < 10 or before['accepted']/before['recorded'] < .8 or before['api_parse_failed'] > 1 or before['guards']:
                raise RuntimeError('Canary not ready for full synthesis')
        stop = threading.Event()
        meter = StrictMeter(calls_path, 'gpt-6.1-sol', max_calls, stop)
        existing_rows = read_rows(ledger) if ledger.exists() else []
        done = {r['id'] for r in existing_rows}
        started_ids = {r['identity']['sample'] for r in old_calls if r['event']=='started'}
        orphaned = started_ids-done
        if orphaned:
            raise RuntimeError('Reconcile interrupted paid calls before resume: '+','.join(sorted(orphaned)))
        repair_left = repair_cap-sum(r['event']=='started' and r['identity'].get('round')==1 for r in old_calls)
        repair_lock = threading.Lock()
        def worker(case):
            nonlocal repair_left
            if stop.is_set():
                return None
            initial = capture_messages(case['state'])
            state = copy.deepcopy(case['state'])
            row = dict(id=case['id'], state=case['state'], messages=initial, status='failed', chosen=None,
                rejected=None, mock=False, model='gpt-6.1-sol', target_origin='fresh_teacher_only',
                rounds=[], human_review_status='pending')
            for index in range(2):
                if stop.is_set():
                    row['status']='interrupted_by_guard'
                    break
                try:
                    plan, call = meter.call(capture_messages(state), dict(sample=case['id'], round=index, run='large_pool_teacher_v2'))
                    errors, checked, length = audit(plan, case, maps, tokenizer)
                    row['rounds'].append(dict(round=index, output=plan, messages=capture_messages(state),
                        rule_errors=errors, call_id=call['call_id'], usage=call.get('usage'),
                        qwen_tokens=length, route_distance_legs=checked.route_distance_legs))
                    if not errors:
                        row.update(status='accepted', chosen=plan, qwen_tokens=length)
                        break
                    row['status']='quarantined'
                    if index==0 and not stop.is_set():
                        with repair_lock:
                            if repair_left<=0:
                                break
                            repair_left-=1
                        state.update(route=plan['days'], meal_slots=plan.get('meal_slots', []), review_round=1,
                            route_modify_opinion='本次草稿确定性检查未通过：'+', '.join(errors)+
                            '。只依据候选输入重排并输出完整合法JSON；理由、notes简洁且只陈述最终安排；未知的细分片区、设施、交通不作断言，不编造来源引用，不提修改过程。')
                    else:
                        break
                except RuntimeError as exc:
                    row['error']=str(exc)
                    # Parse failures remain failures. Do not auto-coerce them
                    # or hide a separate retry in the first-round success rate.
                    row['status']='api_or_schema_failed'
                    break
                except Exception as exc:
                    row.update(status='audit_failed', error=type(exc).__name__)
                    break
            return row
        pending = [c for c in canary_order(cases) if c['id'] not in done]
        if not full:
            pending=pending[:limit or 10]
        dump(OUT/'launch.json', dict(pid=os.getpid(), time=time.time(), full=full, pending=len(pending),
            concurrency=concurrency, global_calls_before=existing+meter.count))
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Bounded submission lets usage/auth guards stop scheduling calls.
            queue = iter(pending)
            active = {executor.submit(worker, c):c for c in [next(queue, None) for _ in range(concurrency)] if c}
            while active:
                future = next(as_completed(active))
                active.pop(future)
                row = future.result()
                if row is not None:
                    append_json(ledger, row)
                    result=summary(cases)
                    print(json.dumps(dict(id=row['id'], status=row['status'], recorded=result['recorded'], accepted=result['accepted'],
                        calls=meter.count, tokens=result['usage']['total_tokens']), ensure_ascii=False), flush=True)
                case = next(queue, None) if not stop.is_set() else None
                if case:
                    active[executor.submit(worker, case)]=case
        result=summary(cases)
        result['stopped_by_guard']=stop.is_set()
        result['finished_at']=time.time()
        dump(OUT/'status.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


def release_preview(cases, maps):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path.home()/'.cache/travelmind-qwen-tokenizer'), local_files_only=True)
    refs = {c['id']:c for c in cases}
    accepted=[]
    for row in read_rows(OUT/'synthesis.ledger.jsonl'):
        if row['status']!='accepted':
            continue
        assert row['messages']==capture_messages(refs[row['id']]['state'])
        errors, _, _ = audit(row['chosen'], refs[row['id']], maps, tokenizer)
        if errors:
            raise RuntimeError('Recheck failed: '+row['id'])
        accepted.append(row)
    sft, dpo=export(accepted)
    if len(sft)!=len(accepted):
        raise RuntimeError('Export duplicates or count mismatch')
    dump(OUT/'travelmind_planner_sft.preview.json', sft)
    dump(OUT/'export_audit.json', dict(accepted=len(accepted), sft=len(sft), dpo=len(dpo),
        all_messages_match=True, human_review_pending=True, training_ready=False))
    return len(sft)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--all', action='store_true')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--concurrency', type=int, default=2, choices=[1,2,3,4])
    p.add_argument('--export-preview', action='store_true')
    args=p.parse_args()
    cases,maps=prepare()
    if args.run:
        run(cases,maps,args.limit,args.all,args.concurrency)
    if args.export_preview:
        print(json.dumps(dict(sft_preview=release_preview(cases,maps))))


if __name__=='__main__':
    main()
