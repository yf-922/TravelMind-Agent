"""Metered teacher arm of the frozen controlled benchmark (no GPU required).

Uses the exact frozen messages, schema, scorer, feedback text and decoding limit
of benchmark_controlled.py. Provider-side caching/scheduling is uncontrollable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.planning.schemas import TravelRoute
from scripts import benchmark_controlled as protocol
from scripts.compare_route_interest import exclusive_ledger
from scripts.post_train_runtime import append_json


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    tmp.replace(path)


def rule_flags(faults, delivered):
    flags = protocol.flags(faults)
    flags['non_duplicate_pass'] &= 'global_duplicate' not in faults
    flags['timeline_transit_pass'] &= not bool(set(faults) & {'time_overlap', 'invalid_time', 'time_order'})
    return {k: bool(v and delivered) for k, v in flags.items()}


def summary(rows, mode):
    # Reuse the common aggregate, but never replace missing usage with a guessed
    # character/token ratio, nor pass invalid JSON on individual rules.
    usable = copy.deepcopy(rows)
    usage_complete = all(r['usage_complete'] for r in usable)
    for r in usable:
        r['prompt_tokens'] = r['prompt_tokens'] or 0
        r['completion_tokens'] = r['completion_tokens'] or 0
    # Common summarizer needs at least one measurable TPOT.
    has_tpot = any(r['tpot_ms'] is not None for r in usable)
    if not has_tpot:
        for r in usable:
            r['tpot_ms'] = 0
    result = protocol.summarize(usable, mode)
    calls = [q for r in rows for q in r['rounds']]
    tpots = [q['timing']['tpot_ms'] for q in calls if q['timing'].get('tpot_ms') is not None]
    result['tpot_p50_ms'] = statistics.median(tpots) if tpots else None
    result['usage_complete'] = usage_complete
    if not usage_complete:
        for k in ['avg_prompt_tokens', 'avg_completion_tokens', 'total_tokens']:
            result[k] = None
    delivered = sum(r['delivery'] for r in rows)
    result['all_rule_pass_given_delivery_pct'] = 100 * sum(r['all_rule_pass'] for r in rows) / delivered if delivered else None
    result['truncated_calls'] = sum(q['timing'].get('finish_reason') == 'length' for q in calls)
    result['single_chunk_calls'] = sum(q['timing'].get('content_chunks') == 1 for q in calls)
    result['total_reasoning_tokens'] = sum(q['timing'].get('reasoning_tokens', 0) for q in calls)
    result['total_visible_completion_tokens'] = sum(q['timing'].get('visible_completion_tokens') or 0 for q in calls) if usage_complete else None
    result['ttft_p50_s'] = statistics.median([q['timing']['ttft_s'] for q in calls if q['timing'].get('ttft_s') is not None]) if any(q['timing'].get('ttft_s') is not None for q in calls) else None
    return result


class TransientTeacherError(RuntimeError):
    """Explicit provider availability failure, distinct from a model answer."""


class Teacher:
    def __init__(self, out, model, url, key, cap):
        self.out, self.model, self.url, self.key, self.cap = out, model, url, key, cap
        self.ledger = out / 'teacher.calls.jsonl'
        self.calls = sum(json.loads(s).get('event') == 'started' for s in self.ledger.read_text(encoding='utf8').splitlines()) if self.ledger.exists() else 0
        self.include_reasoning_effort = True

    def call(self, messages, seed, identity):
        for attempt in range(4):
            try:
                return self._call(messages, seed, dict(identity, transport_attempt=attempt+1))
            except TransientTeacherError:
                if attempt == 3:
                    raise
                # Delays count toward the current task's E2E, not generation TPOT.
                append_json(self.ledger, dict(event='backoff', call_id=self.calls, seconds=60, time=time.time()))
                print('Teacher provider temporarily unavailable; waiting 60s before bounded retry.', flush=True)
                time.sleep(60)

    def _call(self, messages, seed, identity):
        if self.calls >= self.cap:
            raise RuntimeError('Teacher run call budget exhausted')
        self.calls += 1
        payload = dict(model=self.model, messages=messages, temperature=0.0, seed=seed,
                       max_tokens=2048, stream=True, stream_options={'include_usage': True},
                       reasoning_effort='none',
                       response_format={'type': 'json_schema', 'json_schema': {'name': 'TravelRoute', 'strict': False, 'schema': TravelRoute.model_json_schema()}})
        if not self.include_reasoning_effort:
            payload.pop('reasoning_effort')
        append_json(self.ledger, dict(event='started', call_id=self.calls, identity=identity, time=time.time(), request=payload))
        started = time.perf_counter()
        text, usage, ttft, finish, model_returned, done = '', {}, None, None, None, False
        content_chunks = 0
        stream_events = 0
        content_type = None
        try:
            req = urllib.request.Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
            with urllib.request.urlopen(req, timeout=180) as response:
                content_type = response.headers.get('Content-Type') if hasattr(response, 'headers') else None
                for raw in response:
                    line = raw.decode('utf8').strip()
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        done = True
                        continue
                    item = json.loads(data)
                    stream_events += 1
                    if item.get('error'):
                        detail = json.dumps(item['error'], ensure_ascii=False).replace(self.key, '[REDACTED]')
                        raise RuntimeError('Provider stream returned an error: ' + detail[:2000])
                    model_returned = item.get('model', model_returned)
                    if item.get('usage'):
                        usage = item['usage']
                    for choice in item.get('choices') or []:
                        finish = choice.get('finish_reason') or finish
                        delta = (choice.get('delta') or {}).get('content')
                        if delta:
                            content_chunks += 1
                            if ttft is None:
                                ttft = time.perf_counter() - started
                            text += delta
            if not finish or not done:
                raise RuntimeError('Incomplete SSE stream')
        except Exception as exc:
            append_json(self.ledger, dict(event='failed', call_id=self.calls, error_type=type(exc).__name__,
                http_status=getattr(exc, 'code', None), elapsed_s=time.perf_counter()-started,
                diagnostic=str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__,
                partial_output=text, partial_usage=usage, finish_reason=finish, done_received=done,
                content_chunks=content_chunks, stream_events=stream_events, content_type=content_type))
            transient = getattr(exc, 'code', None) in {429, 502, 503, 504} or 'temporarily unavailable' in str(exc).lower()
            error = TransientTeacherError if transient else RuntimeError
            raise error(f'Teacher transport failed: {type(exc).__name__}, HTTP {getattr(exc, "code", None)}') from None
        elapsed = time.perf_counter() - started
        completion = usage.get('completion_tokens')
        reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
        visible = completion - reasoning if completion is not None else None
        timing = dict(prompt_tokens=usage.get('prompt_tokens'), completion_tokens=completion,
                      output_chars=len(text), e2e_s=elapsed, ttft_s=ttft, finish_reason=finish,
                      reasoning_tokens=reasoning, visible_completion_tokens=visible, content_chunks=content_chunks,
                      tpot_ms=(elapsed-ttft)/(completion-1)*1000 if ttft is not None and completion is not None and completion > 1 and content_chunks > 1 else None,
                      visible_tpot_ms=(elapsed-ttft)/(visible-1)*1000 if ttft is not None and visible is not None and visible > 1 and content_chunks > 1 else None)
        append_json(self.ledger, dict(event='completed', call_id=self.calls, timing=timing,
            usage=usage, output_text=text, returned_model=model_returned))
        return text, timing


def run_mode(teacher, out, mode, cases, maps):
    folder = out / ('teacher_' + mode)
    folder.mkdir(exist_ok=True)
    raw = folder / 'results.jsonl'
    rows = [json.loads(s) for s in raw.read_text(encoding='utf8').splitlines()] if raw.exists() else []
    completed = {(r['id'], r['seed']) for r in rows}
    warmup = folder / 'warmup_complete.json'
    if not warmup.exists():
        events = [json.loads(s) for s in teacher.ledger.read_text(encoding='utf8').splitlines()] if teacher.ledger.exists() else []
        finished_ids = {r['call_id'] for r in events if r['event'] == 'completed'}
        warmed = {r['identity']['case'] for r in events if r['event'] == 'started' and r['call_id'] in finished_ids and r['identity'].get('warmup') and r['identity'].get('mode') == mode}
        for case in cases[:5]:
            if case['id'] in warmed:
                continue
            teacher.call(case['messages'], 42, dict(mode=mode, warmup=True, case=case['id']))
        write(warmup, dict(n=5))
    for seed in protocol.SEEDS:
        for case in cases:
            if (case['id'], seed) in completed:
                continue
            rounds, feedback, final = [], '', None
            started = time.perf_counter()
            for turn in range(1 if mode == 'planner-only' else protocol.MAX_REWORK + 1):
                messages = copy.deepcopy(case['messages'])
                if feedback:
                    messages[-1]['content'] += '\n\n确定性规则返工反馈：' + feedback
                text, timing = teacher.call(messages, seed, dict(mode=mode, warmup=False, case=case['id'], seed=seed, round=turn))
                final = None
                try:
                    final = TravelRoute.model_validate_json(text).model_dump(mode='json')
                except ValueError as exc:
                    rounds.append(dict(turn=turn+1, error=type(exc).__name__, faults=['delivery_failure'], timing=timing))
                    # Matches the running local arm: schema failures end the case.
                    break
                faults = protocol.score_case(final, case, maps)
                rounds.append(dict(turn=turn+1, faults=faults, timing=timing, output=final,
                    explanation_faults=protocol.explanation_faults(final)))
                if not faults:
                    break
                feedback = '；'.join(faults)
            faults = rounds[-1]['faults']
            timings = [q['timing'] for q in rounds]
            measured = [q['tpot_ms'] for q in timings if q['tpot_ms'] is not None]
            usage_complete = all(q['prompt_tokens'] is not None and q['completion_tokens'] is not None for q in timings)
            row = dict(id=case['id'], seed=seed, delivery=final is not None,
                all_rule_pass=final is not None and not faults, round0_pass=not rounds[0]['faults'],
                rounds_used=len(rounds), **rule_flags(faults, final is not None), usage_complete=usage_complete,
                prompt_tokens=sum(q['prompt_tokens'] for q in timings) if usage_complete else None,
                completion_tokens=sum(q['completion_tokens'] for q in timings) if usage_complete else None,
                output_chars=sum(q['output_chars'] for q in timings), e2e_s=time.perf_counter()-started,
                tpot_ms=statistics.mean(measured) if measured else None, faults=faults, rounds=rounds)
            append_json(raw, row)
            rows.append(row)
            write(out/'status.json', dict(status='running', mode=mode, completed=len(rows), target=len(cases)*len(protocol.SEEDS),
                calls_used=teacher.calls, last_update=time.strftime('%Y-%m-%d %H:%M:%S')))
            print(json.dumps(dict(mode=mode, n=len(rows), passed=sum(r['all_rule_pass'] for r in rows), calls=teacher.calls)), flush=True)
    result = summary(rows, mode)
    write(folder/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/post_train/controlled_teacher_20261027')
    parser.add_argument('--max-calls', type=int, default=760)
    parser.add_argument('--modes', nargs='+', choices=['planner-only', 'rules-rework'], default=['planner-only'])
    args = parser.parse_args()
    load_local_env()
    model = os.getenv('OPENAI_MODEL') or os.getenv('AI_MODEL')
    url = (os.getenv('OPENAI_BASE_URL') or os.getenv('AI_BASE_URL') or '').rstrip('/') + '/chat/completions'
    key = os.getenv('OPENAI_API_KEY') or os.getenv('AI_API_KEY')
    if not model or not key or not url.startswith('https://'):
        raise RuntimeError('Missing teacher model, HTTPS base URL or key')
    protocol.verify()
    cases = json.loads((protocol.SUITE/'cases.json').read_text(encoding='utf8'))
    maps = json.loads((protocol.SUITE/'maps.json').read_text(encoding='utf8'))['routes']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    binding = dict(model=model, temperature=0.0, max_tokens=2048, concurrency=1, seeds=protocol.SEEDS,
        cases_sha256=hashlib.sha256((protocol.SUITE/'cases.json').read_bytes()).hexdigest(),
        schema_sha256=hashlib.sha256(json.dumps(TravelRoute.model_json_schema(), sort_keys=True).encode()).hexdigest(),
        max_rework=3, reasoning_effort='none', provider_cache='uncontrolled', transport='chat.completions SSE',
        latency_scope='remote API including network; not hardware-matched to local GPU', max_calls=args.max_calls)
    binding_path = out/'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text(encoding='utf8')) != binding:
        raise RuntimeError('Resume binding mismatch')
    write(binding_path, binding)
    write(out/'run_scope.json', dict(modes=args.modes, target=150*len(args.modes)))
    with exclusive_ledger(out/'run.jsonl'):
        teacher = Teacher(out, model, url, key, args.max_calls)
        try:
            results = {mode: run_mode(teacher, out, mode, cases, maps) for mode in args.modes}
            write(out/'summary.json', results)
            events = [json.loads(s) for s in teacher.ledger.read_text(encoding='utf8').splitlines()]
            completed_calls = [r for r in events if r['event'] == 'completed']
            write(out/'api_accounting.json', dict(attempted=teacher.calls,
                successful_calls=len(completed_calls), failed_calls=sum(r['event']=='failed' for r in events),
                abandoned_calls=sum(r['event']=='abandoned' for r in events),
                reported_prompt_tokens=sum((r.get('usage') or {}).get('prompt_tokens', 0) for r in completed_calls),
                reported_completion_tokens=sum((r.get('usage') or {}).get('completion_tokens', 0) for r in completed_calls),
                billing_note='Includes warmup. Failed/abandoned calls may incur unreported charges; no provider price assumed.'))
            columns = list(dict.fromkeys(k for result in results.values() for k in result))
            lines = ['# Teacher API controlled benchmark', '',
                'Same frozen cases and scorer as local models. 50 cases × 3 greedy repetitions; repetitions are not independent cases.', '',
                'API latency includes network and provider scheduling. Cache and hardware are not controlled. Token counts use the provider tokenizer.', '',
                'Completion counts include provider-reported reasoning tokens (also listed separately). Single-chunk responses have no measurable TPOT. Provider may generate reasoning despite reasoning_effort=none.', '',
                '| Metric | ' + ' | '.join(results) + ' |', '|---|' + '---:|'*len(results)]
            lines += ['| ' + k + ' | ' + ' | '.join(str(result.get(k, '—')) for result in results.values()) + ' |' for k in columns]
            (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
            write(out/'status.json', dict(status='complete', calls_used=teacher.calls, modes=args.modes, completed=sum(r['n'] for r in results.values())))
        except Exception as exc:
            write(out/'status.json', dict(status='stopped_on_error', calls_used=teacher.calls, error_type=type(exc).__name__, message=str(exc)))
            raise


if __name__ == '__main__':
    main()
