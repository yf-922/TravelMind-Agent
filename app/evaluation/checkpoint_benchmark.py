"""Measure real production prompt text and checkpoint restore, never simulated waits."""
from __future__ import annotations

import copy
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from app.planning import nodes
from app.planning.graph import restore_modification_state
from app.planning.schemas import TravelPlanState


class PromptCaptured(Exception):
    pass


def capture_prompt(task, state):
    captured = []

    class Capture:
        def invoke(self, messages):
            captured.extend(messages)
            raise PromptCaptured()

    with patch.object(nodes, 'build_structured_llm', return_value=Capture()):
        if task == 'intent':
            fn = nodes.make_intent_node(None)
        elif task == 'query_rewrite':
            fn = nodes.make_query_rewrite_node(None, None)
        elif task == 'planner':
            fn = nodes.make_joint_planner_node(None, knowledge_lookup=lambda *a: [])
        else:
            raise ValueError('unknown prompt task')
        with patch.object(nodes, 'invoke_structured', side_effect=lambda llm, messages, **kw: llm.invoke(messages)):
            try:
                fn(state)
            except PromptCaptured:
                pass
    if not captured:
        raise RuntimeError('production node did not assemble a prompt')
    return [{'role': role, 'content': content} for role, content in captured]


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values)*q)-1)]


def measure_checkpoint_prompts(ledger, *, samples=30, repetitions=3, encoding_name='cl100k_base'):
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    cases = [c for c in ledger['cases'] if c['suite'] == 'task']
    snapshots = [(c, ledger.get('seeds', {}).get(c['id'], {}).get('state')) for c in cases]
    snapshots = [(c, s) for c, s in snapshots if s and s.get('route') and s.get('pois')]
    if not 1 <= samples <= len(snapshots):
        raise ValueError(f'requested {samples} distinct checkpoints; only {len(snapshots)} available')
    if repetitions < 1:
        raise ValueError('repetitions must be positive')
    rows = []
    prompt_rows = []
    timings = {'full': [], 'local': []}
    for index, (case, saved) in enumerate(snapshots[:samples]):
        notes = ('把当天开始时间调整为10:00以后，其余当前要求保留。' if index % 2 == 0
                 else '下午优先选择候选池内的室内景点；没有合适室内候选时明确说明，不新增景点。')
        full_messages = local_messages = None
        for repeat in range(repetitions):
            # Alternate ordering; exclude tokenizer initialization from both timers.
            for arm in (('full', 'local') if repeat % 2 == 0 else ('local', 'full')):
                started = time.perf_counter()
                checkpoint = json.loads(json.dumps(saved, ensure_ascii=False))
                if arm == 'local':
                    state = restore_modification_state(checkpoint, notes)
                    messages = {'planner': capture_prompt('planner', state)}
                else:
                    query = checkpoint['query'] + '\n本次修改：' + notes
                    messages = {'intent': capture_prompt('intent', TravelPlanState(query=query))}
                    # Replay parsed facts, not synthetic model output generation.
                    state = TravelPlanState(**copy.deepcopy(checkpoint))
                    state.query = query
                    state.rewritten_query = query
                    messages['query_rewrite'] = capture_prompt('query_rewrite', state)
                    # Serialize actual saved weather and candidates as provider response replay.
                    state.weather_forecast = json.loads(json.dumps(checkpoint['weather_forecast']))
                    state.pois = json.loads(json.dumps(checkpoint['pois']))
                    state.route = []
                    state.planner_reviewer_dialogue = []
                    state.route_modify_opinion = None
                    messages['planner'] = capture_prompt('planner', state)
                timings[arm].append((time.perf_counter()-started)*1000)
                if arm == 'local':
                    local_messages = messages
                else:
                    full_messages = messages
        def count(groups):
            return {task: sum(len(encoding.encode(m['content'], disallowed_special=())) for m in messages)
                    for task, messages in groups.items()}
        full_tokens, local_tokens = count(full_messages), count(local_messages)
        prompt_rows.append({'id': case['id'], 'full': full_messages, 'local': local_messages})
        rows.append({'id': case['id'], 'modification': notes, 'checkpoint_sha256': hashlib.sha256(
            json.dumps(saved, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            'full_text_tokens_by_node': full_tokens, 'local_text_tokens_by_node': local_tokens,
            'full_text_tokens': sum(full_tokens.values()), 'local_text_tokens': sum(local_tokens.values()),
            'avoided_front_llm_invocations': len(set(full_messages) - set(local_messages)),
            'full_prompt_invocations': len(full_messages), 'local_prompt_invocations': len(local_messages),
            'avoided_external_query_categories': ['weather', 'poi'],
            'avoided_actual_provider_requests': None})
    full_total = sum(r['full_text_tokens'] for r in rows)
    local_total = sum(r['local_text_tokens'] for r in rows)
    summarize = lambda values: {'p50': round(percentile(values, .5), 4),
                                 'p95': round(percentile(values, .95), 4),
                                 'mean': round(statistics.mean(values), 4)}
    return {'measurement_type': 'production-prompt-capture-and-application-checkpoint-restore',
            'samples': samples, 'runs': repetitions, 'timed_runs_per_arm': samples*repetitions,
            'source_binding': ledger.get('binding'), 'encoding': encoding_name,
            'tokenizer_version': __import__('importlib.metadata', fromlist=['version']).version('tiktoken'),
            'prompt_capture_date': date.today().isoformat(),
            'text_input_tokens': {'full': full_total, 'local': local_total,
                                  'reduction_ratio': 1-local_total/full_total},
            'avoided_front_llm_invocations': sum(row['avoided_front_llm_invocations'] for row in rows),
            'avoided_external_query_categories_per_request': ['weather', 'poi'],
            'full_replay_ms': summarize(timings['full']),
            'checkpoint_local_replan_ms': summarize(timings['local']),
            'paired_samples_ms': timings, 'rows': rows, '_prompts': prompt_rows,
            'external_io_latency': None, 'end_to_end_latency_reduction_ratio': None,
            'avoided_nodes': ['intent', 'query_rewrite', 'weather_search', 'attraction_search'],
            'pass': len(rows) == samples,
            'limitations': [
                'Checkpoint seeds are saved real model drafts over fixed provider fixtures, not live prefetch traces.',
                'Token counts cover actual production system/human prompt text, not tool schemas, wire framing or billed model tokens.',
                'cl100k_base is an explicitly named comparison tokenizer, not a verified Grok billing tokenizer.',
                'Full replay reuses stored parsed intent/rewrite fields; it does not regenerate them.',
                'Local timing measures Python restore/serialization/prompt assembly only; no model/API waits, no sleep.',
                'External categories are skipped by design; exact provider request counts are unknown.',
            ]}
