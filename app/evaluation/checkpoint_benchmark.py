"""Capture production prompts and probe API modification routing offline."""
from __future__ import annotations

import copy
from datetime import date
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
from unittest.mock import Mock, patch

from app.planning import nodes
from app.planning.graph import restore_modification_state
from app.planning.modification import refresh_explicit_modification_candidates
from app.planning.schemas import TravelPlanState
from app.providers.amap.poi import poi_to_spot


class PromptCaptured(Exception):
    pass


MODIFICATION_SCENARIOS = (
    {'id': 'start_time', 'label': '调整出发时间'},
    {'id': 'replace_poi', 'label': '替换单个景点'},
    {'id': 'rain_indoor', 'label': '改为室内避雨'},
    {'id': 'niche_candidate', 'label': '新增池外小众场馆并定向补查'},
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _scenario_for_checkpoint(index, checkpoint):
    scenario = dict(MODIFICATION_SCENARIOS[index])
    scenario.update(applicable=True, inapplicable_reason=None, requested_poi=None)
    pois = checkpoint['pois']
    selected = [spot.get('name') for day in checkpoint['route']
                for spot in day.get('spots') or [] if spot.get('name')]
    if scenario['id'] == 'start_time':
        scenario['notes'] = '把当天开始时间调整为10:00以后，其余当前要求保留。'
    elif scenario['id'] == 'replace_poi':
        replacement = next((p['name'] for p in pois if p.get('name') and p['name'] not in selected), None)
        scenario['notes'] = f'把第1天的“{selected[0] if selected else "原景点"}”换成“{replacement or "未找到可替换候选"}”，其他安排不变，不新增景点。'
        scenario['requested_poi'] = replacement
        if not selected or not replacement:
            scenario.update(applicable=False, inapplicable_reason='no_unused_replacement_in_pool')
            scenario['notes'] = '希望替换第1天的单个景点，但当前没有可替换的池内候选；不新增景点，请明确说明候选不足。'
    elif scenario['id'] == 'rain_indoor':
        indoor = next((p['name'] for p in pois if p.get('indoor') is True and p.get('name')), None)
        scenario['notes'] = '预计下雨，请把下午改为原候选池内的室内景点，不新增景点；没有已知室内候选时说明候选不足。'
        scenario['requested_poi'] = indoor
        if not indoor:
            scenario.update(applicable=False, inapplicable_reason='no_verified_indoor_candidate')
    else:
        # Fictional venue facts test lookup routing, never real POI existence.
        name = '离线Mock非遗手作博物馆'
        while any(p.get('name') == name for p in pois):
            name += '分馆'
        scenario['requested_poi'] = name
        scenario['notes'] = f'新增小众非遗手作需求，加入“{name}”，先核验并定向补查，不要重新搜索原有候选。'
    return scenario


def _prepare_offline_pair(saved, scenario):
    """Exercise API lookup decisions once; timer repetitions do not rerun lookups."""
    raw = {'name': scenario['requested_poi'], 'location': '118.8,32.0',
           'type': '博物馆', 'id': 'offline-fixture-not-a-real-provider-poi'}
    targeted_lookup = Mock(return_value=[raw])
    local_checkpoint, notes, trace = refresh_explicit_modification_candidates(
        copy.deepcopy(saved), scenario['notes'], lookup=targeted_lookup,
        key_provider=lambda: 'offline-no-api-key',
    )
    # Independent full-prefetch fixture: a broken local merge cannot
    # silently contaminate both arms and pass the candidate parity check.
    full_pool = copy.deepcopy(saved['pois'])
    if scenario['id'] == 'niche_candidate':
        full_pool = [poi_to_spot(raw), *full_pool]
    weather_prefetch = Mock(return_value=copy.deepcopy(saved.get('weather_forecast', [])))
    poi_prefetch = Mock(return_value=full_pool)
    full_checkpoint = copy.deepcopy(saved)
    full_checkpoint['weather_forecast'] = weather_prefetch()
    full_checkpoint['pois'] = poi_prefetch()
    local_counts = {'weather_prefetch': 0, 'poi_prefetch': 0,
                    'poi_targeted': targeted_lookup.call_count}
    full_counts = {'weather_prefetch': weather_prefetch.call_count,
                   'poi_prefetch': poi_prefetch.call_count, 'poi_targeted': 0}
    return full_checkpoint, local_checkpoint, notes, trace, full_counts, local_counts


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
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)]


def _token_summary(rows):
    full = sum(row['full_text_tokens'] for row in rows)
    local = sum(row['local_text_tokens'] for row in rows)
    ratios = [1 - row['local_text_tokens'] / row['full_text_tokens'] for row in rows]
    return {'full': full, 'local': local, 'reduction_ratio': 1 - local / full if full else None,
            'mean_per_request_reduction_ratio': statistics.mean(ratios) if ratios else None,
            'p50_per_request_reduction_ratio': percentile(ratios, .5) if ratios else None}


def measure_checkpoint_prompts(ledger, *, samples=30, repetitions=3, encoding_name='cl100k_base'):
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    cases = [c for c in ledger['cases'] if c['suite'] == 'task']
    snapshots = [(c, ledger.get('seeds', {}).get(c['id'], {}).get('state')) for c in cases]
    snapshots = [(c, s) for c, s in snapshots if s and s.get('route') and s.get('pois')]
    if len({c['id'] for c, _ in snapshots}) != len(snapshots):
        raise ValueError('duplicate checkpoint case IDs')
    if not 1 <= samples <= len(snapshots):
        raise ValueError(f'requested {samples} distinct checkpoints; only {len(snapshots)} available')
    if repetitions < 1:
        raise ValueError('repetitions must be positive')
    rows, prompt_rows = [], []
    timings = {'full': [], 'local': []}
    for case, saved in snapshots[:samples]:
        for index in range(len(MODIFICATION_SCENARIOS)):
            scenario = _scenario_for_checkpoint(index, saved)
            before = fingerprint(saved)
            full_cp, local_cp, notes, trace, full_calls, local_calls = _prepare_offline_pair(saved, scenario)
            pool_equal = full_cp['pois'] == local_cp['pois']
            if not pool_equal or fingerprint(saved) != before:
                raise RuntimeError('benchmark pool parity or checkpoint isolation violated')
            full_messages = local_messages = None
            row_timings = {'full': [], 'local': []}
            for repeat in range(repetitions):
                for arm in (('full', 'local') if repeat % 2 == 0 else ('local', 'full')):
                    started = time.perf_counter()
                    if arm == 'local':
                        checkpoint = json.loads(json.dumps(local_cp, ensure_ascii=False))
                        state = restore_modification_state(checkpoint, notes)
                        messages = {'planner': capture_prompt('planner', state)}
                    else:
                        checkpoint = json.loads(json.dumps(full_cp, ensure_ascii=False))
                        query = checkpoint['query'] + '\n本次修改：' + notes
                        messages = {'intent': capture_prompt('intent', TravelPlanState(query=query))}
                        # Replay parsed facts; no model output is regenerated.
                        state = TravelPlanState(**checkpoint)
                        state.query = state.rewritten_query = query
                        state.modification_notes = notes
                        messages['query_rewrite'] = capture_prompt('query_rewrite', state)
                        state.route = []
                        state.planner_reviewer_dialogue = []
                        state.route_modify_opinion = None
                        messages['planner'] = capture_prompt('planner', state)
                    elapsed = (time.perf_counter() - started) * 1000
                    timings[arm].append(elapsed)
                    row_timings[arm].append(elapsed)
                    if arm == 'local':
                        local_messages = messages
                    else:
                        full_messages = messages

            def count(groups):
                return {task: sum(len(encoding.encode(m['content'], disallowed_special=())) for m in messages)
                        for task, messages in groups.items()}

            full_tokens, local_tokens = count(full_messages), count(local_messages)
            row_id = case['id'] + ':' + scenario['id']
            prompt_rows.append({'id': row_id, 'full': full_messages, 'local': local_messages})
            rows.append({
                'id': row_id, 'checkpoint_case_id': case['id'], 'scenario': scenario['id'],
                'destination': saved.get('destination'), 'modification': scenario['notes'],
                'effective_modification': notes, 'applicable': scenario['applicable'],
                'inapplicable_reason': scenario['inapplicable_reason'],
                'cache_pool_sufficient': scenario['applicable'] and not local_calls['poi_targeted'],
                'mock_lookup_trace': trace, 'mock_provider_calls': {'full': full_calls, 'local': local_calls},
                'mocked_targeted_lookup_calls': local_calls['poi_targeted'],
                'identical_candidate_pool': pool_equal, 'candidate_pool_sha256': fingerprint(full_cp['pois']),
                'checkpoint_sha256': before,
                'full_text_tokens_by_node': full_tokens, 'local_text_tokens_by_node': local_tokens,
                'full_text_tokens': sum(full_tokens.values()), 'local_text_tokens': sum(local_tokens.values()),
                'avoided_front_llm_invocations': len(set(full_messages) - set(local_messages)),
                'full_prompt_invocations': len(full_messages), 'local_prompt_invocations': len(local_messages),
                'avoided_external_query_categories': ['weather_prefetch', 'poi_prefetch'],
                'avoided_actual_provider_requests': None, 'timings_ms': row_timings,
            })
    summaries = []
    for scenario in MODIFICATION_SCENARIOS:
        subset = [row for row in rows if row['scenario'] == scenario['id']]
        tokens = _token_summary(subset)
        summaries.append({**scenario, 'requests': len(subset),
                          'applicable_requests': sum(row['applicable'] for row in subset),
                          'cache_pool_sufficient_requests': sum(row['cache_pool_sufficient'] for row in subset),
                          'mocked_targeted_lookup_calls': sum(row['mocked_targeted_lookup_calls'] for row in subset),
                          'full_text_tokens': tokens['full'], 'local_text_tokens': tokens['local'],
                          'prompt_token_reduction_ratio': tokens['reduction_ratio'],
                          'applicable_request_tokens': _token_summary([row for row in subset if row['applicable']])})
    summarize = lambda values: {'p50': round(percentile(values, .5), 4),
                                 'p95': round(percentile(values, .95), 4),
                                 'mean': round(statistics.mean(values), 4)}
    return {
        'schema_version': 2,
        'measurement_type': 'production-prompt-capture-and-application-checkpoint-restore',
        'samples': samples, 'modifications_per_checkpoint': len(MODIFICATION_SCENARIOS),
        'modification_requests': len(rows), 'runs': repetitions, 'timed_runs_per_arm': len(rows) * repetitions,
        'source_binding': ledger.get('binding'), 'encoding': encoding_name,
        'tokenizer_version': importlib.metadata.version('tiktoken'),
        'prompt_capture_date': date.today().isoformat(), 'text_input_tokens': _token_summary(rows),
        'avoided_front_llm_invocations': sum(row['avoided_front_llm_invocations'] for row in rows),
        'avoided_external_query_categories_per_request': ['weather_prefetch', 'poi_prefetch'],
        'modification_scenarios': summaries,
        'mocked_targeted_lookup': {'calls': sum(row['mocked_targeted_lookup_calls'] for row in rows),
                                 'real_provider_calls': 0},
        'mock_provider_calls': {arm: {kind: sum(row['mock_provider_calls'][arm][kind] for row in rows)
                                     for kind in ('weather_prefetch', 'poi_prefetch', 'poi_targeted')}
                                for arm in ('full', 'local')},
        'first_three_cache_hit_tokens': _token_summary([row for row in rows if row['cache_pool_sufficient']]),
        'full_replay_ms': summarize(timings['full']),
        'checkpoint_local_replan_ms': summarize(timings['local']),
        'paired_samples_ms': timings, 'rows': rows, '_prompts': prompt_rows,
        'external_io_latency': None, 'end_to_end_latency_reduction_ratio': None,
        'avoided_nodes': ['intent', 'query_rewrite', 'weather_search', 'attraction_search'],
        'pass': len(rows) == samples * len(MODIFICATION_SCENARIOS) and all(row['identical_candidate_pool'] for row in rows),
        'limitations': [
            'Saved real model drafts use fixed provider fixtures, not live prefetch traces.',
            'All four modifications run independently on every checkpoint; unsupported cache-only edits stay in the report.',
            'API-shared quoted-name lookup logic is exercised; generic unquoted new themes and Supervisor LLM routing are not measured.',
            'Full prefetch mocks each return one fixture; call counts do not estimate real HTTP pagination, cache hits or retries.',
            'Targeted POIs are fictional, explicitly marked offline fixtures. They cannot be treated as real travel knowledge.',
            'Token counts cover production system/human prompt text, not tool schemas, wire framing, model outputs or billed tokens.',
            'cl100k_base is a comparison tokenizer, not a verified Grok billing tokenizer.',
            'Full replay reuses parsed facts; it does not regenerate intent or query rewrite outputs.',
            'Repeated timings cover restore/serialization/prompt assembly only; mock lookups run once before timing.',
            'No generated route, edit success, approval, final quality or end-to-end latency is measured.',
        ],
    }
