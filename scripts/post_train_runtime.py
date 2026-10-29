"""Shared offline checks and metered API boundary; no live map/reviewer calls."""
from __future__ import annotations
import copy
import json
import os
import threading
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.planning import nodes
from app.planning.schemas import TravelPlanState, TravelRoute
from app.evaluation.task_robustness import load_assets, grade
from app.evaluation.route_interest import route_key


def append_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def replay_check(plan, initial, maps, fault=None):
    """Inspect RAW model output, before production normalization can hide defects."""
    parsed = TravelRoute.model_validate(plan)
    state = TravelPlanState(**initial)
    state.route = [d.model_dump() for d in parsed.days]
    state.meal_slots = [s.model_dump() for s in parsed.meal_slots]
    state.route_distance_legs = []
    locations = {p['name']: p.get('location') for p in state.pois + state.main_meal_candidates}
    errors = []
    mode = 'walk' if state.max_walking_km is not None else 'drive'
    from app.evaluation.task_robustness import fault_distance
    provider = fault_distance(maps, fault, {})
    for day in state.route:
        events = list(day['spots']) + [dict(s, name=s['restaurant_name']) for s in state.meal_slots if s['day'] == day['day']]
        events.sort(key=lambda s: s['start_time'])
        for a, b in zip(events, events[1:]):
            origin, dest = locations.get(a['name']), locations.get(b['name'])
            leg = None
            if origin and dest:
                try:
                    leg = provider(origin, dest, 'offline', mode=mode)
                except (TimeoutError, RuntimeError):
                    pass
            if not leg or leg.get('duration_min') is None:
                errors.append('road_unverified')
            if leg:
                state.route_distance_legs.append(dict(leg, day=day['day'], **{'from': a['name'], 'to': b['name']}))
                if leg.get('duration_min') is not None and nodes._clock_minutes(b['start_time']) - nodes._clock_minutes(a['end_time']) < float(leg['duration_min']):
                    errors.append('commute_time_conflict')
    errors += list(set(nodes._route_risk_flags(state)) - {'opening_time_unknown', 'user_modification', 'long_road_leg'})
    state.final_plan = {'approved': True, 'days': [{'day': d['day'], 'timeline': [dict(s, type='attraction') for s in d['spots']]} for d in state.route]}
    return sorted(set(errors)), state


def quality_faults(state):
    """Only explicit, observable constraints. No invented stair/commercial labels."""
    faults = []
    query = state.query
    by_name = {p['name']: p for p in state.pois}
    names = [s['name'] for d in state.route for s in d['spots']]
    for p in state.pois:
        if f'必须参观「{p["name"]}」' in query and p['name'] not in names:
            faults.append('required_poi_missing')
        if f'不要参观「{p["name"]}」' in query and p['name'] in names:
            faults.append('excluded_poi')
    for d in state.route:
        if '每天最多两个景点' in query and len(d['spots']) > 2:
            faults.append('pace_count')
        for s in d['spots']:
            start, end = nodes._clock_minutes(s['start_time']), nodes._clock_minutes(s['end_time'])
            if '每个至少90分钟' in query and end - start < 90:
                faults.append('visit_too_short')
            if '10:00后开始' in query and start < 600:
                faults.append('late_start')
    from app.planning.constraints import heat_violations, effective_daily_limit
    if any(len(d['spots']) > effective_daily_limit(state) for d in state.route):
        faults.append('pace_count')
    if heat_violations(state, state.route):
        faults.append('heat_exposure_or_unknown_indoor')
    return sorted(set(faults))


def capture_messages(state):
    captured = []
    class Captured(Exception):
        pass
    def capture(llm, messages):
        captured.extend({'role': {'human': 'user'}.get(role, role), 'content': content} for role, content in messages)
        raise Captured()
    # Existing injectable boundary constructs the exact production text, with RAG disabled.
    planner = nodes.make_joint_planner_node(None, llm=object(), invoke_fn=capture, knowledge_lookup=lambda *a: [])
    try:
        planner(TravelPlanState(**state))
    except Captured:
        pass
    if not captured:
        raise RuntimeError('production message capture failed')
    return captured


class Meter:
    def __init__(self, path, model, max_calls=1000, timeout=90):
        self.path, self.model = Path(path), model
        self.lock = threading.Lock()
        self.count = sum(1 for l in self.path.read_text(encoding='utf-8').splitlines() if json.loads(l).get('event') == 'started') if self.path.exists() else 0
        self.max_calls = max_calls
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get('AI_API_KEY') or os.environ.get('OPENAI_API_KEY'),
            base_url=os.environ.get('AI_BASE_URL') or os.environ.get('OPENAI_BASE_URL') or os.environ.get('OPENAI_API_BASE'),
            timeout=timeout, max_retries=0)

    def call(self, messages, identity):
        payload = {'model': self.model, 'messages': messages, 'temperature': 0.3,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'TravelRoute', 'strict': False, 'schema': TravelRoute.model_json_schema()}}}
        with self.lock:
            if self.count >= self.max_calls:
                raise RuntimeError('MAX_LLM_CALLS reached')
            self.count += 1
            call_id = self.count
            append_json(self.path, {'event': 'started', 'call_id': call_id, 'identity': identity,
                'time': time.time(), 'request': payload})
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**payload)
            raw = response.model_dump(mode='json')
            text = response.choices[0].message.content
            plan = TravelRoute.model_validate_json(text).model_dump(mode='json')
            record = {'event': 'completed', 'call_id': call_id, 'identity': identity, 'response': raw,
                'elapsed_s': time.perf_counter() - start, 'usage': raw.get('usage'), 'parse_success': True}
        except Exception as exc:
            record = {'event': 'failed', 'call_id': call_id, 'identity': identity,
                'elapsed_s': time.perf_counter() - start, 'error_type': type(exc).__name__,
                'http_status': getattr(exc, 'status_code', None), 'parse_success': False}
            if 'raw' in locals():
                record['response'] = raw
                record['usage'] = raw.get('usage')
            with self.lock:
                append_json(self.path, record)
            raise RuntimeError(f"API call {call_id}: {record['error_type']} status={record['http_status']}") from None
        with self.lock:
            append_json(self.path, record)
        return plan, record
