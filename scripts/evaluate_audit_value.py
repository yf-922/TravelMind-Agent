"""Real-model paired audit ablation; fixed upstream, shared first Planner draft."""
from contextlib import ExitStack
from datetime import date
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.planning import nodes
from app.planning.schemas import TravelPlanState
from app.evaluation.rag_protocol import fingerprint
from app.evaluation.route_interest import route_key

HARD = {'duplicate_poi', 'unknown_poi', 'route_structure', 'habit_constraint',
        'opening_time_conflict', 'weather_outdoor_conflict', 'walking_constraint',
        'modification_time_unfulfilled', 'indoor_constraint'}


def assets():
    """Use city pools only, never the frozen test queries or their labels."""
    source = json.loads((ROOT/'knowledge/route_interest_v1/cases.json').read_text(encoding='utf-8'))
    cases = []
    for city in ('南京', '上海', '三亚'):
        pois = next(c['pois'] for c in source if c['destination'] == city)
        indoor = next((p for p in pois if p.get('indoor') is True), pois[0])
        outdoor = next((p for p in pois if p.get('indoor') is False), pois[-1])
        profiles = [
            ('ordinary', '一天最多两个景点，上午10点开始。', {}),
            ('history', '喜欢历史文化，避免游乐设施。', {'attraction_preference': '历史文化'}),
            ('nature', '喜欢自然风光，避免历史展览。', {'attraction_preference': '自然风光'}),
            ('late', '不喜欢早起，10点后开始。', {'habit_preference': '不喜欢早起'}),
            ('slow', '慢节奏，每天最多两个不同景点。', {'habit_preference': '慢节奏'}),
            ('indoor', '只安排室内景点。', {'pois': [indoor, outdoor]}),
            ('rain', '雨天避免露天游玩，优先室内。', {'rain_indoor_priority': True,
             'weather_forecast': [{'date': '2026-10-15', 'is_bad': True, 'day_weather': '雨', 'night_weather': '雨', 'day_temp': '20', 'night_temp': '18'}]}),
            ('single', '最多一个景点，认真游览。', {'max_per_day': 1}),
            ('window', '请把行程改为10:00到15:00游玩，保留午餐休息。', {'modification_notes': '改为10:00到15:00游玩'}),
            ('family', '带孩子，希望内容容易理解，不要安排太密。', {'attraction_preference': '亲子科普', 'habit_preference': '慢节奏'}),
        ]
        for label, request, extra in profiles:
            cases.append({'id': f'{city}-{label}', 'expected': 'feasible',
                          'state': {'query': f'{city}一日游。'+request, 'destination': city,
                                    'days': 1, 'max_per_day': 2, 'pois': pois,
                                    'travel_start_date': '2026-10-15', 'travel_end_date': '2026-10-15',
                                    'weather_forecast': [{'date': '2026-10-15', 'is_bad': False, 'day_weather': '晴', 'night_weather': '晴', 'day_temp': '20', 'night_temp': '18'}],
                                    **extra}})
    # Predeclared impossibility, scoped to this frozen pool/window (not all travel).
    for i in range(10):
        city = ('南京', '上海', '三亚')[i % 3]
        base = dict(next(c['state'] for c in cases if c['state']['destination'] == city))
        if i < 3:
            base.update(query=f'{city}一日游，必须至少安排一个真实候选景点。候选池已穷尽，禁止补查或编造。', pois=[])
            reason = '候选池为空，无法安排至少一个真实景点'
        elif i < 6:
            outdoor = next(p for p in base['pois'] if p.get('indoor') is False)
            base.update(query=f'{city}一日游，只能安排室内景点。候选池仅有户外景点且禁止扩展。', pois=[outdoor])
            reason = '只允许室内，但唯一候选为户外'
        else:
            poi = {**base['pois'][0], 'open_time': '17:00-18:00'}
            base.update(query=f'{city}一日游，必须在10:00到11:00游览唯一候选景点，禁止扩展候选池。',
                        modification_notes='改为10:00到11:00游玩', pois=[poi])
            reason = '唯一候选17点开放，与必须游玩时段不相交'
        cases.append({'id': f'impossible-{i:02d}', 'expected': 'reject', 'impossibility_reason': reason, 'state': base})
    return cases


def objective(state):
    flags = sorted(set(nodes._route_risk_flags(state)) & HARD)
    names = [s['name'] for d in state.route for s in d.get('spots', [])]
    return {'route_valid': bool(names) and not flags, 'hard_violations': flags,
            'selected': names, 'approved': state.approved}


def outcome(state, case):
    result = objective(state)
    plan = nodes.finalize_node(state)['final_plan']
    messages = plan.get('route_issues', [])
    result.update(final_approved=plan['approved'], user_messages=messages,
                  user_failure_explained=any('无法生成满足' in str(m) for m in messages),
                  rework=max(0, state.review_round-1), final_plan=plan)
    if case.get('quality'):
        from scripts.reviewer_quality_cases import quality_grade
        result['quality'] = quality_grade(state, case)
    if case['expected'] == 'reject':
        result['unsafe_release'] = bool(plan['approved'])
        result['correct_rejection'] = not plan['approved'] and result['user_failure_explained']
    return result


def experiment(case, arm, initial, planner, reviewer, time_check):
    state = TravelPlanState(**initial)
    if arm == 'planner_only':
        state.approved = True  # Unaudited baseline releases its generated draft.
        return state
    if arm == 'rules_only':
        state = state.model_copy(update=nodes.route_risk_gate_node(state))
        state.approved = objective(state)['route_valid']
        return state
    for iteration in range(3):
        state = state.model_copy(update=nodes.route_distance_check_node(state))
        state = state.model_copy(update=nodes.route_risk_gate_node(state))
        if arm in ('full', 'forced_reviewer') and (state.review_required or arm == 'forced_reviewer'):
            state = state.model_copy(update=reviewer(state))
        else:
            state.approved = objective(state)['route_valid']
        # Explicit time-only comparator isolates marginal Reviewer benefit.
        if state.time_check_required:
            state = state.model_copy(update=time_check(state))
        if state.approved or iteration == 2:
            break
        state = state.model_copy(update=planner(state))
    return state


def summary(report):
    output = {}
    for arm in ('planner_only', 'rules_only', 'rules_time', 'full', 'forced_reviewer'):
        rows = [r for r in report['rows'] if r['arm'] == arm]
        feasible = [r for r in rows if r['expected'] == 'feasible']
        negative = [r for r in rows if r['expected'] == 'reject']
        output[arm] = {'feasible_planned': sum(c['expected'] == 'feasible' for c in report.get('cases', [])) or 30, 'feasible_recorded': len(feasible),
                       'valid_routes': sum(r.get('outcome', {}).get('route_valid', False) for r in feasible),
                       'successful_approved_routes': sum(r.get('outcome', {}).get('route_valid', False) and r.get('outcome', {}).get('final_approved', False) for r in feasible),
                       'negative_planned': 10, 'negative_recorded': len(negative),
                       'unsafe_releases': sum(r.get('outcome', {}).get('unsafe_release', False) for r in negative),
                       'rejected_with_user_error': sum(r.get('outcome', {}).get('correct_rejection', False) for r in negative),
                       'errors': sum(bool(r.get('error')) for r in rows)}
        output[arm]['quality_passed'] = sum(r.get('outcome', {}).get('quality', {}).get('passed', False) for r in rows)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--out', type=Path, default=ROOT/'data/eval/audit_value_real.json')
    p.add_argument('--max-calls', type=int, default=240)
    p.add_argument('--supplemental', action='store_true', help='Two predeclared feasible replacements; original failed cases stay in original report')
    p.add_argument('--reviewer-quality', action='store_true', help='Controlled preference/pace/weather/commute drafts, with clean controls')
    p.add_argument('--case-ids', nargs='+', help='Explicit unexecuted cases, written to a separate report')
    a = p.parse_args()
    load_local_env()
    cases = assets()
    if a.reviewer_quality:
        from scripts.reviewer_quality_cases import build_cases
        cases = build_cases()
    if a.case_ids:
        missing = set(a.case_ids)-{c['id'] for c in cases}
        if missing: p.error('unknown case IDs: '+str(missing))
        cases = [c for c in cases if c['id'] in a.case_ids]
    if a.supplemental:
        base = next(c['state'] for c in cases if c['id'] == '三亚-ordinary')
        cases = [{'id': '三亚-'+label, 'expected': 'feasible',
                  'state': {**base, 'query': '三亚一日游。'+query, **extra}}
                 for label, query, extra in [
                     ('island', '想体验海岛自然风光，最多安排一个景点。', {'max_per_day': 1}),
                     ('coast', '想看海岸地貌，慢节奏，每天最多两个景点。', {'habit_preference': '慢节奏'})]]
    model = os.getenv('GROK_MODEL', 'grok-4.6')
    report = {'status': 'dry_run', 'method': 'real_joint_planner_shared_draft_paired_audit_tail',
              'model': model, 'cases': cases, 'asset_fingerprint': fingerprint(cases),
              'code_fingerprint': fingerprint({str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in [Path(__file__), ROOT/'app/planning/nodes.py', ROOT/'app/planning/helpers.py']}),
              'calls_limit': a.max_calls, 'cost_status': 'unpriced', 'rows': [], 'calls': [],
              'labels': 'author_draft_hard_constraints_not_human_gold',
              'limits': 'Fixed upstream and replay. No live search, optional tips/meals or external Judge. One run per case; no statistical stability claim. 30 cases do not relabel old 39.'}
    if a.out.exists():
        previous = json.loads(a.out.read_text(encoding='utf-8'))
        if previous['asset_fingerprint'] != fingerprint(cases): raise ValueError('assets changed')
        if previous['code_fingerprint'] != report['code_fingerprint']:
            raise ValueError('implementation changed; use a separate report, never silently resume')
        report = previous
        if any(c['status'] == 'in_flight' for c in report['calls']): raise RuntimeError('unsettled call; no automatic replay')
    def save():
        report['summary'] = summary(report)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'{len(cases)} requests; shared initial drafts; actual call cap={a.max_calls}; cost=unpriced; no model retries', flush=True)
    if not a.execute:
        save(); return
    from app.llm.grok import build_chat_grok, use_grok_responses_api
    from scripts.compare_route_interest import read_gateway_usage
    read_gateway_usage(os.environ['GROK_BASE_URL'].rstrip('/'), os.environ['GROK_API_KEY'])
    current = {'case': '', 'arm': ''}
    def factory(schema, **kw):
        client = build_chat_grok(model=model, temperature=kw.get('temperature', 0))
        client.max_tokens = 2048
        client.max_retries = 0
        structured = client.with_structured_output(schema, method='json_schema' if use_grok_responses_api() else 'function_calling', strict=True, include_raw=True)
        class Metered:
            def invoke(self, messages):
                if len(report['calls']) >= a.max_calls: raise RuntimeError('call cap reached')
                entry = {**current, 'task': kw.get('task_type'), 'status': 'in_flight'}
                report['calls'].append(entry); save()
                start = time.perf_counter()
                try:
                    response = structured.invoke(messages)
                    entry.update(status='completed', usage=response['raw'].usage_metadata)
                    if not entry['usage']: raise RuntimeError('actual usage missing')
                    if response.get('parsing_error') or response.get('parsed') is None: raise ValueError('invalid structured result')
                    return response['parsed']
                except Exception as exc:
                    entry.update(status='failed', error=type(exc).__name__+': '+str(exc)[:200]); raise
                finally:
                    entry['elapsed_ms'] = (time.perf_counter()-start)*1000; save()
        return Metered()
    from functools import partial
    maps = json.loads((ROOT/'knowledge/route_interest_v1/maps.json').read_text(encoding='utf-8'))['routes']
    with ExitStack() as stack:
        stack.enter_context(patch.object(nodes, 'build_structured_llm', factory))
        stack.enter_context(patch.object(nodes, 'invoke_structured', lambda llm, messages, **kw: llm.invoke(messages)))
        stack.enter_context(patch.object(nodes, 'amap_key', return_value='replay'))
        stack.enter_context(patch.object(nodes, 'plan_route_distance', side_effect=lambda origin, destination, key, mode='drive': maps.get(route_key(origin, destination, mode))))
        stack.enter_context(patch.object(nodes, 'plan_transport_leg', return_value=None))
        stack.enter_context(patch.object(nodes, 'recommend_chain_hotel', return_value=None))
        stack.enter_context(patch.object(nodes, 'lookup_live_ticket_prices', return_value={}))
        planner = nodes.make_joint_planner_node(model, knowledge_lookup=lambda *a: [])
        reviewer = nodes.make_reviewer_node(model)
        time_check = nodes.make_time_check_node(model)
        report['status'] = 'running'; save()
        for case in cases:
            current.update(case=case['id'], arm='initial')
            initial = report.setdefault('initial_states', {}).get(case['id'])
            if not initial:
                start = TravelPlanState(**case['state'])
                try:
                    if case.get('quality'):
                        start.review_round = 1
                    else:
                        start = start.model_copy(update=planner(start))
                    initial = start.model_dump(mode='json')
                    report['initial_states'][case['id']] = initial; save()
                except Exception as exc:
                    report['status'] = 'stopped_on_provider_failure'; report['stop_reason'] = str(exc)[:250]; save(); return
            arms = ('planner_only', 'rules_only', 'full', 'forced_reviewer') if a.reviewer_quality else ('planner_only', 'rules_only', 'rules_time', 'full')
            for arm in arms:
                if any(r['id'] == case['id'] and r['arm'] == arm for r in report['rows']): continue
                current.update(arm=arm)
                row = {'id': case['id'], 'expected': case['expected'], 'arm': arm}
                start_time = time.perf_counter()
                before = len(report['calls'])
                try:
                    state = experiment(case, arm, initial, planner, reviewer, time_check)
                    row['outcome'] = outcome(state, case)
                except Exception as exc:
                    row['error'] = type(exc).__name__+': '+str(exc)[:250]
                row.update(elapsed_ms=(time.perf_counter()-start_time)*1000,
                           extra_calls=len(report['calls'])-before)
                report['rows'].append(row); save()
                print(case['id'], arm, row.get('outcome', {}).get('route_valid'), row.get('error', ''), flush=True)
                if row.get('error') or any(c['status'] == 'failed' for c in report['calls'][before:]):
                    report['status'] = 'stopped_on_provider_failure'; save(); return
    report['status'] = 'complete_pending_human_review'; save()


if __name__ == '__main__': main()
