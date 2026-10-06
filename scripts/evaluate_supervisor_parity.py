"""Bounded real-model comparison of planning/audit runtimes on frozen facts."""

import argparse
import asyncio
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.llm_usage import bind_run, take_run_snapshot, unbind_run
from app.core.eval_safety import require_call_budget, require_external_calls
from app.multi_agent_core.memory import SQLiteAgentMemoryStore
from app.multi_agent_core.runtime import TravelSupervisor, production_nodes
from app.planning.schemas import TravelPlanState
from tests.eval.graders.code_graders import grade_code
from tests.eval.harness import build_state_from_fixture, load_fixtures


def supervisor_outcome(events):
    if not events:
        raise ValueError('Supervisor returned no events')
    last = events[-1]
    if last['type'] == 'modification_warning':
        return TravelPlanState(**last['pending_state']), 'requires_confirmation'
    if last['type'] != 'result' or 'checkpoint' not in last:
        raise ValueError('Supervisor did not return a final checkpoint')
    return TravelPlanState(**last['checkpoint']), 'result'


def call_budget(engines):
    return 90 * ("langgraph" in engines) + 18 * ("supervisor" in engines)


async def evaluate(case_id, timeout, engines=('langgraph', 'supervisor'), force_review=False):
    fixture = load_fixtures(case_id)[0]
    state = build_state_from_fixture(fixture)
    state.max_review_rounds = 1
    state.max_time_check_rounds = 2
    if force_review:
        state.route_modify_opinion = "本次机制验收必须独立审核当前路线是否满足约束"
    nodes = production_nodes(state.model_name)
    # Isolate planning, road verification and audit. No meal/ticket/hotel calls.
    nodes['meal_enrichment'] = lambda s: {}
    nodes['spot_tips'] = lambda s: {}
    nodes['finalize'] = lambda s: {'final_plan': {'approved': s.approved, 'days': s.route}}
    records = []
    with tempfile.TemporaryDirectory() as directory:
        for engine in engines:
            started = time.perf_counter()
            token = bind_run('parity-' + engine)
            stages = []
            try:
                if engine == 'supervisor':
                    runtime = TravelSupervisor(nodes, SQLiteAgentMemoryStore(Path(directory) / 'memory.db'), engine)
                    async def consume():
                        events = []
                        async for event in runtime.stream(state.model_copy(deep=True), modification=True):
                            events.append(event)
                            if event.get('type') == 'stage':
                                stages.append(event['node'])
                        return supervisor_outcome(events)
                    final, outcome = await asyncio.wait_for(consume(), timeout)
                else:
                    import app.planning.graph as graph
                    with patch.object(graph, 'make_planner_node', lambda *a: nodes['planner']), \
                         patch.object(graph, 'make_reviewer_node', lambda *a: nodes['reviewer']), \
                         patch.object(graph, 'make_time_check_node', lambda *a: nodes['time_check']), \
                         patch.object(graph, 'make_meal_enrichment_node', lambda *a: nodes['meal_enrichment']), \
                         patch.object(graph, 'make_spot_tips_node', lambda *a: nodes['spot_tips']), \
                         patch.object(graph, 'make_finalize_node', lambda *a: nodes['finalize']):
                        compiled = graph.build_modification_graph(state.model_name)
                        result = await asyncio.wait_for(compiled.ainvoke(state.model_copy(deep=True), config={'recursion_limit': 30}), timeout)
                        final = TravelPlanState(**result)
                        outcome = 'result'
                grading = grade_code(final, fixture)
                records.append({'engine': engine, 'completed': True, 'approved': final.approved,
                                'outcome': outcome,
                                'nodes': stages, 'risk_flags': final.route_risk_flags,
                                'grading': grading, 'review_round': final.review_round,
                                'time_check_round': final.time_check_round})
            except Exception as error:
                records.append({'engine': engine, 'completed': False, 'error_type': type(error).__name__,
                                'error_location': [f'{f.name}:{f.lineno}' for f in traceback.extract_tb(error.__traceback__)[-4:]]})
            finally:
                unbind_run(token)
            records[-1]['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 2)
            records[-1]['usage'] = take_run_snapshot('parity-' + engine)
            print(json.dumps(records[-1], ensure_ascii=True), flush=True)
    return {'case_id': case_id, 'records': records,
            'scope': 'real-model planning/audit and live road API over identical frozen POI/weather; enrichment omitted',
            'boundary': 'One run per engine; sequential execution; no statistical quality or latency claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='nanjing-1d-sunny-nightlife')
    parser.add_argument('--allow-external-calls', action='store_true')
    parser.add_argument('--timeout', type=float, default=150)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-llm-calls', type=int)
    parser.add_argument('--engine', choices=('both', 'langgraph', 'supervisor'), default='both')
    parser.add_argument('--force-review', action='store_true', help='Require an independent audit; not a natural-request quality trial')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation' / 'supervisor_parity_smoke.json')
    args = parser.parse_args()
    # Main graph: conservatively allow one model call per recursion step (30).
    # Supervisor: two planner/reviewer/time-check rounds (6). Each structured
    # invocation may retry three times. This is a preflight estimate, not a limiter.
    engines = ('langgraph', 'supervisor') if args.engine == 'both' else (args.engine,)
    upper = call_budget(engines)
    budget = {'provider_attempts_upper_bound': upper, 'engines': len(engines),
              'scope': 'frozen POI/weather; live model and route APIs'}
    if args.dry_run:
        print(json.dumps(budget))
        return
    require_external_calls(parser, allowed=args.allow_external_calls, operation='Supervisor parity')
    require_call_budget(parser, estimated_upper_bound=upper, maximum=args.max_llm_calls)
    report = asyncio.run(evaluate(args.case, args.timeout, engines, args.force_review))
    report['forced_review'] = args.force_review
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
