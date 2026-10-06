"""Real-provider authenticated SSE smoke using temporary application storage."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.eval_safety import require_call_budget, require_external_calls


def parse_events(body):
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith('data: ')]


def evaluate(query, timeout, modification=None, confirmation_modification=None):
    from fastapi.testclient import TestClient
    from app.core import database
    from app.multi_agent_core.memory import SQLiteAgentMemoryStore

    async def no_background(*args, **kwargs):
        return None

    with tempfile.TemporaryDirectory() as directory:
        private_store = SQLiteAgentMemoryStore(Path(directory) / 'private.db')
        with patch.object(database, '_DB_PATH', Path(directory) / 'application.db'), \
             patch.dict(os.environ, {'SEMANTIC_MEMORY_ENABLED': '0', 'PLAN_TIMEOUT_SECONDS': str(timeout)}):
            database.init_db()
            import app.main as main
            with patch('app.multi_agent_core.memory.SQLiteAgentMemoryStore', return_value=private_store), \
                 patch.object(main, 'run_profile_update_agent', no_background), \
                 patch.object(main, 'run_semantic_memory_update', no_background), TestClient(main.app) as client:
                from app.core.auth import create_token
                with database.get_conn() as conn:
                    conn.execute('INSERT INTO users VALUES (?,?,?,?)', ('smoke-owner', 'smoke-owner', 'unused', '2026-10-06'))
                headers = {'Authorization': 'Bearer ' + create_token('smoke-owner')}
                started = time.perf_counter()
                response = client.post('/api/plan/stream', json={
                    'query': query, 'engine': 'supervisor', 'max_review_rounds': 1,
                    'max_spots': 8, 'max_per_day': 2,
                }, headers=headers)
                initial_elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                events = parse_events(response.text)
                terminal = next((e for e in reversed(events) if e['type'] in {'result', 'error', 'modification_warning'}), {})
                plan = terminal.get('plan') or {}
                run_id = response.headers.get('X-Agent-Run-ID')
                trace = main.agent_runs.get_owned(run_id, 'smoke-owner') if run_id else None
                saved = False
                isolation = None
                constraints_preserved = None
                if terminal.get('plan_id'):
                    detail = client.get('/api/history/' + terminal['plan_id'], headers=headers)
                    saved = detail.status_code == 200
                    checkpoint = detail.json().get('planner_state') or {}
                    constraints_preserved = 'max_walking_km' in checkpoint and 'rain_indoor_priority' in checkpoint
                    isolation = client.get('/api/history/' + terminal['plan_id'], headers={
                        'Authorization': 'Bearer ' + create_token('other-owner'),
                    }).status_code == 403
                session = checkpoint.get('memory_session_id') if saved else None
                from app.multi_agent_core.runtime import memory_scope
                scope = memory_scope('smoke-owner', session) if session else 'smoke-owner'
                follow_ups = []
                original_checkpoint = checkpoint if saved else None
                current_plan_id = terminal.get('plan_id')
                for notes in (modification, confirmation_modification):
                    if not notes or not current_plan_id:
                        continue
                    before = len(private_store.load(scope, 'planner'))
                    follow_started = time.perf_counter()
                    changed = client.post('/api/plan/stream', json={
                        'query': query, 'engine': 'supervisor', 'plan_id': current_plan_id,
                        'modification_notes': notes, 'max_review_rounds': 1,
                        'max_spots': 8, 'max_per_day': 2,
                    }, headers=headers)
                    changed_events = parse_events(changed.text)
                    changed_terminal = next((e for e in reversed(changed_events) if e['type'] in {'result', 'error', 'modification_warning'}), {})
                    confirmation = None
                    if changed_terminal.get('type') == 'modification_warning':
                        pending_id = changed_terminal.get('pending_id')
                        denied = client.post('/api/plan/confirm_modification', json={'pending_id': pending_id}, headers={
                            'Authorization': 'Bearer ' + create_token('other-owner'),
                        })
                        confirmed = client.post('/api/plan/confirm_modification', json={
                            'pending_id': pending_id, 'parent_plan_id': current_plan_id,
                        }, headers=headers)
                        confirm_events = parse_events(confirmed.text)
                        confirmation = {
                            'other_user_denied': denied.status_code == 404,
                            'http_status': confirmed.status_code,
                            'nodes': [e['node'] for e in confirm_events if e['type'] == 'stage'],
                        }
                        changed_terminal = next((e for e in reversed(confirm_events) if e['type'] in {'result', 'error'}), {})
                    next_checkpoint = None
                    if changed_terminal.get('plan_id'):
                        current_plan_id = changed_terminal['plan_id']
                        next_checkpoint = client.get('/api/history/' + current_plan_id, headers=headers).json().get('planner_state') or {}
                    follow_ups.append({
                        'elapsed_ms': round((time.perf_counter() - follow_started) * 1000, 2),
                        'outcome': changed_terminal.get('type'), 'success': changed_terminal.get('success'),
                        'error_code': changed_terminal.get('code'),
                        'nodes': [e['node'] for e in changed_events if e['type'] == 'stage'],
                        'confirmation': confirmation,
                        'planner_memory_before': before, 'planner_memory_after': len(private_store.load(scope, 'planner')),
                        'same_memory_session': next_checkpoint.get('memory_session_id') == session if next_checkpoint else None,
                        'candidate_pool_reused': next_checkpoint.get('pois') == original_checkpoint.get('pois') if next_checkpoint else None,
                        'route_changed': next_checkpoint.get('route') != original_checkpoint.get('route') if next_checkpoint else None,
                        'route_issues': (changed_terminal.get('plan') or {}).get('route_issues'),
                    })
                    print(json.dumps({'follow_up': follow_ups[-1]}, ensure_ascii=True), flush=True)
                return {
                    'http_status': response.status_code, 'elapsed_ms': initial_elapsed_ms,
                    'workflow_elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
                    'outcome': terminal.get('type'), 'success': terminal.get('success'),
                    'error_code': terminal.get('code'), 'nodes': [e['node'] for e in events if e['type'] == 'stage'],
                    'approved': plan.get('approved'), 'days_count': len(plan.get('days') or []),
                    'degraded_services': plan.get('degraded_services'),
                    'route_issues': plan.get('route_issues'),
                    'quality_gate': plan.get('quality_gate'),
                    'saved_history_verified': saved, 'saved_constraints_verified': constraints_preserved,
                    'selected_attractions': [spot.get('name') for day in (checkpoint.get('route') or []) for spot in day.get('spots', [])] if saved else [],
                    'indoor_sources': {poi.get('name'): poi.get('indoor_source') for poi in checkpoint.get('pois', [])} if saved else {},
                    'other_user_denied': isolation,
                    'private_memory_counts': {role: len(private_store.load(scope, role)) for role in ('planner', 'reviewer', 'time_check')},
                    'trace': trace,
                    'follow_ups': follow_ups,
                    'scope': 'Authenticated ASGI API + real production nodes/providers + temporary SQLite. Chroma and background preference extraction disabled; not a browser/network deployment benchmark.',
                }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', default='南京2026年10月10日一日游，想看历史文化，每天最多两个景点，晚上不安排景点')
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--allow-external-calls', action='store_true')
    parser.add_argument('--max-llm-calls', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--modification')
    parser.add_argument('--confirmation-modification')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation' / 'supervisor_full_api.json')
    args = parser.parse_args()
    budget = 60 + (60 if args.modification else 0) + (120 if args.confirmation_modification else 0)
    if args.dry_run:
        print(json.dumps({'provider_attempts_upper_bound': budget, 'includes': 'intent/rewrite/planning/audit/meal/tips plus optional modification/confirmation; live map API calls additional'}))
        return
    require_external_calls(parser, allowed=args.allow_external_calls, operation='Full Supervisor API smoke')
    require_call_budget(parser, estimated_upper_bound=budget, maximum=args.max_llm_calls)
    report = evaluate(args.query, args.timeout, args.modification, args.confirmation_modification)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
