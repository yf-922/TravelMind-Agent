"""Read-only run audit; never prints keys or full requests/responses."""
from collections import Counter
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # Ignore a concurrent writer's unfinished final line.
    return rows


def audit():
    root = ROOT/'data/post_train'
    synth = read(root/'raw_synthesis_ledger.jsonl')
    calls = read(root/'raw_synthesis_ledger.calls.jsonl')
    evaluation = read(root/'eval/api_planner-only.jsonl')
    eval_calls = read(root/'eval/api_planner-only.calls.jsonl')
    terminal = [r for r in calls if r['event'] != 'started']
    starts = [r for r in calls if r['event'] == 'started']
    settled = {r['call_id'] for r in terminal}
    pending = [r for r in starts if r['call_id'] not in settled]
    accepted = [r for r in synth if r.get('status') == 'accepted']
    infeasible_rain = sum(bool(r['state'].get('rain_indoor_priority')) and
        sum(p.get('indoor') is True for p in r['state']['pois']) < r['state']['days'] for r in synth)
    unique = {json.dumps(r['messages'], sort_keys=True, ensure_ascii=False) for r in accepted}
    latency = sorted(r['elapsed_s'] for r in terminal)
    result = {'time': time.time(), 'synthesis_recorded': len(synth), 'synthesis_status': dict(Counter(r['status'] for r in synth)),
        'recorded_infeasible_rain_states': infeasible_rain,
        'accepted_unique_messages': len(unique), 'accepted_cities': dict(Counter(r['state']['destination'] for r in accepted)),
        'first_round_faults': dict(Counter(f for r in synth for f in (r.get('rounds') or [{}])[0].get('rule_errors', []))),
        'chosen_multi_spot_day_rate': sum(any(len(d['spots']) > 1 for d in r['chosen']['days']) for r in accepted)/max(1, len(accepted)),
        'natural_preference_pairs': sum(r.get('preference_source') == 'natural_repair' for r in accepted),
        'synthesis_call_started': len(starts), 'synthesis_call_completed': sum(r['event'] == 'completed' for r in terminal),
        'synthesis_call_failed': sum(r['event'] == 'failed' for r in terminal), 'inflight': len(pending),
        'oldest_inflight_s': max([time.time() - r['time'] for r in pending], default=0),
        'latency_p50_s': latency[len(latency)//2] if latency else None,
        'known_synthesis_tokens': sum((r.get('usage') or {}).get('total_tokens', 0) for r in terminal),
        'api_eval_recorded': len(evaluation), 'api_eval_success': sum(bool(r.get('grade', {}).get('task_success')) for r in evaluation),
        'api_eval_errors': sum(bool(r.get('error')) for r in evaluation),
        'known_api_eval_tokens': sum((r.get('usage') or {}).get('total_tokens', 0) for r in eval_calls if r.get('event') != 'started'),
        'api_error_types': dict(Counter(r.get('error_type') for r in terminal if r['event'] == 'failed'))}
    result['total_started_including_preflight'] = len(starts) + sum(r['event'] == 'started' for r in eval_calls) + len([r for r in read(root/'api_preflight.calls.jsonl') if r['event'] == 'started'])
    out = root/'progress_snapshot.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return result


if __name__ == '__main__':
    audit()
