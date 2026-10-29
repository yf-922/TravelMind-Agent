"""Paired Base/SFT development check, using validation only, never test witnesses.

Fixed local vLLM endpoint, no paid API keys or external fallback. This diagnostic
does not replace the final four-model test or establish generalization.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.planning.schemas import TravelRoute
from app.evaluation.task_robustness import load_assets
from scripts.post_train_runtime import append_json, replay_check, quality_faults
from scripts.run_hard_post_train_synthesis import audit_plan


def score(plan, row, maps, hard_cases):
    if row['id'] in hard_cases:
        return audit_plan(plan, hard_cases[row['id']], maps)[0]
    errors, state = replay_check(plan, row['state'], maps)
    errors += quality_faults(state)
    if not any(d['spots'] for d in plan['days']):
        errors.append('empty_route')
    return sorted(set(errors))


def run(tag, model, output):
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT/'data/post_train/release_v1/validation.ledger.jsonl'
    rows = [json.loads(l) for l in source.read_text(encoding='utf-8').splitlines()]
    _, maps = load_assets()
    hard = {c['id']: c for c in json.loads((ROOT/'data/post_train/hard_supplement_cases.json').read_text(encoding='utf-8'))}
    binding = hashlib.sha256(source.read_bytes()+json.dumps(maps, sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()
    metadata = {'binding': binding, 'model': model, 'tag': tag, 'sample_count': len(rows),
        'suite': 'development_validation_not_final_test', 'temperature': .3, 'seed': 42, 'max_tokens': 2048}
    meta_path = output/f'{tag}.meta.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError('Validation binding changed; use a new output directory')
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    path = output/f'{tag}.jsonl'
    existing = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done = {r['id'] for r in existing}

    def worker(row):
        started = time.perf_counter()
        request = {'model': model, 'messages': row['messages'], 'temperature': .3, 'seed': 42,
            'max_tokens': 2048, 'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'TravelRoute', 'strict': False, 'schema': TravelRoute.model_json_schema()}}}
        result = {'id': row['id'], 'tag': tag, 'model': model, 'request': request,
            'parse_success': False, 'passed': False, 'faults': [], 'suite': 'validation_only'}
        try:
            req = urllib.request.Request('http://10.14.3.82:8000/v1/chat/completions',
                data=json.dumps(request, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'})
            # trust_env=False equivalent: no proxy and no paid endpoint fallback.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=90) as reply:
                raw = json.load(reply)
            result['response'] = raw
            result['usage'] = raw.get('usage')
            if raw['choices'][0]['finish_reason'] != 'stop':
                raise ValueError('incomplete_generation')
            plan = TravelRoute.model_validate_json(raw['choices'][0]['message']['content']).model_dump(mode='json')
            result['parse_success'] = True
            result['plan'] = plan
            result['faults'] = score(plan, row, maps, hard)
            result['passed'] = not result['faults']
        except Exception as exc:
            result['error_type'] = type(exc).__name__
        result['elapsed_s'] = time.perf_counter()-started
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, row) for row in rows if row['id'] not in done]
        for future in as_completed(futures):
            result = future.result()
            append_json(path, result)
            print(json.dumps({k: result.get(k) for k in ['id', 'tag', 'passed', 'faults', 'error_type']}), flush=True)
    results = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
    summary = {'tag': tag, 'n': len(results), 'parsed': sum(r['parse_success'] for r in results),
        'passed': sum(r['passed'] for r in results),
        'faults': dict(Counter(e for r in results for e in r['faults'])),
        'errors': sum('error_type' in r for r in results),
        'total_tokens': sum((r.get('usage') or {}).get('total_tokens', 0) for r in results),
        'scope': '30 same-source validation states; no independent test peeking'}
    (output/f'{tag}.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return summary


def compare(output):
    sets = {tag: {r['id']: r for r in [json.loads(l) for l in (output/f'{tag}.jsonl').read_text(encoding='utf-8').splitlines()]}
            for tag in ['base', 'sft']}
    if sets['base'].keys() != sets['sft'].keys() or len(sets['base']) != 30:
        raise ValueError('Incomplete or mismatched paired validation')
    improvements = [k for k in sets['base'] if not sets['base'][k]['passed'] and sets['sft'][k]['passed']]
    regressions = [k for k in sets['base'] if sets['base'][k]['passed'] and not sets['sft'][k]['passed']]
    result = {'improved': improvements, 'regressed': regressions,
        'base_passed': sum(r['passed'] for r in sets['base'].values()),
        'sft_passed': sum(r['passed'] for r in sets['sft'].values()),
        'n': 30, 'dpo_auto_start': False,
        'decision': 'Inspect paired cases and explanation quality before DPO; not a final four-model result'}
    (output/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', choices=['base', 'sft'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'data/post_train/sft_validation_v1')
    args = parser.parse_args()
    run(args.tag, args.model, args.output)
