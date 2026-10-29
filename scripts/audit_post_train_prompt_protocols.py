"""Read-only comparison of exported training prompts and recorded eval requests."""
import collections
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.post_train_runtime import capture_messages

DATA = ROOT / 'data/post_train'
OUT = DATA / 'prompt_protocol_audit_20261025'
RELEASES = dict(v1='release_v1', v2='release_v2', v3='release_v3',
                mc='release_migration_control', v4_clean='release_v4_clean',
                v4='release_v4', large='v2_large_pool_teacher_contract_v3/sft_candidate')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    OUT.mkdir(exist_ok=True)
    cases = json.loads((DATA/'representative_eval50_v1/cases.json').read_text(encoding='utf-8'))
    frozen_system = cases[0]['messages'][0]['content']
    current_system = capture_messages(cases[0]['state'])[0]['content']
    systems = {digest(frozen_system): frozen_system, digest(current_system): current_system}
    result = {'scope': 'Fixed instructions must match; case-specific facts naturally differ.',
              'frozen_eval_system_sha256': digest(frozen_system),
              'current_production_system_sha256': digest(current_system),
              'frozen_eval_human_matches_current': sum(c['messages'][-1]['content'] == capture_messages(c['state'])[-1]['content'] for c in cases),
              'training': {}, 'evaluation': {}}
    server_files = []
    for tag, release in RELEASES.items():
        folder = DATA/release
        name = 'travelmind_planner_sft' if tag == 'large' else 'travelmind_planner_sft_train'
        info = json.loads((folder/'dataset_info.json').read_text(encoding='utf-8'))[name]
        path = folder/info['file_name']
        rows = json.loads(path.read_text(encoding='utf-8'))
        counts = collections.Counter()
        messages = []
        for row in rows:
            msgs = [{'role': {'human': 'user', 'gpt': 'assistant'}.get(m['from'], m['from']),
                     'content': m['value']} for m in row['conversations'][:-1]]
            messages.append(msgs)
            system = '\n'.join(m['content'] for m in msgs if m['role'] == 'system')
            systems[digest(system)] = system
            counts[digest(system)] += 1
        ledger_path = folder/('synthesis.ledger.jsonl' if tag == 'large' else 'train.ledger.jsonl')
        ledger = read_rows(ledger_path)
        ledger_messages = {json.dumps(r['messages'], sort_keys=True, ensure_ascii=False) for r in ledger}
        recapture = collections.Counter()
        human_matches = 0
        for row in ledger:
            try:
                current = capture_messages(row['state'])
                recapture['exact_match' if current == row['messages'] else 'different'] += 1
                human_matches += current[-1]['content'] == row['messages'][-1]['content']
            except Exception as exc:
                recapture['error_'+type(exc).__name__] += 1
        result['training'][tag] = dict(file=str(path.relative_to(ROOT)), count=len(rows),
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), system_hash_counts=dict(counts),
            matching_frozen_eval_system=counts[digest(frozen_system)],
            matching_current_production_system=counts[digest(current_system)],
            exported_inputs_found_in_ledger=sum(json.dumps(m, sort_keys=True, ensure_ascii=False) in ledger_messages for m in messages),
            current_production_recapture=dict(recapture), human_matches_current=human_matches,
            human_contract_count=sum('planner-contract-v1' in m[-1]['content'] for m in messages))
        server_files.extend([path.relative_to(ROOT).as_posix(), (folder/'dataset_info.json').relative_to(ROOT).as_posix()])
    for tag in ['base', *RELEASES]:
        path = DATA/'representative_eval50_v1/runs'/f'all_versions_v2_{tag}'/'results.jsonl'
        rows = read_rows(path)
        by_id = {c['id']: c for c in cases}
        requests = [r.get('request') for r in rows]
        request_hashes = collections.Counter(digest(json.dumps({k: v for k, v in req.items() if k != 'model'},
                                                             sort_keys=True, ensure_ascii=False)) for req in requests if req)
        result['evaluation'][tag] = dict(count=len(rows), requests_recorded=sum(bool(r) for r in requests),
            messages_match_frozen_cases=sum(r.get('request', {}).get('messages') == by_id[r['id']]['messages'] for r in rows),
            system_hash_counts=dict(collections.Counter(digest(r['request']['messages'][0]['content']) for r in rows if r.get('request'))),
            request_hash_multiset=dict(request_hashes))
    remote_code = ('import json,hashlib\nfrom pathlib import Path\np=Path("/home/jiale/TravelMind-posttrain")\n'
                   'files='+repr(server_files)+'\n'
                   'print(json.dumps({s:hashlib.sha256((p/s).read_bytes()).hexdigest() if (p/s).exists() else None for s in files}))\n')
    remote = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'jiale-10.14.3.82', 'python3 -'], input=remote_code,
                            capture_output=True, text=True, timeout=60)
    if remote.returncode:
        result['server_verification'] = {'error': remote.stderr[:1000]}
    else:
        hashes = json.loads(remote.stdout)
        result['server_verification'] = {s: {'server_sha256': h,
            'matches_local': h == hashlib.sha256((ROOT/s).read_bytes()).hexdigest()} for s, h in hashes.items()}
    for h, text in systems.items():
        (OUT/(h+'.system.txt')).write_text(text, encoding='utf-8')
        (OUT/(h+'.vs_frozen.diff')).write_text(''.join(difflib.unified_diff(
            frozen_system.splitlines(True), text.splitlines(True), fromfile='frozen_eval', tofile=h)), encoding='utf-8')
    (OUT/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'evaluation'}, ensure_ascii=False, indent=2))
    for tag, value in result['evaluation'].items():
        print(tag, json.dumps({k: v for k, v in value.items() if k != 'request_hash_multiset'}, ensure_ascii=False))
    print('all_eval_request_multisets_equal', all(v['request_hash_multiset'] == result['evaluation']['base']['request_hash_multiset'] for v in result['evaluation'].values()))


if __name__ == '__main__':
    main()
