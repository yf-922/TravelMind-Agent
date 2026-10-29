"""Run both current-protocol smokes before sequential v1/v2 SFT on GPU0."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/jiale/TravelMind-posttrain')
PYTHON = '/home/jiale/.venvs/travelmind-train-smoke/bin/python'
RUN = 'current-20261025'
RELEASE = ROOT/'data/post_train/release_current_protocol_v1_v2'
BASE = '/home/jiale/models/Qwen2.5-7B-Instruct'


def status(stage, **extra):
    row = dict(stage=stage, updated_at=time.time(), dpo_auto_start=False, **extra)
    target = ROOT/f'logs/{RUN}-status.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2))
    tmp.replace(target)
    print(json.dumps(row), flush=True)


def run_logged(command, name, env):
    with (ROOT/f'logs/{RUN}-{name}.log').open('x') as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, check=True)


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from scripts.launch_sft_4090_job import stop_owned_service
    manifest = json.loads((RELEASE/'training_manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Training input hash mismatch: '+relative)
    for tag in ['v1', 'v2']:
        for parent in ['saves', 'models']:
            path = ROOT/parent/f'TravelMind-Planner-{tag}-{RUN}'
            if path.exists():
                raise RuntimeError('Refusing to overwrite: '+str(path))
    if shutil.disk_usage(ROOT).free < 42*1024**3:
        raise RuntimeError('Need at least 42 GiB free before this two-model run')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    started = time.time()
    status('stopping_owned_base_service', started_at=started)
    restore = False
    completed = []
    try:
        restore = True
        stop_owned_service()
        # Validate exported records and select the longest records for memory stress.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
        roles = {'system': 'system', 'human': 'user', 'gpt': 'assistant'}
        lengths = {}
        shared_validation = None
        for tag, expected in [('v1', 156), ('v2', 250)]:
            folder = RELEASE/tag
            datasets = {}
            for split, count in [('train', expected), ('validation', 30)]:
                rows = json.loads((folder/f'travelmind_planner_sft_{split}.json').read_text())
                if len(rows) != count:
                    raise RuntimeError(f'{tag} {split} count mismatch')
                measured = []
                for row in rows:
                    messages = [dict(role=roles[m['from']], content=m['value'])
                                for m in row['conversations']]
                    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
                        raise RuntimeError('Expected Round-0 input and one final answer')
                    json.loads(messages[-1]['content'])
                    tokens = len(tokenizer.apply_chat_template(messages, tokenize=True,
                                                               add_generation_prompt=False))
                    if tokens > 3584:
                        raise RuntimeError('Export exceeds cutoff; refusing truncation')
                    measured.append((tokens, row))
                datasets[split] = [r for _, r in sorted(measured, key=lambda pair: pair[0], reverse=True)]
                lengths[f'{tag}_{split}'] = dict(count=len(rows), max=max(n for n, _ in measured),
                    tokens=sum(n for n, _ in measured))
                if split == 'validation':
                    if shared_validation is not None and rows != shared_validation:
                        raise RuntimeError('Shared validation changed between versions')
                    shared_validation = rows
            smoke = RELEASE/f'{tag}-smoke'
            smoke.mkdir(exist_ok=False)
            for split, count in [('train', 16), ('validation', 4)]:
                (smoke/f'travelmind_planner_sft_{split}.json').write_text(
                    json.dumps(datasets[split][:count], ensure_ascii=False))
            shutil.copyfile(folder/'dataset_info.json', smoke/'dataset_info.json')
        (ROOT/f'logs/{RUN}-lengths.json').write_text(json.dumps(lengths, indent=2))
        for tag in ['v1', 'v2']:
            status('smoke_training', version=tag, started_at=started)
            env['TRAVELMIND_TOKEN_LEDGER'] = str(ROOT/f'logs/{RUN}-{tag}-smoke-tokens.jsonl')
            run_logged([PYTHON, 'scripts/post_train_quality_entry.py', 'train',
                f'configs/post_train/qwen25_7b_planner_sft_{tag}_current_smoke.yaml'],
                f'{tag}-smoke', env)
            state = json.loads((ROOT/f'saves/TravelMind-Planner-{tag}-{RUN}-smoke/trainer_state.json').read_text())
            evals = [r['eval_loss'] for r in state['log_history'] if 'eval_loss' in r]
            if state['global_step'] != 5 or not evals or not all(math.isfinite(x) for x in evals):
                raise RuntimeError('Smoke did not complete five steps and finite validation')
            status('smoke_passed', version=tag, eval_loss=evals[-1], started_at=started)
        for tag in ['v1', 'v2']:
            status('full_sft_training', version=tag, completed=completed, started_at=started)
            env['TRAVELMIND_TOKEN_LEDGER'] = str(ROOT/f'logs/{RUN}-{tag}-full-tokens.jsonl')
            run_logged([PYTHON, 'scripts/post_train_quality_entry.py', 'train',
                f'configs/post_train/qwen25_7b_planner_sft_{tag}_current.yaml'],
                f'{tag}-full', env)
            if shutil.disk_usage(ROOT).free < 16*1024**3:
                raise RuntimeError('Insufficient free space for CPU merge')
            status('cpu_merge', version=tag, completed=completed, started_at=started)
            run_logged([PYTHON, 'scripts/post_train_train_entry.py', 'merge', '--base', BASE,
                '--adapter', str(ROOT/f'saves/TravelMind-Planner-{tag}-{RUN}'),
                '--output', str(ROOT/f'models/TravelMind-Planner-{tag}-{RUN}')],
                f'{tag}-merge', env)
            completed.append(tag)
        status('complete_pending_evaluation', completed=completed, started_at=started,
               elapsed_s=time.time()-started)
    except Exception as exc:
        status('failed', error=str(exc), completed=completed, started_at=started)
        raise
    finally:
        if restore:
            with open('/home/jiale/vllm_qwen.log', 'a') as stream:
                subprocess.Popen(['bash', str(ROOT/'scripts/serve_post_train_4090.sh')],
                    stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env=env, start_new_session=True)


if __name__ == '__main__':
    main()
