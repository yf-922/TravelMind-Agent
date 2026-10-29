"""Adopt running A without restarting it, then train B for 96 total steps."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/jiale/TravelMind-posttrain')
PYTHON = '/home/jiale/.venvs/travelmind-train-smoke/bin/python'
RUN = 'count-ab-total96-20261025'
DATA = ROOT/'data/post_train/count_infeasible_ab_20261025'
A = ROOT/'saves/TravelMind-count-ab-20261025-A'
V2 = ROOT/'saves/TravelMind-Planner-v2-current-20261025'
B = ROOT/f'saves/TravelMind-{RUN}-B'


def status(stage, **extra):
    row = dict(stage=stage, updated_at=time.time(), formal_steps=dict(A_predecessor=48, A_new=48, B=96),
               smoke_weights_used=False, dpo_auto_start=False, **extra)
    path = ROOT/f'logs/{RUN}-status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2))
    tmp.replace(path)
    print(json.dumps(row), flush=True)


def read_state(folder, steps):
    state = json.loads((folder/'trainer_state.json').read_text())
    if state['global_step'] != steps:
        raise RuntimeError(f'{folder.name}: expected {steps} steps, got {state["global_step"]}')
    metrics = [float(v) for r in state['log_history'] for k,v in r.items()
               if k.startswith('eval_') and k.endswith('loss')]
    if not metrics or not all(math.isfinite(x) for x in metrics):
        raise RuntimeError('Missing or nonfinite validation metrics')
    if not (folder/'adapter_model.safetensors').exists():
        raise RuntimeError('Missing final adapter')
    return state


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from scripts.launch_sft_4090_job import stop_owned_service
    amendment = json.loads((DATA/'matched_total_steps.json').read_text())
    for relative, expected in amendment['sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Amended experiment binding mismatch: '+relative)
    if B.exists():
        raise RuntimeError('Refusing to overwrite B')
    pid = amendment['adopt_A_pid']
    proc = Path('/proc')/str(pid)
    started = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TRAVELMIND_TOKEN_LEDGER=str(ROOT/f'logs/{RUN}-B-full-tokens.jsonl'))
    try:
        status('adopting_A_training', A_pid=pid, started_at=started)
        # The old supervisor alone was stopped; A's optimizer and process are unchanged.
        while proc.exists():
            try:
                command = (proc/'cmdline').read_bytes()
                if not command or b'count_ab_A.yaml' not in command:
                    break
                if proc.stat().st_uid != os.getuid():
                    raise RuntimeError('A PID owner changed')
            except FileNotFoundError:
                break
            if time.time()-started > 3600:
                raise RuntimeError('A completion timeout; inspect original A log')
            time.sleep(5)
        predecessor = read_state(V2, 48)
        a_state = read_state(A, 48)
        if predecessor['best_model_checkpoint'] != str(V2.relative_to(ROOT)/'checkpoint-48'):
            raise RuntimeError('A predecessor is not the recorded 48-step v2 checkpoint')
        status('A_complete_B_loading', A_total_steps=96, started_at=started)
        stop_owned_service()
        with (ROOT/f'logs/{RUN}-B-full.log').open('x') as stream:
            subprocess.run([PYTHON, 'scripts/post_train_quality_entry.py', 'train',
                'configs/post_train/count_ab_B_total96.yaml'], env=env,
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
        b_state = read_state(B, 96)
        totals = dict(A=predecessor['global_step']+a_state['global_step'], B=b_state['global_step'])
        if totals != {'A':96,'B':96}:
            raise RuntimeError('Formal total steps do not match')
        status('complete_pending_evaluation', verified_total_steps=totals, started_at=started,
               elapsed_s=time.time()-started)
    except Exception as exc:
        status('failed', error=str(exc), started_at=started)
        raise
    finally:
        # Do not race an adopted training process with a restored inference server.
        if not proc.exists():
            with open('/home/jiale/vllm_qwen.log', 'a') as stream:
                subprocess.Popen(['bash', str(ROOT/'scripts/serve_post_train_4090.sh')],env=env,
                    stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)


if __name__=='__main__':
    main()
