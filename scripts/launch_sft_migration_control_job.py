"""GPU0-only versioned SFT job; never overwrite v1 or start DPO."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.launch_sft_4090_job import stop_owned_service

ROOT = Path('/home/jiale/TravelMind-posttrain')
PYTHON = '/home/jiale/.venvs/travelmind-train-smoke/bin/python'
MODEL = ROOT/'models/TravelMind-Planner-7B-SFT-migration-control'
ADAPTER = ROOT/'saves/TravelMind-Planner-SFT-migration-control'


def status(stage, **values):
    row = dict(values)
    row.update(stage=stage, updated_at=time.time(), dpo_auto_start=False)
    path = ROOT/'logs/sft-migration-control-status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2))
    tmp.replace(path)
    print(json.dumps(row), flush=True)


def main():
    os.chdir(ROOT)
    (ROOT/'logs').mkdir(exist_ok=True)
    release = ROOT/'data/post_train/release_migration_control'
    audit = json.loads((release/'audit.json').read_text())
    for name, sha in audit['sha256'].items():
        if hashlib.sha256((release/name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('release hash mismatch: '+name)
    if audit['over_cutoff'] or not audit['validation_ledger_unchanged']:
        raise RuntimeError('release did not pass audits')
    if ADAPTER.exists() or MODEL.exists() or (release/'TRAINING_STARTED.json').exists():
        raise RuntimeError('Version already exists; refusing overwrite')
    started = time.time()
    binding = dict(started_at=started, train=audit['train'], validation=audit['validation'],
        sha256=audit['sha256'], base='/home/jiale/models/Qwen2.5-7B-Instruct', dpo_auto_start=False)
    with (release/'TRAINING_STARTED.json').open('x') as stream:
        json.dump(binding, stream, indent=2)
    status('stopping_base_service', **binding)
    stop_owned_service()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    done = threading.Event()
    def monitor():
        with (ROOT/'logs/sft-migration-control-gpu.jsonl').open('a') as stream:
            while not done.is_set():
                r = subprocess.run(['nvidia-smi', '-i', '0',
                    '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
                stream.write(json.dumps({'time': time.time(), 'gpu0': r.stdout.strip()})+'\n')
                stream.flush()
                done.wait(5)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    code = -1
    try:
        status('sft_training', **binding)
        with (ROOT/'logs/sft-migration-control-training.log').open('w') as stream:
            code = subprocess.run([PYTHON, 'scripts/post_train_train_entry.py', 'train',
                'configs/post_train/qwen25_7b_planner_sft_migration_control.yaml'], env=env,
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        if code:
            raise RuntimeError('SFT training failed; inspect sft-migration-control-training.log')
        status('cpu_merge', **binding)
        with (ROOT/'logs/sft-migration-control-merge.log').open('w') as stream:
            code = subprocess.run([PYTHON, 'scripts/post_train_train_entry.py', 'merge',
                '--base', binding['base'], '--adapter', str(ADAPTER), '--output', str(MODEL)], env=env,
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        if code:
            raise RuntimeError('SFT merge failed; inspect sft-migration-control-merge.log')
    finally:
        done.set()
        monitor_thread.join(timeout=6)
        with open('/home/jiale/vllm_qwen.log', 'a') as stream:
            service = subprocess.Popen(['bash', str(ROOT/'scripts/serve_post_train_4090.sh')],
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        status('sft_complete_pending_validation' if code == 0 else 'sft_failed', exit_code=code,
            elapsed_s=time.time()-started, base_service_pid=service.pid,
            base_service_readiness='starting_not_yet_verified', **binding)


if __name__ == '__main__':
    main()
