"""Continue frozen C-SFT into C-DPO on GPU0; never calls external APIs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/jiale/TravelMind-posttrain')
PY = '/home/jiale/.venvs/travelmind-train-smoke/bin/python'
RUN = ROOT / 'logs/c-posttrain-20261028'
RUN.mkdir(parents=True, exist_ok=True)
STATUS = RUN / 'status.json'
SFT_PID = ROOT / 'logs/c-sft-training.pid'
SFT = ROOT / 'saves/TravelMind-C-SFT-20261028'
DPO = ROOT / 'saves/TravelMind-C-DPO-20261028'
PARENT = ROOT / 'models/TravelMind-C-SFT-DPO-parent-20261028'


def status(stage: str, **extra) -> None:
    row = dict(stage=stage, updated_at=time.time(), teacher_api_calls=0, **extra)
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(row, ensure_ascii=False, indent=2))
    temp.replace(STATUS)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def digest_tree(path: Path) -> dict:
    files = sorted(path.glob('*.safetensors'))
    if not files:
        raise RuntimeError('No safetensors found under ' + str(path))
    return {'path': str(path), 'bytes': sum(x.stat().st_size for x in files),
            'files': [{'name': x.name, 'bytes': x.stat().st_size,
                       'sha256': hashlib.sha256(x.read_bytes()).hexdigest()} for x in files]}


def verify_training(path: Path, expected_steps: int) -> dict:
    state = json.loads((path / 'trainer_state.json').read_text())
    if state.get('global_step') != expected_steps:
        raise RuntimeError(f'{path.name}: expected {expected_steps} optimizer steps, got {state.get("global_step")}')
    if not (path / 'adapter_model.safetensors').is_file():
        raise RuntimeError('Missing adapter weights: ' + str(path))
    losses = [float(v) for row in state.get('log_history', []) for k, v in row.items()
              if k in {'loss', 'eval_loss', 'train_loss'}]
    if not losses or any(not __import__('math').isfinite(v) for v in losses):
        raise RuntimeError('Missing/non-finite training losses: ' + str(path))
    return {'global_step': state['global_step'], 'loss_entries': len(losses),
            'last_loss': losses[-1], 'adapter_sha256': hashlib.sha256((path / 'adapter_model.safetensors').read_bytes()).hexdigest()}


def wait_for_sft() -> dict:
    if not SFT_PID.is_file():
        raise RuntimeError('C-SFT PID marker missing')
    pid = int(SFT_PID.read_text())
    started = time.time()
    while True:
        stat = Path(f'/proc/{pid}/stat')
        alive = stat.exists() and stat.read_text().split()[2] != 'Z'
        if not alive:
            break
        if time.time() - started > 6 * 3600:
            raise TimeoutError('C-SFT exceeded six-hour guard')
        time.sleep(15)
    return verify_training(SFT, 96)


def run() -> None:
    os.chdir(ROOT)
    if STATUS.exists():
        raise RuntimeError('C training continuation status already exists; inspect before restart')
    started = time.time()
    try:
        status('waiting_for_c_sft', pid=int(SFT_PID.read_text()))
        sft_metrics = wait_for_sft()
        if PARENT.exists() or DPO.exists():
            raise RuntimeError('C-DPO parent/output exists; refuse overwrite')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   TRAVELMIND_TOKEN_LEDGER=str(RUN / 'c-dpo-token-ledger.jsonl'))
        status('merging_c_sft_parent', c_sft=sft_metrics)
        with (RUN / 'c-sft-parent-merge.log').open('x') as stream:
            subprocess.run([PY, 'scripts/post_train_train_entry.py', 'merge', '--base',
                            '/home/jiale/models/Qwen2.5-7B-Instruct', '--adapter', str(SFT), '--output', str(PARENT)],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        parent_hash = digest_tree(PARENT)
        (RUN / 'c-dpo-lineage.json').write_text(json.dumps({
            'c_sft_metrics': sft_metrics, 'base': '/home/jiale/models/Qwen2.5-7B-Instruct',
            'parent_merged_from': str(SFT), 'parent_weights': parent_hash,
            'dpo_config_sha256': hashlib.sha256((ROOT / 'configs/post_train/c_planner_dpo.yaml').read_bytes()).hexdigest(),
            'dpo_training_pairs': 433, 'teacher_api_calls': 0,
        }, ensure_ascii=False, indent=2))
        status('training_c_dpo', parent=parent_hash)
        with (RUN / 'c-dpo-training.log').open('x') as stream:
            subprocess.run([PY, '-X', 'utf8', '-u', 'scripts/post_train_quality_entry.py', 'train',
                            'configs/post_train/c_planner_dpo.yaml'], cwd=ROOT, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        # 433 pairs, per-device batch 1, accumulation 8, two epochs.
        dpo_steps = ((433 + 8 - 1) // 8) * 2
        dpo_metrics = verify_training(DPO, dpo_steps)
        status('training_complete_ready_for_evaluation', c_sft=sft_metrics, c_dpo=dpo_metrics,
               parent_weights=parent_hash, elapsed_s=time.time() - started)
    except Exception as exc:
        status('failed', error=repr(exc), elapsed_s=time.time() - started)
        raise


if __name__ == '__main__':
    run()
