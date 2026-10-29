"""GPU0: smoke both arms, then train 48 steps each; preserve LoRA only."""
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
RUN = 'count-ab-20261025'
DATA = ROOT/'data/post_train/count_infeasible_ab_20261025'


def status(stage, **extra):
    row = dict(stage=stage, updated_at=time.time(), dpo_auto_start=False, **extra)
    target = ROOT/f'logs/{RUN}-status.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2))
    tmp.replace(target)
    print(json.dumps(row), flush=True)


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from scripts.launch_sft_4090_job import stop_owned_service
    for name in ['manifest.json','training_binding.json']:
        manifest = json.loads((DATA/name).read_text())
        for relative, sha in manifest['sha256'].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != sha:
                raise RuntimeError('Input hash mismatch: '+relative)
    for arm in ['A','B']:
        for suffix in ['', '-smoke']:
            path = ROOT/f'saves/TravelMind-{RUN}-{arm}{suffix}'
            if path.exists():
                raise RuntimeError('Refusing overwrite: '+str(path))
    if shutil.disk_usage(ROOT).free < 6*1024**3:
        raise RuntimeError('Need 6 GiB free for adapter checkpoints and cache')
    # Assert source checkpoints are complete before taking the inference service down.
    for base in ['/home/jiale/models/Qwen2.5-7B-Instruct',
                 str(ROOT/'models/TravelMind-Planner-v2-current-20261025')]:
        index = json.loads((Path(base)/'model.safetensors.index.json').read_text())
        if not all((Path(base)/name).is_file() for name in set(index['weight_map'].values())):
            raise RuntimeError('Missing base shards')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    started = time.time()
    completed = []
    restore = False
    try:
        status('stopping_owned_base_service', started_at=started)
        restore = True
        stop_owned_service()
        for smoke in [True, False]:
            for arm in ['A','B']:
                name = arm+('-smoke' if smoke else '-full')
                status('smoke' if smoke else 'training', arm=arm, completed=completed, started_at=started)
                env['TRAVELMIND_TOKEN_LEDGER'] = str(ROOT/f'logs/{RUN}-{name}-tokens.jsonl')
                config = f'configs/post_train/count_ab_{arm}'+('_smoke' if smoke else '')+'.yaml'
                with (ROOT/f'logs/{RUN}-{name}.log').open('x') as stream:
                    subprocess.run([PYTHON, 'scripts/post_train_quality_entry.py', 'train', config],
                        env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
                suffix = '-smoke' if smoke else ''
                folder = ROOT/f'saves/TravelMind-{RUN}-{arm}{suffix}'
                state = json.loads((folder/'trainer_state.json').read_text())
                metrics = [float(v) for r in state['log_history'] for k,v in r.items()
                           if k.startswith('eval_') and k.endswith('loss')]
                if state['global_step'] != (5 if smoke else 48) or not metrics or not all(math.isfinite(v) for v in metrics):
                    raise RuntimeError('Incomplete or nonfinite training: '+name)
                if not (folder/'adapter_model.safetensors').exists():
                    raise RuntimeError('Final adapter missing: '+name)
                status('smoke_passed' if smoke else 'arm_complete', arm=arm,
                    steps=state['global_step'], final_evaluation={k:v for r in state['log_history'][-3:] for k,v in r.items() if k.startswith('eval_')},
                    completed=completed, started_at=started)
                if not smoke:
                    completed.append(arm)
        status('complete_pending_evaluation', completed=completed, started_at=started,
               elapsed_s=time.time()-started, merge_performed=False)
    except Exception as exc:
        status('failed', completed=completed, started_at=started, error=str(exc))
        raise
    finally:
        if restore:
            with open('/home/jiale/vllm_qwen.log', 'a') as stream:
                subprocess.Popen(['bash', str(ROOT/'scripts/serve_post_train_4090.sh')], env=env,
                    stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


if __name__=='__main__':
    main()
