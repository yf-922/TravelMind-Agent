"""Sequential GPU0 DPO checks, or explicitly requested full training."""
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
RUN = 'shared-dpo-smoke-20261026'
FULL = '--full' in sys.argv
if FULL:
    RUN = 'shared-dpo-full-20261026'
DATA = ROOT/'data/post_train/shared_dpo_20261026/release'
TEMP = ROOT/'models'/f'{RUN}-temporary-merged'
BASE = '/home/jiale/models/Qwen2.5-7B-Instruct'
V2 = str(ROOT/'models/TravelMind-Planner-v2-current-20261025')
ADAPTERS = {'A': 'TravelMind-count-ab-20261025-A',
            'B': 'TravelMind-count-ab-total96-20261025-B'}


def status(stage, **values):
    row = dict(stage=stage, updated_at=time.time(), full_training_requested=FULL, **values)
    path = ROOT/f'logs/{RUN}-status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, indent=2))
    temporary.replace(path)
    print(json.dumps(row), flush=True)


def main():
    import yaml
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from scripts.launch_sft_4090_job import stop_owned_service
    for filename, digest in json.loads((DATA/'manifest.json').read_text()).items():
        if hashlib.sha256((DATA/filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Dataset changed: '+filename)
    binding = json.loads((DATA.parent/'model_binding.json').read_text())
    for arm, name in ADAPTERS.items():
        adapter = ROOT/'saves'/name
        if hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest() != binding[arm]['sha256']:
            raise RuntimeError('Adapter identity mismatch: '+arm)
        expected = V2 if arm == 'A' else BASE
        if json.loads((adapter/'adapter_config.json').read_text())['base_model_name_or_path'] != expected:
            raise RuntimeError('Adapter parent mismatch: '+arm)
    if TEMP.exists():
        raise RuntimeError('Temporary merge already exists; inspect before retry')
    for arm in ['v2','A','B']:
        if (ROOT/f'saves/{RUN}-{arm}').exists():
            raise RuntimeError('Refusing to overwrite smoke output: '+arm)
    if shutil.disk_usage(ROOT).free < 18*1024**3:
        raise RuntimeError('Need at least 18 GiB for temporary SFT merge')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    started = time.time()
    completed = []
    results = []
    restore = False
    try:
        status('stopping_owned_service', completed=completed, started_at=started)
        restore = True
        stop_owned_service()
        for arm in ['v2', 'A', 'B']:
            base = V2
            if arm in ADAPTERS:
                status('merging_sft_on_cpu', arm=arm, completed=completed, started_at=started)
                TEMP.mkdir()
                with (ROOT/f'logs/{RUN}-{arm}-merge.log').open('x') as stream:
                    subprocess.run([PYTHON, 'scripts/post_train_train_entry.py', 'merge',
                        '--base', V2 if arm == 'A' else BASE,
                        '--adapter', str(ROOT/'saves'/ADAPTERS[arm]), '--output', str(TEMP)],
                        env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                base = str(TEMP)
            config = yaml.safe_load((ROOT/'configs/post_train/qwen25_7b_planner_dpo.yaml').read_text())
            output = ROOT/f'saves/{RUN}-{arm}'
            config.update(model_name_or_path=base, dataset_dir=str(DATA), dataset='shared_dpo_train',
                eval_dataset='shared_dpo_validation', output_dir=str(output), max_samples=16,
                max_steps=5, save_strategy='steps', save_steps=5, eval_strategy='steps', eval_steps=5,
                load_best_model_at_end=False, save_total_limit=1, dataloader_num_workers=0)
            if FULL:
                config.pop('max_samples', None)
                config.update(max_steps=-1, num_train_epochs=2.0,
                    save_strategy='epoch', eval_strategy='epoch',
                    load_best_model_at_end=True, save_total_limit=2)
            cfg = ROOT/(f'configs/post_train/shared_{arm}_planner_dpo_'+('full.yaml' if FULL else 'smoke.yaml'))
            cfg.write_text(yaml.safe_dump(config, sort_keys=False))
            provenance = dict(arm=arm, sft_base=V2 if arm in ['v2','A'] else BASE,
                sft_adapter=str(ROOT/'saves'/ADAPTERS[arm]) if arm in ADAPTERS else None,
                sft_adapter_sha256=binding[arm]['sha256'] if arm in ADAPTERS else None,
                reference='merged SFT base with fresh DPO adapter disabled',
                dataset_manifest=json.loads((DATA/'manifest.json').read_text()),
                temporary_base_requires_reconstruction=arm in ADAPTERS,
                config=config)
            (ROOT/f'logs/{RUN}-{arm}-binding.json').write_text(json.dumps(provenance, indent=2))
            status('full_training' if FULL else 'smoke_training', arm=arm, completed=completed, started_at=started)
            with (ROOT/f'logs/{RUN}-{arm}.log').open('x') as stream:
                subprocess.run([PYTHON, 'scripts/post_train_train_entry.py', 'train', str(cfg)],
                    env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            state = json.loads((output/'trainer_state.json').read_text())
            losses = [float(v) for row in state['log_history'] for k,v in row.items()
                      if k in ['loss', 'eval_loss', 'train_loss']]
            expected_steps = math.ceil(len(json.loads((DATA/'shared_dpo_train.json').read_text())) / 8)*2 if FULL else 5
            if state['global_step'] != expected_steps or not losses or not all(math.isfinite(v) for v in losses):
                raise RuntimeError('Incomplete/nonfinite smoke: '+arm)
            if not (output/'adapter_model.safetensors').is_file():
                raise RuntimeError('Adapter not saved: '+arm)
            metrics = [json.loads(line) for line in (ROOT/'logs/gpu_training_metrics.jsonl').read_text().splitlines()]
            metrics = next(row for row in reversed(metrics) if row['stage_config'] == str(cfg))
            results.append(dict(arm=arm, steps=state['global_step'], metrics=metrics, log_history=state['log_history']))
            (ROOT/f'logs/{RUN}-results.json').write_text(json.dumps(results, indent=2))
            completed.append(arm)
            if arm in ADAPTERS:
                # Only this run's exclusively created directory may be removed.
                if TEMP.resolve() != (ROOT/'models'/f'{RUN}-temporary-merged').resolve():
                    raise RuntimeError('Unexpected temporary merge path')
                shutil.rmtree(TEMP)
            status('arm_passed', arm=arm, completed=completed, started_at=started)
        status('complete', completed=completed, elapsed_s=time.time()-started, started_at=started)
    except Exception as exc:
        status('failed', completed=completed, error=str(exc), started_at=started)
        raise
    finally:
        if restore:
            with open('/home/jiale/vllm_qwen.log', 'a') as stream:
                process = subprocess.Popen(['bash', str(ROOT/'scripts/serve_post_train_4090.sh')],
                    env=env, stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
            (ROOT/f'logs/{RUN}-restore.json').write_text(json.dumps(dict(
                pid=process.pid, readiness='starting_not_verified', time=time.time())))


if __name__ == '__main__':
    main()
