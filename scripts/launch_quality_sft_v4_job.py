"""GPU0-only clean-only/augmented controlled SFT; restore Base on termination."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/jiale/TravelMind-posttrain')
sys.path.insert(0,str(ROOT))
from scripts.launch_sft_4090_job import stop_owned_service
PYTHON='/home/jiale/.venvs/travelmind-train-smoke/bin/python'
ARMS=[('v4_clean','v4-clean'),('v4','v4')]


def status(stage,**kw):
    row=dict(stage=stage,updated_at=time.time(),dpo_auto_start=False,**kw)
    tmp=ROOT/'logs/quality-sft-v4-status.tmp'
    tmp.write_text(json.dumps(row,indent=2))
    tmp.replace(ROOT/'logs/quality-sft-v4-status.json')
    print(json.dumps(row),flush=True)


def main():
    os.chdir(ROOT)
    if ROOT != Path.cwd() or not str(ROOT).startswith('/home/jiale/'):
        raise RuntimeError('Wrong workspace')
    (ROOT/'logs').mkdir(exist_ok=True)
    bindings={}
    for arm,suffix in ARMS:
        release=ROOT/'data/post_train'/('release_'+arm)
        audit=json.loads((release/'audit.json').read_text())
        for name,digest in audit['sha256'].items():
            if hashlib.sha256((release/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Release hash mismatch '+arm)
        if not audit['validation_ledger_unchanged'] or audit['over_cutoff']:
            raise RuntimeError('Invalid release')
        for target in [ROOT/('saves/TravelMind-Planner-SFT-'+suffix),ROOT/('models/TravelMind-Planner-7B-SFT-'+suffix),release/'TRAINING_STARTED.json']:
            if target.exists():
                raise RuntimeError('Refusing overwrite '+str(target))
        bindings[arm]=dict(train=audit['train'],sha256=audit['sha256'],base='/home/jiale/models/Qwen2.5-7B-Instruct')
    if __import__('shutil').disk_usage(ROOT).free < 38*1024**3:
        raise RuntimeError('Insufficient space for two merged models and checkpoints')
    start=time.time();completed=[];failed=None
    status('stopping_owned_base',started_at=start)
    stop_owned_service()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    try:
        for arm,suffix in ARMS:
            release=ROOT/'data/post_train'/('release_'+arm)
            with (release/'TRAINING_STARTED.json').open('x') as f:
                json.dump(dict(started_at=time.time(),**bindings[arm]),f,indent=2)
            status('sft_training',arm=arm,completed_arms=completed,started_at=start)
            arm_env=dict(env,TRAVELMIND_TOKEN_LEDGER=str(ROOT/('logs/quality-'+arm+'-tokens.jsonl')))
            with (ROOT/('logs/quality-'+arm+'-training.log')).open('w') as log:
                subprocess.run([PYTHON,'scripts/post_train_quality_entry.py','train',
                    'configs/post_train/qwen25_7b_planner_sft_'+arm+'.yaml'],env=arm_env,stdout=log,stderr=subprocess.STDOUT,check=True)
            ledger=Path(arm_env['TRAVELMIND_TOKEN_LEDGER'])
            rows=[json.loads(l) for l in ledger.read_text().splitlines()]
            if not rows:
                raise RuntimeError('Actual token accounting missing')
            totals={k:sum(r[k] for r in rows) for k in ['samples','nonpadding_tokens','supervised_tokens']}
            (ROOT/('logs/quality-'+arm+'-token-summary.json')).write_text(json.dumps(totals,indent=2))
            status('cpu_merge',arm=arm,actual_training=totals,completed_arms=completed,started_at=start)
            with (ROOT/('logs/quality-'+arm+'-merge.log')).open('w') as log:
                subprocess.run([PYTHON,'scripts/post_train_train_entry.py','merge','--base',bindings[arm]['base'],
                    '--adapter',str(ROOT/('saves/TravelMind-Planner-SFT-'+suffix)),
                    '--output',str(ROOT/('models/TravelMind-Planner-7B-SFT-'+suffix))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            completed.append(arm)
    except Exception as exc:
        failed=type(exc).__name__
        raise
    finally:
        with open('/home/jiale/vllm_qwen.log','a') as log:
            service=subprocess.Popen(['bash',str(ROOT/'scripts/serve_post_train_4090.sh')],
                env=dict(env,MAX_MODEL_LEN='8192'),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        status('training_complete' if len(completed)==len(ARMS) else 'training_failed',
            completed_arms=completed,error_type=failed,elapsed_s=time.time()-start,base_service_pid=service.pid,
            base_service_readiness='starting_not_yet_verified')


if __name__=='__main__':
    main()
