"""Evaluate v2/A/B DPO adapters on the frozen 50x3 suite, then stop vLLM."""
import json, os, signal, subprocess, sys, time, urllib.request, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_shared_prompt_all_versions as shared
from scripts.evaluate_representative50 import SUITE, verify

HOST='jiale-10.14.3.82'; REMOTE='/home/jiale/TravelMind-posttrain'; PY='/home/jiale/.venvs/travelmind-train-smoke/bin/python'
BASE='/home/jiale/models/Qwen2.5-7B-Instruct'; V2=REMOTE+'/models/TravelMind-Planner-v2-current-20261025'
DPO={'v2_dpo':(V2,REMOTE+'/saves/shared-dpo-full-20261026-v2'),
     'A_dpo':(V2,REMOTE+'/saves/shared-dpo-full-20261026-A'),
     'B_dpo':(BASE,REMOTE+'/saves/shared-dpo-full-20261026-B')}
SFT={'A_dpo':REMOTE+'/saves/TravelMind-count-ab-20261025-A','B_dpo':REMOTE+'/saves/TravelMind-count-ab-total96-20261025-B'}
TEMP=REMOTE+'/models/dpo-eval-temporary-sft'
OUT=SUITE/'dpo_eval_50x3_20261027'; OUT.mkdir(parents=True,exist_ok=True)

def ssh(cmd,check=True): return subprocess.run(['ssh',HOST,cmd],check=check,text=True,capture_output=True)
def stop():
    ssh("pkill -TERM -f '/home/jiale/.local/bin/vllm serve' 2>/dev/null || true",False)
    time.sleep(4)
def ready(name):
    for _ in range(180):
        try:
            with urllib.request.urlopen('http://10.14.3.82:8000/v1/models',timeout=2) as r:
                if any(x.get('id')==name for x in json.load(r).get('data',[])): return
        except Exception: pass
        time.sleep(2)
    raise RuntimeError('vLLM readiness timeout '+name)
def launch(tag,base,adapter):
    name='dpo-'+tag
    cmd=f"nohup /home/jiale/.local/bin/vllm serve {base} --host 0.0.0.0 --port 8000 --served-model-name {name} --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 --enforce-eager --enable-lora --max-lora-rank 16 --lora-modules {name}={adapter} > {REMOTE}/logs/dpo-eval-{tag}.log 2>&1 < /dev/null &"
    ssh(cmd); ready(name); return name
def main():
    verify(); cases=json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))
    shared.maps=json.loads((SUITE/'maps.json').read_text(encoding='utf-8'))['routes']
    shared.MODELS={tag:(base,'dpo-'+tag) for tag,(base,adapter) in DPO.items()}
    report={'status':'running','started_at':time.time(),'variants':[]}
    (OUT/'status.json').write_text(json.dumps(report,indent=2))
    stop()
    try:
        for tag,(base,adapter) in DPO.items():
            temporary=False
            try:
                if tag in SFT:
                    ssh(f"rm -rf {TEMP} && mkdir -p {TEMP} && {PY} {REMOTE}/scripts/post_train_train_entry.py merge --base {base} --adapter {SFT[tag]} --output {TEMP}")
                    base=TEMP; temporary=True
                name=launch(tag,base,adapter)
                shared.MODELS[tag]=(base,name)
                for mode in ['planner-only','rules-rework']:
                    run=OUT/f'{tag}_{mode}'; run.mkdir(parents=True,exist_ok=True)
                    rs=shared.run_one(tag,mode,cases,[42,43,44],4)
                    sm=shared.summarize(rs,cases)
                    (run/'summary.json').write_text(json.dumps(sm,ensure_ascii=False,indent=2),encoding='utf-8')
                report['variants'].append({'tag':tag,'status':'complete'}); (OUT/'status.json').write_text(json.dumps(report,indent=2))
                stop()
            finally:
                if temporary: ssh(f'rm -rf {TEMP}',False)
    except Exception as exc:
        report.update(status='failed',error=str(exc)); (OUT/'status.json').write_text(json.dumps(report,indent=2)); raise
    finally:
        stop()
    report.update(status='complete',elapsed_s=time.time()-report['started_at']); (OUT/'status.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__': main()
