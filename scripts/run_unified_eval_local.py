"""Drive remote GPU0 vLLM while scoring the frozen 50x3 suite locally."""
import json, os, subprocess, time, urllib.request
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; HOST='jiale-10.14.3.82'
sys.path.insert(0,str(ROOT))
REMOTE='/home/jiale/TravelMind-posttrain'; PY='/home/jiale/.venvs/travelmind-train-smoke/bin/python'
BASE='/home/jiale/models/Qwen2.5-7B-Instruct'; V2=REMOTE+'/models/TravelMind-Planner-v2-current-20261025'
AD={'v2_sft':None,'v2_dpo':REMOTE+'/saves/shared-dpo-full-20261026-v2',
    'A_sft':REMOTE+'/saves/TravelMind-count-ab-20261025-A','A_dpo':REMOTE+'/saves/shared-dpo-full-20261026-A',
    'B_sft':REMOTE+'/saves/TravelMind-count-ab-total96-20261025-B','B_dpo':REMOTE+'/saves/shared-dpo-full-20261026-B'}
PARENT={'v2_sft':V2,'v2_dpo':V2,'A_sft':V2,'A_dpo':V2,'B_sft':BASE,'B_dpo':BASE}
TEMP=REMOTE+'/models/unified-eval-merged'; STATE=ROOT/'data/post_train/unified_eval_20261026.json'
def ssh(cmd,check=True): return subprocess.run(['ssh',HOST,cmd],check=check,text=True,capture_output=True)
def stop(): ssh("pkill -f '/home/jiale/.local/bin/vllm serve' 2>/dev/null || true",False); time.sleep(4)
def wait_model(name):
    for _ in range(180):
        try:
            with urllib.request.urlopen('http://10.14.3.82:8000/v1/models',timeout=2) as r:
                if any(x.get('id')==name for x in json.load(r).get('data',[])): return
        except Exception: pass
        time.sleep(2)
    raise RuntimeError('vLLM readiness timeout: '+name)
def main():
    report={'status':'running','started_at':time.time(),'variants':[]}; STATE.write_text(json.dumps(report,indent=2))
    stop()
    for tag in ['v2_sft','v2_dpo','A_sft','A_dpo','B_sft','B_dpo']:
        temporary=False; base=PARENT[tag]; adapter=AD[tag]
        try:
            if tag.endswith('_dpo') and not tag.startswith('v2'):
                sft=AD[tag.replace('_dpo','_sft')]
                ssh(f"rm -rf {TEMP} && {PY} {REMOTE}/scripts/post_train_train_entry.py merge --base {PARENT[tag]} --adapter {sft} --output {TEMP}")
                base=TEMP; temporary=True
            name='eval-'+tag
            lora=f' --enable-lora --max-lora-rank 16 --lora-modules {name}={adapter}' if adapter else ''
            cmd=f"nohup /home/jiale/.local/bin/vllm serve {base} --host 0.0.0.0 --port 8000 --served-model-name {name} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --enforce-eager{lora} > {REMOTE}/logs/unified-eval-{tag}.log 2>&1 < /dev/null &"
            ssh(cmd); wait_model(name)
            for mode in ['planner-only','rules-rework']:
                # evaluate_representative50 reads the frozen 50 cases locally and calls remote endpoint.
                os.system(f'python scripts/evaluate_representative50.py --model-name {name} --tag unified_{tag}_{mode} --mode {mode} --concurrency 4')
            stop(); report['variants'].append({'tag':tag,'status':'complete'}); STATE.write_text(json.dumps(report,indent=2))
        except Exception as exc:
            report.update(status='failed',error=f'{tag}: {exc}'); STATE.write_text(json.dumps(report,indent=2)); stop(); raise
        finally:
            if temporary: ssh(f'rm -rf {TEMP}',False)
    report.update(status='complete',elapsed_s=time.time()-report['started_at']); STATE.write_text(json.dumps(report,indent=2)); stop()
if __name__=='__main__': main()
