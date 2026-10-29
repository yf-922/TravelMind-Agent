"""Run the frozen 50x3 benchmark for SFT/DPO variants on one GPU0 server."""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT=Path('/home/jiale/TravelMind-posttrain'); PY='/home/jiale/.venvs/travelmind-train-smoke/bin/python'
BASE='/home/jiale/models/Qwen2.5-7B-Instruct'; V2=str(ROOT/'models/TravelMind-Planner-v2-current-20261025')
AD={'v2_sft':None,'v2_dpo':str(ROOT/'saves/shared-dpo-full-20261026-v2'),
    'A_sft':str(ROOT/'saves/TravelMind-count-ab-20261025-A'),'A_dpo':str(ROOT/'saves/shared-dpo-full-20261026-A'),
    'B_sft':str(ROOT/'saves/TravelMind-count-ab-total96-20261025-B'),'B_dpo':str(ROOT/'saves/shared-dpo-full-20261026-B')}
PARENT={'v2_sft':V2,'v2_dpo':V2,'A_sft':V2,'A_dpo':V2,'B_sft':BASE,'B_dpo':BASE}
FULL=['v2_sft','v2_dpo','A_sft','A_dpo','B_sft','B_dpo']
TEMP=ROOT/'models/unified-eval-merged'
LOG=ROOT/'logs/unified-eval-20261026.json'
def save(x): LOG.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def stop():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            cmd=(p/'cmdline').read_bytes()
            if p.stat().st_uid==os.getuid() and b'/home/jiale/.local/bin/vllm' in cmd:
                os.kill(int(p.name),signal.SIGTERM)
        except (FileNotFoundError,PermissionError,ProcessLookupError): pass
    for _ in range(60):
        try:
            m=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']))
            if m<500:return
        except Exception: pass
        time.sleep(1)
    raise RuntimeError('GPU0 did not clear')
def merge(base,adapter):
    if TEMP.exists(): raise RuntimeError('stale temp merge')
    TEMP.mkdir()
    subprocess.run([PY,'scripts/post_train_train_entry.py','merge','--base',base,'--adapter',adapter,'--output',str(TEMP)],check=True)
    return str(TEMP)
def serve(base, name, adapter=None):
    args=['/home/jiale/.local/bin/vllm','serve',base,'--host','0.0.0.0','--port','8000','--served-model-name',name,'--dtype','bfloat16','--max-model-len','4096','--gpu-memory-utilization','0.90','--enforce-eager']
    if adapter: args += ['--enable-lora','--max-lora-rank','16','--lora-modules',f'{name}={adapter}']
    out=open(ROOT/f'logs/unified-eval-{name}.log','w')
    p=subprocess.Popen(args,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(180):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=2) as r:
                if any(item.get('id') == name for item in json.load(r).get('data',[])): return p
        except Exception: pass
        time.sleep(2)
    p.terminate(); raise RuntimeError('vLLM failed: '+name)
def main():
    os.chdir(ROOT); stop(); report={'status':'running','variants':[],'started_at':time.time()}; save(report)
    for tag in FULL:
        base=PARENT[tag]; adapter=AD[tag]; temporary=False
        try:
            # DPO adapter was trained against a merged SFT base. Reconstruct it.
            if tag.endswith('_dpo'):
                source=V2 if tag.startswith('v2') else (V2 if tag.startswith('A') else BASE)
                sft=AD[tag.replace('_dpo','_sft')]
                if sft:
                    base=merge(source,sft); temporary=True
                else:
                    base=source
            name='eval-'+tag
            proc=serve(base,name,adapter)
            for mode in ['planner-only','rules-rework']:
                out=ROOT/f'data/post_train/representative_eval50_v1/runs/unified_{tag}_{mode}'
                subprocess.run([PY,'scripts/evaluate_representative50.py','--model-name',name,'--tag',f'unified_{tag}_{mode}','--mode',mode,'--concurrency','4'],check=True)
            proc.terminate(); proc.wait(timeout=30); stop()
            report['variants'].append({'tag':tag,'status':'complete'}); save(report)
        except Exception as e:
            report.update(status='failed',error=f'{tag}: {e}'); save(report); raise
        finally:
            if temporary and TEMP.exists():
                import shutil; shutil.rmtree(TEMP)
    report.update(status='complete',elapsed_s=time.time()-report['started_at']); save(report)
if __name__=='__main__': main()
