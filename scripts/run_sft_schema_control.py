"""Current-schema Base/v2/A/B controls for the completed DPO 50x3 run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_shared_prompt_all_versions as shared
from scripts.evaluate_representative50 import SUITE, verify
from scripts.run_sft_validation_job import remote
from app.planning.schemas import TravelRoute
from app.planning.prompts import JOINT_PLANNER_SYSTEM
from scripts.post_train_runtime import capture_messages

OUT = SUITE/'sft_schema_control_20261027'
BASE = '/home/jiale/models/Qwen2.5-7B-Instruct'
V2 = '/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-v2-current-20261025'
MODELS = {'base': (BASE, None), 'v2_sft': (V2, None),
          'A_sft': (V2, '/home/jiale/TravelMind-posttrain/saves/TravelMind-count-ab-20261025-A'),
          'B_sft': (BASE, '/home/jiale/TravelMind-posttrain/saves/TravelMind-count-ab-total96-20261025-B')}

def stop():
    return remote('''
import os,signal,time,subprocess
from pathlib import Path
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\\0')
  if p.stat().st_uid==os.getuid() and b'/home/jiale/.local/bin/vllm' in args and b'serve' in args:
   os.kill(int(p.name),signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError,PermissionError): pass
for _ in range(60):
 r=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
 if int(r.strip())<500: print(r.strip()); break
 time.sleep(1)
else: raise RuntimeError('GPU0 still occupied; no unrelated process was killed')
''')

def start(tag, base, adapter):
    name='schema-control-'+tag
    args=['/home/jiale/.local/bin/vllm','serve',base,'--host','0.0.0.0','--port','8000',
          '--served-model-name',name+'-parent' if adapter else name,'--dtype','bfloat16',
          '--max-model-len','8192','--gpu-memory-utilization','0.90','--enforce-eager']
    if adapter: args+=['--enable-lora','--max-lora-rank','16','--lora-modules',name+'='+adapter]
    remote('''
import os,subprocess
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
h='/home/jiale/python-dev/extracted/usr/include'
env['CPATH']=h+'/python3.10:'+h+'/x86_64-linux-gnu/python3.10:'+h
with open('/home/jiale/TravelMind-posttrain/logs/schema-control-'''+tag+'''.log','a') as f:
 p=subprocess.Popen('''+repr(args)+''',env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(p.pid)
''')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(150):
        try:
            with opener.open('http://10.14.3.82:8000/v1/models',timeout=3) as r:
                inventory=json.load(r)
            if any(m['id']==name and (not adapter or m.get('parent')) for m in inventory['data']):
                shared.atomic(OUT/f'{tag}-inventory.json',inventory)
                return name
        except (OSError,ValueError): pass
        time.sleep(2)
    raise RuntimeError('Model not ready: '+name)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=verify()
    cases=json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))
    dpo=json.loads((SUITE/'shared_prompt_current_v2/v2_dpo_planner-only/protocol.json').read_text(encoding='utf-8'))
    schema=hashlib.sha256(json.dumps(TravelRoute.model_json_schema(),sort_keys=True).encode()).hexdigest()
    assert schema==dpo['schema_sha256'], 'Current schema differs from DPO run'
    assert hashlib.sha256(JOINT_PLANNER_SYSTEM.encode()).hexdigest()==dpo['system_sha256']
    for c in cases:
        digest=hashlib.sha256(json.dumps(capture_messages(c['state']),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        assert digest==dpo['request_messages_sha256'][c['id']], 'Input differs from DPO'
    binding=dict(models=MODELS,schema_sha256=schema,system_sha256=dpo['system_sha256'],suite=manifest['hashes'],
                 seeds=[42,43,44],temperature=.3,max_tokens=2048,context=8192,concurrency=4)
    bp=OUT/'binding.json'
    if bp.exists(): assert json.loads(bp.read_text())==json.loads(json.dumps(binding)), 'Binding changed'
    shared.atomic(bp,binding)
    shared.OUT=OUT
    shared.maps=json.loads((SUITE/'maps.json').read_text(encoding='utf-8'))['routes']
    status=dict(stage='starting',started_at=time.time(),completed=[],target=1200)
    def save(**kw):
        status.update(kw,updated_at=time.time()); shared.atomic(OUT/'status.json',status)
    try:
        stop()
        for tag,(base,adapter) in MODELS.items():
            save(stage='loading_model',model=tag)
            name=start(tag,base,adapter); shared.MODELS[tag]=(base,name)
            for mode in ['planner-only','rules-rework']:
                save(stage='evaluating',model=tag,mode=mode)
                rows=shared.run_one(tag,mode,cases,[42,43,44],4)
                errors=[r for r in rows if r.get('error_type') in ['HTTPError','URLError','TimeoutError']]
                if errors: raise RuntimeError(f'{tag}/{mode}: {len(errors)} transport/service errors; inspect before continuing')
                shared.atomic(OUT/f'{tag}_{mode}/summary.json',shared.summarize(rows,cases))
                status['completed'].append(tag+'/'+mode); save()
            stop()
        save(stage='complete',elapsed_s=time.time()-status['started_at'])
    except Exception as exc:
        save(stage='failed',error=str(exc)); raise
    finally:
        try: save(vllm_shutdown_memory_mib=int(stop().strip()),vllm_shutdown_verified=True)
        except Exception as exc: save(vllm_shutdown_verified=False,shutdown_error=str(exc))

if __name__=='__main__': main()
