"""Seven frozen planner arms x two modes x 50 tasks x 3 decoding seeds.

Manages only its OWN vLLM process group, always closes it on exit. No training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import shlex

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_online_holdout50 import OUT,verify,dump

HOST='jiale-10.14.3.82'
REMOTE='/home/jiale/TravelMind-posttrain'
PY='/home/jiale/.venvs/travelmind-train-smoke/bin/python'
BASE='/home/jiale/models/Qwen2.5-7B-Instruct'
TAGS=['base','v2_sft','A_sft','B_sft','v2_dpo','A_dpo','B_dpo']
V2=REMOTE+'/models/TravelMind-Planner-v2-current-20261025'
SAVES=REMOTE+'/saves/'
MODELS={'base':(BASE,None),'v2_sft':(V2,None),
        'A_sft':(V2,SAVES+'TravelMind-count-ab-20261025-A'),
        'B_sft':(BASE,SAVES+'TravelMind-count-ab-total96-20261025-B'),
        'v2_dpo':(V2,SAVES+'shared-dpo-full-20261026-v2'),
        'A_dpo':(V2,SAVES+'shared-dpo-full-20261026-A'),
        'B_dpo':(BASE,SAVES+'shared-dpo-full-20261026-B')}


def remote(source):
    r=subprocess.run(['ssh',HOST,PY+' -'],input=source,text=True,capture_output=True,encoding='utf-8')
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    return r.stdout


def stop_owned():
    return remote('''
from pathlib import Path
import os,signal,time
p=Path('/home/jiale/TravelMind-posttrain/logs/online-holdout-vllm.pid')
if p.exists():
 pid=int(p.read_text())
 proc=Path('/proc')/str(pid)
 if proc.exists():
  args=(proc/'cmdline').read_bytes()
  if proc.stat().st_uid!=os.getuid() or b'holdout-' not in args:
   raise RuntimeError('PID owner/command mismatch; refusing to stop unrelated process')
  os.killpg(pid,signal.SIGTERM)
  for _ in range(100):
   if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
   time.sleep(.2)
  else:
   os.killpg(pid,signal.SIGKILL)
 p.unlink()
 print('owned service stopped')
''')


def start(tag,base,adapter):
    name='holdout-'+tag
    args=['/home/jiale/.local/bin/vllm','serve',base,'--host','0.0.0.0','--port','8000',
          '--served-model-name',name+'-parent' if adapter else name,'--dtype','bfloat16','--max-model-len','8192',
          '--gpu-memory-utilization','0.90','--enforce-eager','--no-enable-prefix-caching']
    if adapter:args+=['--enable-lora','--max-lora-rank','16','--lora-modules',name+'='+adapter]
    remote('''
import os,subprocess,socket
from pathlib import Path
s=socket.socket()
if s.connect_ex(('127.0.0.1',8000))==0:raise RuntimeError('Port8000 occupied; no unrelated server stopped')
s.close()
used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>500:raise RuntimeError('GPU0 busy; no unrelated GPU process stopped')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
h='/home/jiale/python-dev/extracted/usr/include'
env['CPATH']=h+'/python3.10:'+h+'/x86_64-linux-gnu/python3.10:'+h
with open('/home/jiale/TravelMind-posttrain/logs/online-holdout-vllm.log','a') as f:
 p=subprocess.Popen('''+repr(args)+''',env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
Path('/home/jiale/TravelMind-posttrain/logs/online-holdout-vllm.pid').write_text(str(p.pid))
print(p.pid)
''')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(150):
        try:
            with opener.open('http://10.14.3.82:8000/v1/models',timeout=3) as r:inventory=json.load(r)
            names={m['id'] for m in inventory['data']}
            if name in names:
                selected=next(m for m in inventory['data'] if m['id']==name)
                if adapter and not selected.get('parent'):raise RuntimeError('Expected adapter is not bound to parent')
                return inventory
        except (OSError,ValueError):pass
        time.sleep(2)
    raise RuntimeError('Owned vLLM readiness timeout; inspect server log')


def resource_state():
    return json.loads(remote('''
import json,subprocess,socket
memory=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
s=socket.socket();occupied=s.connect_ex(('127.0.0.1',8000))==0;s.close()
print(json.dumps(dict(memory_mib=memory,port8000_occupied=occupied,ready=memory<500 and not occupied)))
'''))


def merge_parent(tag):
    base=V2 if tag=='A_dpo' else BASE
    adapter=SAVES+('TravelMind-count-ab-20261025-A' if tag=='A_dpo' else 'TravelMind-count-ab-total96-20261025-B')
    path=remote('''
import tempfile
print(tempfile.mkdtemp(prefix='online-holdout-owned-',dir='/home/jiale/TravelMind-posttrain/models'))
''').strip()
    cmd=[PY,REMOTE+'/scripts/post_train_train_entry.py','merge','--base',base,'--adapter',adapter,'--output',path]
    r=subprocess.run(['ssh',HOST,shlex.join(cmd)],text=True,capture_output=True,encoding='utf-8')
    if r.returncode:
        cleanup_parent(path)
        raise RuntimeError('SFT parent merge failed: '+r.stderr[-2000:])
    return path


def cleanup_parent(path):
    remote('''
from pathlib import Path
import shutil
p=Path('''+repr(path)+''').resolve()
root=Path('/home/jiale/TravelMind-posttrain/models').resolve()
if p.parent!=root or not p.name.startswith('online-holdout-owned-'):raise RuntimeError('Unexpected temporary parent path')
if p.exists():shutil.rmtree(p)
''')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',choices=['online-api'],required=True)
    p.add_argument('--tags',nargs='+',choices=TAGS,default=TAGS)
    p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--wait-for-gpu',action='store_true',help='Queue without stopping the existing GPU0/port8000 workload')
    args=p.parse_args();manifest=verify()
    result_dir=OUT/'matrix';result_dir.mkdir(parents=True,exist_ok=True)
    binding=dict(tags=args.tags,seeds=[42,43,44],modes=['planner-only','online-rework'],suite=manifest['hashes'],
        audit=args.audit,reviewer_policy='fixed audit model, planner only varies',target_tasks=len(args.tags)*300,
        model_bindings={k:list(v) for k,v in MODELS.items()},planner_temperature=.3,planner_max_tokens=2048,context_length=8192,
        audit_temperature=0,audit_seed='not supported by online Responses; not injected')
    if args.audit=='online-api':
        from app.core.env import load_local_env
        from app.llm.openai import resolve_openai_base_url,resolve_openai_model
        import os
        load_local_env()
        url=resolve_openai_base_url();model=resolve_openai_model()
        key=os.getenv('OPENAI_API_KEY') or os.getenv('AI_API_KEY')
        if not url or not key:raise RuntimeError('Online API endpoint/credential missing')
        binding.update(audit_url=url,reviewer_model=model)
    else:url=None;model='holdout-base';key=None
    bp=result_dir/'binding.json'
    if bp.exists() and json.loads(bp.read_text(encoding='utf-8'))!=binding:raise RuntimeError('Matrix binding changed')
    dump(bp,binding)
    completed=[];started=time.time()
    temporary=None
    try:
        for tag in args.tags:
            state=resource_state()
            while not state['ready']:
                dump(result_dir/'status.json',dict(stage='waiting_for_gpu0',next_model=tag,resource=state,completed=completed,updated_at=time.time()))
                if not args.wait_for_gpu:raise RuntimeError('GPU0/port8000 busy; no unrelated job stopped')
                time.sleep(30);state=resource_state()
            base,adapter=MODELS[tag]
            if tag in {'A_dpo','B_dpo'}:
                dump(result_dir/'status.json',dict(stage='merging_sft_parent',model=tag,completed=completed,updated_at=time.time()))
                temporary=merge_parent(tag);base=temporary
            inventory=start(tag,base,adapter);dump(result_dir/(tag+'-inventory.json'),inventory)
            dump(result_dir/(tag+'-weights.json'),dict(base=base,adapter=adapter,original_binding=MODELS[tag],
                parent='Corresponding SFT merged BF16 before DPO adapter; no concatenated adapter substitution'))
            dump(result_dir/'status.json',dict(stage='evaluating',model=tag,completed=completed,started_at=started,updated_at=time.time()))
            cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'scripts/evaluate_online_holdout50.py'),
                 '--tag',tag,'--model-name','holdout-'+tag,'--reviewer-model',model,
                 '--output',str(result_dir),'--concurrency',str(args.concurrency)]
            env=None
            if url:
                import os
                env=dict(os.environ,EVAL_AUDIT_API_KEY=key)
                cmd+=['--audit-url',url,'--audit-production-adapter']
            with (result_dir/(tag+'.log')).open('a',encoding='utf-8') as f:
                r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError('Evaluation runner failed for '+tag)
            sm=json.loads((result_dir/tag/'summary.json').read_text(encoding='utf-8'))
            transport={'HTTPError','URLError','TimeoutError','APIConnectionError','APITimeoutError','AuthenticationError','RateLimitError'}
            if any(transport & (set(sm[m]['all']['errors'])|set(sm[m]['all']['call_errors'])) for m in ['planner-only','online-rework']):
                raise RuntimeError('Recorded service exceptions; investigate before other arms, preserve protocol')
            completed.append(tag)
            stop_owned()
            if temporary:cleanup_parent(temporary);temporary=None
            subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/report_online_holdout50.py')],cwd=ROOT,check=True)
        dump(result_dir/'status.json',dict(stage='complete',completed=completed,elapsed_s=time.time()-started))
    except Exception as exc:
        dump(result_dir/'status.json',dict(stage='failed',error=str(exc),completed=completed,elapsed_s=time.time()-started));raise
    finally:
        stop_owned()
        if temporary:cleanup_parent(temporary)
        dump(result_dir/'shutdown.json',dict(stopped_owned_vllm=True,time=time.time()))


if __name__=='__main__':main()
