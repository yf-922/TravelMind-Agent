"""Sequential full-BF16 seven-arm evaluation with production graph and online audit roles."""
from __future__ import annotations
import argparse,hashlib,json,os,random,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_online_holdout50 import OUT,verify,dump
from scripts.prepare_controlled_holdout_weights import HOST,PY,OUT as WEIGHTS

TAGS=['base','v2_sft','A_sft','B_sft','v2_dpo','A_dpo','B_dpo']

def remote(code):
    r=subprocess.run(['ssh',HOST,PY+' -'],input=code,text=True,capture_output=True,encoding='utf-8')
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    return r.stdout.strip()

def resource():
    return json.loads(remote('''
import json,socket,subprocess
used,free=map(int,subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(','))
s=socket.socket();busy=s.connect_ex(('127.0.0.1',8000))==0;s.close()
print(json.dumps(dict(used_mib=used,free_mib=free,port_busy=busy,ready=used<500 and free>22000 and not busy)))
'''))

def stop_owned():
    remote('''
from pathlib import Path
import os,signal,time
p=Path('/home/jiale/TravelMind-posttrain/logs/controlled-holdout-vllm.pid')
if p.exists():
 pid=int(p.read_text());proc=Path('/proc')/str(pid)
 if proc.exists():
  args=(proc/'cmdline').read_bytes()
  if proc.stat().st_uid!=os.getuid() or b'controlled-' not in args:raise RuntimeError('refuse to stop non-owned PID')
  os.killpg(pid,signal.SIGTERM)
  for _ in range(100):
   if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
   time.sleep(.2)
  else:os.killpg(pid,signal.SIGKILL)
 p.unlink()
''')

def wait_gpu_zero():
    time.sleep(4)
    for _ in range(30):
        state=resource()
        if state['used_mib']<500 and state['free_mib']>23000 and not state['port_busy']:
            return state
        time.sleep(2)
    raise RuntimeError('Owned vLLM stopped but GPU memory/port did not return to idle; refusing next model')

def start(tag,path):
    name='controlled-'+tag
    args=['/home/jiale/.local/bin/vllm','serve',path,'--host','0.0.0.0','--port','8000','--served-model-name',name,
          '--dtype','bfloat16','--max-model-len','8192','--gpu-memory-utilization','0.90','--enforce-eager','--no-enable-prefix-caching']
    remote('''
import os,socket,subprocess
from pathlib import Path
s=socket.socket();busy=s.connect_ex(('127.0.0.1',8000))==0;s.close()
used,free=map(int,subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(','))
if busy or used>500 or free<22000:raise RuntimeError('GPU0 or port 8000 not clean')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
h='/home/jiale/python-dev/extracted/usr/include';env['CPATH']=h+'/python3.10:'+h+'/x86_64-linux-gnu/python3.10:'+h
with open('/home/jiale/TravelMind-posttrain/logs/controlled-holdout-vllm.log','a') as f:
 p=subprocess.Popen('''+repr(args)+''',env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
Path('/home/jiale/TravelMind-posttrain/logs/controlled-holdout-vllm.pid').write_text(str(p.pid))
print(p.pid)
''')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(240):
        try:
            with opener.open('http://10.14.3.82:8000/v1/models',timeout=3) as r:inventory=json.load(r)
            if name in {m['id'] for m in inventory['data']}:return inventory
        except (OSError,ValueError):pass
        time.sleep(2)
    tail=remote("from pathlib import Path;print(Path('/home/jiale/TravelMind-posttrain/logs/controlled-holdout-vllm.log').read_text(errors='replace')[-5000:])")
    raise RuntimeError('vLLM readiness failed: '+tail)

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',choices=['local-rules'],required=True)
    p.add_argument('--wait-for-gpu',action='store_true');p.add_argument('--tags',nargs='+',choices=TAGS,default=TAGS)
    p.add_argument('--run-id',required=True);args=p.parse_args();suite=verify()
    root=OUT/args.run_id;root.mkdir(parents=True,exist_ok=True)
    preflight=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_controlled_holdout_weights.py'),'preflight'],
       cwd=ROOT,capture_output=True,text=True)
    if preflight.returncode:raise RuntimeError('Rolling weight preflight failed: '+preflight.stderr[-2000:])
    rolling_evidence=json.loads(preflight.stdout)
    protocol_sources=['app/planning/graph.py','app/planning/nodes.py','app/planning/prompts.py','app/planning/schemas.py',
      'app/planning/constraints.py','app/evaluation/online_replay.py','scripts/evaluate_controlled_online_holdout.py',
      'scripts/run_controlled_online_holdout_matrix.py','scripts/prepare_controlled_holdout_weights.py',
      'scripts/report_controlled_online_holdout.py','scripts/build_representative_eval50.py','scripts/post_train_runtime.py']
    bindings=dict(run_id=args.run_id,tags=args.tags,seeds=[1001,1002,1003],modes=['planner-only','online-rework'],
       max_tasks=len(args.tags)*300,planner_temperature=.3,top_p=.95,max_tokens=2048,
       planner_streaming=True,concurrency=1,retry_budget='up to three additional Planner calls; deterministic local rubric feedback',
       reviewer='disabled',time_check='disabled',teacher_api_calls=0,weight_form='full BF16',
       models={t:'rolling:'+t for t in args.tags},suite_sha256=suite['hashes']['cases.json'],
       protocol_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in protocol_sources},
       prefix_cache=False,arm_order_seed=271027,arm_order='randomized sequential; single GPU cannot hold multiple BF16 arms',
       warmup='5 Planner-only examples per arm, discarded')
    bp=root/'binding.json'
    if bp.exists() and json.loads(bp.read_text(encoding='utf-8'))!=bindings:raise RuntimeError('run binding differs')
    dump(bp,bindings);dump(root/'adapter_parent_evidence.json',rolling_evidence)
    completed=[];order=list(args.tags);random.Random(271027).shuffle(order);started=time.time()
    try:
      for tag in order:
        state=resource()
        while not state['ready']:
          dump(root/'status.json',dict(stage='waiting_for_gpu0',model=tag,resource=state,completed=completed,updated_at=time.time()))
          if not args.wait_for_gpu:raise RuntimeError('GPU0 busy')
          time.sleep(20);state=resource()
        dump(root/'status.json',dict(stage='starting_model',model=tag,completed=completed,updated_at=time.time()))
        merged=None
        try:
          prepare=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_controlled_holdout_weights.py'),'prepare','--tag',tag],
              cwd=ROOT,capture_output=True,text=True)
          if prepare.returncode:raise RuntimeError('Rolling merge failed for '+tag+': '+prepare.stderr[-2500:])
          merged=json.loads(prepare.stdout);dump(root/(tag+'-weights.json'),merged)
          inventory=start(tag,merged['path']);dump(root/(tag+'-inventory.json'),inventory)
          cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'scripts/evaluate_controlled_online_holdout.py'),
            '--tag',tag,'--model-name','controlled-'+tag,
            '--output',str(root),'--temperature','0.3','--max-tokens','2048']
          with (root/(tag+'.log')).open('a',encoding='utf-8') as f:
            r=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ),stdout=f,stderr=subprocess.STDOUT)
          if r.returncode:raise RuntimeError('evaluator failed for '+tag)
          completed.append(tag)
        finally:
          stop_owned();idle=wait_gpu_zero()
          cleanup=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_controlled_holdout_weights.py'),'cleanup','--tag',tag],
              cwd=ROOT,capture_output=True,text=True)
          if cleanup.returncode:raise RuntimeError('Could not clean rolling weights for '+tag+': '+cleanup.stderr[-2000:])
          dump(root/(tag+'-cleanup.json'),dict(gpu_idle=idle,cleanup=json.loads(cleanup.stdout)))
        dump(root/'status.json',dict(stage='gpu_and_disk_released',model=tag,gpu=idle,completed=completed,elapsed_s=time.time()-started,updated_at=time.time()))
        report=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/report_controlled_online_holdout.py'),'--run-dir',str(root)],cwd=ROOT,capture_output=True,text=True)
        if report.returncode:raise RuntimeError(report.stderr[-2000:])
        dump(root/'status.json',dict(stage='completed_arm',model=tag,completed=completed,elapsed_s=time.time()-started,updated_at=time.time()))
      dump(root/'status.json',dict(stage='complete',completed=completed,elapsed_s=time.time()-started))
    except Exception as exc:
      dump(root/'status.json',dict(stage='failed',error=str(exc),completed=completed,elapsed_s=time.time()-started));raise
    finally:
      stop_owned();dump(root/'shutdown.json',dict(stopped_owned_vllm=True,time=time.time()))
if __name__=='__main__':main()
