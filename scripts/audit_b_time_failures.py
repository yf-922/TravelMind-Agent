"""Re-run only the frozen B time-format failures with raw outputs retained."""
import copy, json, time, urllib.request
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from app.planning.schemas import TravelRoute
from scripts.post_train_runtime import capture_messages
from scripts.run_shared_prompt_all_versions import invalid_clock_fields
from scripts.build_representative_eval50 import score_case
from scripts.run_sft_validation_job import remote, ready
from scripts.evaluate_representative50 import SUITE, verify

IDS=['representative50-1-04','representative50-2-04','representative50-4-03','representative50-4-08','representative50-5-08']
SEEDS=[43,44]
MODEL='count-ab-B'; OUT=ROOT/'data/post_train/count_infeasible_ab_20261025/evaluation_total96/time_audit'

def start():
    remote(r'''
import os,signal,subprocess,time
from pathlib import Path
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().split(b'\0')
  if p.stat().st_uid==os.getuid() and b'/home/jiale/.local/bin/vllm' in cmd and b'serve' in cmd: os.kill(int(p.name),signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
for _ in range(60):
 r=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 if int(r.stdout.strip())<500: break
 time.sleep(1)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
args=['/home/jiale/.local/bin/vllm','serve','/home/jiale/models/Qwen2.5-7B-Instruct','--host','0.0.0.0','--port','8000','--served-model-name','count-ab-B','--dtype','bfloat16','--max-model-len','8192','--gpu-memory-utilization','0.90','--enforce-eager','--enable-lora','--max-lora-rank','16','--lora-modules','count-ab-B=/home/jiale/TravelMind-posttrain/saves/TravelMind-count-ab-total96-20261025-B/checkpoint-96']
with open('/home/jiale/vllm_count_ab_audit.log','a') as f: subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
''')
    ready(MODEL)

def main():
    verify(); cases=json.loads((SUITE/'cases.json').read_text(encoding='utf-8')); cases=[c for c in cases if c['id'] in IDS]
    _, maps = (None, json.loads((SUITE/'maps.json').read_text(encoding='utf-8'))['routes'])
    start(); OUT.mkdir(parents=True,exist_ok=True); rows=[]
    for mode in ['planner-only','rules-rework']:
      for c in cases:
       for seed in SEEDS:
        feedback=''
        for round_no in range(1 if mode=='planner-only' else 4):
         msg=capture_messages(copy.deepcopy(c['state']))
         if feedback: msg[-1]['content'] += '\n\n确定性规则返工反馈：'+feedback
         req={'model':MODEL,'messages':msg,'temperature':.3,'seed':seed,'max_tokens':2048,'response_format':{'type':'json_schema','json_schema':{'name':'TravelRoute','strict':False,'schema':TravelRoute.model_json_schema()}}}
         t=time.perf_counter(); rec={'id':c['id'],'seed':seed,'mode':mode,'round':round_no,'request':req}
         try:
          q=urllib.request.Request('http://10.14.3.82:8000/v1/chat/completions',data=json.dumps(req,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
          with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(q,timeout=120) as h: raw=json.load(h)
          rec['raw_response']=raw; rec['usage']=raw.get('usage')
          plan=TravelRoute.model_validate_json(raw['choices'][0]['message']['content']).model_dump(mode='json'); rec['schema_valid']=True
          bad=invalid_clock_fields(plan); rec['invalid_time_fields']=bad
          rec['faults']=['invalid_time_format'] if bad else score_case(plan,c,maps); rec['plan']=plan
          if not rec['faults'] or mode=='planner-only': rows.append(rec); break
          feedback='；'.join(rec['faults'])
         except Exception as e: rec.update(schema_valid=False,error_type=type(e).__name__,error=str(e)); rows.append(rec); break
         finally: rec['elapsed_s']=time.perf_counter()-t
    (OUT/'raw_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'invalid':sum(bool(r.get('invalid_time_fields')) for r in rows),'errors':sum('error' in r for r in rows)},ensure_ascii=False))
if __name__=='__main__': main()
