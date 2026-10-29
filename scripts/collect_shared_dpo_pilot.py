"""Collect fresh local-model negatives on train-only pilot; never use eval answers."""
import json
import sys
import time
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_count_ab import switch, MODELS
from scripts.run_sft_validation_job import remote
from scripts.build_shared_dpo import OUT, digest
from scripts.regen_v1_v2_current_protocol import sources,saved_maps,audit,make_tokenizer
from scripts.prepare_count_infeasible_ab import check
from scripts.post_train_runtime import append_json
from app.planning.schemas import TravelRoute

def status(stage,**kw):
    (OUT/'pilot_status.json').write_text(json.dumps(dict(stage=stage,updated_at=time.time(),teacher_calls=0,**kw),indent=2),encoding='utf-8')

def preflight():
    result=remote('''
import json,hashlib
from pathlib import Path
root=Path('/home/jiale/TravelMind-posttrain')
paths={'A':root/'saves/TravelMind-count-ab-20261025-A','B':root/'saves/TravelMind-count-ab-total96-20261025-B'}
out={}
for tag,p in paths.items():
 cfg=json.loads((p/'adapter_config.json').read_text())
 out[tag]={'config':cfg,'bytes':(p/'adapter_model.safetensors').stat().st_size,'sha256':hashlib.sha256((p/'adapter_model.safetensors').read_bytes()).hexdigest()}
print(json.dumps(out))
''')
    (OUT/'model_binding.json').write_text(result,encoding='utf-8')

def main():
    preflight();rows=json.loads((OUT/'pilot30.json').read_text(encoding='utf-8'))
    _,_,tr,va=sources();cases={c['id']:c for c in tr+va};maps=saved_maps();tok=make_tokenizer()
    path=OUT/'pilot_model_outputs.jsonl'
    done={(r['tag'],r['id']) for r in [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]} if path.exists() else set()
    try:
        for tag in ['v2-current','A','B']:
            if all((tag,r['id']) in done for r in rows):continue
            status('loading',model=tag);switch(tag)
            name='count-ab-'+tag
            # Verify the requested adapter is actually advertised before inference.
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open('http://10.14.3.82:8000/v1/models',timeout=10) as h: inventory=json.load(h)
            advertised={m['id']:m for m in inventory['data']}
            if name not in advertised:raise RuntimeError('Missing requested model '+name)
            if tag in ['A','B'] and advertised[name].get('parent') is None:
                raise RuntimeError('Expected LoRA adapter, got base model')
            for row in rows:
                if (tag,row['id']) in done:continue
                rec=dict(tag=tag,id=row['id'],category=row['category'],messages_sha256=digest(row['messages']),teacher_calls=0)
                request=dict(model=name,messages=row['messages'],temperature=.3,seed=42,max_tokens=2048,
                             response_format={'type':'json_schema','json_schema':dict(name='TravelRoute',strict=False,schema=TravelRoute.model_json_schema())})
                t=time.perf_counter()
                try:
                    req=urllib.request.Request('http://10.14.3.82:8000/v1/chat/completions',data=json.dumps(request,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
                    with opener.open(req,timeout=120) as h:raw=json.load(h)
                    rec['response']=raw;rec['usage']=raw.get('usage')
                    if raw['choices'][0]['finish_reason']!='stop':raise ValueError('incomplete_generation')
                    plan=TravelRoute.model_validate_json(raw['choices'][0]['message']['content']).model_dump(mode='json')
                    rec['plan']=plan
                    if row['id'].startswith('count-'):
                        from scripts.build_shared_dpo import read,DATA
                        countcase=next(r for r in read(DATA/'count_infeasible_ab_20261025/train.ledger.json') if r['id']==row['id'])
                        faults=check(countcase,plan)
                    else:faults=audit(cases[row['id']],plan,maps,tok)[0]
                    rec['faults']=faults;rec['passed']=not faults
                except Exception as exc:
                    rec.update(passed=False,error_type=type(exc).__name__,error=str(exc)[:1500])
                rec['elapsed_s']=time.perf_counter()-t
                append_json(path,rec);done.add((tag,row['id']))
                status('collecting',model=tag,done=len(done),target=3*len(rows))
        status('completed',done=len(done),target=3*len(rows))
    except Exception as exc:
        status('failed',error=str(exc));raise
    finally:
        # switch('base') uses an evaluation alias, so restore the original name.
        from scripts.diagnose_sft_transfer import switch as restore
        from scripts import run_shared_prompt_all_versions as shared
        shared.MODELS['base']=('/home/jiale/models/Qwen2.5-7B-Instruct','Qwen2.5-7B-Instruct')
        restore('base',max_model_len=8192)

if __name__=='__main__':main()
