"""Rebuild v1/v2 SFT targets from current production messages with a fresh teacher.

Source inputs are the 250 v2 training states plus its shared 30-state
validation split. Old targets are used only for provenance/row selection and are
never sent to the teacher or reused as answers. This script does not train.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.core.env import load_local_env
from app.planning.prompts import JOINT_PLANNER_SYSTEM
from app.planning.schemas import TravelRoute, TravelPlanState
from scripts.post_train_runtime import Meter, append_json, capture_messages, replay_check, quality_faults
from scripts.prepare_sft_v2_release import read_rows, explanation_faults
from scripts.run_large_pool_teacher_v2 import strict_schema, validate_raw
from scripts.prepare_heldout_evidence import opening_evidence_faults
from scripts.prepare_targeted_city_sft import additional_faults
from scripts.run_hard_post_train_synthesis import audit_plan
from scripts.audit_sft_contract_targets import aggregate_duration_faults
from scripts.prepare_quality_sft_v4 import strict_start_faults
from scripts.export_llamafactory_dataset import export
from scripts.compare_route_interest import exclusive_ledger

DATA=ROOT/'data/post_train'
OUT=DATA/'release_current_protocol_v1_v2'
SOURCE_V2=DATA/'release_v2/train.ledger.jsonl'
SOURCE_V1=DATA/'release_v1/train.ledger.jsonl'
VALIDATION=DATA/'release_v2/validation.ledger.jsonl'
RUN_CALL_CAP=400
GLOBAL_CAP=1500
INPUT_FIELDS={'query','destination','travel_start_date','travel_end_date','attraction_preference',
    'food_preference','habit_preference','max_walking_km','rain_indoor_priority','days','max_per_day',
    'min_rating','max_spots','pois','weather_forecast','weather_note','meal_duration_min'}
LOCK=threading.Lock()

def digest_bytes(b): return hashlib.sha256(b).hexdigest()
def digest(value): return digest_bytes(json.dumps(value,ensure_ascii=False,sort_keys=True).encode())
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)
def load_json(path): return json.loads(path.read_text(encoding='utf-8'))

def make_case(source, split):
    # Whitelist only planning inputs. In particular, strip all route/history/review fields.
    state=TravelPlanState(**{k:copy.deepcopy(v) for k,v in source['state'].items() if k in INPUT_FIELDS}).model_dump(mode='json')
    return dict(id=source['id'],split=split,state=state)

def sources():
    v1=read_rows(SOURCE_V1); v2=read_rows(SOURCE_V2); val=read_rows(VALIDATION)
    v1ids={r['id'] for r in v1}; v2by={r['id']:r for r in v2}
    if not v1ids.issubset(v2by) or any(v2by[r['id']]['state']!=r['state'] for r in v1):
        raise RuntimeError('v1 is not an identical-state subset of v2; refuse derived release')
    if len(v2)!=250 or len(v1)!=156 or len(val)!=30:
        raise RuntimeError('Unexpected frozen source cardinality')
    train=[make_case(r,'train') for r in v2]
    validation=[make_case(r,'validation') for r in val]
    if set(x['id'] for x in train)&set(x['id'] for x in validation): raise RuntimeError('train/validation ID overlap')
    # Reused old input families carry independently authored case rubrics for a stricter audit.
    refs={}
    for filename in ['new_city_sft_v1/cases.json','hard_supplement_cases.json']:
        for c in load_json(DATA/filename): refs[c['id']]=c
    for case in train:
        ref=refs.get(case['id'])
        if ref:
            for k in ['rubric','spots_per_day','source_case','category','suite']:
                if k in ref: case[k]=copy.deepcopy(ref[k])
    return v1,v2,train,validation

def saved_maps():
    maps=load_json(DATA/'v2_large_pool_teacher_v1/maps.json')['routes']
    # These are the real offline drive responses used to audit the historical source rows.
    return maps

def make_tokenizer():
    from transformers import AutoTokenizer
    path=Path.home()/'.cache/travelmind-qwen-tokenizer'
    return AutoTokenizer.from_pretrained(str(path),local_files_only=True)

def audit(case,plan,maps,tokenizer):
    state=case['state']; errors=[]
    errors+=replay_check(plan,state,maps)[0]
    checked_state=TravelPlanState(**state)
    errors+=quality_faults(checked_state.model_copy(update={
        'route':[d.model_dump() if hasattr(d,'model_dump') else d for d in plan['days']],
        'meal_slots':[x.model_dump() if hasattr(x,'model_dump') else x for x in plan.get('meal_slots',[])]}))
    errors+=opening_evidence_faults(plan,dict(state=state))
    errors+=explanation_faults(plan)+aggregate_duration_faults(plan)
    if case.get('category')=='strict_start': errors+=strict_start_faults(plan,state['query'])
    if case['id'].startswith('newcity-'):
        errors+=additional_faults(plan,case)
    if case.get('rubric'):
        audit_case=copy.deepcopy(case)
        audit_case['state']=TravelPlanState(**state).model_dump(mode='json')
        errors+=audit_plan(plan,audit_case,maps)[0]
    if sorted(d['day'] for d in plan['days'])!=list(range(1,state['days']+1)):
        errors.append('day_structure')
    if any(not d['spots'] for d in plan['days']):
        errors.append('empty_day_not_independently_proven_infeasible')
    if plan.get('meal_slots'):
        errors.append('unrequested_meal_slots')
    text=json.dumps({k:plan.get(k) for k in ['reasoning','notes','selection_reasons']},ensure_ascii=False)
    if any(t in text for t in ['上一版','上轮','本轮修正','审核反馈','原方案']): errors.append('repair_narrative_leakage')
    messages=capture_messages(state)
    sequence=messages+[{'role':'assistant','content':json.dumps(plan,ensure_ascii=False)}]
    tokens=len(tokenizer.apply_chat_template(sequence,tokenize=True,add_generation_prompt=False))
    if tokens>3584: errors.append('over_cutoff_no_silent_truncation')
    return sorted(set(errors)),tokens

def run(canary_size=10,full=False,concurrency=4):
    load_local_env()
    v1,v2,train,val=sources(); cases=train+val
    maps=saved_maps(); tokenizer=make_tokenizer()
    for c in cases:
        c['messages']=capture_messages(c['state'])
        if c['messages'][0]['content']!=JOINT_PLANNER_SYSTEM: raise RuntimeError('not current production system')
    binding=dict(source_v1=digest_bytes(SOURCE_V1.read_bytes()),source_v2=digest_bytes(SOURCE_V2.read_bytes()),
        source_validation=digest_bytes(VALIDATION.read_bytes()),system_sha256=digest_bytes(JOINT_PLANNER_SYSTEM.encode()),
        schema_sha256=digest(TravelRoute.model_json_schema()),maps_sha256=digest_bytes((DATA/'v2_large_pool_teacher_v1/maps.json').read_bytes()),
        samples={'v1_train':156,'v2_train':250,'shared_validation':30},teacher='gpt-6.1-sol',
        max_total_calls=RUN_CALL_CAP,max_global_started_calls=GLOBAL_CAP,one_rule_repair_max=True,
        instruction='fresh answer from current messages; old targets never included in request',cutoff_len=3584)
    OUT.mkdir(parents=True,exist_ok=True)
    bp=OUT/'binding.json'
    if bp.exists() and load_json(bp)!=binding: raise RuntimeError('Immutable synthesis binding changed')
    write(bp,binding)
    cases_path=OUT/'cases.json'
    public_cases=[{k:v for k,v in c.items() if k not in ['rubric','spots_per_day','source_case','category','suite']} for c in cases]
    if cases_path.exists() and load_json(cases_path)!=public_cases: raise RuntimeError('Frozen cases changed')
    write(cases_path,public_cases)
    ledger=OUT/'synthesis.ledger.jsonl'; calls=OUT/'synthesis.calls.jsonl'
    own=read_rows(ledger) if ledger.exists() else []
    by_id={r['id']:r for r in own}
    done={k for k,r in by_id.items() if r['status']=='accepted'}
    call_rows=read_rows(calls) if calls.exists() else []
    started=sum(x.get('event')=='started' for x in call_rows)
    existing_global=0
    for p in DATA.rglob('*'):
        if not p.is_file() or p.resolve()==calls.resolve(): continue
        # Older ledgers use both *.calls.jsonl and the bare name calls.jsonl.
        if p.name!='calls.jsonl' and not p.name.endswith('.calls.jsonl'): continue
        try:
            existing_global+=sum(json.loads(line).get('event')=='started'
                for line in p.read_text(encoding='utf-8').splitlines() if line.strip())
        except (OSError,UnicodeDecodeError,json.JSONDecodeError):
            raise RuntimeError('Cannot safely count global call ledger: '+str(p))
    future_room=min(RUN_CALL_CAP-started,GLOBAL_CAP-existing_global-started)
    if future_room<0:
        raise RuntimeError(f'call cap already exceeded: external={existing_global}, own={started}, per_run={RUN_CALL_CAP}')
    if future_room==0 and any(r['status']!='accepted' for r in own):
        raise RuntimeError('No global/API budget remains to retry failed records')
    # Completed calls not yet represented by an accepted durable row can be
    # retried; their spend remains charged in the immutable call log.
    # The Meter itself has no global ledger lock, so acquire both locks for the complete run.
    with exclusive_ledger(DATA/'teacher_global_budget.jsonl'),exclusive_ledger(OUT/'synthesis.ledger.jsonl'):
        meter=Meter(calls,'gpt-6.1-sol',RUN_CALL_CAP,90)
        stop=threading.Event(); repair_lock=threading.Lock()
        max_repairs=120
        repairs_used=sum(x.get('event')=='started' and x.get('identity',{}).get('round')==1 for x in call_rows)
        repair_left=max(0,max_repairs-repairs_used)
        def worker(case):
            nonlocal repair_left
            if stop.is_set(): return None
            messages=case['messages']; row=dict(id=case['id'],split=case['split'],state=case['state'],messages=messages,
                status='failed',mock=False,model='gpt-6.1-sol',target_origin='fresh_teacher_current_production',
                old_answer_used=False,rounds=[],chosen=None)
            state=TravelPlanState(**case['state'])
            for round_no in range(2):
                try:
                    req_messages=messages if not round_no else copy.deepcopy(messages)
                    if round_no:
                        req_messages[-1]['content'] += ('\n\n【当前答案返工要求】上一轮输出未通过本地确定性检查：'
                            + '；'.join(row['rounds'][-1]['rule_errors'])
                            + '。请只根据原始输入和这些明确错误重新规划；不得引用或保留上一轮的不一致内容。')
                    if round_no and req_messages[0]['content']!=JOINT_PLANNER_SYSTEM: raise RuntimeError('repair system mismatch')
                    plan,call=meter.call(req_messages,dict(sample=case['id'],split=case['split'],round=round_no,run='v1_v2_current_protocol'))
                    errors,tokens=audit(case,plan,maps,tokenizer)
                    row['rounds'].append(dict(round=round_no,messages=req_messages,output=plan,rule_errors=errors,
                        qwen_tokens=tokens,call_id=call['call_id'],usage=call.get('usage')))
                    if not errors:
                        row.update(status='accepted',chosen=plan,qwen_tokens=tokens,final_messages=req_messages)
                        break
                    row['status']='quarantined'
                    if round_no==0 and not stop.is_set():
                        with repair_lock:
                            if repair_left<=0: break
                            repair_left-=1
                    else: break
                except Exception as exc:
                    row.update(status='api_or_validation_failed',error_type=type(exc).__name__,error=str(exc)[:500])
                    if isinstance(exc,RuntimeError) and any(f'status={x}' in str(exc) for x in [400,401,403,404,429]): stop.set()
                    break
            return row
        pending=[c for c in cases if c['id'] not in done]
        if not full: pending=pending[:canary_size]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            active={pool.submit(worker,c):c for c in pending[:concurrency]}
            queue=iter(pending[concurrency:])
            while active:
                future=next(as_completed(active)); active.pop(future)
                row=future.result()
                if row is not None:
                    append_json(ledger,row)
                    # Store one current row per input; call history remains append-only.
                    current=read_rows(ledger)
                    current=[x for x in current if x['id']!=row['id']]+[row]
                    tmp=ledger.with_suffix('.tmp')
                    tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in current),encoding='utf-8')
                    tmp.replace(ledger)
                    print(json.dumps(dict(id=row['id'],split=row['split'],status=row['status'],recorded=len(current),
                        calls=meter.count,errors=row.get('rounds',[]) [-1].get('rule_errors',[]) if row.get('rounds') else row.get('error_type')),ensure_ascii=False),flush=True)
                nxt=next(queue,None) if not stop.is_set() else None
                if nxt: active[pool.submit(worker,nxt)]=nxt
        result=read_rows(ledger)
        result_summary=dict(recorded=len(result),accepted=sum(x['status']=='accepted' for x in result),
            statuses=dict(Counter(x['status'] for x in result)),calls=meter.count,
            usage=sum((x.get('usage') or {}).get('total_tokens',0) for x in call_rows if x.get('event') in ['completed','failed']),
            pending=len(cases)-len({x['id'] for x in result}),training_started=False,
            accepted_train=sum(x['status']=='accepted' and x['split']=='train' for x in result),
            accepted_validation=sum(x['status']=='accepted' and x['split']=='validation' for x in result),
            rule_errors=dict(Counter(e for x in result for rr in x['rounds'] for e in rr['rule_errors'])))
        write(OUT/'status.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),**result_summary))
        print(json.dumps(result_summary,ensure_ascii=False),flush=True)

def publish():
    v1,v2,train,val=sources(); refs={c['id']:c for c in train+val}
    maps=saved_maps(); tokenizer=make_tokenizer()
    ledger=read_rows(OUT/'synthesis.ledger.jsonl')
    byid={x['id']:x for x in ledger}
    if len(byid)!=280: raise RuntimeError('Need all 250 train + 30 validation records before publish')
    for case in train+val:
        row=byid.get(case['id'])
        if not row or row['status']!='accepted' or row.get('old_answer_used') or row['messages']!=capture_messages(case['state']):
            raise RuntimeError('Accepted current-protocol answer missing/mismatch: '+case['id'])
        errors,_=audit(case,row['chosen'],maps,tokenizer)
        if errors: raise RuntimeError('full offline re-audit failed: '+case['id']+':'+','.join(errors))
    train_by={x['id']:byid[x['id']] for x in train}
    v1ids={x['id'] for x in v1}
    releases={'v1': [train_by[x['id']] for x in train if x['id'] in v1ids], 'v2':[train_by[x['id']] for x in train]}
    for tag,subset in releases.items():
        folder=OUT/tag; folder.mkdir(parents=True,exist_ok=True)
        valrows=[byid[x['id']] for x in val]
        train_export,_=export(subset); val_export,_=export(valrows)
        write(folder/'train.ledger.jsonl',subset);write(folder/'validation.ledger.jsonl',valrows)
        write(folder/'travelmind_planner_sft_train.json',train_export);write(folder/'travelmind_planner_sft_validation.json',val_export)
        info={
          'travelmind_planner_sft_train':{'file_name':'travelmind_planner_sft_train.json','formatting':'sharegpt','columns':{'messages':'conversations'},'tags':{'role_tag':'from','content_tag':'value','user_tag':'human','assistant_tag':'gpt','system_tag':'system'}},
          'travelmind_planner_sft_validation':{'file_name':'travelmind_planner_sft_validation.json','formatting':'sharegpt','columns':{'messages':'conversations'},'tags':{'role_tag':'from','content_tag':'value','user_tag':'human','assistant_tag':'gpt','system_tag':'system'}}}
        write(folder/'dataset_info.json',info)
        audit_info=dict(version=tag,train_count=len(subset),validation_count=len(valrows),
            train_system_sha256=digest_bytes(JOINT_PLANNER_SYSTEM.encode()),all_inputs_exact_current_capture=True,
            all_answers_fresh_teacher=True,teacher='gpt-6.1-sol',old_answers_reused=0,
            training_status='ready_for_separate_review_and_training_decision',training_started=False,
            train_ids=[r['id'] for r in subset],validation_ids=[r['id'] for r in valrows],
            source_binding=load_json(OUT/'binding.json'))
        write(folder/'audit.json',audit_info)
    print(json.dumps({'published':['v1','v2'],'train_counts':{'v1':156,'v2':250},'shared_validation':30,'training_started':False},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',action='store_true');p.add_argument('--all',action='store_true')
    p.add_argument('--limit',type=int,default=10);p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--publish',action='store_true')
    a=p.parse_args()
    if a.publish: publish();return
    v1,v2,train,val=sources(); maps=saved_maps()
    if not a.run:
        print(json.dumps(dict(v1_unique=156,v2_unique=250,shared_validation=30,
            total_inputs=280,new_teacher_initial_calls=280,available_after_previous_calls=568,
            saved_road_entries=len(maps),states_with_correct_current_messages=all(capture_messages(c['state'])[0]['content']==JOINT_PLANNER_SYSTEM for c in train+val)),ensure_ascii=False));return
    if a.concurrency not in [1,2,3,4]: raise ValueError('concurrency must be 1..4')
    run(canary_size=a.limit,full=a.all,concurrency=a.concurrency)

if __name__=='__main__': main()
