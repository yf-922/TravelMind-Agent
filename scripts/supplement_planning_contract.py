"""Small evidence-backed supplement under the new production contract."""
import argparse
import copy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.core.env import load_local_env
from app.planning.schemas import TravelPlanState
from app.planning.constraints import heat_violations
from scripts.post_train_runtime import capture_messages, Meter, append_json
from scripts.prepare_targeted_city_sft import feasible_plan, additional_faults, clock
from scripts.run_hard_post_train_synthesis import audit_plan
from scripts.prepare_sft_v2_release import explanation_faults, read_rows
from scripts.export_llamafactory_dataset import export
OUT=ROOT/'data/post_train/contract_supplement_v3'
SOURCE=ROOT/'data/post_train/new_city_sft_v1'

def write_frozen(path, value):
    text=json.dumps(value,ensure_ascii=False,indent=2)
    if path.exists() and path.read_text(encoding='utf-8')!=text:raise RuntimeError('Frozen artifact changed: '+path.name)
    path.write_text(text,encoding='utf-8')

def check(plan,case,maps):
    errors=audit_plan(plan,case,maps)[0]+additional_faults(plan,case)+explanation_faults(plan)
    text=plan['reasoning']+' '+plan.get('notes','')
    names=[s['name'] for d in plan['days'] for s in d['spots']]
    if any(t in text for t in ['不重复','无重复','各不相同']) and len(set(names))!=len(names):errors.append('false_unique_claim')
    return sorted(set(errors))

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads((SOURCE/'cases.json').read_text(encoding='utf-8'))
    evidence=json.loads((SOURCE/'indoor_evidence.json').read_text(encoding='utf-8'))
    maps=json.loads((SOURCE/'maps.json').read_text(encoding='utf-8'))['routes']
    cases=[]
    for city in ['成都','苏州']:
        base=next(c for c in source if c['state']['destination']==city and c['category']=='rain')
        full=base['state']['pois']; known=next(p for p in full if p.get('indoor') is True)
        ev=next(e for e in evidence if e['name']==known['name'])
        unknown=[p for p in full if p.get('indoor') is None]
        for category in ['sparse','multi_day','weather_boundary','explanation']:
            for variant in range(4):
                days=2 if category=='sparse' else 3 if category=='multi_day' else 1
                per_day=2 if category=='explanation' else 1
                pool=copy.deepcopy([known]+unknown[variant:variant+2])
                if category=='sparse':pool=pool[:2]
                for p in pool:
                    if p['name']==known['name']:
                        p.update(indoor_source='已保存官方展厅范围证据（非知识库source）',indoor_scope='仅馆内展厅，不含庭院、排队和通勤')
                    else:p.update(indoor_source='unknown',indoor_scope='未核实室内外范围')
                earliest,latest,duration,temp=600,1020,60+15*(variant%3),'24'
                required=pool[0]['name']
                if category=='weather_boundary':
                    temp='37.9' if variant==0 else '38'
                    if variant==0:earliest,latest,required=720,840,pool[1]['name']
                    elif variant==1:earliest,latest=720,840
                    elif variant==2:earliest,latest,duration,required=600,660,60,pool[1]['name']
                    else:earliest,latest,required=900,1020,pool[1]['name']
                start=date(2026,11,3)
                query=(f'{city}{days}日游，2026年11月3日开始。每天最多{per_day}个景点，且每天恰好{per_day}个景点，'
                    f'每个至少{duration}分钟。全程景点不重复；不能因候选不足重复填满。'
                    f'每天{clock(earliest)}后开始，{clock(latest)}前结束。必须参观「{required}」。'
                    '只规划景点，不安排餐馆。解释须与最终点数、景点及分钟数一致，不提不存在的旧路线。当前没有知识库source，selection_reasons的source_ids必须为空数组；室内证据标签不能作source_ids。'
                    f'「{known["name"]}」仅限有证据的馆内展厅，不含庭院、排队和通勤，其他候选属性未知。')
                if category=='weather_boundary':
                    query=query.replace(f'每天{clock(earliest)}后开始，{clock(latest)}前结束。', f'每天不早于{clock(earliest)}开始、不晚于{clock(latest)}结束，允许取端点。')
                    query+=f'模拟最高温{temp}°C，按统一高温协议安排。'
                state=dict(destination=city,days=days,max_per_day=3,pois=pool,query=query,main_meal_candidates=[],
                    travel_start_date=start.isoformat(),travel_end_date=(start+timedelta(days=days-1)).isoformat(),
                    candidate_pool_status='sufficient',rain_indoor_priority=False,
                    weather_forecast=[dict(date=(start+timedelta(days=d)).isoformat(),day_weather='晴',night_weather='晴',day_temp=temp,night_temp='20',is_bad=False) for d in range(days)])
                case=dict(id=f'newcity-contract-{city}-{category}-{variant}',source_id=base['id'],suite='task',category=category,
                    training_use='allowed',spots_per_day=per_day,state=state,
                    rubric=dict(required=[required],excluded=[],min_count=days*per_day,max_count=days*per_day,min_visit=duration,earliest=earliest,latest=latest))
                case['feasibility_witness']=feasible_plan(case,maps)
                errors=check(case['feasibility_witness'],case,maps)
                if errors:raise RuntimeError(str((case['id'],errors)))
                cases.append(case)
    forbidden={json.dumps(r['messages'],ensure_ascii=False,sort_keys=True) for r in read_rows(ROOT/'data/post_train/release_v1/validation.ledger.jsonl')}
    train=read_rows(ROOT/'data/post_train/release_v2/train.ledger.jsonl')
    forbidden.update(json.dumps(r['messages'],ensure_ascii=False,sort_keys=True) for r in train)
    keys=[json.dumps(capture_messages(c['state']),ensure_ascii=False,sort_keys=True) for c in cases]
    if len(set(keys))!=len(cases) or set(keys)&forbidden:raise RuntimeError('duplicate training/dev messages')
    if any(c['state']['destination'] in {'丽江','景德镇'} for c in cases):raise RuntimeError('test leakage')
    write_frozen(OUT/'cases.json',cases)
    write_frozen(OUT/'maps.json',dict(routes=maps))
    write_frozen(OUT/'indoor_evidence.json',evidence)
    cleaned=[r for r in train if r['id']!='synth-0127']
    cleaned_path=OUT/'historical_train_cleaned.ledger.jsonl'
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cleaned)
    if cleaned_path.exists() and cleaned_path.read_text(encoding='utf-8')!=payload:raise RuntimeError('Historical copy changed')
    cleaned_path.write_text(payload,encoding='utf-8')
    write_frozen(OUT/'preflight.json',dict(cases=len(cases),categories=dict(Counter(c['category'] for c in cases)),
        feasibility_witnesses_passed=len(cases),maps_source='saved new_city_sft_v1 real directed roads, no new map calls',
        quarantined=['synth-0127'],historical_kept=len(cleaned),historical_messages='old captured protocol retained; do not claim they match the new production prompt',
        old_release_modified=False,test_cities_untouched=True,teacher_call_cap=48,new_prompt_protocol='planner-contract-v1'))
    return cases,maps

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--limit',type=int);args=parser.parse_args()
    cases,maps=build()
    if not args.run:print(json.dumps(dict(preflight_passed=len(cases),paid_calls=0)));return
    load_local_env()
    own=OUT/'synthesis.calls.jsonl'
    external=sum(json.loads(l).get('event')=='started' for p in (ROOT/'data/post_train').rglob('*.calls.jsonl') if p!=own for l in p.read_text(encoding='utf-8').splitlines() if l.strip())
    if external+48>1000:raise RuntimeError('Global teacher budget insufficient')
    meter=Meter(own,'gpt-6.1-sol',48,90)
    ledger=OUT/'synthesis.jsonl';done={r['id'] for r in read_rows(ledger)} if ledger.exists() else set()
    pending=[c for c in cases if c['id'] not in done]
    if args.limit:pending=pending[:args.limit]
    for case in pending:
        state=copy.deepcopy(case['state'])
        row=dict(id=case['id'],parent_seed=case['source_id'],state=state,messages=capture_messages(state),
            rubric=case['rubric'],mock=False,model='gpt-6.1-sol',protocol='planner-contract-v1',status='failed',chosen=None,rejected=None,rounds=[])
        for rnd in range(2):
            if meter.count>=48:break
            messages=capture_messages(state)
            try:
                plan,call=meter.call(messages,dict(sample=case['id'],round=rnd,run='contract_supplement_v3'))
                errors=check(plan,case,maps)
                row['rounds'].append(dict(round=rnd,messages=messages,output=plan,rule_errors=errors,usage=call.get('usage'),call_id=call['call_id']))
                if not errors:row.update(status='accepted',chosen=plan);break
                state=copy.deepcopy(case['state']);state.update(route=plan['days'],route_modify_opinion='确定性检查未通过：'+','.join(errors)+'。没有知识库source，source_ids必须全部为空数组。输出独立完整最终方案，解释中不要提旧版或本轮修正。')
            except Exception as exc:
                row['error_type']=type(exc).__name__;break
        append_json(ledger,row)
        print(json.dumps(dict(id=row['id'],status=row['status'],calls=meter.count)),flush=True)
    records=read_rows(ledger) if ledger.exists() else []
    accepted=[r for r in records if r['status']=='accepted']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(Path.home()/'.cache/travelmind-qwen-tokenizer'),local_files_only=True)
    lengths={r['id']:len(tokenizer.apply_chat_template(r['messages']+[dict(role='assistant',content=json.dumps(r['chosen'],ensure_ascii=False))],tokenize=True)) for r in accepted}
    if any(n>3584 for n in lengths.values()):raise RuntimeError('No truncation allowed; quarantine long sample')
    sft,_=export(accepted)
    (OUT/'travelmind_contract_sft.json').write_text(json.dumps(sft,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=dict(stage='completed' if len(records)==len(cases) else 'partial',requested=len(cases),completed=len(records),accepted=len(accepted),
      calls=meter.count,global_started=external+meter.count,max_tokens=max(lengths.values(),default=0),lengths=lengths,
      tokens=sum((rd.get('usage') or {}).get('total_tokens',0) for r in records for rd in r['rounds']),
      failed=[r['id'] for r in records if r['status']!='accepted'],human_certified=False,agent_semantic_review='pending',training_started=False)
    (OUT/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
