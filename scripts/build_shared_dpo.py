"""Proof-checked shared DPO candidate release, excluding evaluation outputs."""
import copy
import json
import hashlib
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.planning.schemas import TravelRoute
from scripts.regen_v1_v2_current_protocol import sources,saved_maps,audit,make_tokenizer
from scripts.build_verified_preferences import subtle_variants
from scripts.prepare_count_infeasible_ab import check
from scripts.post_train_runtime import capture_messages

DATA=ROOT/'data/post_train'
OUT=DATA/'shared_dpo_20261026'

def read(path):
    text=path.read_text(encoding='utf-8')
    try:return json.loads(text)
    except json.JSONDecodeError:return [json.loads(l) for l in text.splitlines() if l.strip()]

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def variants(plan,state,maps):
    yield from subtle_variants(plan,state,maps)
    for di,d in enumerate(plan['days']):
        for si,s in enumerate(d['spots']):
            if s['start_time'].startswith('0'):
                p=copy.deepcopy(plan);p['days'][di]['spots'][si]['start_time']=s['start_time'][1:]
                yield 'time_format',p
            p=copy.deepcopy(plan);p['days'][di]['spots'][si]['end_time']=s['start_time']
            yield 'nonpositive_duration',p
            if len(d['spots'])>1:
                p=copy.deepcopy(plan);p['days'][di]['spots'][si]['name']=d['spots'][(si+1)%len(d['spots'])]['name']
                yield 'duplicate',p
            p=copy.deepcopy(plan);p['days'][di]['spots'][si]['name']='候选范围外虚构景点'
            yield 'membership',p

def conv(row):
    return dict(conversations=[dict(from_={'user':'human','assistant':'gpt','system':'system'}[m['role']],value=m['content']) for m in row['messages']],
                chosen={'from':'gpt','value':json.dumps(row['chosen'],ensure_ascii=False)},
                rejected={'from':'gpt','value':json.dumps(row['rejected'],ensure_ascii=False)})

def main():
    tok=make_tokenizer();maps=saved_maps();_,_,train,val=sources();byid={r['id']:r for r in train+val}
    counts=Counter();skips=Counter();allrows={};used=set()
    for split in ['train','validation']:
        source=read(DATA/f'release_current_protocol_v1_v2/v2/{split}.ledger.jsonl')
        countrows=read(DATA/f'count_infeasible_ab_20261025/{split}.ledger.json')
        accepted=[]
        for row in source+countrows:
            iscount='required_total' in row
            case=row if iscount else byid[row['id']]
            messages=capture_messages(row['state'])
            if row['messages']!=messages:skips['messages_changed']+=1;continue
            def errors(p):
                try:
                    TravelRoute.model_validate(p)
                    return check(case,p) if iscount else audit(case,p,maps,tok)[0]
                except Exception as e:return ['schema_or_time:'+type(e).__name__]
            chosen=row['chosen']
            bad=errors(chosen)
            if bad:skips['chosen_recheck:'+','.join(bad)]+=1;continue
            if iscount:
                rejected=copy.deepcopy(chosen)
                if row['expected']=='feasible':
                    for day in rejected['days']:day['spots']=[]
                    rejected.update(reasoning='候选不足，无法完成规划。',notes='无法满足本次数量要求。',selection_reasons=[])
                    options=[('false_refusal',rejected)]
                else:
                    witness=next(r for r in countrows if r['pair_id']==row['pair_id'] and r['expected']=='feasible')
                    options=[('count_infeasible_partial',copy.deepcopy(witness['chosen']))]
            else:
                options=list(variants(chosen,row['state'],maps))
            options.sort(key=lambda x:(counts[(split,x[0])],digest([row['id'],x[0],x[1]])))
            mh=digest(messages)
            if mh in used:skips['duplicate_message']+=1;continue
            found=None
            for kind,negative in options:
                defects=errors(negative)
                if not defects:continue
                lengths=[len(tok.apply_chat_template(messages+[dict(role='assistant',content=json.dumps(p,ensure_ascii=False))],tokenize=True,add_generation_prompt=False)) for p in [chosen,negative]]
                if max(lengths)>3584:continue
                found=dict(id=row['id'],source_family=row.get('source_id',row['id']),split=split,state=row['state'],messages=messages,
                           chosen=chosen,rejected=negative,category=kind,chosen_faults=[],rejected_faults=defects,
                           tokens=lengths,provenance='deterministic_negative_existing_verified_chosen',teacher_calls=0)
                break
            if found:accepted.append(found);counts[(split,found['category'])]+=1;used.add(mh)
            else:skips['no_verified_negative_within_cutoff']+=1
        allrows[split]=accepted
    # Pair/source family separation, not merely serialized message separation.
    families=[{r['source_family'] for r in allrows[s]} for s in ['train','validation']]
    if families[0]&families[1]:raise RuntimeError('train/validation source-family overlap')
    forbidden=[]
    for path in [DATA/'representative_eval50_v1/cases.json',DATA/'heldout_evidence_v1/cases.json',DATA/'count_infeasible_ab_20261025/test.ledger.json']:
        forbidden+=read(path)
    testmessages={digest(capture_messages(c['state'])) for c in forbidden}
    if used&testmessages:raise RuntimeError('test message leakage')
    OUT.mkdir(parents=True,exist_ok=True)
    info={}
    for split,rows in allrows.items():
        (OUT/f'{split}.ledger.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        exported=[]
        for r in rows:
            x=conv(r)
            x['conversations']=[{'from':m['from_'],'value':m['value']} for m in x['conversations']]
            exported.append(x)
        name='shared_dpo_'+split
        (OUT/f'{name}.json').write_text(json.dumps(exported,ensure_ascii=False,indent=2),encoding='utf-8')
        info[name]=dict(file_name=name+'.json',formatting='sharegpt',ranking=True,columns=dict(messages='conversations',chosen='chosen',rejected='rejected'),tags=dict(role_tag='from',content_tag='value',user_tag='human',assistant_tag='gpt',system_tag='system'))
    preview=[]
    categories=sorted({r['category'] for r in allrows['train']})
    for index in range(4):
        for category in categories:
            group=[r for r in allrows['train'] if r['category']==category]
            if index<len(group):preview.append(group[index])
    (OUT/'pilot30.json').write_text(json.dumps(preview[:30],ensure_ascii=False,indent=2),encoding='utf-8')
    summary=dict(stage='candidate_not_training_release',teacher_calls=0,training_started=False,
        sizes={s:len(r) for s,r in allrows.items()},categories={s:dict(Counter(r['category'] for r in rs)) for s,rs in allrows.items()},skips=dict(skips),
        source_family_overlap=0,test_message_overlap=0,shared_for=['v2','A','B'],
        limitations=['No natural model negatives collected yet','Rule-certified preferences, not expert subjective rankings','Existing test cities are already seen in previous diagnostics; do not call them pristine holdout','Invalid-time negatives intentionally fail strict schema; JSON syntax stays valid'],
        schema_sha256=digest(TravelRoute.model_json_schema()))
    (OUT/'dataset_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    (OUT/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.name!='manifest.json'},indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True))

if __name__=='__main__':main()
