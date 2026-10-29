"""Freeze proof pairs, preferentially replacing synthetic with audited natural negatives."""
import json
import hashlib
from collections import Counter
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_shared_dpo import OUT,read,digest,conv
from scripts.regen_v1_v2_current_protocol import make_tokenizer

def main():
    status=read(OUT/'pilot_status.json')
    if status.get('stage')!='completed' or status.get('done')!=90:
        raise RuntimeError('Need complete three-model 90-call pilot before release')
    tok=make_tokenizer();rs=read(OUT/'pilot_model_outputs.jsonl')
    pairs=read(OUT/'train.ledger.jsonl');byid={r['id']:r for r in pairs}
    counts=Counter();potential={};skips=Counter()
    for r in rs:
        counts[(r['tag'],'passed' if r['passed'] else 'failed')]+=1
        pair=byid[r['id']]
        if r['messages_sha256']!=digest(pair['messages']):raise RuntimeError('Pilot messages differ')
        if r.get('error_type'):
            skips['generation_or_validation_exception']+=1;continue
        if r.get('passed') or not r.get('faults') or not r.get('plan'):continue
        negative=r['plan']
        n=len(tok.apply_chat_template(pair['messages']+[dict(role='assistant',content=json.dumps(negative,ensure_ascii=False))],tokenize=True,add_generation_prompt=False))
        if n>3584:skips['natural_over_cutoff']+=1;continue
        potential.setdefault(r['id'],[]).append((r,n))
    replacements=Counter()
    for ident,options in potential.items():
        # Prefer the least confounded real negative. Deterministic tie breaking
        # rotates contributing model, rather than overweighting one SFT base.
        options.sort(key=lambda x:(len(x[0]['faults']),replacements[x[0]['tag']],x[0]['tag']))
        r,n=options[0];pair=byid[ident]
        pair.update(rejected=r['plan'],rejected_faults=r['faults'],tokens=[pair['tokens'][0],n],
                    synthetic_category=pair['category'],category='natural_'+pair['category'],
                    provenance='real_model_negative_verified_chosen',negative_model=r['tag'])
        replacements[r['tag']]+=1
    release=OUT/'release';release.mkdir(exist_ok=True);info={}
    for split,rows in [('train',pairs),('validation',read(OUT/'validation.ledger.jsonl'))]:
        exported=[]
        for r in rows:
            x=conv(r);x['conversations']=[{'from':m['from_'],'value':m['value']} for m in x['conversations']];exported.append(x)
        (release/f'{split}.ledger.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        name='shared_dpo_'+split
        (release/f'{name}.json').write_text(json.dumps(exported,ensure_ascii=False,indent=2),encoding='utf-8')
        info[name]=dict(file_name=name+'.json',formatting='sharegpt',ranking=True,columns=dict(messages='conversations',chosen='chosen',rejected='rejected'),tags=dict(role_tag='from',content_tag='value',user_tag='human',assistant_tag='gpt',system_tag='system'))
    audit=read(OUT/'audit.json')
    audit['limitations']=[s for s in audit['limitations'] if s!='No natural model negatives collected yet']
    audit.update(stage='audited_initial_release',natural_replacements=dict(replacements),pilot_counts={tag:{s:counts[tag,s] for s in ['passed','failed']} for tag in ['v2-current','A','B']},pilot_skips=dict(skips),
                 local_generation_calls=len(rs),reported_tokens=sum((r.get('usage') or {}).get('total_tokens',0) for r in rs),
                 teacher_gap_calls_required=0,training_started=False,
                 release_note='346 train +54 validation unique prompts; do not pad to 600 via duplicate prompts. Validation is synthetic-rule preference validation; generation evaluation remains necessary.')
    audit['categories']['train']=dict(Counter(r['category'] for r in pairs))
    (release/'dataset_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    (release/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    (release/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in release.iterdir() if p.name!='manifest.json'},indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=True))

if __name__=='__main__':main()
