"""Freeze clean-only and augmented SFT arms after explicit agent review."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_quality_sft_v4 import OUT, check_candidate, grounded_target, frozen, dump
from scripts.prepare_sft_v2_release import read_rows, explanation_faults
from scripts.audit_sft_contract_targets import aggregate_duration_faults
from scripts.post_train_runtime import capture_messages
from scripts.export_llamafactory_dataset import export


def canonical_target(row,case=None):
    original=row['original_teacher_target']
    reason=case.get('infeasibility_proof',{}).get('explanation') if case else None
    target=grounded_target(original,row['state'],reason)
    if reason is None:
        target['notes']='开放时间依据输入快照，不代表实时运营保证。未提供的交通、预约、客流及室内范围不作推断。'
    return target


def prepare():
    from transformers import AutoTokenizer
    import yaml
    tokenizer=AutoTokenizer.from_pretrained(str(Path.home()/'.cache/travelmind-qwen-tokenizer'),local_files_only=True)
    cases={c['id']:c for c in json.loads((OUT/'cases.json').read_text(encoding='utf-8'))}
    records=read_rows(OUT/'synthesis.jsonl')
    if len(records)!=len(cases) or len({r['id'] for r in records})!=len(records):
        raise RuntimeError('Synthesis incomplete or duplicated')
    review=json.loads((OUT/'agent_review.json').read_text(encoding='utf-8'))
    if not review['release_allowed'] or set(review['reviewed_ids'])!=set(cases):
        raise RuntimeError('Agent review incomplete')
    if review['synthesis_sha256']!=hashlib.sha256((OUT/'synthesis.jsonl').read_bytes()).hexdigest():
        raise RuntimeError('Reviewed synthesis changed')
    maps=json.loads((OUT/'maps.json').read_text(encoding='utf-8'))['routes']
    history=read_rows(OUT/'historical_cleaned.ledger.jsonl')
    supplemental=[];quarantine=[]
    for source in records:
        if source['status']!='candidate_accepted':
            quarantine.append(dict(id=source['id'],reason='teacher_not_rule_accepted'));continue
        r=copy.deepcopy(source);target=canonical_target(r,cases[r['id']])
        faults=check_candidate(target,cases[r['id']],maps)+explanation_faults(target)+aggregate_duration_faults(target)
        if faults:
            quarantine.append(dict(id=r['id'],reason=faults));continue
        r.update(chosen=target,semantic_review_status='agent_reviewed_grounded_target_not_human_certification')
        supplemental.append(r)
    for r in history:
        r['chosen']=canonical_target(r)
    if len(supplemental)<36:
        raise RuntimeError('Too few diverse clean supplemental targets; no automatic training')
    val=ROOT/'data/post_train/release_v1'
    datasets={}
    for arm,records in [('v4_clean',history),('v4',history+supplemental)]:
        folder=ROOT/'data/post_train'/('release_'+arm)
        if (folder/'TRAINING_STARTED.json').exists():
            raise RuntimeError('Trained release is immutable')
        folder.mkdir(parents=True,exist_ok=True)
        selected=[];lengths=[];supervised=[];rejected=[];seen=set()
        for r in records:
            if r['messages']!=capture_messages(r['state']):
                raise RuntimeError('Current production capture mismatch')
            key=json.dumps(r['messages'],ensure_ascii=False,sort_keys=True)
            if key in seen:
                raise RuntimeError('Duplicate prompt')
            seen.add(key)
            prompt=tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)
            n=len(tokenizer.apply_chat_template(r['messages']+[dict(role='assistant',content=json.dumps(r['chosen'],ensure_ascii=False))],tokenize=True))
            if n>3584:
                rejected.append(dict(id=r['id'],reason='over_cutoff',tokens=n));continue
            selected.append(r);lengths.append(n);supervised.append(n-len(prompt))
        if len(selected)<len(records)*.95:
            raise RuntimeError('More than 5% targets over cutoff; revise inputs, never truncate')
        sft,_=export([dict(r,status='accepted') for r in selected])
        if len(sft)!=len(selected):
            raise RuntimeError('Silent export filtering')
        payloads={'train.ledger.jsonl':''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),
            'travelmind_planner_sft_train.json':json.dumps(sft,ensure_ascii=False,indent=2)}
        for name in ['validation.ledger.jsonl','travelmind_planner_sft_validation.json','dataset_info.json']:
            payloads[name]=(val/name).read_text(encoding='utf-8')
        for name,text in payloads.items():
            p=folder/name
            if p.exists() and p.read_text(encoding='utf-8')!=text:
                raise RuntimeError('Release changed')
            p.write_text(text,encoding='utf-8')
        stats=dict(train=len(selected),validation=30,supplemental=sum(r['id'] in cases for r in selected),
            source_ids=[r['id'] for r in selected],max_train_tokens=max(lengths),total_unique_chat_tokens=sum(lengths),
            total_unique_supervised_tokens=sum(supervised),mean_chat_tokens=sum(lengths)/len(lengths),
            expected_sample_exposure_at_48_steps=48*16/len(selected),expected_chat_tokens_at_48_steps=48*16*sum(lengths)/len(lengths),
            budget='same base, LoRA, seed, 48 optimizer updates, accumulation16; actual nonpadding and supervised tokens logged separately; exposure is not equal',
            over_cutoff=0,quarantined=quarantine+rejected,
            validation_ledger_unchanged=(folder/'validation.ledger.jsonl').read_bytes()==(val/'validation.ledger.jsonl').read_bytes(),
            target_authorship='all prose mechanically reauthored; original teacher targets retained in ledger',
            human_certified=False,dpo_auto_start=False,
            sha256={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in payloads})
        frozen(folder/'audit.json',stats)
        datasets[arm]=stats
        config=yaml.safe_load((ROOT/'configs/post_train/qwen25_7b_planner_sft_v3.yaml').read_text(encoding='utf-8'))
        suffix='v4-clean' if arm=='v4_clean' else 'v4'
        config.update(dataset_dir='data/post_train/release_'+arm,output_dir='saves/TravelMind-Planner-SFT-'+suffix,
            max_steps=48,load_best_model_at_end=False,save_strategy='steps',save_steps=16,
            eval_strategy='steps',eval_steps=16,save_total_limit=1,logging_steps=1)
        p=ROOT/'configs/post_train'/('qwen25_7b_planner_sft_'+arm+'.yaml')
        payload=yaml.safe_dump(config,sort_keys=False,allow_unicode=True)
        if p.exists() and p.read_text(encoding='utf-8')!=payload:
            raise RuntimeError('Training config changed')
        p.write_text(payload,encoding='utf-8')
    frozen(OUT/'release_summary.json',datasets)
    print(json.dumps({k:{n:v[n] for n in ['train','supplemental','max_train_tokens','expected_sample_exposure_at_48_steps']} for k,v in datasets.items()}))


if __name__=='__main__':
    prepare()
