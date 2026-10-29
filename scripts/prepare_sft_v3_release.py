"""Audited contract migration plus curated supplement, never rewrite source ledgers."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_sft_v2_release import read_rows, explanation_faults
from scripts.post_train_runtime import capture_messages, replay_check, quality_faults
from scripts.run_hard_post_train_synthesis import audit_plan
from scripts.supplement_planning_contract import check
from scripts.export_llamafactory_dataset import export
from app.evaluation.task_robustness import load_assets
from transformers import AutoTokenizer
from scripts.audit_sft_contract_targets import aggregate_duration_faults
OUT=ROOT/'data/post_train/release_v3'
REJECT={
 'synth-0127':'confirmed repair-context narrative leakage',
 'newcity-contract-成都-explanation-1':'claims 150 total minutes but 90+75=165',
 'newcity-contract-成都-multi_day-0':'60分钟以上 may imply strictly greater, actual60; exclude ambiguous target',
 'newcity-contract-苏州-sparse-0':'query says after10:00, target starts10:00',
 'newcity-contract-苏州-sparse-1':'query says after10:00, target starts10:00',
 'newcity-contract-苏州-multi_day-0':'query says after10:00, target starts10:00',
}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'TRAINING_STARTED.json').exists():raise RuntimeError('Frozen trained release')
 folder=ROOT/'data/post_train'
 old=read_rows(folder/'release_v2/train.ledger.jsonl');supp=read_rows(folder/'contract_supplement_v3/synthesis.jsonl')
 val=read_rows(folder/'release_v1/validation.ledger.jsonl')
 _,maps=load_assets()
 newmaps=json.loads((folder/'new_city_sft_v1/maps.json').read_text(encoding='utf-8'))['routes']
 newcases={c['id']:c for c in json.loads((folder/'new_city_sft_v1/cases.json').read_text(encoding='utf-8'))}
 hard={c['id']:c for c in json.loads((folder/'hard_supplement_cases.json').read_text(encoding='utf-8'))}
 scases={c['id']:c for c in json.loads((folder/'contract_supplement_v3/cases.json').read_text(encoding='utf-8'))}
 tokenizer=AutoTokenizer.from_pretrained(str(Path.home()/'.cache/travelmind-qwen-tokenizer'),local_files_only=True)
 selected=[];quarantine=[];lengths=[];seen=set()
 forbidden={json.dumps(capture_messages(r['state']),ensure_ascii=False,sort_keys=True) for r in val}
 for source in old+supp:
  r=copy.deepcopy(source);errors=[]
  if r['id'] in REJECT:errors.append(REJECT[r['id']])
  if r['id'] in scases:errors+=check(r['chosen'],scases[r['id']],newmaps)
  elif r['id'] in newcases:errors+=audit_plan(r['chosen'],newcases[r['id']],newmaps)[0]
  elif r['id'] in hard:errors+=audit_plan(r['chosen'],hard[r['id']],maps)[0]
  else:
   faults,state=replay_check(r['chosen'],r['state'],maps);errors+=faults+quality_faults(state)
  errors+=explanation_faults(r['chosen'])+aggregate_duration_faults(r['chosen'])
  if r['id'] not in scases:
   r['original_messages']=copy.deepcopy(r['messages'])
   r['messages']=capture_messages(r['state'])
   r['prompt_origin']='audited_contract_migration_of_historical_teacher_target_not_new_teacher_call'
  else:r['prompt_origin']='actual_teacher_boundary_capture'
  r['protocol']='planner-contract-v1'
  r['rejected']=None;r['rejected_rule_errors']=[]
  key=json.dumps(r['messages'],ensure_ascii=False,sort_keys=True)
  if key in forbidden or r['state']['destination'] in ['丽江','景德镇']:errors.append('validation_or_test_leakage')
  if key in seen:errors.append('duplicate_prompt')
  seen.add(key)
  n=len(tokenizer.apply_chat_template(r['messages']+[dict(role='assistant',content=json.dumps(r['chosen'],ensure_ascii=False))],tokenize=True))
  if n>3584:errors.append('over_cutoff')
  if errors:quarantine.append(dict(id=r['id'],reasons=sorted(set(errors))));continue
  selected.append(r);lengths.append(n)
 payloads={}
 for split,rows in [('train',selected),('validation',val)]:
  payloads[split+'.ledger.jsonl']=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
  sft,_=export(rows);assert len(sft)==len(rows)
  payloads['travelmind_planner_sft_'+split+'.json']=json.dumps(sft,ensure_ascii=False,indent=2)
 # Keep the same loss-validation dataset as v2; actual deployment validation uses new contract captures.
 payloads['validation.ledger.jsonl']=(folder/'release_v1/validation.ledger.jsonl').read_text(encoding='utf-8')
 info={f'travelmind_planner_sft_{split}':dict(file_name=f'travelmind_planner_sft_{split}.json',formatting='sharegpt',columns=dict(messages='conversations'),tags=dict(role_tag='from',content_tag='value',user_tag='human',assistant_tag='gpt',system_tag='system')) for split in ['train','validation']}
 payloads['dataset_info.json']=json.dumps(info,indent=2)
 for name,text in payloads.items():
  path=OUT/name
  if path.exists() and path.read_text(encoding='utf-8')!=text:raise RuntimeError('Immutable draft changed')
  path.write_text(text,encoding='utf-8')
 review=dict(reviewer='Codex agent, not human certification',supplement_read=32,excluded_supplement=5,
  checks=['compare reasoning/notes/selection reasons to final schedule','counts and total minutes','unknown indoor remains unknown','no invented citations','strict start-time wording'],
  limitations=['Historical targets pass deterministic replay, not exhaustive semantic recertification','Sparse/multiday supplement chiefly one-stop-per-day, limited complex-road learning','Cannot attribute changes solely to added data: prompt migration and cleanup also changed'],
  per_sample=[dict(id=r['id'],decision='quarantine' if r['id'] in REJECT else 'keep',reason=REJECT.get(r['id'],'reviewed final schedule, explanations, indoor scope and references; no definite contradiction found')) for r in supp])
 payload=json.dumps(review,ensure_ascii=False,indent=2);(OUT/'agent_review.json').write_text(payload,encoding='utf-8')
 summary=dict(train=len(selected),validation=len(val),new_kept=sum(r['id'] in scases for r in selected),historical_migrated=sum(r['id'] not in scases for r in selected),
  validation_ledger_unchanged=(OUT/'validation.ledger.jsonl').read_bytes()==(folder/'release_v1/validation.ledger.jsonl').read_bytes(),
  loss_validation_protocol='unchanged historical capture for continuity, not deployment protocol; loss alone cannot establish quality',
  over_cutoff=0,max_train_tokens=max(lengths),quarantined=quarantine,cities=dict(Counter(r['state']['destination'] for r in selected)),
  historical_messages_preserved=True,new_protocol='planner-contract-v1',max_optimizer_steps=48,comparison='same base/LoRA/LR/seed/48steps as v2; changed dataset size changes epoch exposure',
  sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in list(payloads)+['agent_review.json']},dpo_auto_start=False)
 (OUT/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
