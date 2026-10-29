"""Exact 249-row diagnostic subset; keep known source defects visible."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_sft_v2_release import read_rows, explanation_faults
from scripts.post_train_runtime import capture_messages
from scripts.export_llamafactory_dataset import export
from app.planning.schemas import TravelPlanState
from app.planning.constraints import heat_violations,effective_daily_limit,hot_day
from app.planning.nodes import _clock_minutes,requires_indoor_only
OUT=ROOT/'data/post_train/release_migration_control'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'TRAINING_STARTED.json').exists():raise RuntimeError('Frozen training release')
 source=ROOT/'data/post_train/release_v3'
 rows=[r for r in read_rows(source/'train.ledger.jsonl') if 'original_messages' in r]
 assert len(rows)==249
 checks=[];known=[]
 for r in rows:
  s=TravelPlanState(**r['state']);plan=r['chosen'];route=plan['days'];errors=[]
  if capture_messages(r['state'])!=r['messages']:errors.append('capture_mismatch')
  if heat_violations(s,route):errors.append('new_heat_boundary_conflict')
  if any(len(d['spots'])>effective_daily_limit(s) for d in route):errors.append('new_daily_cap_conflict')
  by={p['name']:p for p in s.pois}
  if requires_indoor_only(s) and any(by[x['name']].get('indoor') is not True for d in route for x in d['spots']):errors.append('new_indoor_constraint_conflict')
  m=re.search(r'(\d{1,2}:\d{2})后开始',s.query)
  issue=[]
  if m and any(_clock_minutes(x['start_time'])<=_clock_minutes(m[1]) for d in route for x in d['spots']):
   issue.append('preexisting strict after versus inclusive start; target unchanged from source')
   known.append(dict(id=r['id'],fault=issue[0],boundary=m[1],origin='present in original teacher target, not introduced by migration'))
  claims=[]
  for reason in plan.get('selection_reasons',[]):
   text=reason['reason'];poi=by.get(reason['name'],{})
   if poi.get('indoor') is not True and '室内' in text:
    claims.append(dict(name=reason['name'],indoor=poi.get('indoor'),text=text))
  checks.append(dict(id=r['id'],new_contract_conflicts=errors,existing_issues=issue,
    hot_days=[d for d in range(1,s.days+1) if hot_day(s,d)],effective_daily_cap=effective_daily_limit(s),
    selected=[dict(day=d['day'],name=x['name'],indoor=by[x['name']].get('indoor'),start=x['start_time'],end=x['end_time']) for d in route for x in d['spots']],indoor_claims_needing_context=claims))
 if any(c['new_contract_conflicts'] for c in checks):raise RuntimeError('Migration conflict found; do not train control')
 # Two flagged non-indoor POI reasons refer to another indoor visit / midday rest, not upgrading that POI.
 review=dict(n=249,scope='all raw schedules checked for new contract contradictions; contextual claims screening, not exhaustive semantic certification',
  new_contract_route_conflicts=0,preexisting_strict_start_issues=known,
  contextual_review={'synth-0070':'Shanghai Yuyuan reason describes combination with a separate indoor museum; does not relabel Yuyuan',
   'synth-0042':'Luhuitou reason refers to indoor midday rest preceding evening visit, not Luhuitou indoor status'},
  provenance_risks='legacy true labels remain unverified snapshot declarations; scoped new-city rain visits come from explicit user exhibition scope. No label has been upgraded during migration.',
  causal_limits='diagnostic fixed48-step comparison to v3; fewer rows changes exposure frequency and epochs; not proof that 27 samples are intrinsically harmful',
  disposition='Keep exact v3 historical targets including four existing issues to isolate removal of supplement; diagnostic-only, not certified production training release',checks=checks)
 (OUT/'semantic_audit.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
 payloads={'train.ledger.jsonl':''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)}
 sft,_=export(rows);assert len(sft)==249
 payloads['travelmind_planner_sft_train.json']=json.dumps(sft,ensure_ascii=False,indent=2)
 for name in ['validation.ledger.jsonl','travelmind_planner_sft_validation.json','dataset_info.json']:
  payloads[name]=(source/name).read_text(encoding='utf-8')
 for name,text in payloads.items():
  path=OUT/name
  if path.exists() and path.read_text(encoding='utf-8')!=text:raise RuntimeError('Frozen release content changed')
  path.write_text(text,encoding='utf-8')
 audit=dict(train=249,validation=30,over_cutoff=0,max_train_tokens=json.loads((source/'audit.json').read_text(encoding='utf-8'))['max_train_tokens'],
  validation_ledger_unchanged=(OUT/'validation.ledger.jsonl').read_bytes()==(source/'validation.ledger.jsonl').read_bytes(),
  exact_historical_subset_of_v3=True,known_issues=known,diagnostic_only=True,dpo_auto_start=False,
  sha256={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in list(payloads)+['semantic_audit.json']})
 (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in audit.items() if k!='sha256'},ensure_ascii=False))
if __name__=='__main__':main()
