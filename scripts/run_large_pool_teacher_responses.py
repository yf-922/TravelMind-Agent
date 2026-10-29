"""Fresh-teacher synthesis using the production-compatible Responses API.

The chat transport trial is retained separately. This run has a distinct
immutable ledger, includes every probe in the global budget, and launches no
training. Input messages remain the exact production capture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_large_pool_teacher_v2 as engine
from scripts.post_train_runtime import append_json
from scripts.prepare_quality_sft_v4 import frozen
from scripts.audit_teacher_duration_scope import duration_claim_faults

BASE_AUDIT=engine.audit


def scoped_audit(plan,case,maps,tokenizer):
    errors,checked,tokens=BASE_AUDIT(plan,case,maps,tokenizer)
    # The old audit treated "first day each 120, next day each 90" as two
    # global claims. Replace only that narrow check, not any route constraint.
    errors=[e for e in errors if e!='explanation_visit_duration_mismatch']
    errors+=duration_claim_faults(plan)
    return sorted(set(errors)),checked,tokens


def seed_same_prompt_outputs(cases,maps):
    """Reuse this experiment's fresh raw answers, never historical v2 targets."""
    ledger=engine.OUT/'synthesis.ledger.jsonl'
    if ledger.exists():
        return
    source=ROOT/'data/post_train/v2_large_pool_teacher_contract_v2/synthesis.ledger.jsonl'
    if not source.exists():
        return
    from transformers import AutoTokenizer
    from scripts.post_train_runtime import capture_messages
    from scripts.prepare_sft_v2_release import read_rows
    tokenizer=AutoTokenizer.from_pretrained(str(Path.home()/'.cache/travelmind-qwen-tokenizer'),local_files_only=True)
    refs={c['id']:c for c in cases}
    decisions=[]
    for row in read_rows(source):
        if row['id'] not in refs or row['messages']!=capture_messages(refs[row['id']]['state']):
            continue
        case=refs[row['id']]
        for round_row in row['rounds']:
            faults,_,_=scoped_audit(round_row['output'],case,maps,tokenizer)
            round_row['original_rule_errors']=round_row['rule_errors']
            round_row['rule_errors']=faults
        if not row['rounds']:
            continue
        final=row['rounds'][-1]
        if final['rule_errors']:
            continue
        row.update(status='accepted',chosen=final['output'],qwen_tokens=final['qwen_tokens'],
            carried_from=str(source.relative_to(ROOT)),carried_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            audit_note='same production messages; day-scoped duration audit corrected; raw teacher answer unchanged')
        append_json(ledger,row)
        decisions.append(dict(id=row['id'],source=str(source.relative_to(ROOT)),status='accepted'))
    frozen(engine.OUT/'seed_review.json',decisions)


class ResponsesMeter(engine.StrictMeter):
    def call(self,messages,identity):
        payload=dict(model=self.model,input=messages,
            text=dict(format=dict(type='json_schema',name='TravelRoute',strict=True,schema=engine.strict_schema())),
            reasoning=dict(effort='low'),max_output_tokens=3072,tools=[],tool_choice='none',store=False)
        with self.lock:
            if self.stop.is_set() or self.count>=self.max_calls or self.used_tokens>=self.max_tokens:
                raise RuntimeError('Budget or stop guard reached')
            self.count+=1
            call_id=self.count
            append_json(self.path,dict(event='started',call_id=call_id,identity=identity,time=time.time(),request=payload))
        start=time.perf_counter();raw=None
        try:
            response=self.client.responses.create(**payload)
            raw=response.model_dump(mode='json')
            if raw.get('status')!='completed':
                raise ValueError('response_not_complete')
            if any(o.get('type') not in ['message','reasoning'] for o in raw.get('output',[])):
                raise ValueError('unexpected_tool_output')
            if ((raw.get('tool_usage') or {}).get('web_search') or {}).get('num_requests',0):
                raise ValueError('unexpected_web_search_usage')
            plan=engine.validate_raw(response.output_text)
            record=dict(event='completed',parse_success=True)
        except Exception as exc:
            record=dict(event='failed',parse_success=False,error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
            if hasattr(exc,'absolute_path'):
                record['validation_path']=list(exc.absolute_path)
            if record['http_status'] in [400,401,403,404,429]:
                self.stop.set()
        record.update(call_id=call_id,identity=identity,elapsed_s=time.perf_counter()-start)
        if raw is not None:
            u=raw.get('usage') or {}
            record.update(response=raw,usage=dict(prompt_tokens=u.get('input_tokens',0),
                completion_tokens=u.get('output_tokens',0),total_tokens=u.get('total_tokens',0)))
        with self.lock:
            self.used_tokens+=int((record.get('usage') or {}).get('total_tokens') or 0)
            reported=int((record.get('usage') or {}).get('prompt_tokens') or 0)
            if reported>20000:
                record['usage_warning']='reported_input_over_20000_no_client_side_billing_explanation'
            excessive=reported>40000 or (reported>20000 and record['event']=='failed')
            if excessive or self.used_tokens>=self.max_tokens:
                record['guard_triggered']='excessive_reported_input' if excessive else 'total_token_cap'
                self.stop.set()
            append_json(self.path,record)
        if record['event']=='failed':
            raise RuntimeError(f"Call {call_id}: {record['error_type']} status={record.get('http_status')}")
        return plan,record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--all',action='store_true')
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--concurrency',type=int,choices=[1,2,3,4],default=2)
    p.add_argument('--export-preview',action='store_true')
    args=p.parse_args()
    engine.OUT=ROOT/'data/post_train/v2_large_pool_teacher_contract_v3'
    engine.StrictMeter=ResponsesMeter
    engine.audit=scoped_audit
    cases,maps=engine.prepare()
    frozen(engine.OUT/'driver.meta.json',dict(driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        production_prompt_sha256=hashlib.sha256((ROOT/'app/planning/prompts.py').read_bytes()).hexdigest(),
        scoped_audit_sha256=hashlib.sha256((ROOT/'scripts/audit_teacher_duration_scope.py').read_bytes()).hexdigest(),
        usage_policy=dict(warn_input_over=20000,stop_invalid_input_over=20000,stop_any_input_over=40000,total_reported_cap=2500000),
        transport='responses',tools_disabled=True,strict_json=True,official_reference='https://developers.openai.com/api/docs/guides/structured-outputs'))
    if args.run:
        seed_same_prompt_outputs(cases,maps)
        # Earlier transport/prompt trials have used thirty calls, all counted
        # globally; existing fresh valid answers are carried without API calls.
        engine.run(cases,maps,args.limit,args.all,args.concurrency,max_calls=270)
    if args.export_preview:
        print(json.dumps(dict(sft_preview=engine.release_preview(cases,maps))))


if __name__=='__main__':
    main()
