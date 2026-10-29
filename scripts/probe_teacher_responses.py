"""One metered Responses transport probe using unchanged production messages."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.core.env import load_local_env
from scripts.post_train_runtime import append_json, capture_messages
from scripts.run_large_pool_teacher_v2 import strict_schema, validate_raw, OUT
from scripts.prepare_sft_v2_release import read_rows
from scripts.compare_route_interest import exclusive_ledger


def main():
    load_local_env()
    path=OUT/'responses_transport_probe.calls.jsonl'
    with exclusive_ledger(ROOT/'data/post_train/teacher_global_budget.jsonl'):
        records=read_rows(path) if path.exists() else []
        if records:
            raise RuntimeError('Probe is immutable and already attempted')
        count=sum(r.get('event')=='started' for p in (ROOT/'data/post_train').rglob('*.calls.jsonl') for r in read_rows(p))
        if count>=1000:
            raise RuntimeError('Global call budget exhausted')
        case=next(c for c in json.loads((OUT/'cases.json').read_text(encoding='utf-8')) if c['id']=='synth-0085')
        messages=capture_messages(case['state'])
        payload=dict(model='gpt-6.1-sol', input=messages,
            text=dict(format=dict(type='json_schema',name='TravelRoute',strict=True,schema=strict_schema())),
            reasoning=dict(effort='low'), max_output_tokens=3072, tools=[], tool_choice='none', store=False)
        append_json(path,dict(event='started',call_id=1,identity=dict(sample=case['id'],round=0,run='responses_probe'),
            time=time.time(),request=payload))
        from openai import OpenAI
        client=OpenAI(api_key=os.environ.get('AI_API_KEY'),base_url=os.environ.get('AI_BASE_URL'),timeout=90,max_retries=0)
        start=time.perf_counter();raw=None
        try:
            response=client.responses.create(**payload)
            raw=response.model_dump(mode='json')
            if raw.get('status')!='completed':
                raise ValueError('response_incomplete')
            plan=validate_raw(response.output_text)
            result=dict(event='completed',parse_success=True,plan=plan)
        except Exception as exc:
            result=dict(event='failed',parse_success=False,error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
        result.update(call_id=1,elapsed_s=time.perf_counter()-start)
        if raw:
            usage=raw.get('usage') or {}
            result.update(response=raw,usage=dict(prompt_tokens=usage.get('input_tokens',0),completion_tokens=usage.get('output_tokens',0),total_tokens=usage.get('total_tokens',0)))
        append_json(path,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['response','plan']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
