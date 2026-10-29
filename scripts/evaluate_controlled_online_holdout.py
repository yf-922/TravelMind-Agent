"""Controlled Planner evaluation with production prompts and local deterministic rework."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy,hashlib,json,math,os,random,sys,threading,time,urllib.error,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.online_replay import HTTPBackend,ProductionAuditBackend,SESSION,Session,production_replay,run_session
from app.planning import graph, nodes
from app.planning.schemas import TravelPlanState
from unittest.mock import patch
from scripts.build_online_holdout50 import OUT as SUITE,score_soft
from scripts.build_representative_eval50 import score_case
from scripts.prepare_heldout_evidence import opening_window
from scripts.post_train_runtime import append_json
from app.core.env import load_local_env
from scripts.prepare_sft_v2_release import explanation_faults

class StreamingProductionBackend(ProductionAuditBackend):
    """Stream Planner calls to local vLLM; auditing is deterministic and local."""
    def __init__(self,*,planner_temperature=0.0,**kwargs):
        super().__init__(**kwargs);self.stream_url=self.base_url+'/chat/completions';self.planner_temperature=planner_temperature

    def __call__(self,task,schema,messages,seed,temperature,session):
        if task!='joint_planner':
            raise RuntimeError('External reviewer/time_check calls are disabled in local-rules mode')
        planner_round=sum(c['task']=='joint_planner' for c in session.calls[:-1])
        request_seed=seed+10000*planner_round
        payload=dict(model=self.models[task],messages=messages,temperature=self.planner_temperature,top_p=.95,seed=request_seed,
            max_tokens=self.max_tokens,stream=True,stream_options={'include_usage':True},
            response_format=(dict(type='json_object') if getattr(self,'teacher_json_mode',False) else
                dict(type='json_schema',json_schema=dict(name=schema.__name__,strict=True,
                    schema=schema.model_json_schema()))))
        session.calls[-1]['request']=copy.deepcopy(payload)
        session.calls[-1]['sampling']={'repetition_seed':seed,'planner_round':planner_round+1,'request_seed':request_seed}
        req=urllib.request.Request(self.stream_url,data=json.dumps(payload,ensure_ascii=False).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
        retry_wait=0.0;attempts=0;attempt_errors=[]
        for attempt in range(3):
            attempts=attempt+1;started=time.perf_counter();first_token=None;chunks=[];usage={};finish='stop'
            try:
                with self.opener.open(req,timeout=self.timeout) as response:
                    for raw_line in response:
                        line=raw_line.decode('utf-8',errors='replace').strip()
                        if not line.startswith('data:'):continue
                        data=line[5:].strip()
                        if data=='[DONE]':break
                        event=json.loads(data)
                        if event.get('usage'):usage=event['usage']
                        for choice in event.get('choices',[]):
                            delta=choice.get('delta',{}).get('content')
                            if isinstance(delta,str) and delta:
                                if first_token is None:first_token=time.perf_counter()
                                chunks.append(delta)
                            if choice.get('finish_reason'):finish=choice['finish_reason']
                ended=time.perf_counter();break
            except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
                attempt_error=dict(type=type(exc).__name__,message=str(getattr(exc,'reason',exc))[:500])
                if isinstance(exc,urllib.error.HTTPError):
                    session.calls[-1]['http_status']=exc.code
                    session.calls[-1]['error_body']=exc.read().decode('utf-8',errors='replace')[:4000]
                    attempt_error['http_status']=exc.code
                    if exc.code not in {408,425,429,500,502,503,504}:
                        session.calls[-1].update(retry_attempts=attempt,retry_wait_s=retry_wait,
                            retry_errors=attempt_errors+[attempt_error])
                        raise
                attempt_errors.append(attempt_error)
                if attempt==2:
                    session.calls[-1].update(retry_attempts=attempt,retry_wait_s=retry_wait,retry_errors=attempt_errors)
                    raise
                delay=min(2**attempt,4);time.sleep(delay);retry_wait+=delay
        session.calls[-1]['retry_attempts']=attempts-1
        session.calls[-1]['retry_wait_s']=retry_wait
        session.calls[-1]['retry_errors']=attempt_errors
        content=''.join(chunks);prompt=usage.get('prompt_tokens',0);completion=usage.get('completion_tokens',0)
        ttft=(first_token-started) if first_token is not None else None
        duration=ended-started
        tpot=(1000*(duration-ttft)/(completion-1)) if ttft is not None and completion>1 else None
        return dict(choices=[dict(finish_reason=finish,message=dict(content=content))],
            usage=dict(prompt_tokens=prompt,completion_tokens=completion,total_tokens=usage.get('total_tokens',prompt+completion)),
            stream_metrics=dict(ttft_s=ttft,e2e_s=duration,tpot_ms=tpot,completion_tokens=completion,
                                first_content_chunk_observed=first_token is not None))

    def _audited_call(self,task,schema,messages,temperature,session):
        if task not in {'reviewer','time_check'}:raise RuntimeError('Unexpected audit role: '+task)
        from app.llm.openai import build_chat_openai
        cache_key=(task,schema)
        with self._lock:
            if cache_key not in self._auditors:
                llm=_build_audit_client(self.models[task],temperature)
                self._auditors[cache_key]=llm.with_structured_output(schema,method='json_schema',strict=True,include_raw=True)
            auditor=self._auditors[cache_key]
        session.calls[-1]['request']=dict(model=self.models[task],messages=messages,temperature=temperature,
            schema=schema.model_json_schema(),strict=True,adapter='production app.llm.openai; client retries controlled by evaluator')
        retry_wait=0.0;retry_errors=[]
        for attempt in range(3):
            try:
                reply=auditor.invoke(messages);break
            except Exception as exc:
                if not _transient_error(exc) or attempt==2:
                    session.calls[-1].update(retry_attempts=attempt,retry_wait_s=retry_wait,
                        retry_errors=retry_errors+[dict(type=type(exc).__name__,message=str(exc)[:500])])
                    raise
                retry_errors.append(dict(type=type(exc).__name__,message=str(exc)[:500]))
                delay=min(2**attempt,4);time.sleep(delay);retry_wait+=delay
        session.calls[-1].update(retry_attempts=attempt,retry_wait_s=retry_wait,retry_errors=retry_errors)
        raw=reply.get('raw');raw_dump=raw.model_dump(mode='json') if hasattr(raw,'model_dump') else raw
        content=getattr(raw,'content','')
        if isinstance(content,list):
            content=''.join(block.get('text','') for block in content if isinstance(block,dict) and block.get('type') in {'text','output_text'})
        usage=getattr(raw,'usage_metadata',None) or {}
        if not usage:usage=(getattr(raw,'response_metadata',None) or {}).get('token_usage',{})
        normalized=dict(prompt_tokens=usage.get('input_tokens',usage.get('prompt_tokens',0)),
            completion_tokens=usage.get('output_tokens',usage.get('completion_tokens',0)),total_tokens=usage.get('total_tokens',0))
        if not normalized['total_tokens']:normalized['total_tokens']=normalized['prompt_tokens']+normalized['completion_tokens']
        metadata=getattr(raw,'response_metadata',None) or {};finish=metadata.get('finish_reason') or ('length' if metadata.get('status')=='incomplete' else 'stop')
        return dict(choices=[dict(finish_reason=finish,message=dict(content=content))],usage=normalized,
            production_adapter_raw=raw_dump,parsing_error=str(reply.get('parsing_error')) if reply.get('parsing_error') else None)


class StreamingTeacherBackend(StreamingProductionBackend):
    """Teacher Planner over the configured OpenAI-compatible API; local rules still audit."""
    def __init__(self, *, api_key, base_url, model, **kwargs):
        super().__init__(planner_model=model,base_url=base_url,**kwargs)
        self.key = api_key
        self.base_url = base_url.rstrip('/')
        self.models['joint_planner'] = model
        self.stream_url = self.base_url + '/chat/completions'
        self.teacher_json_mode = True
        # Unlike the local vLLM path, the teacher endpoint may require the
        # machine's configured HTTPS proxy.
        self.opener = urllib.request.build_opener()


def _transient_error(exc):
    if isinstance(exc,(TimeoutError,ConnectionError,OSError,urllib.error.URLError)):return True
    try:
        import httpx
        if isinstance(exc,httpx.TransportError):return True
    except ImportError:pass
    return type(exc).__name__ in {'APITimeoutError','APIConnectionError','RateLimitError','InternalServerError'}

def _build_audit_client(model,temperature):
    """Production OpenAI/Responses settings with retries owned by this evaluator."""
    from app.core.env import load_local_env
    from app.core.http import choose_http_proxy
    from app.llm.openai import resolve_openai_base_url,resolve_reasoning_effort
    load_local_env()
    import httpx
    from langchain_openai import ChatOpenAI
    proxy=choose_http_proxy()
    return ChatOpenAI(model=model,api_key=os.getenv('OPENAI_API_KEY') or os.getenv('AI_API_KEY'),
        base_url=resolve_openai_base_url(),temperature=temperature,reasoning_effort=resolve_reasoning_effort(),
        use_responses_api=(os.getenv('OPENAI_USE_RESPONSES_API') or '1').strip().lower() not in {'0','false','no','off'},
        timeout=120,max_retries=0,http_client=httpx.Client(proxy=proxy,trust_env=False),
        http_async_client=httpx.AsyncClient(proxy=proxy,trust_env=False),http_socket_options=())

def verify_frozen_suite_assets(path=SUITE):
    frozen=json.loads((path/'FROZEN.json').read_text(encoding='utf-8'))
    for name,digest in frozen['hashes'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen holdout data changed: '+name)
    if frozen.get('cases')!=50:raise RuntimeError('Controlled protocol requires the frozen 50-case suite')
    return frozen

def grade(result,case,maps):
    if not result.get('plan'):
        return dict(result,delivery=False,faults=['no_parsed_plan'],soft_faults=[],explanation_faults=[],
        rule_dimensions=dict(in_pool=False,opening=False,timeline_transit=False,no_duplicate=False),rules_pass=False,
                    local_rule_pass=False,
                    infeasible_recognized=False if case['expected']=='explained_infeasible' else None,
                    refusal_compliant=False if case['expected']=='explained_infeasible' else None,
                    infeasible_hallucination=False if case['expected']=='explained_infeasible' else None)
    plan=result['plan'];faults=score_case(plan,case,maps);raw_faults=score_case(result['raw_plan'],case,maps)
    pool_names={p['name'] for p in case['state']['pois']}
    visited=[s['name'] for d in plan.get('days',[]) for s in d.get('spots',[])]
    outsider=set(visited)-pool_names
    outsider_occurrences=sum(name not in pool_names for name in visited)
    poi_by_name={p['name']:p for p in case['state']['pois']}
    def minutes(value):
        h,m=map(int,value.split(':'));return 60*h+m
    opening_ok=True;timeline_ok=True
    for day in plan.get('days',[]):
        spots=sorted(day.get('spots',[]),key=lambda x:x.get('start_time','99:99'))
        for spot in spots:
            poi=poi_by_name.get(spot['name'])
            if not poi:opening_ok=False;continue
            try:
                start,end=minutes(spot['start_time']),minutes(spot['end_time'])
                window=opening_window(poi)
                opening_ok &= bool(window and start>=window[0] and end<=window[1] and start<=window[2])
            except (ValueError,TypeError,IndexError):opening_ok=False
        for previous,current in zip(spots,spots[1:]):
            try:
                gap=minutes(current['start_time'])-minutes(previous['end_time']);timeline_ok &= gap>=0
                a=poi_by_name.get(previous['name']);b=poi_by_name.get(current['name'])
                from app.evaluation.route_interest import route_key
                route=maps.get(route_key(a['location'],b['location'],'walk' if case['state'].get('max_walking_km') is not None else 'drive')) if a and b else None
                timeline_ok &= bool(route and route.get('duration_min') is not None and gap>=float(route['duration_min']))
            except (ValueError,TypeError,KeyError):timeline_ok=False
    dimensions=dict(in_pool=not outsider,opening=opening_ok,timeline_transit=timeline_ok,
                    no_duplicate='global_duplicate' not in faults)
    soft=score_soft(plan,case)
    infeasible=case['expected']=='explained_infeasible'
    return dict(result,delivery=bool(result['parse_success']),faults=faults,raw_faults=raw_faults,
        rule_dimensions=dimensions,rules_pass=bool(case['expected']=='feasible' and result['parse_success'] and not faults),
        local_rule_pass=bool(result['parse_success'] and not faults),
        online_approved=None,correct_task=bool(result['parse_success'] and not faults and
            (case['expected']=='explained_infeasible' or result['mode']=='planner-only' or result['approved'])),
        outsider_names=sorted(outsider),entity_total=len(visited),entity_outside=outsider_occurrences,
        infeasible_recognized=(not any(f in faults for f in ['infeasible_not_explained','infeasible_wrong_reason'])) if infeasible else None,
        refusal_compliant=(not faults) if infeasible else None,
        infeasible_hallucination=bool(visited) if infeasible else None,
        soft_faults=soft,explanation_faults=explanation_faults(plan),
        false_approval=result['mode']=='online-rework' and result['approved'] and bool(faults))

def pct(x,n):return 100*x/n if n else None
def qtile(v,q):return sorted(v)[max(0,math.ceil(q*len(v))-1)] if v else None

def planner_only_prefix(rework_result):
    """Project Round 0 from the same production trajectory, without another model call."""
    planner_calls=[c for c in rework_result.get('calls',[]) if c['task']=='joint_planner']
    first=planner_calls[0] if planner_calls else None
    event=next((e for e in rework_result.get('events',[]) if e['node']=='planner'),None)
    parsed=first.get('parsed') if first and first.get('status')=='completed' else None
    update=(event or {}).get('update') or {}
    plan=None
    if parsed is not None and event:
        plan=dict(parsed,days=copy.deepcopy(update.get('route',[])),
            meal_slots=copy.deepcopy(update.get('meal_slots',[])),
            selection_reasons=copy.deepcopy(update.get('selection_reasons',[])))
    usage=copy.deepcopy(first.get('usage') or {}) if first else {}
    return dict(rework_result,id=rework_result['id'],seed=rework_result['seed'],mode='planner-only',
        parse_success=bool(plan),approved=False,plan=plan,raw_plan=parsed,first_raw_plan=parsed,
        calls=[copy.deepcopy(first)] if first else [],
        calls_by_role={role:int(role=='joint_planner' and first is not None) for role in ['joint_planner','reviewer','time_check']},
        usage=usage,planner_calls=int(first is not None),reworks=0,
        elapsed_s=first.get('elapsed_s',0) if first else 0,
        active_elapsed_s=first.get('elapsed_s',0) if first else 0,
        wall_elapsed_s=first.get('wall_elapsed_s',first.get('elapsed_s',0)) if first else 0,
        retry_wait_s=first.get('retry_wait_s',0) if first else 0,
        error_type=first.get('error_type') if first and first.get('status')!='completed' else None,
        source='same Rules+Rework trajectory Round 0')

def local_rules_rework(state,case,maps,session,planner=None):
    """Run production Planner up to four times, using only local rubric feedback."""
    planner=planner or nodes.make_joint_planner_node(None)
    attempts=[];feedback=''
    for round_no in range(4):
        if feedback:
            state=state.model_copy(update={'route_modify_opinion':feedback})
        update=planner(state)
        state=state.model_copy(update=update)
        parsed=session.calls[-1].get('parsed') if session.calls else {}
        plan=dict(parsed or {},days=copy.deepcopy(state.route or []),
                  meal_slots=copy.deepcopy(state.meal_slots or []),
                  selection_reasons=copy.deepcopy(state.selection_reasons or []),
                  modification_concern=(parsed or {}).get('modification_concern',''))
        faults=score_case(plan,case,maps)
        attempt=dict(round=round_no,plan=plan,faults=faults)
        attempts.append(attempt)
        session.events.append(dict(node='planner',round=round_no,update=copy.deepcopy(update)))
        session.events.append(dict(node='local_rule_check',round=round_no,
                                   faults=faults,planner_update=copy.deepcopy(update)))
        if not faults:break
        feedback='确定性规则校验反馈（请逐项修正）：'+'；'.join(faults)
    passed=bool(attempts and not attempts[-1]['faults'])
    return state,attempts,passed

def run_local_rules_session(session):
    """Planner + local verifier only; no Reviewer or time_check clients are reachable."""
    token=SESSION.set(session);started=time.perf_counter()
    state=TravelPlanState(**copy.deepcopy(session.case['state']))
    result=dict(id=session.case['id'],seed=session.seed,tag=session.tag,mode='online-rework',
                engine='production_joint_planner+local_deterministic_rules',parse_success=False,
                approved=False,online_approved=None)
    try:
        state,attempts,passed=local_rules_rework(state,session.case,session.maps,session)
        result.update(local_rule_attempts=attempts,local_rule_pass=passed,
                      rule_errors_by_round=[x['faults'] for x in attempts],
                      approved=passed)
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc)[:500])
    finally:
        planner_calls=[c for c in session.calls if c['task']=='joint_planner']
        valid=[c for c in planner_calls if c.get('status')=='completed']
        last_ok=bool(planner_calls and planner_calls[-1].get('status')=='completed')
        result.update(elapsed_s=time.perf_counter()-started,calls=session.calls,events=session.events,
            road_requests=session.roads,final_state=state.model_dump(mode='json'),
            parse_success=last_ok,planner_calls=len(planner_calls),reworks=max(0,len(planner_calls)-1),
            calls_by_role={role:sum(c['task']==role for c in session.calls)
                           for role in ['joint_planner','reviewer','time_check']},
            usage={k:sum((c.get('usage') or {}).get(k,0) or 0 for c in session.calls)
                   for k in ['prompt_tokens','completion_tokens','total_tokens']})
        if valid and last_ok:
            result['raw_plan']=valid[-1]['parsed'];result['first_raw_plan']=valid[0]['parsed']
            result['plan']=dict(valid[-1]['parsed'],days=copy.deepcopy(state.route or []),
                meal_slots=copy.deepcopy(state.meal_slots or []),
                selection_reasons=copy.deepcopy(state.selection_reasons or []))
        SESSION.reset(token)
    return result

def subtract_retry_wait(result):
    retry_wait=sum(c.get('retry_wait_s',0) for c in result.get('calls',[]))
    for call in result.get('calls',[]):
        call['wall_elapsed_s']=call.get('elapsed_s',0)
        call['elapsed_s']=max(0,call['wall_elapsed_s']-call.get('retry_wait_s',0))
    result['wall_elapsed_s']=result.get('elapsed_s',0)
    result['retry_wait_s']=retry_wait
    result['active_elapsed_s']=max(0,result['wall_elapsed_s']-retry_wait)
    result['elapsed_s']=result['active_elapsed_s']
    return result

def summarize(rows,cases):
    n=len(rows);delivered=[r for r in rows if r['delivery']];passed=[r for r in rows if r['rules_pass']]
    feasible_ids={c['id'] for c in cases if c['expected']=='feasible'}
    feasible_rows=[r for r in rows if r['id'] in feasible_ids]
    infeasible_rows=[r for r in rows if r.get('infeasible_recognized') is not None]
    route_rows=[r for r in feasible_rows if r['delivery']]
    dimension={k:sum(r['rule_dimensions'][k] for r in route_rows) for k in ['in_pool','opening','timeline_transit','no_duplicate']}
    calls=[c for r in rows for c in r.get('calls',[])];planner=[c for c in calls if c['task']=='joint_planner']
    feasible_planner=[c for r in feasible_rows for c in r.get('calls',[]) if c['task']=='joint_planner']
    usage={key:sum((c.get('usage') or {}).get(key,0) or 0 for c in calls) for key in ['prompt_tokens','completion_tokens','total_tokens']}
    stream=[c.get('response',{}).get('stream_metrics',{}) for c in feasible_planner]
    ttft=[x['ttft_s'] for x in stream if x.get('ttft_s') is not None]
    tpot=[x['tpot_ms'] for x in stream if x.get('tpot_ms') is not None]
    lat=[r.get('active_elapsed_s',r['elapsed_s']) for r in feasible_rows]
    seed_rates=[pct(sum(r['rules_pass'] for r in feasible_rows if r['seed']==seed),
                    sum(r['seed']==seed for r in feasible_rows)) for seed in sorted({r['seed'] for r in feasible_rows})]
    seed_mean=sum(seed_rates)/len(seed_rates) if seed_rates else None
    seed_std=math.sqrt(sum((x-seed_mean)**2 for x in seed_rates)/len(seed_rates)) if seed_rates else None
    feasible_cases=[c for c in cases if c['expected']=='feasible']
    per_case={c['id']:sum(r['rules_pass'] for r in rows if r['id']==c['id']) for c in feasible_cases}
    return dict(n=n,delivery_rate=pct(sum(r['delivery'] for r in feasible_rows),len(feasible_rows)),
        overall_delivery_rate=pct(len(delivered),n),all_rule_pass_rate=pct(len(passed),len(feasible_rows)),
        overall_all_rule_pass_rate=pct(len(passed),n),
        all_rule_denominator=len(feasible_rows),feasible_n=len(feasible_rows),infeasible_n=len(infeasible_rows),
        online_approval_rate=None,
        local_rule_acceptance_rate=pct(sum(bool(r.get('local_rule_pass')) for r in rows),n),
        correct_task_rate=pct(sum(r.get('correct_task',False) for r in rows),n),
        pass_at_1_by_seed_pct=seed_rates,pass_at_1_mean_pct=seed_mean,pass_at_1_seed_std_pp=seed_std,
        all_3_pass_rate_pct=pct(sum(v==3 for v in per_case.values()),len(per_case)),
        cases_passes_0_1_2_3=dict(Counter(str(v) for v in per_case.values())),
        subrule_pass_rate={k:pct(v,len(feasible_rows)) for k,v in dimension.items()},subrule_denominator=len(feasible_rows),
        subrule_delivered_denominator=len(route_rows),
        in_pool_entity_faithfulness=pct(sum(max(0,r.get('entity_total',0)-r.get('entity_outside',0)) for r in feasible_rows),
                                        sum(r.get('entity_total',0) for r in feasible_rows)),
        avg_prompt_tokens=usage['prompt_tokens']/n if n else None,
        avg_completion_tokens=usage['completion_tokens']/n if n else None,
        avg_planner_completion_tokens=sum((c.get('usage') or {}).get('completion_tokens',0) or 0 for c in planner)/len(planner) if planner else None,
        avg_planner_output_chars=sum(len(c.get('raw_content','')) for c in planner)/len(planner) if planner else None,
        avg_planner_completion_tokens_feasible=sum((c.get('usage') or {}).get('completion_tokens',0) or 0 for c in feasible_planner)/len(feasible_planner) if feasible_planner else None,
        avg_planner_output_chars_feasible=sum(len(c.get('raw_content','')) for c in feasible_planner)/len(feasible_planner) if feasible_planner else None,
        avg_output_chars=sum(len(c.get('raw_content','')) for c in planner)/n if n else None,
        total_tokens=usage['total_tokens'],mean_tokens_per_task=usage['total_tokens']/n if n else None,
        feasible_e2e_latency_p50_s=qtile(lat,.5),feasible_e2e_latency_p95_s=qtile(lat,.95),
        infeasible_e2e_latency_p50_s=qtile([r.get('active_elapsed_s',r['elapsed_s']) for r in infeasible_rows],.5),
        infeasible_e2e_latency_p95_s=qtile([r.get('active_elapsed_s',r['elapsed_s']) for r in infeasible_rows],.95),
        e2e_latency_p50_s=qtile(lat,.5),e2e_latency_p95_s=qtile(lat,.95),
        ttft_p50_s=qtile(ttft,.5),normalized_tpot_p50_ms=qtile(tpot,.5),
        planner_calls=len(planner),reviewer_calls=0,
        time_check_calls=sum(r['calls_by_role'].get('time_check',0) for r in rows),
        output_chars=sum(sum(len(c.get('raw_content','')) for c in r.get('calls',[])) for r in rows),
        hard_faults=dict(Counter(f for r in rows for f in r.get('faults',[]))),
        soft_faults=dict(Counter(f for r in rows for f in r.get('soft_faults',[]))),
        role_token_totals={role:{k:sum((c.get('usage') or {}).get(k,0) or 0 for c in calls if c['task']==role)
            for k in ['prompt_tokens','completion_tokens','total_tokens']} for role in ['joint_planner','reviewer','time_check']},
        local_rule_feedback_rounds=sum(e['node']=='local_rule_check' for r in rows for e in r.get('events',[])),
        errors=dict(Counter(r.get('error_type') for r in rows if r.get('error_type'))),
        infeasible_recognition_rate=pct(sum(bool(r['infeasible_recognized']) for r in rows if r.get('infeasible_recognized') is not None),sum(r.get('infeasible_recognized') is not None for r in rows)),
        refusal_compliance_rate=pct(sum(bool(r['refusal_compliant']) for r in rows if r.get('refusal_compliant') is not None),sum(r.get('refusal_compliant') is not None for r in rows)),
        infeasible_hallucination_rate=pct(sum(bool(r['infeasible_hallucination']) for r in rows if r.get('infeasible_hallucination') is not None),sum(r.get('infeasible_hallucination') is not None for r in rows)),
        retry_attempts=sum(c.get('retry_attempts',0) for c in calls),retry_wait_s=sum(c.get('retry_wait_s',0) for c in calls),
        planner_only_source='projected from matching Rules+Rework run Round 0',
        paired_mode_rows_are_not_independent_calls=True)

def summarize_rework(planner_rows,rework_rows,success_key='rules_pass'):
    p={(r['id'],r['seed']):r for r in planner_rows};rr={(r['id'],r['seed']):r for r in rework_rows}
    keys=set(p)&set(rr);failed=[k for k in keys if not p[k].get(success_key,False)]
    recovered=sum(rr[k].get(success_key,False) for k in failed)
    rework_calls=sum(max(0,r['planner_calls']-1) for r in rework_rows)
    return dict(n=len(keys),planner_only_rule_failures=len(failed),
        rework_trigger_rate_pct=pct(sum(r['planner_calls']>1 for r in rework_rows),len(rr)),
        rule_fail_trigger_proxy_pct=pct(len(failed),len(p)),avg_rework_turns=rework_calls/len(rr) if rr else None,
        rework_recovery_rate_pct=pct(recovered,len(failed)),recovered=recovered,
        paired_initial_failures=len(failed),corrected_failures=recovered,
        regression_after_rework=sum(p[k].get(success_key,False) and not rr[k].get(success_key,False) for k in keys),
        success_key=success_key,
        total_token_delta_pct=pct(sum(p[k]['usage']['total_tokens'] for k in keys)-sum(rr[k]['usage']['total_tokens'] for k in keys),
                                 sum(p[k]['usage']['total_tokens'] for k in keys)))

def run(args):
    load_local_env()
    manifest=verify_frozen_suite_assets(args.suite);cases=json.loads((args.suite/'cases.json').read_text(encoding='utf-8'))
    maps=json.loads((args.suite/'maps.json').read_text(encoding='utf-8'))['routes']
    if args.limit:cases=cases[:args.limit]
    out=args.output/args.tag;out.mkdir(parents=True,exist_ok=True)
    if args.teacher_api:
        api_key=os.getenv('AI_API_KEY') or os.getenv('OPENAI_API_KEY')
        base_url=args.base_url or os.getenv('AI_BASE_URL') or os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE')
        if not api_key or not base_url:raise RuntimeError('Teacher API requires a configured API key and base URL')
        backend=StreamingTeacherBackend(planner_temperature=args.temperature,base_url=base_url,model=args.model_name,
            reviewer_model='disabled',api_key=api_key,audit_url=base_url,audit_api_key='disabled',max_tokens=args.max_tokens,timeout=90)
    else:
        backend=StreamingProductionBackend(planner_temperature=args.temperature,base_url=args.base_url,planner_model=args.model_name,reviewer_model='disabled',
            api_key='local-only',audit_url=args.base_url,audit_api_key='disabled',max_tokens=args.max_tokens)
    binding=dict(suite=manifest['hashes']['cases.json'],model=args.model_name,
        planner_temperature=args.temperature,top_p=.95,max_tokens=args.max_tokens,concurrency=1,seeds=args.seeds,
        modes=args.modes,weight_form='provider-managed teacher model' if args.teacher_api else 'full BF16 local weights',
        prefix_cache='provider-managed/unspecified' if args.teacher_api else False,warmup_cases=5,
        rework_policy='initial Planner call plus at most three extra Planner calls',
        planner_provider='configured OpenAI-compatible teacher API' if args.teacher_api else 'local vLLM',
        structured_output='json_object with local TravelRoute Pydantic validation' if args.teacher_api else 'strict json_schema',
        audit_mode='local deterministic score_case feedback; no Reviewer/time_check API clients',
        audit_teacher_api_calls=0,planner_api_calls_metered=bool(args.teacher_api))
    bp=out/'binding.json'
    if bp.exists() and json.loads(bp.read_text(encoding='utf-8'))!=binding:raise RuntimeError('Binding differs; select another run id')
    bp.write_text(json.dumps(binding,indent=2),encoding='utf-8')
    result_path=out/'results.jsonl';old=[json.loads(x) for x in result_path.read_text(encoding='utf-8').splitlines() if x.strip()] if result_path.exists() else []
    if old:raise RuntimeError('A partial/old run exists in this tag; use a fresh tag for the controlled protocol')
    lock=threading.Lock()
    def ledger(x):
        with lock:append_json(out/'calls.jsonl',x)
    random.Random(271027).shuffle(cases)
    with production_replay(maps) as compiled:
        warm=[]
        for case in cases[:5]:
            warm_case=copy.deepcopy(case)
            s=Session(warm_case,args.seeds[0],backend,maps,tag='warmup-'+args.tag)
            warm.append(subtract_retry_wait(run_session(compiled,s,'planner-only')))
        warm_calls=[c for x in warm for c in x.get('calls',[])]
        (out/'warmup.json').write_text(json.dumps(dict(n=len(warm),discarded=True,parsed=sum(x['parse_success'] for x in warm),
            usage={k:sum((c.get('usage') or {}).get(k,0) or 0 for c in warm_calls) for k in ['prompt_tokens','completion_tokens','total_tokens']}),indent=2),encoding='utf-8')
        if set(args.modes)!={'planner-only','online-rework'}:
            raise ValueError('Controlled paired run requires planner-only and online-rework modes')
        tasks=[(c,seed) for c in cases for seed in args.seeds]
        random.Random(271027).shuffle(tasks)
        # One serialized production closure supplies both paired measurements.
        for c,seed in tasks:
            c=copy.deepcopy(c)
            session=Session(c,seed,backend,maps,tag=args.tag,ledger=ledger)
            closure=subtract_retry_wait(run_local_rules_session(session))
            for projected in (planner_only_prefix(closure),closure):
                row=grade(projected,c,maps);row['mock']=False
                with lock:
                    append_json(result_path,row);old.append(row)
                    (out/'status.json').write_text(json.dumps(dict(completed=len(old),target=len(cases)*len(args.seeds)*2,
                        source_pairs=sum(r['mode']=='online-rework' for r in old),last_case=c['id'],last_mode=row['mode'],updated_at=time.time()),indent=2),encoding='utf-8')
    summaries={m:summarize([r for r in old if r['mode']==m],cases) for m in args.modes}
    for mode in args.modes:
        for group,expected in [('feasible','feasible'),('infeasible','explained_infeasible')]:
            group_cases=[c for c in cases if c['expected']==expected]
            group_ids={c['id'] for c in group_cases}
            summaries[mode][group]=summarize([r for r in old if r['mode']==mode and r['id'] in group_ids],group_cases)
    if 'planner-only' in summaries and 'online-rework' in summaries:
        summaries['rework_efficiency']=summarize_rework([r for r in old if r['mode']=='planner-only'],[r for r in old if r['mode']=='online-rework'])
        for group,expected in [('feasible','feasible'),('infeasible','explained_infeasible')]:
            group_ids={c['id'] for c in cases if c['expected']==expected}
            summaries['rework_efficiency'][group]=summarize_rework(
                [r for r in old if r['mode']=='planner-only' and r['id'] in group_ids],
                [r for r in old if r['mode']=='online-rework' and r['id'] in group_ids],
                success_key='rules_pass' if group=='feasible' else 'correct_task')
    summaries['binding']=binding
    (out/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summaries.items() if k in args.modes+['rework_efficiency']},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,default=SUITE);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tag',required=True);p.add_argument('--model-name',required=True)
    p.add_argument('--base-url',default=None);p.add_argument('--teacher-api',action='store_true')
    p.add_argument('--max-tokens',type=int,default=2048);p.add_argument('--temperature',type=float,choices=[0.3],default=0.3)
    p.add_argument('--seeds',nargs='+',type=int,default=[1001,1002,1003])
    p.add_argument('--modes',nargs='+',choices=['planner-only','online-rework'],default=['planner-only','online-rework'])
    p.add_argument('--limit',type=int);a=p.parse_args()
    if a.seeds!=[1001,1002,1003]: p.error('controlled protocol requires shared seeds 1001, 1002, 1003')
    run(a)

if __name__=='__main__':main()
