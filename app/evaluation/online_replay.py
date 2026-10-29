"""Production LangGraph replay with frozen upstream facts and audited LLM IO.

Never implements a second reviewer/rework policy. build_graph, planner,
distance checker, risk gate, reviewer, time_check and finalize run unchanged.
Global boundary patches are installed ONCE per run; ContextVar isolates cases.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
import copy
import json
import threading
import time
from unittest.mock import patch
import urllib.error
import urllib.request

from app.planning import graph, nodes
from app.planning.schemas import TravelPlanState, TravelRoute
from app.evaluation.route_interest import route_key

SESSION = ContextVar('online_holdout_session')
ROAD_MAPS = {}


class Session:
    def __init__(self, case, seed, backend, maps, *, tag='mock', ledger=None):
        self.case, self.seed, self.backend, self.maps = case, seed, backend, maps
        self.tag, self.ledger = tag, ledger
        self.calls, self.events, self.roads = [], [], []
        self.lock = threading.Lock()

    def call(self, schema, task, messages, temperature):
        normalized = [dict(role={'human':'user','ai':'assistant'}.get(role, role), content=text)
                      for role, text in messages]
        with self.lock:
            rec = dict(case_id=self.case['id'], seed=self.seed, tag=self.tag, task=task,
                       call_index=len(self.calls), schema=schema.__name__, messages=normalized,
                       temperature=temperature, status='started')
            self.calls.append(rec)
            if self.ledger: self.ledger(dict(rec))
        t = time.perf_counter()
        try:
            raw = self.backend(task, schema, normalized, self.seed, temperature, self)
            # Save raw response BEFORE validation, including truncated/invalid generations.
            rec['response'] = raw
            rec['usage'] = raw.get('usage') or {}
            choice = raw['choices'][0]
            rec['finish_reason'] = choice.get('finish_reason')
            rec['raw_content'] = choice['message']['content']
            if choice.get('finish_reason') != 'stop':
                raise ValueError('incomplete_generation')
            result = schema.model_validate_json(rec['raw_content'])
            rec['parsed'] = result.model_dump(mode='json')
            rec['status'] = 'completed'
            return result
        except Exception as exc:
            rec.update(status='failed', error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                rec['http_status'] = exc.code
                rec['error_body'] = exc.read().decode('utf-8', errors='replace')[:4000]
            # No request URL/API headers in failure messages.
            raise
        finally:
            rec['elapsed_s'] = time.perf_counter() - t
            if self.ledger: self.ledger(copy.deepcopy(rec))


class StructuredBoundary:
    def __init__(self, schema, task, temperature):
        self.schema, self.task, self.temperature = schema, task, temperature
        self.model_name = 'online-replay-' + task

    def invoke(self, messages):
        return SESSION.get().call(self.schema, self.task, messages, self.temperature)


def structured_factory(schema, *, model=None, temperature=0, task_type=None, **kwargs):
    return StructuredBoundary(schema, task_type or schema.__name__, temperature)


class HTTPBackend:
    """Fixed role assignment; planner changes, audit roles stay fixed across arms."""
    def __init__(self, *, base_url, planner_model, reviewer_model, time_model=None, api_key='local',
                 timeout=120, max_tokens=2048, audit_url=None, audit_api_key=None):
        self.base_url, self.audit_url = base_url.rstrip('/'), (audit_url or base_url).rstrip('/')
        self.models = dict(joint_planner=planner_model, reviewer=reviewer_model, time_check=time_model or reviewer_model)
        self.key, self.audit_key = api_key, audit_api_key or api_key
        self.timeout, self.max_tokens = timeout, max_tokens
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(self, task, schema, messages, seed, temperature, session):
        if task not in self.models:
            raise RuntimeError('Unexpected paid/model role: ' + task)
        url = self.base_url if task == 'joint_planner' else self.audit_url
        key = self.key if task == 'joint_planner' else self.audit_key
        payload = dict(model=self.models[task], messages=messages, temperature=temperature, seed=seed,
            max_tokens=self.max_tokens, response_format=dict(type='json_schema',
            json_schema=dict(name=schema.__name__, strict=True, schema=schema.model_json_schema())))
        # The exact selected serving/model payload is saved, never credentials.
        session.calls[-1]['request'] = copy.deepcopy(payload)
        req = urllib.request.Request(url+'/chat/completions', data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={'Content-Type':'application/json', 'Authorization':'Bearer '+key})
        with self.opener.open(req, timeout=self.timeout) as response:
            return json.load(response)


class ProductionAuditBackend(HTTPBackend):
    """Use the existing online OpenAI adapter for audit roles (Responses if configured).

    include_raw changes observability only; prompt/schema/temperature/provider
    configuration comes from the production adapter. Audit seed is not injected
    because Responses does not expose the local planner's seed control.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._auditors = {}
        self._lock = threading.Lock()

    def __call__(self, task, schema, messages, seed, temperature, session):
        if task == 'joint_planner':
            return super().__call__(task,schema,messages,seed,temperature,session)
        if task not in {'reviewer','time_check'}:
            raise RuntimeError('Unexpected audit role')
        from app.llm.openai import build_chat_openai
        cache_key=(task,schema)
        with self._lock:
            if cache_key not in self._auditors:
                self._auditors[cache_key]=build_chat_openai(model=self.models[task],temperature=temperature).with_structured_output(
                    schema,method='json_schema',strict=True,include_raw=True)
            auditor=self._auditors[cache_key]
        session.calls[-1]['request']=dict(model=self.models[task],messages=messages,temperature=temperature,
            schema=schema.model_json_schema(),strict=True,adapter='production app.llm.openai',
            seed_policy='online API adapter; no Responses seed parameter')
        reply=auditor.invoke(messages)
        raw=reply.get('raw')
        raw_dump=raw.model_dump(mode='json') if hasattr(raw,'model_dump') else raw
        content=getattr(raw,'content','')
        if isinstance(content,list):
            content=''.join(block.get('text','') for block in content if isinstance(block,dict) and block.get('type') in {'text','output_text'})
        usage=getattr(raw,'usage_metadata',None) or {}
        if not usage:
            usage=(getattr(raw,'response_metadata',None) or {}).get('token_usage',{})
        normalized=dict(prompt_tokens=usage.get('input_tokens',usage.get('prompt_tokens',0)),
            completion_tokens=usage.get('output_tokens',usage.get('completion_tokens',0)),total_tokens=usage.get('total_tokens',0))
        if not normalized['total_tokens']:normalized['total_tokens']=normalized['prompt_tokens']+normalized['completion_tokens']
        metadata=getattr(raw,'response_metadata',None) or {}
        finish=metadata.get('finish_reason') or ('length' if metadata.get('status')=='incomplete' else 'stop')
        return dict(choices=[dict(finish_reason=finish,message=dict(content=content))],usage=normalized,
            production_adapter_raw=raw_dump,parsing_error=str(reply.get('parsing_error')) if reply.get('parsing_error') else None)
class MockBackend:
    """Known witness + scripted auditor, ONLY plumbing verification."""
    def __call__(self, task, schema, messages, seed, temperature, session):
        if task == 'joint_planner':
            obj = session.case['feasibility_witness']
        elif task == 'reviewer':
            obj = dict(reasoning='Mock protocol check only.', approved=True, score=100,
                       route_modify_opinion='', issues=[])
        elif task == 'time_check':
            obj = dict(reasoning='Mock protocol check only.', violations=[])
        else:
            raise RuntimeError('Unexpected role '+task)
        return dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(obj, ensure_ascii=False)))],
                    usage=dict(prompt_tokens=0, completion_tokens=0, total_tokens=0), mock=True)


def replay_distance(origin, destination, key, mode='drive'):
    # Production route_distance_check uses raw ThreadPoolExecutor, which does
    # not propagate ContextVar. The frozen union matrix is immutable per run.
    session = SESSION.get(None)
    result = copy.deepcopy((session.maps if session else ROAD_MAPS).get(route_key(origin, destination, mode)))
    if session:
        with session.lock:
            session.roads.append(dict(origin=origin, destination=destination, mode=mode, found=bool(result)))
    # Missing evidence is NOT estimated. Production checker sees an unavailable leg.
    return result


def no_update(state):
    return {}


@contextmanager
def production_replay(maps=None):
    """Freeze pre-planner inputs; leave all planning/audit/repair edges intact."""
    with ExitStack() as stack:
        stack.enter_context(patch.dict(ROAD_MAPS, maps or {}, clear=True))
        stack.enter_context(patch.object(nodes, 'build_structured_llm', side_effect=structured_factory))
        stack.enter_context(patch.object(nodes, 'search_planning_knowledge', return_value=[]))
        stack.enter_context(patch.object(nodes, 'amap_key', return_value='offline-evidence-only'))
        stack.enter_context(patch.object(nodes, 'plan_route_distance', side_effect=replay_distance))
        stack.enter_context(patch.object(nodes, 'recommend_chain_hotel', return_value=None))
        stack.enter_context(patch.object(nodes, 'lookup_live_ticket_prices', return_value={}))
        stack.enter_context(patch.object(nodes, 'enrich_restaurant_prices', side_effect=lambda rows,*a,**kw:rows))
        stack.enter_context(patch.object(nodes, 'plan_transport_leg', return_value=None))
        # External retrieval/input interpretation held fixed; no hidden teacher calls.
        for name in ['make_intent_node','make_query_rewrite_node','make_candidate_react_node']:
            stack.enter_context(patch.object(graph, name, return_value=no_update))
        for name in ['weather_search_node','attraction_search_node','candidate_search_node',
                     'candidate_validator_node','main_meal_candidate_search_node']:
            stack.enter_context(patch.object(graph, name, new=no_update))
        stack.enter_context(patch.object(graph, 'make_spot_tips_node', return_value=lambda state:
            dict(spot_tips={}, spot_guides={}, spot_tips_status='skipped')))
        yield graph.build_graph()


def run_session(compiled, session, mode):
    """No rubric/witness is exposed to production nodes or real LLM backend."""
    token = SESSION.set(session)
    state = TravelPlanState(**copy.deepcopy(session.case['state']))
    started = time.perf_counter()
    result = dict(id=session.case['id'], seed=session.seed, tag=session.tag, mode=mode,
                  engine='production_langgraph', parse_success=False, approved=False)
    acc = state.model_dump(mode='json')
    try:
        if mode == 'planner-only':
            update = nodes.make_joint_planner_node(None)(state)
            acc.update(update)
            session.events.append(dict(node='planner', update=update))
        elif mode == 'online-rework':
            # Exactly the online run_stream recursion bound; exhaustion is recorded, not silently extended.
            limit=2*(state.max_review_rounds+1)+2*state.max_time_check_rounds+20
            for event in compiled.stream(state, config={'recursion_limit':limit}, stream_mode='updates'):
                for node, update in event.items():
                    session.events.append(dict(node=node, update=copy.deepcopy(update)))
                    if update: acc.update(update)
        else:
            raise ValueError('Unknown mode')
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)[:500] if not isinstance(exc,urllib.error.URLError) else 'transport_error')
    finally:
        result.update(elapsed_s=time.perf_counter()-started, calls=session.calls, events=session.events,
                      road_requests=session.roads, final_state=acc)
        planner_calls=[c for c in session.calls if c['task']=='joint_planner']
        valid=[c for c in planner_calls if c.get('status')=='completed']
        if valid:
            result.update(parse_success=bool(planner_calls[-1].get('status')=='completed'),
                          first_raw_plan=valid[0]['parsed'], raw_plan=valid[-1]['parsed'])
            raw=valid[-1]['parsed']
            result['plan']=dict(raw, days=copy.deepcopy(acc.get('route', [])), meal_slots=copy.deepcopy(acc.get('meal_slots', [])),
                                selection_reasons=copy.deepcopy(acc.get('selection_reasons', [])))
        result['approved']=bool((acc.get('final_plan') or {}).get('approved',acc.get('approved',False)))
        result['planner_calls']=len(planner_calls)
        result['reworks']=max(0,len(planner_calls)-1)
        result['calls_by_role']={role:sum(c['task']==role for c in session.calls) for role in ['joint_planner','reviewer','time_check']}
        result['usage']={key:sum(c.get('usage',{}).get(key,0) or 0 for c in session.calls)
                         for key in ['prompt_tokens','completion_tokens','total_tokens']}
        SESSION.reset(token)
    return result
