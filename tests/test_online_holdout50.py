import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.evaluation.online_replay import MockBackend, Session, production_replay, run_session
from app.evaluation.route_interest import route_key
from scripts.build_online_holdout50 import OUT, CITIES, score_soft, verify
from scripts.evaluate_online_holdout50 import score_result


@pytest.fixture
def example():
    pois=[dict(name=n,location=dict(lng=116.3+i*.01,lat=39.9),open_time='09:00-18:00',indoor=None)
          for i,n in enumerate(['甲博物馆','乙公园'])]
    state=dict(query='参观两个景点，不安排餐馆。',destination='北京',days=1,pois=pois,
               attraction_preference='历史文化',travel_start_date='2026-11-11',travel_end_date='2026-11-11',
               weather_forecast=[dict(date='2026-11-11',day_temp='26',night_temp='20',day_weather='晴',night_weather='晴',is_bad=False)],candidate_pool_status='ready')
    plan=dict(reasoning='使用给定候选。',notes='按候选规划。',days=[dict(day=1,theme='文化',spots=[
        dict(name='甲博物馆',period='morning',start_time='09:00',end_time='10:15'),
        dict(name='乙公园',period='afternoon',start_time='14:00',end_time='15:15')])],meal_slots=[],selection_reasons=[])
    case=dict(id='unit-only',city='北京',category='unit',stratum='normal',expected='feasible',state=state,
        rubric=dict(required=[],excluded=[],counts=[[2,2]],earliest=540,latest=1080,min_visit=75),
        soft_rubric={},feasibility_witness=plan)
    maps={route_key(a['location'],b['location'],'drive'):dict(duration_min=10,distance_km=2,mode='taxi_or_car',source='unit_mock')
          for a in pois for b in pois if a!=b}
    return case,maps


class RejectFirst(MockBackend):
    def __call__(self,task,schema,messages,seed,temperature,session):
        raw=super().__call__(task,schema,messages,seed,temperature,session)
        if task=='reviewer' and sum(c['task']=='reviewer' for c in session.calls)==1:
            obj=dict(reasoning='第1天建议调整安排。',approved=False,score=50,
                     route_modify_opinion='请把第1天乙公园调整到15:00开始，其他要求保持。',issues=['注意休息。'])
            raw['choices'][0]['message']['content']=json.dumps(obj,ensure_ascii=False)
        if task=='joint_planner' and sum(c['task']=='joint_planner' for c in session.calls)>1:
            obj=copy.deepcopy(session.case['feasibility_witness'])
            obj['days'][0]['spots'][1].update(start_time='15:00',end_time='16:15')
            raw['choices'][0]['message']['content']=json.dumps(obj,ensure_ascii=False)
        return raw


def test_production_rework_receives_prior_route_and_reviewer_instruction(example):
    case,maps=example
    session=Session(case,42,RejectFirst(),maps)
    with production_replay(maps) as compiled:
        result=run_session(compiled,session,'online-rework')
    assert not result.get('error_type')
    calls=[c for c in result['calls'] if c['task']=='joint_planner']
    assert len(calls)==2 and result['reworks']==1
    text=calls[1]['messages'][-1]['content']
    assert '上一版路线（仅供修改参考）' in text
    assert '乙公园' in text and '14:00' in text
    assert '调整到15:00' in text
    assert '确定性规则返工反馈' not in text
    assert not any('feasibility_witness' in c['messages'][-1]['content'] for c in calls)
    assert result['approved']
    distances=[e['update'] for e in result['events'] if e['node']=='route_distance_check']
    assert all(x['route_distance_legs'][0]['duration_min']==10 for x in distances)
    assert score_result(result,case,maps)['passed']


def test_time_repair_uses_production_day_poi_detail(example):
    case,maps=example
    class TimeReject(MockBackend):
        def __call__(self,task,schema,messages,seed,temperature,session):
            raw=super().__call__(task,schema,messages,seed,temperature,session)
            if task=='time_check' and sum(c['task']==task for c in session.calls)==1:
                raw['choices'][0]['message']['content']=json.dumps(dict(reasoning='Mock targeted violation.',
                    violations=[dict(day=1,spot_name='乙公园',detail='请调整为15:00，测试定向时间反馈。')]),ensure_ascii=False)
            return raw
    with production_replay(maps) as g:
        r=run_session(g,Session(case,43,TimeReject(),maps),'online-rework')
    planner=[c for c in r['calls'] if c['task']=='joint_planner']
    assert len(planner)==2
    assert '【开放时间修正（第1轮）】' in planner[1]['messages'][-1]['content']
    assert 'Day1 乙公园' in planner[1]['messages'][-1]['content']


def test_reviewer_rejection_budget_is_real_production_budget(example):
    case,maps=example
    class RejectAll(MockBackend):
        def __call__(self,task,schema,messages,seed,temperature,session):
            raw=super().__call__(task,schema,messages,seed,temperature,session)
            if task=='reviewer':
                obj=json.loads(raw['choices'][0]['message']['content']);obj.update(approved=False,score=0,route_modify_opinion='请重新安排。')
                raw['choices'][0]['message']['content']=json.dumps(obj)
            return raw
    with production_replay(maps) as g:
        r=run_session(g,Session(case,42,RejectAll(),maps),'online-rework')
    assert r['planner_calls']==4
    assert not r['approved']
    assert not score_result(r,case,maps)['passed']


def test_invalid_response_is_saved_before_schema_validation(example):
    case,maps=example
    class Invalid(MockBackend):
        def __call__(self,*args):
            return dict(choices=[dict(finish_reason='length',message=dict(content='{"days":['))],
                        usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15))
    records=[]
    with production_replay(maps) as g:
        r=run_session(g,Session(case,42,Invalid(),maps,ledger=records.append),'online-rework')
    failed=[x for x in records if x['status']=='failed']
    assert failed and failed[0]['raw_content']=='{"days":['
    assert failed[0]['finish_reason']=='length'
    assert not r['approved']
    assert r['usage']['total_tokens']==15*len(r['calls'])


def test_parallel_cases_do_not_share_messages_or_context(example):
    case,maps=example
    other=copy.deepcopy(case);other['id']='other';other['state']['query']='另一个请求，只用给定候选。'
    with production_replay(maps) as g:
        with ThreadPoolExecutor(max_workers=2) as pool:
            rs=list(pool.map(lambda c:run_session(g,Session(c,42,MockBackend(),maps),'online-rework'),[case,other]))
    for r,c in zip(rs,[case,other]):
        assert not r.get('error_type')
        assert all(x['case_id']==c['id'] for x in r['calls'])
        assert c['state']['query'] in r['calls'][0]['messages'][-1]['content']


def test_soft_preference_is_separate_from_hard_rules(example):
    c,maps=example;c=copy.deepcopy(c)
    c['soft_rubric']=dict(preferred_names=['未选中的偏好候选'],minimum_hits=1)
    assert score_soft(c['feasibility_witness'],c)==['preferred_category_not_selected']


def test_new_suite_frozen_balance_witnesses_and_provenance():
    if not (OUT/'FROZEN.json').exists():pytest.skip('Capture not yet complete')
    from collections import Counter
    from scripts.build_representative_eval50 import score_case
    manifest=verify()
    cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
    maps=json.loads((OUT/'maps.json').read_text(encoding='utf-8'))['routes']
    assert len(cases)==50 and len({c['id'] for c in cases})==50
    assert Counter(c['city'] for c in cases)==dict.fromkeys(CITIES,5)
    assert Counter(c['stratum'] for c in cases)=={'normal':40,'hard':10}
    assert Counter(c['expected'] for c in cases)=={'feasible':45,'explained_infeasible':5}
    assert manifest['known_train_message_overlap']==0
    for c in cases:
        assert not score_case(c['feasibility_witness'],c,maps),c['id']
        assert not score_soft(c['feasibility_witness'],c),c['id']
        if c['stratum']=='normal' or c['expected']=='explained_infeasible':assert len(c['state']['pois'])>=18
        for p in c['state']['pois']:
            assert p['amap_id'] and p['provenance']['raw_sha256']
            assert '暂停开放' not in p['name']
            assert p['indoor'] is None


def test_rubric_never_drives_online_feedback(example):
    case,maps=example
    case=copy.deepcopy(case);case['rubric']['required']=['不应泄露给模型的评分器专用秘密']
    with production_replay(maps) as g:
        r=run_session(g,Session(case,44,MockBackend(),maps),'online-rework')
    assert all('评分器专用秘密' not in json.dumps(c['messages'],ensure_ascii=False) for c in r['calls'])
    scored=score_result(r,case,maps)
    assert 'required_missing' in scored['faults']
    assert scored['false_approval']
