from types import SimpleNamespace
import sys

from scripts.evaluate_controlled_online_holdout import local_rules_rework, summarize_rework
from app.evaluation.online_replay import SESSION, Session
from app.planning.schemas import TravelPlanState


def test_local_rules_rework_uses_fault_feedback_and_allows_three_extra_calls(monkeypatch):
    import copy,json
    from scripts.build_online_holdout50 import OUT
    from scripts.evaluate_controlled_online_holdout import local_rules_rework
    from scripts.build_representative_eval50 import score_case
    cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
    maps=json.loads((OUT/'maps.json').read_text(encoding='utf-8'))['routes']
    case=next(c for c in cases if c['expected']=='feasible')
    witness=copy.deepcopy(case['feasibility_witness'])
    broken=copy.deepcopy(witness);broken['days'][0]['spots'][0]['name']='__outside_pool__'
    updates=[dict(route=broken['days'],meal_slots=[],selection_reasons=[]),
             dict(route=witness['days'],meal_slots=[],selection_reasons=[])]
    seen=[];session=SimpleNamespace(events=[])
    state=TravelPlanState(**case['state'])
    def planner(current):
        seen.append(current.route_modify_opinion)
        update=updates[min(len(seen)-1,len(updates)-1)]
        # Preserve the response explanation fields used by the deterministic rubric.
        session.calls[-1]['parsed']=dict(witness if len(seen)>1 else broken)
        return update
    session.calls=[]
    def tracked_planner(current):
        session.calls.append({'task':'joint_planner','status':'completed'})
        return planner(current)
    _,attempts,passed=local_rules_rework(state,case,maps,session,planner=tracked_planner)
    assert passed and len(attempts)==2
    assert attempts[0]['faults'] and attempts[1]['faults']==[]
    assert '确定性规则校验反馈' in seen[1]
    assert not any(e['node'] in {'reviewer','time_check'} for e in session.events)

    session.events=[];session.calls=[];seen.clear()
    def always_bad(current):
        session.calls.append({'task':'joint_planner','status':'completed'})
        session.calls[-1]['parsed']=broken
        return dict(route=broken['days'],meal_slots=[],selection_reasons=[])
    _,attempts,passed=local_rules_rework(state,case,maps,session,planner=always_bad)
    assert not passed and len(attempts)==4


def test_recovery_is_paired_by_case_and_seed():
    def row(case, seed, passed, calls, tokens):
        return dict(id=case, seed=seed, rules_pass=passed, planner_calls=calls,
                    usage={'total_tokens': tokens})
    planner=[row('a',42,False,1,100),row('a',43,True,1,100),row('b',42,False,1,100)]
    rework=[row('a',42,True,2,180),row('a',43,False,1,90),row('b',42,False,2,190)]
    m=summarize_rework(planner,rework)
    assert m['planner_only_rule_failures']==2
    assert m['corrected_failures']==1
    assert m['rework_recovery_rate_pct']==50
    assert m['avg_rework_turns']==2/3
    assert m['regression_after_rework']==1


def test_in_pool_faithfulness_counts_repeated_outside_entities(tmp_path):
    import copy,json
    from pathlib import Path
    from scripts.build_online_holdout50 import OUT
    from scripts.evaluate_controlled_online_holdout import grade

    cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
    case=next(c for c in cases if c['expected']=='feasible')
    plan=copy.deepcopy(case['feasibility_witness'])
    spot=copy.deepcopy(plan['days'][0]['spots'][0])
    spot['name']='Invented Place'
    plan['days'][0]['spots'].extend([spot,copy.deepcopy(spot)])
    result=dict(plan=plan,raw_plan=plan,parse_success=True,approved=False,mode='planner-only')
    maps=json.loads((OUT/'maps.json').read_text(encoding='utf-8'))['routes']
    measured=grade(result,case,maps)
    assert measured['entity_outside']==2
    assert measured['outsider_names']==['Invented Place']


def test_infeasible_is_correct_refusal_not_a_feasible_rule_pass():
    import json
    from scripts.build_online_holdout50 import OUT
    from scripts.evaluate_controlled_online_holdout import grade, summarize

    cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
    case=next(c for c in cases if c['expected']=='explained_infeasible')
    plan=case['feasibility_witness']
    row=grade(dict(plan=plan,raw_plan=plan,parse_success=True,approved=False,mode='planner-only',
                   id=case['id'],seed=1001,calls=[],calls_by_role={'reviewer':0,'time_check':0},usage={},
                   elapsed_s=1.0,planner_calls=1),case,{})
    assert row['correct_task'] is True
    assert row['rules_pass'] is False
    assert row['refusal_compliant'] is True
    summary=summarize([row],[case])
    assert summary['all_rule_pass_rate'] is None
    assert summary['infeasible_recognition_rate']==100
    assert summary['refusal_compliance_rate']==100
    assert summary['overall_all_rule_pass_rate']==0


def test_planner_only_is_projection_of_first_call_in_same_rework_trace():
    from scripts.evaluate_controlled_online_holdout import planner_only_prefix

    closure=dict(id='c',seed=1001,calls=[
        dict(task='joint_planner',status='completed',parsed={'days':[]},usage={'total_tokens':11},elapsed_s=1.2),
        dict(task='reviewer',status='completed',usage={'total_tokens':7}),
        dict(task='joint_planner',status='completed',parsed={'days':[]},usage={'total_tokens':13},elapsed_s=1.4),
    ],events=[dict(node='planner',update={'route':[{'day':1,'spots':[]}], 'meal_slots':[]})],
       usage={'total_tokens':31},elapsed_s=8.0,active_elapsed_s=7.0,planner_calls=2,reworks=1,
       calls_by_role={'joint_planner':2,'reviewer':1,'time_check':1})
    projected=planner_only_prefix(closure)
    assert projected['plan']['days']==[{'day':1,'spots':[]}]
    assert projected['calls']==closure['calls'][:1]
    assert projected['usage']['total_tokens']==11
    assert projected['elapsed_s']==1.2
    assert projected['planner_calls']==1 and projected['reworks']==0


def test_planner_retry_wait_is_excluded_from_stream_latency(monkeypatch):
    import urllib.error
    import scripts.evaluate_controlled_online_holdout as evaluator
    from app.evaluation.online_replay import Session

    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __iter__(self):
            return iter([b'data: {"choices":[{"delta":{"content":"{}"}}]}\n',
                         b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n',
                         b'data: [DONE]\n'])
    calls=[]
    def open_once(*args,**kwargs):
        calls.append(1)
        if len(calls)==1:raise urllib.error.URLError('temporary')
        return Response()
    monkeypatch.setattr(evaluator.time,'sleep',lambda _:None)
    backend=evaluator.StreamingProductionBackend(base_url='http://local/v1',planner_model='p',reviewer_model='r',planner_temperature=.3)
    backend.opener=SimpleNamespace(open=open_once)
    session=Session({'id':'c'},1001,backend,{},tag='test')
    class Schema:
        @classmethod
        def model_json_schema(cls):return {}
    session.calls.append({'task':'joint_planner','call_index':0})
    result=backend('joint_planner',Schema,[],1001,.3,session)
    assert len(calls)==2 and session.calls[-1]['retry_attempts']==1
    assert session.calls[-1]['retry_wait_s']==1
    assert result['stream_metrics']['e2e_s']>=0
    trace=dict(elapsed_s=5.0,calls=[dict(elapsed_s=4.0,retry_wait_s=1.0)])
    evaluator.subtract_retry_wait(trace)
    assert trace['elapsed_s']==4.0 and trace['calls'][0]['elapsed_s']==3.0
    assert trace['wall_elapsed_s']==5.0 and trace['retry_wait_s']==1.0


def test_actual_rework_trigger_and_rule_failure_proxy_are_distinct():
    planner=[dict(id='a',seed=42,rules_pass=True,usage={'total_tokens':12})]
    rework=[dict(id='a',seed=42,rules_pass=True,planner_calls=2,usage={'total_tokens':10})]
    m=summarize_rework(planner,rework)
    assert m['rework_trigger_rate_pct']==100
    assert m['rule_fail_trigger_proxy_pct']==0


def test_fixed_seed_schedule_is_identical_for_each_repair_depth():
    from scripts.evaluate_controlled_online_holdout import StreamingProductionBackend
    from app.evaluation.online_replay import Session

    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __iter__(self):
            return iter([b'data: {"choices":[{"delta":{"content":"{}"}}]}\n',
                         b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n',
                         b'data: [DONE]\n'])
    backend=StreamingProductionBackend(base_url='http://local/v1',planner_model='p',reviewer_model='r',planner_temperature=.3)
    backend.opener=SimpleNamespace(open=lambda *a,**kw:Response())
    case={'id':'c'};session=Session(case,1001,backend,{},tag='test')
    class Schema:
        @classmethod
        def model_json_schema(cls):return {}
    for index in range(2):
        session.calls.append({'task':'joint_planner','call_index':index})
        response=backend('joint_planner',Schema,[],1001,.3,session)
        assert session.calls[-1]['sampling']['request_seed']==1001+10000*index
        assert response['stream_metrics']['completion_tokens']==2


def test_report_placeholders_keep_column_count_and_online_sampling_language(tmp_path, monkeypatch):
    from scripts.report_controlled_online_holdout import main

    monkeypatch.setattr(sys, 'argv', ['report_controlled_online_holdout.py', '--run-dir', str(tmp_path)])
    main()
    report=(tmp_path/'controlled_ablation_report.md').read_text(encoding='utf-8')
    table=[line for line in report.split('## 返工效率',1)[0].splitlines() if line.startswith('|')]
    header=next(line for line in table if line.startswith('|模型|模式|'))
    expected_columns=len(header.split('|'))-2
    model_rows=[line for line in table if (line.startswith('|base|') or line.startswith('|v2_sft|'))
                and ('|planner-only|' in line or '|online-rework|' in line)]
    assert len(model_rows)==4
    assert all(len(line.split('|'))-2==expected_columns for line in model_rows)
    assert 'Pass@1 Mean±Std %' in header and 'All-3-Pass %' in header
    assert 'Feasible All-Rule %' in header and 'Overall All-Rule %' in header
    assert 'temperature=0.3' in report and '1001/1002/1003' in report
    assert '贪心 seed' not in report and 'temperature=0 时' not in report
    assert '45 个可行题' in report and '不可行识别率' in report
