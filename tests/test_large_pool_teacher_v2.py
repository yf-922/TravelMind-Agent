import copy
import json
import threading
from types import SimpleNamespace

import pytest

from scripts.run_large_pool_teacher_v2 import strict_schema, validate_raw, evidence_faults, StrictMeter, usage_totals


def target():
    return dict(reasoning='按候选坐标规划，未提供设施信息。', days=[dict(day=1, theme='参观',
        spots=[dict(name='候选甲', period='morning', start_time='09:00', end_time='10:30')])],
        meal_slots=[], selection_reasons=[dict(name='候选甲', reason='安排90分钟。', source_ids=[])],
        notes='', modification_concern='')


def test_strict_schema_requires_all_nested_properties_and_disallows_extras():
    schema=strict_schema()
    assert schema['required']==list(schema['properties'])
    assert schema['additionalProperties'] is False
    for node in schema['$defs'].values():
        assert node['required']==list(node['properties'])
        assert node['additionalProperties'] is False
        assert all('default' not in prop for prop in node['properties'].values())


def test_raw_validation_does_not_extract_json_or_coerce_schema():
    value=target()
    assert validate_raw(json.dumps(value))==value
    with pytest.raises(ValueError):
        validate_raw('以下是方案：'+json.dumps(value))
    value['reasoning']=['not a string']
    with pytest.raises(Exception):
        validate_raw(json.dumps(value))
    value=target()
    value['days'][0]['spots'][0]['invented']=1
    with pytest.raises(Exception):
        validate_raw(json.dumps(value))


def test_ferry_and_micro_area_claims_need_evidence_without_modifying_answer():
    value=target()
    value['reasoning']='全天安排陆家嘴片区景点。'
    before=copy.deepcopy(value)
    assert evidence_faults(value, {}, [dict(content='区域 浦东新区')])==['unsupported_micro_area:陆家嘴']
    assert not evidence_faults(value, {}, [dict(content='已知陆家嘴片区')])
    value['days'][0]['spots'][0]['name']='三亚西岛海洋文化旅游区'
    assert 'ferry_timetable_unverified' in evidence_faults(value, {}, [dict(content='陆家嘴')])
    assert before['reasoning']==value['reasoning']


def test_negated_access_guarantees_are_not_affirmative_claims():
    value=target()
    value['notes']='不能保证无需预约，请以官方信息为准。'
    assert not evidence_faults(value, {}, [dict(content='无设施证据')])
    value['notes']='无需预约。'
    assert evidence_faults(value, {}, [dict(content='无设施证据')])==['unsupported_facility_or_access_claim']


def fake_meter(tmp_path, content, usage):
    meter=StrictMeter.__new__(StrictMeter)
    meter.path=tmp_path/'calls.jsonl'
    meter.model='gpt-6.1-sol'
    meter.lock=threading.Lock()
    meter.count=0
    meter.max_calls=1
    meter.used_tokens=0
    meter.max_tokens=2500000
    meter.stop=threading.Event()
    response=dict(choices=[dict(finish_reason='stop', message=dict(content=content))], usage=usage)
    def create(**kwargs):
        assert kwargs['response_format']['json_schema']['strict'] is True
        assert kwargs['store'] is False
        return SimpleNamespace(model_dump=lambda **kw: response)
    meter.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return meter


def test_usage_anomaly_trips_guard_and_still_accounts_paid_response(tmp_path):
    meter=fake_meter(tmp_path, json.dumps(target()), dict(prompt_tokens=50000, completion_tokens=1000, total_tokens=51000))
    meter.call([dict(role='user',content='new input only')], dict(sample='mock',round=0))
    assert meter.stop.is_set()
    records=[json.loads(l) for l in meter.path.read_text(encoding='utf-8').splitlines()]
    assert records[-1]['guard_triggered']=='reported_input_over_20000'
    assert usage_totals(records)['total_tokens']==51000
    with pytest.raises(RuntimeError):
        meter.call([], dict(sample='next',round=0))
    assert meter.count==1


def test_invalid_raw_response_is_logged_as_failure_and_not_cleaned(tmp_path):
    meter=fake_meter(tmp_path, 'Here is JSON: '+json.dumps(target()), dict(total_tokens=500))
    with pytest.raises(RuntimeError):
        meter.call([], dict(sample='mock',round=0))
    records=[json.loads(l) for l in meter.path.read_text(encoding='utf-8').splitlines()]
    assert [r['event'] for r in records]==['started','failed']
    assert records[-1]['parse_success'] is False
    assert records[-1]['response']['choices'][0]['message']['content'].startswith('Here is JSON:')


def test_responses_meter_preserves_messages_and_forbids_tools(tmp_path):
    from scripts.run_large_pool_teacher_responses import ResponsesMeter
    value=target()
    meter=fake_meter(tmp_path, '', {})
    meter.__class__=ResponsesMeter
    messages=[dict(role='system',content='actual production system'),dict(role='user',content='actual expanded input')]
    raw=dict(status='completed',output=[dict(type='message')],usage=dict(input_tokens=4000,output_tokens=500,total_tokens=4500))
    def create(**payload):
        assert payload['input']==messages
        assert payload['tools']==[] and payload['tool_choice']=='none'
        assert payload['text']['format']['strict'] is True
        return SimpleNamespace(output_text=json.dumps(value),model_dump=lambda **kw:raw)
    meter.client=SimpleNamespace(responses=SimpleNamespace(create=create))
    answer,call=meter.call(messages,dict(sample='mock',round=0))
    assert answer==value and call['usage']['total_tokens']==4500
    records=[json.loads(l) for l in meter.path.read_text(encoding='utf-8').splitlines()]
    assert records[0]['request']['input']==messages


def test_responses_with_unrequested_tool_output_is_not_training_answer(tmp_path):
    from scripts.run_large_pool_teacher_responses import ResponsesMeter
    meter=fake_meter(tmp_path, '', {})
    meter.__class__=ResponsesMeter
    raw=dict(status='completed',output=[dict(type='web_search_call')],usage=dict(input_tokens=28000,output_tokens=1000,total_tokens=29000))
    meter.client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kw:SimpleNamespace(output_text=json.dumps(target()),model_dump=lambda **kw:raw)))
    with pytest.raises(RuntimeError):
        meter.call([],dict(sample='mock',round=0))
    assert meter.stop.is_set()
    records=[json.loads(l) for l in meter.path.read_text(encoding='utf-8').splitlines()]
    assert records[-1]['parse_success'] is False
    assert records[-1]['usage']['total_tokens']==29000


def test_day_scoped_duration_claims_are_not_applied_to_other_days():
    from scripts.audit_teacher_duration_scope import duration_claim_faults
    value=target()
    value['days'][0]['spots'][0].update(start_time='09:00',end_time='11:00')
    value['days'].append(dict(day=2,theme='次日',spots=[dict(name='候选乙',period='morning',start_time='09:30',end_time='11:00')]))
    value['reasoning']='首日每处120分钟，次日每处90分钟。全程每处至少90分钟。'
    assert duration_claim_faults(value)==[]
    value['reasoning']='首日每处90分钟，次日每处90分钟。'
    assert duration_claim_faults(value)==['explanation_visit_duration_mismatch']
    value['reasoning']='每处120分钟。'
    assert duration_claim_faults(value)==['explanation_visit_duration_mismatch']


def test_current_capture_uses_current_production_system_not_historical_prompt():
    from scripts.post_train_runtime import capture_messages
    from app.planning.prompts import JOINT_PLANNER_SYSTEM
    messages=capture_messages(dict(query='南京1日游',destination='南京',days=1))
    assert messages[0]['content']==JOINT_PLANNER_SYSTEM
    assert '不自行联网' in messages[0]['content']
    assert 'reasoning必须为字符串' in messages[0]['content']


@pytest.mark.parametrize('reported,stopped',[(30000,False),(41000,True)])
def test_valid_tool_free_response_has_warning_and_bounded_usage_policy(tmp_path,reported,stopped):
    from scripts.run_large_pool_teacher_responses import ResponsesMeter
    meter=fake_meter(tmp_path, '', {})
    meter.__class__=ResponsesMeter
    raw=dict(status='completed',output=[dict(type='message')],
        tool_usage=dict(web_search=dict(num_requests=0)),
        usage=dict(input_tokens=reported,output_tokens=500,total_tokens=reported+500))
    meter.client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kw:SimpleNamespace(output_text=json.dumps(target()),model_dump=lambda **kw:raw)))
    meter.call([],dict(sample='mock',round=0))
    records=[json.loads(l) for l in meter.path.read_text(encoding='utf-8').splitlines()]
    assert records[-1]['usage_warning']
    assert meter.stop.is_set()==stopped
    assert bool(records[-1].get('guard_triggered'))==stopped
