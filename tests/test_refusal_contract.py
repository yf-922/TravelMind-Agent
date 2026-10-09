import pytest
from tests.eval.run_refusal_contract import evaluate, load_cases


@pytest.mark.parametrize('case', load_cases(), ids=lambda case: case['id'])
def test_rule_checks_override_injected_optimistic_model_verdict(case):
    result = evaluate(case)
    assert result['passed'], result


def test_valid_route_is_not_rejected_by_same_audit_chain():
    case = {'id': 'valid_control', 'category': 'valid', 'query': '南京一日游',
            'pois': [{'name': '展馆', 'open_time': '09:00-18:00', 'indoor': True}],
            'route': [{'day': 1, 'spots': [{'name': '展馆', 'period': 'morning',
                                         'start_time': '10:00', 'end_time': '11:00'}]}],
            'expected_flag': 'route_structure'}
    result = evaluate(case)
    assert result['final_approved'] is True
    assert result['final_risk_flags'] == []
