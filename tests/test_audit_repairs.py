from types import SimpleNamespace
import pytest
from app.llm.grok import parse_structured_response
from app.planning.schemas import TravelPlanState, RouteReview
from app.planning import nodes
from scripts.reviewer_quality_cases import build_cases
from tests.eval.run_refusal_contract import evaluate, load_cases


def test_preference_conflict_cannot_skip_semantic_reviewer():
    case = next(c for c in build_cases() if c['id'] == 'quality-preference-02')
    update = nodes.route_risk_gate_node(TravelPlanState(**case['state']))
    assert update['review_required'] and 'preference_review' in update['route_risk_flags']


def test_road_travel_gap_is_hard_conflict():
    case = next(c for c in build_cases() if c['id'] == 'quality-commute-01')
    state = TravelPlanState(**case['state'])
    a, b = state.route[0]['spots']
    state.route_distance_legs = [{'day': 1, 'from': a['name'], 'to': b['name'],
                                  'mode': 'drive', 'duration_min': 30}]
    result = nodes.route_risk_gate_node(state)
    assert 'commute_time_conflict' in result['route_risk_flags']
    assert result['review_required']


@pytest.mark.parametrize('case', load_cases(), ids=lambda c: c['id'])
def test_failed_final_route_explains_failure_to_user(case):
    result = evaluate(case)
    assert not result['final_approved']
    assert any('无法生成满足全部要求' in m for m in result['user_route_issues'])


def test_valid_raw_json_is_recovered_without_model_retry():
    raw = SimpleNamespace(content='```json\n{"reasoning":"valid","approved":true,"score":90}\n```', tool_calls=[])
    result = parse_structured_response({'parsed': None, 'raw': raw, 'parsing_error': ValueError('missing tool')}, RouteReview)
    assert result.approved


def test_truncated_or_missing_required_fields_fail_explicitly():
    raw = SimpleNamespace(content='{"approved":true}', tool_calls=[], response_metadata={'finish_reason': 'length'})
    with pytest.raises(ValueError, match='finish_reason=length'):
        parse_structured_response({'parsed': None, 'raw': raw}, RouteReview)


def test_original_required_window_cannot_be_silently_relaxed():
    state = TravelPlanState(query='必须在10:00到11:00游览', days=1,
        route=[{'day': 1, 'spots': [{'name': 'A', 'start_time': '17:00', 'end_time': '18:00'}]}])
    assert nodes.explicit_modification_time_violation(state)


def test_invalid_planner_response_clears_stale_route(monkeypatch):
    monkeypatch.setattr(nodes, 'build_structured_llm', lambda *a, **kw: object())
    def fail(*a, **kw): raise ValueError('invalid structured response')
    case = build_cases()[0]
    state = TravelPlanState(**case['state'])
    update = nodes.make_joint_planner_node(None, knowledge_lookup=lambda *a: [], invoke_fn=fail)(state)
    assert update['route'] == [] and update['approved'] is False
    assert update['review_round'] > state.max_review_rounds
    assert '格式无效' in update['reviewer_issues'][0]


def test_invalid_reviewer_response_is_not_approved(monkeypatch):
    monkeypatch.setattr(nodes, 'build_structured_llm', lambda *a, **kw: object())
    def fail(*a, **kw): raise ValueError('invalid structured response')
    monkeypatch.setattr(nodes, 'invoke_structured', fail)
    result = nodes.make_reviewer_node(None)(TravelPlanState(**build_cases()[0]['state']))
    assert result['approved'] is False
    assert '未获审核通过' in result['reviewer_issues'][0]


def test_missing_request_fields_are_clarified_without_review():
    state = TravelPlanState(query='帮我安排一个旅行', missing_fields=['目的地', '出行开始日期'])
    result = nodes.clarify_request_node(state)
    assert result['approved'] is False
    assert result['route'] == []
    assert not result['review_required']
    assert '目的地城市' in result['reviewer_issues'][0]
    assert '出发日期' in result['reviewer_issues'][0]
