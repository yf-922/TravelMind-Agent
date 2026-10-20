from scripts.evaluate_reviewer_ablation import cases_v2, run_arm
from app.planning import nodes
from app.planning.schemas import TravelPlanState


def test_corrected_assets_are_consistent():
    cases = cases_v2()
    assert len(cases) == 60
    for case in cases:
        r = case['rubric']
        if r.get('min_match'):
            assert set(r.get('acceptable', [])) - set(r.get('excluded', []))
        if case['category'] == 'map':
            assert '至少两个' in case['state']['query']


def test_reviewer_adds_semantic_repair_without_exposing_gold_labels(monkeypatch):
    state = TravelPlanState(query='喜欢文化', days=1, pois=[{'name': 'A', 'open_time': '09:00-18:00'}],
        route=[{'day': 1, 'spots': [{'name': 'A', 'period': 'morning', 'start_time': '10:00', 'end_time': '11:00'}]}])
    monkeypatch.setattr(nodes, 'route_distance_check_node', lambda state: {})
    monkeypatch.setattr(nodes, 'finalize_node', lambda state: {'final_plan': {'approved': state.approved}})
    calls = []
    def reviewer(state):
        calls.append('reviewer')
        return {'approved': len(calls) > 1, 'route_modify_opinion': '延长游览时间'}
    def planner(state):
        calls.append('planner')
        return {'route': state.route}
    rules, repairs = run_arm(state.model_dump(), 'rules', planner, reviewer)
    assert rules.approved and repairs == 0 and calls == []
    reviewed, repairs = run_arm(state.model_dump(), 'rules_reviewer', planner, reviewer)
    assert reviewed.approved and repairs == 1
    assert calls == ['reviewer', 'planner', 'reviewer']


def test_repair_loop_is_bounded(monkeypatch):
    state = TravelPlanState(query='旅行', days=1, route=[{'day': 1, 'spots': []}])
    monkeypatch.setattr(nodes, 'route_distance_check_node', lambda state: {})
    monkeypatch.setattr(nodes, 'finalize_node', lambda state: {'final_plan': {'approved': state.approved}})
    _, repairs = run_arm(state.model_dump(), 'rules', lambda state: {'route': state.route}, lambda state: {})
    assert repairs == 3
