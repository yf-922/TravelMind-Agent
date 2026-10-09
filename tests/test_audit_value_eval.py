from scripts.evaluate_audit_value import assets, objective, summary
from app.planning.schemas import TravelPlanState


def test_audit_assets_predeclare_thirty_feasible_and_ten_rejects():
    cases = assets()
    assert len(cases) == len({c['id'] for c in cases}) == 40
    assert sum(c['expected'] == 'feasible' for c in cases) == 30
    assert sum(c['expected'] == 'reject' for c in cases) == 10
    assert all(c.get('impossibility_reason') for c in cases if c['expected'] == 'reject')


def test_empty_route_is_not_valid_even_when_model_approves():
    assert not objective(TravelPlanState(query='test', days=1, route=[], approved=True))['route_valid']


def test_failed_runs_never_count_as_success_and_unrun_stays_visible():
    report = {'rows': [{'id': 'a', 'arm': 'full', 'expected': 'feasible', 'error': 'timeout'},
                       {'id': 'b', 'arm': 'full', 'expected': 'reject', 'outcome': {
                           'unsafe_release': False, 'correct_rejection': False}}]}
    s = summary(report)['full']
    assert s['feasible_planned'] == 30 and s['feasible_recorded'] == 1
    assert s['successful_approved_routes'] == 0 and s['errors'] == 1
    assert s['rejected_with_user_error'] == 0


def test_quality_cases_have_all_four_dimensions_and_clean_controls():
    from collections import Counter
    from scripts.reviewer_quality_cases import build_cases, quality_grade
    cases = build_cases()
    assert len(cases) == 30
    assert Counter(c['quality']['category'] for c in cases) == {
        'preference': 8, 'pace': 8, 'weather': 7, 'commute': 7}
    assert sum(c['quality']['expected_clean'] for c in cases) == 6
    for case in cases:
        assert quality_grade(TravelPlanState(**case['state']), case)['passed'] == case['quality']['expected_clean']


def test_quality_report_counts_unsafe_approval_and_failure_separately():
    from scripts.summarize_reviewer_quality import analyze
    cases = [{'id': 'bad', 'quality': {'category': 'preference', 'expected_clean': False}}]
    rows = [{'id': 'bad', 'arm': 'full', 'outcome': {'final_approved': True,
             'quality': {'passed': False}}}, {'id': 'bad', 'arm': 'forced_reviewer', 'error': 'parse failure'}]
    out = analyze({'cases': cases, 'rows': rows, 'calls': [], 'status': 'partial'})
    assert out['by_category']['preference']['full']['unsafe_quality_approvals'] == 1
    assert out['by_category']['preference']['forced_reviewer']['errors'] == 1
    assert out['by_category']['preference']['forced_reviewer']['quality_passed'] == 0
