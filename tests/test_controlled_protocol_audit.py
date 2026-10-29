import copy

from scripts.audit_controlled_protocol import clock, diagnostic_period_repair, dimensions


def plan():
    return dict(days=[dict(day=1, spots=[dict(name='A', period='09:00-10:00', start_time='09:00', end_time='10:00')])])


def test_clock_rejects_invalid_minutes_and_dates():
    assert clock('09:60') is None
    assert clock('24:00') is None
    assert clock('2026-01-01 09:00') is None
    assert clock('09:59') == 599


def test_period_diagnostic_does_not_mutate_answer_or_fix_time():
    original = plan()
    before = copy.deepcopy(original)
    fixed = diagnostic_period_repair(original)
    assert original == before
    assert fixed['days'][0]['spots'][0]['period'] == 'morning'
    fixed['days'][0]['spots'][0]['start_time'] = '09:60'
    assert diagnostic_period_repair(fixed)['days'][0]['spots'][0]['start_time'] == '09:60'


def test_unparseable_plan_gets_no_micro_credit():
    assert not any(dimensions(None, {}, ['delivery_failure']).values())


def test_period_failure_does_not_imply_time_overlap(monkeypatch):
    monkeypatch.setattr('scripts.audit_controlled_protocol.opening_window', lambda p: (480, 1080, 1000))
    case = {'state': {'pois': [{'name': 'A'}]}}
    flags = dimensions(plan(), case, ['route_structure'])
    assert flags['timeline']
    assert not flags['period']
    assert flags['opening']


def test_actual_overlap_and_duplicates_are_independent(monkeypatch):
    monkeypatch.setattr('scripts.audit_controlled_protocol.opening_window', lambda p: (480, 1080, 1000))
    p = plan()
    p['days'][0]['spots'].append(dict(name='A', period='morning', start_time='09:30', end_time='10:30'))
    flags = dimensions(p, {'state': {'pois': [{'name': 'A'}]}}, ['route_structure'])
    assert not flags['timeline']
    assert not flags['duplicate']
    assert flags['pool']
