import pytest

from scripts.evaluate_supervisor_parity import supervisor_outcome


def test_confirmation_is_not_misclassified_as_runtime_error():
    state, outcome = supervisor_outcome([
        {"type": "modification_warning", "pending_state": {"query": "trip", "approved": False}}
    ])
    assert outcome == "requires_confirmation"
    assert state.approved is False


def test_final_checkpoint_is_distinct_from_confirmation():
    state, outcome = supervisor_outcome([
        {"type": "result", "checkpoint": {"query": "trip", "approved": True}}
    ])
    assert outcome == "result"
    assert state.approved is True


@pytest.mark.parametrize("events", [[], [{"type": "stage", "node": "planner"}]])
def test_incomplete_event_stream_has_explicit_error(events):
    with pytest.raises(ValueError):
        supervisor_outcome(events)
