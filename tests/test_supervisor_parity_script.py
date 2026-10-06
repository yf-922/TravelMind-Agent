import pytest

from scripts.evaluate_supervisor_parity import supervisor_outcome
from scripts.evaluate_supervisor_parity import call_budget


def test_selected_engine_budget_includes_all_retry_attempts():
    assert call_budget(("supervisor",)) == 18
    assert call_budget(("langgraph",)) == 90
    assert call_budget(("langgraph", "supervisor")) == 108


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
