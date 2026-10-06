from scripts.evaluate_supervisor_api import parse_events


def test_parse_events_preserves_failed_results_and_node_sequence():
    events = parse_events('data: {"type":"stage","node":"planner"}\n\ndata: {"type":"result","success":false}\n\n')
    assert events == [{"type": "stage", "node": "planner"}, {"type": "result", "success": False}]
