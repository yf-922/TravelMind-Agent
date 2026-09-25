from tests.eval.run_adversarial_risk_gate import evaluate


def test_adversarial_discovery_never_crashes_or_silently_approves_malformed_routes():
    report = evaluate(count=40)
    assert report["crash_rate"] == 0
    assert report["unsafe_skip_rate"] == 0
    # Discovery cases intentionally include failures; this prevents a fake
    # benchmark where every generated input is labelled as pass.
    assert any(record.get("flags") for record in report["records"])
