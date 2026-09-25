from scripts.evaluate_risk_gate import constructed_cases, evaluate


def test_saved_real_routes_preserve_natural_skip_and_fault_recall():
    report = evaluate()

    assert report["metrics"]["natural_skip_rate"] == 1.0
    assert report["metrics"]["fault_escalation_recall"] == 1.0
    assert report["metrics"]["projected_token_reduction_vs_always_full"] > 0.5
    assert report["metrics"]["projected_latency_reduction_vs_always_full"] > 0.4


def test_constructed_cases_cover_labelled_faults_and_clean_contrasts():
    report = evaluate()

    assert len(constructed_cases()) >= 20
    assert report["metrics"]["constructed_pass_rate"] == 1.0
    assert report["metrics"]["constructed_clean_skip_rate"] == 1.0
    assert report["metrics"]["constructed_fault_recall"] == 1.0
    assert all(value == 1.0 for value in report["metrics"]["constructed_category_recall"].values())
    assert all(not row["unexpected_flags"] and not row["missing_flags"] for row in report["constructed"])
