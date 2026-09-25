from app.core.observability import prometheus_text, request_finished, request_started, snapshot


def test_metrics_are_safe_and_count_requests_without_prompt_content():
    before = snapshot()
    request_started()
    request_finished("/api/health", 200, 12.5)
    after = snapshot()

    assert after["requests_total"] == before["requests_total"] + 1
    assert after["in_flight"] == before["in_flight"]
    assert after["status_counts"][200] >= before["status_counts"].get(200, 0) + 1
    assert after["latency_avg_ms"] >= 0
    assert after["latency_p95_ms"] >= 0
    text = prometheus_text()
    assert "travelmind_requests_total" in text
    assert "travelmind_request_latency_ms_p95" in text
    assert "travelmind_llm_calls_total" in text
    assert "api/health" not in text
