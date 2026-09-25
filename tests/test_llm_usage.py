from app.core import llm_usage
from app.planning import helpers


def test_invoke_structured_records_estimated_tokens_and_latency(monkeypatch):
    llm_usage.reset_for_tests()

    class FakeResult:
        def model_dump_json(self):
            return '{"ok": true}'

    class FakeLLM:
        model_name = "test-model"

        def invoke(self, messages):
            return FakeResult()

    result = helpers.invoke_structured(FakeLLM(), [("system", "系统"), ("human", "用户需求")], retries=1)
    assert result is not None
    snapshot = llm_usage.snapshot()
    assert snapshot["calls_total"] == 1
    assert snapshot["total_tokens"] > 0
    assert snapshot["estimated_calls"] == 1
    assert snapshot["latency_sum_ms"] >= 0


def test_usage_metadata_is_preferred_over_estimate():
    llm_usage.reset_for_tests()

    class Result:
        usage_metadata = {"input_tokens": 11, "output_tokens": 7}

    llm_usage.record_call(
        schema="Demo", model="demo", input_chars=1000, output="long output",
        usage=llm_usage.extract_usage(Result()), latency_ms=12.5,
    )
    snapshot = llm_usage.snapshot()
    assert snapshot["input_tokens"] == 11
    assert snapshot["output_tokens"] == 7
    assert snapshot["usage_exact_ratio"] == 1.0
    assert "travelmind_llm_calls_total" in "\n".join(llm_usage.prometheus_lines())


def test_invoke_structured_retries_transient_timeout():
    llm_usage.reset_for_tests()

    class FakeLLM:
        model_name = "test-model"

        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary")
            return {"ok": True}

    llm = FakeLLM()
    assert helpers.invoke_structured(llm, [("human", "test")], retries=2) == {"ok": True}
    assert llm.calls == 2
    snapshot = llm_usage.snapshot()
    assert snapshot["calls_total"] == 2
    assert snapshot["failed_total"] == 1


def test_invoke_structured_does_not_retry_validation_error():
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise ValueError("bad schema")

    llm = FakeLLM()
    try:
        helpers.invoke_structured(llm, [("human", "test")], retries=3)
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError should be propagated")
    assert llm.calls == 1


def test_usage_delta_reports_window_and_recomputes_exact_ratio(monkeypatch):
    monkeypatch.setenv("LLM_INPUT_USD_PER_1K", "0.01")
    llm_usage.reset_for_tests()
    before = llm_usage.snapshot()
    llm_usage.record_call(
        schema="A", model="m", input_chars=20, output="abcd", usage=(10, 2), latency_ms=5
    )
    llm_usage.record_call(
        schema="B", model="m", input_chars=20, output="abcd", latency_ms=7
    )

    delta = llm_usage.usage_delta(before, llm_usage.snapshot())

    assert delta["calls_total"] == 2
    assert delta["input_tokens"] == 20
    assert delta["output_tokens"] == 4
    assert delta["usage_exact_ratio"] == 0.5
    assert delta["latency_sum_ms"] == 12.0
    assert delta["cost_rates_configured"] is True
