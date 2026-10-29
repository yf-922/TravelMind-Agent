from __future__ import annotations

import pytest

from app.llm import factory
from app.llm.grok import (
    resolve_grok_base_url,
    resolve_grok_model,
    resolve_grok_max_retries,
    resolve_grok_reasoning_effort,
    resolve_grok_request_timeout,
    use_grok_responses_api,
)
from app.llm.openai import resolve_openai_base_url, resolve_openai_model, resolve_reasoning_effort, use_openai_responses_api


def test_factory_accepts_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert factory.resolve_llm_provider() == "openai"


def test_factory_accepts_openai_compatible_proxy_alias(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai-proxy")
    monkeypatch.setenv("AI_PROVIDER", "openai-proxy")
    monkeypatch.setenv("AI_API_KEY", "proxy-key")
    monkeypatch.setenv("AI_BASE_URL", "https://proxy.example/v1/")
    monkeypatch.setenv("AI_MODEL", "proxy-model")
    monkeypatch.setenv("OPENAI_MODEL", "proxy-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1/")
    assert factory.resolve_llm_provider() == "openai"
    assert resolve_openai_model() == "proxy-model"
    assert resolve_openai_base_url() == "https://proxy.example/v1"


def test_openai_proxy_can_disable_responses_api(monkeypatch):
    monkeypatch.setenv("OPENAI_USE_RESPONSES_API", "0")
    assert use_openai_responses_api() is False


def test_factory_accepts_grok(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "grok")
    assert factory.resolve_llm_provider() == "grok"


def test_factory_error_lists_all_supported_providers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")
    with pytest.raises(ValueError, match="openai, grok, deepseek, doubao"):
        factory.resolve_llm_provider()


def test_openai_model_defaults_to_terra(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "")
    monkeypatch.setenv("AI_MODEL", "")
    assert resolve_openai_model() == "gpt-5.6-terra"
    assert resolve_openai_model("gpt-5.6-sol") == "gpt-5.6-sol"


def test_openai_reasoning_effort_validation(monkeypatch):
    monkeypatch.setenv("TRAVELMIND_OPENAI_REASONING_EFFORT", "low")
    assert resolve_reasoning_effort() == "low"
    monkeypatch.setenv("TRAVELMIND_OPENAI_REASONING_EFFORT", "invalid")
    with pytest.raises(ValueError, match="reasoning effort"):
        resolve_reasoning_effort()


def test_grok_configuration(monkeypatch):
    monkeypatch.setenv("GROK_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("GROK_MODEL", "grok-test")
    assert resolve_grok_base_url() == "https://example.test/v1"
    assert resolve_grok_model() == "grok-test"
    assert resolve_grok_model("grok-override") == "grok-override"
    monkeypatch.delenv("GROK_USE_RESPONSES_API", raising=False)
    assert use_grok_responses_api() is True
    monkeypatch.setenv("GROK_USE_RESPONSES_API", "0")
    assert use_grok_responses_api() is False
    monkeypatch.delenv("GROK_REASONING_EFFORT", raising=False)
    assert resolve_grok_reasoning_effort() == "low"
    monkeypatch.setenv("GROK_REASONING_EFFORT", "")
    assert resolve_grok_reasoning_effort() is None


def test_grok_latency_budget_is_bounded_and_fail_fast_by_default(monkeypatch):
    monkeypatch.delenv("GROK_REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("GROK_MAX_RETRIES", raising=False)
    assert resolve_grok_request_timeout() == 45.0
    assert resolve_grok_max_retries() == 0

    monkeypatch.setenv("GROK_REQUEST_TIMEOUT_SECONDS", "999")
    monkeypatch.setenv("GROK_MAX_RETRIES", "99")
    assert resolve_grok_request_timeout() == 120.0
    assert resolve_grok_max_retries() == 2

    monkeypatch.setenv("GROK_REQUEST_TIMEOUT_SECONDS", "invalid")
    monkeypatch.setenv("GROK_MAX_RETRIES", "invalid")
    assert resolve_grok_request_timeout() == 45.0
    assert resolve_grok_max_retries() == 0
