"""Grok/xAI OpenAI-compatible client factory."""

from __future__ import annotations

import os
from typing import Any, TypeVar

from pydantic import BaseModel

from app.core.env import load_local_env
from app.core.http import choose_http_proxy


DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1"
DEFAULT_GROK_MODEL = "grok-4.6"

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def resolve_grok_base_url() -> str:
    return os.getenv("GROK_BASE_URL", DEFAULT_GROK_BASE_URL).strip().rstrip("/")


def resolve_grok_model(model: str | None = None) -> str:
    return (model or os.getenv("GROK_MODEL", DEFAULT_GROK_MODEL)).strip()


def use_grok_responses_api() -> bool:
    return os.getenv("GROK_USE_RESPONSES_API", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def resolve_grok_reasoning_effort() -> str | None:
    effort = os.getenv("GROK_REASONING_EFFORT", "low").strip().lower()
    return effort or None


def resolve_grok_request_timeout() -> float:
    """Bound one provider attempt so a slow gateway cannot consume the whole plan budget."""
    try:
        value = float(os.getenv("GROK_REQUEST_TIMEOUT_SECONDS", "45"))
    except ValueError:
        value = 45.0
    return min(max(value, 5.0), 120.0)


def resolve_grok_max_retries() -> int:
    """Client retries multiply tail latency, so custom gateways default to fail-fast."""
    try:
        value = int(os.getenv("GROK_MAX_RETRIES", "0"))
    except ValueError:
        value = 0
    return min(max(value, 0), 2)


def build_chat_grok(*, model: str | None = None, temperature: float = 0) -> Any:
    """Create a Grok client through the OpenAI-compatible chat endpoint."""
    load_local_env()
    api_key = os.getenv("GROK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 GROK_API_KEY。请在 .env.local 中配置后重试。")
    try:
        import httpx
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少 httpx 或 langchain-openai。请先安装依赖。") from exc

    proxy = choose_http_proxy()
    return ChatOpenAI(
        model=resolve_grok_model(model),
        api_key=api_key,
        base_url=resolve_grok_base_url(),
        temperature=temperature,
        use_responses_api=use_grok_responses_api(),
        reasoning_effort=resolve_grok_reasoning_effort(),
        timeout=resolve_grok_request_timeout(),
        max_retries=resolve_grok_max_retries(),
        http_client=httpx.Client(proxy=proxy, trust_env=False),
        http_async_client=httpx.AsyncClient(proxy=proxy, trust_env=False),
        http_socket_options=(),
    )


def build_structured_grok(
    schema: type[SchemaT],
    *,
    model: str | None = None,
    temperature: float = 0,
) -> Any:
    """Create a Grok client bound to a strict Pydantic schema."""
    llm = build_chat_grok(model=model, temperature=temperature)
    method = "json_schema" if use_grok_responses_api() else "function_calling"
    return llm.with_structured_output(schema, method=method, strict=True)
