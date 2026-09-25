"""OpenAI structured-output client factory."""

from __future__ import annotations

import os
from typing import Any, TypeVar

from pydantic import BaseModel

from app.core.env import load_local_env
from app.core.http import choose_http_proxy


DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING_EFFORT = "none"

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def resolve_openai_model(model: str | None = None) -> str:
    return (model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)).strip()


def resolve_reasoning_effort() -> str:
    effort = os.getenv(
        "TRAVELMIND_OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
    ).strip().lower()
    supported = {"none", "low", "medium", "high", "xhigh", "max"}
    if effort not in supported:
        raise ValueError(
            f"未知的 OpenAI reasoning effort：{effort}，支持：{', '.join(sorted(supported))}"
        )
    return effort


def build_chat_openai(*, model: str | None = None, temperature: float = 0) -> Any:
    """Create an OpenAI client using the Responses API."""
    load_local_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY。请在 .env.local 中配置后重试。")
    try:
        import httpx
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少 httpx 或 langchain-openai。请先安装依赖。") from exc

    proxy = choose_http_proxy()
    return ChatOpenAI(
        model=resolve_openai_model(model),
        api_key=api_key,
        temperature=temperature,
        reasoning_effort=resolve_reasoning_effort(),
        use_responses_api=True,
        timeout=120,
        max_retries=2,
        http_client=httpx.Client(proxy=proxy, trust_env=False),
        http_async_client=httpx.AsyncClient(proxy=proxy, trust_env=False),
        http_socket_options=(),
    )


def build_structured_openai(
    schema: type[SchemaT],
    *,
    model: str | None = None,
    temperature: float = 0,
) -> Any:
    """Create an OpenAI client bound to a strict Pydantic schema."""
    llm = build_chat_openai(model=model, temperature=temperature)
    return llm.with_structured_output(schema, method="json_schema", strict=True)
