"""Small, opt-in model-routing policy for interviewable cost/quality trade-offs.

The unified LLM factory calls this policy for declared task types. Routing is
disabled by default and never changes the configured provider.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field


TaskClass = Literal["deterministic", "short", "complex"]


class RouteDecision(BaseModel):
    task_type: str
    task_class: TaskClass
    model: str
    fallback_model: str | None = None
    reason: str
    routing_enabled: bool = False
    estimated_input_tokens: int = Field(default=0, ge=0)


class ModelRouter:
    """Deterministic policy: cheap model for bounded work, strong model for planning."""

    def __init__(
        self,
        *,
        primary_model: str,
        small_model: str,
        fallback_model: str | None = None,
        enabled: bool = False,
        short_token_threshold: int = 1200,
    ) -> None:
        self.primary_model = primary_model.strip() or "primary"
        self.small_model = small_model.strip() or self.primary_model
        self.fallback_model = fallback_model.strip() if fallback_model else None
        self.enabled = bool(enabled)
        self.short_token_threshold = max(1, int(short_token_threshold))

    @classmethod
    def from_env(cls, primary_model: str | None = None) -> "ModelRouter":
        # Keep this default aligned with factory.DEFAULT_PROVIDER without
        # importing the factory back into the router.
        provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        configured_primary = os.getenv(
            {"grok": "GROK_MODEL", "deepseek": "DEEPSEEK_MODEL",
             "doubao": "DOUBAO_MODEL", "openai": "OPENAI_MODEL"}.get(provider, "OPENAI_MODEL"),
            "primary",
        )
        primary = primary_model.strip() if primary_model and primary_model.strip() else configured_primary
        return cls(
            primary_model=primary,
            small_model=os.getenv("LLM_SMALL_MODEL", primary),
            fallback_model=os.getenv("LLM_FALLBACK_MODEL") or None,
            enabled=os.getenv("LLM_ROUTING_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"},
            short_token_threshold=int(os.getenv("LLM_ROUTING_SHORT_TOKEN_THRESHOLD", "1200")),
        )

    def classify(self, task_type: str, estimated_input_tokens: int = 0) -> TaskClass:
        name = task_type.strip().lower()
        if name in {"query_rewrite", "intent", "intent_extract", "profile_extract", "memory_extract"}:
            return "short"
        if name in {"time_check", "route_distance", "tool_validation", "deterministic"}:
            return "deterministic"
        if estimated_input_tokens <= self.short_token_threshold and name in {"meal_recommend", "spot_tips"}:
            return "short"
        return "complex"

    def choose(self, task_type: str, estimated_input_tokens: int = 0) -> RouteDecision:
        task_class = self.classify(task_type, estimated_input_tokens)
        if not self.enabled:
            return RouteDecision(
                task_type=task_type,
                task_class=task_class,
                model=self.primary_model,
                fallback_model=self.fallback_model,
                reason="routing disabled; preserve configured primary model",
                routing_enabled=False,
                estimated_input_tokens=max(0, int(estimated_input_tokens)),
            )
        if task_class in {"short", "deterministic"}:
            model = self.small_model
            reason = "bounded task routed to lower-cost model"
        else:
            model = self.primary_model
            reason = "complex planning/review task kept on primary model"
        return RouteDecision(
            task_type=task_type,
            task_class=task_class,
            model=model,
            fallback_model=self.fallback_model,
            reason=reason,
            routing_enabled=True,
            estimated_input_tokens=max(0, int(estimated_input_tokens)),
        )


def route_model(
    task_type: str,
    estimated_input_tokens: int = 0,
    primary_model: str | None = None,
) -> RouteDecision:
    """Choose a model while preserving an explicit caller override as primary."""
    return ModelRouter.from_env(primary_model=primary_model).choose(task_type, estimated_input_tokens)
