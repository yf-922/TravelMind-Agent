"""LLM-as-Judge：对固定输入和固定候选池做结构化、可追溯评分。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.llm.factory import build_structured_llm, resolve_llm_provider
from app.planning.helpers import invoke_structured


JUDGE_PROMPT_VERSION = "travelmind-judge-v1"
JUDGE_SYSTEM = """你是严格、可复现的旅行规划评测器。只依据给出的用户请求、候选景点和行程评分，不能补充外部事实。
逐项给 1-5 分：5=证据充分且完全满足；3=基本满足但有明确不足；1=明显不满足或无法验证。
候选池外景点必须把 groundedness 评为 1；行程为空必须把 completeness 评为 1。
仅在每项都不低于 3 且没有候选池外景点时 pass_case=true。理由必须引用输入中的具体名称或字段，不输出思维过程。"""


class JudgeScore(BaseModel):
    groundedness: int = Field(ge=1, le=5, description="景点是否全部来自候选池")
    constraint_satisfaction: int = Field(ge=1, le=5, description="是否响应用户请求")
    route_completeness: int = Field(ge=1, le=5, description="行程是否非空且可交付")
    reviewer_consistency: int = Field(ge=1, le=5, description="Reviewer 结论是否与行程一致")
    clarity: int = Field(ge=1, le=5, description="输出是否清晰")
    pass_case: bool
    failure_categories: list[str] = Field(default_factory=list, description="如 candidate_hallucination / empty_itinerary / constraint_miss")
    reason: str = Field(min_length=1, max_length=500)


def judge_itinerary(
    *, user_request: str, destination: str, candidates: list[dict[str, Any]],
    itinerary: list[dict[str, Any]], review: dict[str, Any], model: str | None = None,
) -> JudgeScore:
    """temperature=0 + versioned prompt + structured schema，使评分可复跑、可比较。"""
    llm = build_structured_llm(JudgeScore, model=model, temperature=0, task_type="judge")
    payload = {
        "user_request": user_request, "destination_hint": destination,
        "candidate_names": [item.get("name") for item in candidates],
        "itinerary": itinerary, "review": review,
    }
    return invoke_structured(llm, [
        ("system", JUDGE_SYSTEM),
        ("human", json.dumps(payload, ensure_ascii=False)),
    ])


def judge_metadata(model: str | None = None) -> dict[str, str | None]:
    return {"prompt_version": JUDGE_PROMPT_VERSION, "provider": resolve_llm_provider(), "model_override": model}
