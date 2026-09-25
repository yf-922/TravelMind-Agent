"""评估 harness：从 fixture 构造状态，跑可切换编排模式的隔离子图。

设计要点：
- 冻结 planner 的输入（destination/dates/preferences/pois/weather）成 fixture，
  跳过 intent 与 attraction_search，使 planner/reviewer 的评估可复现、零外部成本。
- 被选模式中的 LLM 节点真实执行，它们才是被评估对象。
- 完整模式复用生产的 planner / reviewer / time_check 节点与路由；精简模式只移除
  对应审核阶段，其他输入保持一致，用于公平消融。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.planning.nodes import (
    make_planner_node,
    make_reviewer_node,
    make_time_check_node,
    route_after_planner,
    route_after_review,
    route_after_time_check,
)
from app.planning.schemas import TravelPlanState

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EvaluationMode = Literal[
    "planner_only",
    "planner_reviewer",
    "planner_reviewer_time_check",
]
EVALUATION_MODES: tuple[EvaluationMode, ...] = (
    "planner_only",
    "planner_reviewer",
    "planner_reviewer_time_check",
)


# ─── Fixture 加载 ─────────────────────────────────────────────

def load_fixtures(only: str | None = None) -> list[dict[str, Any]]:
    """加载 fixtures/ 下所有用例（按 id 排序）；only 给定时只取该 id。"""
    cases: list[dict[str, Any]] = []
    for fp in sorted(FIXTURES_DIR.glob("*.json")):
        fx = json.loads(fp.read_text(encoding="utf-8"))
        fx.setdefault("id", fp.stem)
        if only and fx["id"] != only:
            continue
        cases.append(fx)
    return cases


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def build_state_from_fixture(fx: dict[str, Any]) -> TravelPlanState:
    """用 fixture 字段直接构造 TravelPlanState（已 seed planner 所需全部输入）。

    pois 里允许携带额外的 `indoor` 真值标签——透传进 state，
    供代码打分器做天气合规判定；planner/reviewer 不读它。
    """
    return TravelPlanState(
        query=fx.get("query") or f"{fx.get('destination', '')}{fx.get('days', '')}日游",
        destination=fx.get("destination"),
        travel_start_date=_parse_date(fx.get("travel_start_date")),
        travel_end_date=_parse_date(fx.get("travel_end_date")),
        days=int(fx.get("days", 0)),
        attraction_preference=fx.get("attraction_preference"),
        food_preference=fx.get("food_preference"),
        habit_preference=fx.get("habit_preference"),
        max_per_day=int(fx.get("max_per_day", 3)),
        min_rating=float(fx.get("min_rating", 4.5)),
        max_spots=int(fx.get("max_spots", 30)),
        max_review_rounds=int(fx.get("max_review_rounds", 3)),
        model_name=fx.get("model_name"),
        pois=list(fx.get("pois", [])),
        weather_forecast=list(fx.get("weather_forecast", [])),
        weather_note=fx.get("weather_note"),
    )


# ─── mini-graph ──────────────────────────────────────────────

def build_evaluation_graph(
    mode: EvaluationMode = "planner_reviewer_time_check",
    model_name: str | None = None,
):
    """Build one comparable evaluation graph from production nodes.

    ``planner_only`` evaluates one-shot generation. ``planner_reviewer`` keeps
    the production review loop but stops before time checking. The full mode
    includes both review loops and is behaviorally aligned with production.
    """
    if mode not in EVALUATION_MODES:
        raise ValueError(f"unknown evaluation mode: {mode}")

    g = StateGraph(TravelPlanState)
    g.add_node("planner", make_planner_node(model_name))
    g.add_edge(START, "planner")
    if mode == "planner_only":
        g.add_edge("planner", END)
        return g.compile()

    g.add_node("reviewer", make_reviewer_node(model_name))
    g.add_edge("planner", "reviewer")
    g.add_conditional_edges(
        "reviewer", route_after_review,
        {"planner": "planner", "time_check": END},
    )
    if mode == "planner_reviewer":
        return g.compile()

    # Replace the reviewer's completion target with the production time-check
    # stage. Rebuild this small graph to keep the two modes explicit.
    g = StateGraph(TravelPlanState)
    g.add_node("planner", make_planner_node(model_name))
    g.add_node("reviewer", make_reviewer_node(model_name))
    g.add_node("time_check", make_time_check_node(model_name))
    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner", route_after_planner,
        {"reviewer": "reviewer", "time_check": "time_check"},
    )
    g.add_conditional_edges(
        "reviewer", route_after_review,
        {"planner": "planner", "time_check": "time_check"},
    )
    g.add_conditional_edges(
        "time_check", route_after_time_check,
        {"planner": "planner", "reviewer": "reviewer", "meal_search": END, "spot_tips": END},
    )
    return g.compile()


def run_evaluation_loop(
    fx: dict[str, Any],
    mode: EvaluationMode = "planner_reviewer_time_check",
) -> TravelPlanState:
    """Run one selected evaluation mode and return its final state.

    最终 state 含评估所需全部信号：route / approved / review_round /
    reviewer_issues / route_modify_opinion / planner_reviewer_dialogue。
    """
    state = build_state_from_fixture(fx)
    app = build_evaluation_graph(mode, state.model_name)
    config = {
        "recursion_limit": (
            2 * (state.max_review_rounds + 1)
            + 2 * state.max_time_check_rounds
            + 10
        )
    }
    result = app.invoke(state, config=config)
    return TravelPlanState(**result) if isinstance(result, dict) else result


def build_planner_reviewer_graph(model_name: str | None = None):
    """Backward-compatible alias for the full production-like eval graph."""
    return build_evaluation_graph("planner_reviewer_time_check", model_name)


def run_planner_reviewer_loop(fx: dict[str, Any]) -> TravelPlanState:
    """Backward-compatible alias for the full production-like eval loop."""
    return run_evaluation_loop(fx, "planner_reviewer_time_check")
