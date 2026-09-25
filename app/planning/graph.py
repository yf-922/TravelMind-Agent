"""LangGraph 图构建与流水线入口。"""

from __future__ import annotations

from typing import Any, AsyncIterator

from langgraph.graph import END, START, StateGraph

from app.planning.schemas import TravelPlanState
from app.planning.helpers import invoke_structured  # re-export for convenience
from app.planning.nodes import (
    attraction_search_node,
    finalize_node,
    make_finalize_node,
    make_intent_node,
    make_meal_enrichment_node,
    make_planner_node,
    make_query_rewrite_node,
    make_reviewer_node,
    make_spot_tips_node,
    make_time_check_node,
    route_distance_check_node,
    route_risk_gate_node,
    weather_search_node,
    route_after_intent,
    route_after_planner,
    route_after_review,
    route_after_risk_gate,
    route_after_time_check,
)


# ─── 构图 ────────────────────────────────────────────────────

def build_graph(
    model_name: str | None = None,
    profile_hint: str = "",
    memory_writer=None,
    user_id: str | None = None,
):
    g = StateGraph(TravelPlanState)

    g.add_node("query_rewrite",    make_query_rewrite_node(model_name, user_id))
    g.add_node("intent",           make_intent_node(model_name, profile_hint=profile_hint))
    g.add_node("weather_search",   weather_search_node)
    g.add_node("attraction_search", attraction_search_node)
    g.add_node("planner",          make_planner_node(model_name))
    g.add_node("route_distance_check", route_distance_check_node)
    g.add_node("risk_gate", route_risk_gate_node)
    g.add_node("reviewer",         make_reviewer_node(model_name))
    g.add_node("time_check",       make_time_check_node(model_name))
    g.add_node("meal_enrichment",  make_meal_enrichment_node(model_name))
    g.add_node("spot_tips",        make_spot_tips_node(model_name))
    g.add_node("finalize",         make_finalize_node(memory_writer))

    g.add_edge(START,   "intent")
    g.add_conditional_edges(
        "intent", route_after_intent,
        {
            "query_rewrite": "query_rewrite",
            "weather_search": "weather_search",
            END: END,
        }
    )
    # query_rewrite 与 weather_search 都只依赖 intent 的输出，可以并行执行。
    # attraction_search 使用两者结果，因此设置多前驱屏障，确保天气/改写都完成后再检索。
    g.add_edge("query_rewrite", "attraction_search")
    g.add_edge("weather_search", "attraction_search")
    g.add_edge("attraction_search", "planner")
    # planner 输出：time_check_done=False 时进 reviewer 走主循环；True 时进 time_check 重新核查
    g.add_conditional_edges(
        "planner", route_after_planner,
        {"reviewer": "route_distance_check", "time_check": "route_distance_check"},
    )
    g.add_edge("route_distance_check", "risk_gate")
    g.add_conditional_edges(
        "risk_gate", route_after_risk_gate,
        {"reviewer": "reviewer", "time_check": "time_check",
         "meal_search": "meal_enrichment", "spot_tips": "spot_tips"},
    )
    # reviewer：通过或达最大轮数 → time_check 阶段；否则打回 planner
    g.add_conditional_edges(
        "reviewer", route_after_review,
        {"planner": "planner", "time_check": "time_check"},
    )
    # time_check：无违规/达上限 → meal_search；有违规且未达上限 → planner 修正
    g.add_conditional_edges(
        "time_check", route_after_time_check,
        {"planner": "planner", "reviewer": "reviewer", "meal_search": "meal_enrichment", "spot_tips": "spot_tips"},
    )
    # The dependent search -> recommend meal subflow and attraction tips are
    # independent branches. Finalize waits for both.
    g.add_edge("meal_enrichment", "finalize")
    g.add_edge("spot_tips",      "finalize")
    g.add_edge("finalize",       END)

    return g.compile()


# ─── 流水线入口 ───────────────────────────────────────────────


# ─── 分阶段流式入口（SSE 事件源）───────────────────────────────

# 节点名 → 进度文案。同时充当“哪些事件需要透出”的过滤白名单。
# planner/reviewer 文案在运行时按轮次/通过位动态拼接，这里留占位。
_NODE_LABELS: dict[str, str] = {
    "query_rewrite":     "🔎 正在结合用户画像改写查询",
    "intent":            "🧭 正在理解出行意图（目的地 / 日期 / 偏好）",
    "weather_search":    "🌦 正在查询出行天气",
    "attraction_search": "🗺 正在调用高德搜索景点池",
    "planner":           "✍️ 正在规划逐日行程",
    "route_distance_check": "🛣 正在核验道路步行/驾车距离",
    "risk_gate": "🧮 正在评估是否需要升级审核",
    "reviewer":          "🔍 正在评审行程",
    "time_check":        "⏱ 正在核查景点开放时间",
    "meal_search":       "🍽 正在搜索周边餐厅",
    "meal_recommend":    "🍴 正在为每天挑选餐厅",
    "meal_enrichment":   "🍽 正在检索并推荐沿线餐厅",
    "spot_tips":         "💡 正在为每个景点生成游玩贴士",
    "finalize":          "📦 正在收敛生成最终行程",
}


def _stage_event(node: str, acc: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    """据节点名与累积状态构造一条 stage 进度事件（在 on_chain_start 时触发）。

    acc 反映的是节点运行前的状态：
    - planner 尚未递增 review_round，所以当前轮 = acc["review_round"] + 1
    - reviewer 读取 planner 刚写入的 review_round，直接用即可
    """
    label = _NODE_LABELS[node]
    ev: dict[str, Any] = {"type": "stage", "node": node, "label": label}
    if node == "planner":
        rnd = (acc.get("review_round") or 0) + 1
        ev["round"] = rnd
        ev["label"] = f"{label}（第 {rnd} 轮）"
    elif node == "reviewer":
        rnd = acc.get("review_round") or 1
        ev["round"] = rnd
        ev["label"] = f"{label}：第 {rnd} 轮"
    elif node == "time_check":
        # on_chain_start 时 acc 中 time_check_round 是节点运行前的值，+1 得当前轮次
        rnd = (acc.get("time_check_round") or 0) + 1
        ev["round"] = rnd
        ev["label"] = f"{label}（第 {rnd} 轮）"
    return ev


def _route_spot_count(route: Any) -> int:
    """Return a display-only spot count without exposing model reasoning."""
    if not isinstance(route, list):
        return 0
    return sum(len(day.get("spots", [])) for day in route if isinstance(day, dict))


def _preview_names(items: Any, limit: int = 3) -> str:
    """Pick a few verified item names for a public progress update."""
    if not isinstance(items, list):
        return ""
    names = [str(item.get("name") or "").strip() for item in items if isinstance(item, dict)]
    names = list(dict.fromkeys(name for name in names if name))
    return "、".join(names[:limit])


def _meal_candidate_preview(entries: Any, limit: int = 3) -> str:
    if not isinstance(entries, list):
        return ""
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for period in ("lunch", "dinner"):
            value = entry.get(period)
            if isinstance(value, dict):
                candidates.extend(value.get("candidates") or [])
    return _preview_names(candidates, limit)


def _meal_pick_preview(meals: Any, limit: int = 2) -> str:
    if not isinstance(meals, list):
        return ""
    lines: list[str] = []
    for meal in meals[:limit]:
        if not isinstance(meal, dict):
            continue
        lunch = (meal.get("lunch") or {}).get("name") if isinstance(meal.get("lunch"), dict) else ""
        dinner = (meal.get("dinner") or {}).get("name") if isinstance(meal.get("dinner"), dict) else ""
        day = meal.get("day") or len(lines) + 1
        picks = " / ".join(part for part in (lunch, dinner) if part)
        if picks:
            lines.append(f"第 {day} 天：{picks}")
    return "；".join(lines)


def _stage_summary(node: str, state_before: dict[str, Any], update: dict[str, Any]) -> str:
    """Build a factual completion report using only structured state fields."""
    state = {**state_before, **update}
    destination = state.get("destination") or "目的地"
    days = state.get("days") or 0

    if node == "intent":
        return f"已识别：{destination}，{days} 天行程。"
    if node == "weather_search":
        weather = "已获取天气预报" if state.get("weather_forecast") else (state.get("weather_note") or "天气暂不可用")
        return weather + "。"
    if node == "query_rewrite":
        return "已结合本次需求和用户偏好，整理检索约束。"
    if node == "attraction_search":
        pois = state.get("pois") or []
        sample = _preview_names(pois)
        suffix = f"示例：{sample}。" if sample else ""
        return f"已建立 {len(pois)} 个候选景点池。{suffix}"
    if node == "planner":
        return (f"第 {state.get('review_round') or 1} 轮行程草案已生成："
                f"{len(state.get('route') or []) or days} 天、{_route_spot_count(state.get('route'))} 个景点。")
    if node == "route_distance_check":
        return state.get("route_distance_note") or "道路距离核验已完成。"
    if node == "risk_gate":
        flags = state.get("route_risk_flags") or []
        if state.get("review_skipped"):
            return "低风险路线，跳过 Reviewer/Time Check，直接进入结果增强。"
        return f"检测到 {len(flags)} 项风险，已升级到独立审核。"
    if node == "reviewer":
        issues = state.get("reviewer_issues") or []
        if state.get("approved"):
            return "行程评审通过，进入开放时间核验。"
        return f"评审发现 {len(issues)} 项需调整内容，已交回规划 Agent 修改。"
    if node == "time_check":
        return f"已完成开放时间核验：发现 {len(state.get('time_violations') or [])} 项时间冲突。"
    if node == "meal_search":
        entries = state.get("meal_candidates") or []
        sample = _meal_candidate_preview(entries)
        suffix = f"候选示例：{sample}。" if sample else ""
        return f"已完成 {len(entries)} 天路线周边餐厅搜索。{suffix}"
    if node == "meal_recommend":
        meals = state.get("meals") or []
        sample = _meal_pick_preview(meals)
        suffix = f"推荐：{sample}。" if sample else ""
        return f"已完成 {len(meals)} 天的午晚餐推荐。{suffix}"
    if node == "meal_enrichment":
        meals = state.get("meals") or []
        sample = _meal_pick_preview(meals)
        suffix = f"推荐：{sample}。" if sample else ""
        return f"已完成 {len(meals)} 天的沿线餐厅检索与推荐。{suffix}"
    if node == "spot_tips":
        return f"已生成 {len(state.get('spot_tips') or {})} 条景点游玩提示。"
    if node == "finalize":
        return "行程已整理完成，正在展示结果。"
    return "一个规划阶段已完成，正在继续下一步。"


# ─── 修改模式专用迷你图 ────────────────────────────────────────

def build_modification_graph(model_name: str | None = None, memory_writer=None):
    """从 checkpoint 重规划，并重新核验道路距离、审查意见和开放时间。"""
    g = StateGraph(TravelPlanState)
    g.add_node("planner",        make_planner_node(model_name))
    g.add_node("route_distance_check", route_distance_check_node)
    g.add_node("risk_gate", route_risk_gate_node)
    g.add_node("reviewer",       make_reviewer_node(model_name))
    g.add_node("time_check",     make_time_check_node(model_name))
    g.add_node("meal_enrichment", make_meal_enrichment_node(model_name))
    g.add_node("spot_tips",      make_spot_tips_node(model_name))
    g.add_node("finalize",       make_finalize_node(memory_writer))
    g.add_edge(START, "planner")
    g.add_edge("planner", "route_distance_check")
    g.add_edge("route_distance_check", "risk_gate")
    g.add_conditional_edges(
        "risk_gate", route_after_risk_gate,
        {"reviewer": "reviewer", "time_check": "time_check",
         "meal_search": "meal_enrichment", "spot_tips": "spot_tips"},
    )
    g.add_conditional_edges(
        "reviewer", route_after_review,
        {"planner": "planner", "time_check": "time_check"},
    )
    g.add_conditional_edges(
        "time_check", route_after_time_check,
        {"planner": "planner", "reviewer": "reviewer", "meal_search": "meal_enrichment", "spot_tips": "spot_tips"},
    )
    g.add_edge("meal_enrichment", "finalize")
    g.add_edge("spot_tips",      "finalize")
    g.add_edge("finalize",       END)
    return g.compile()


def build_confirm_graph(model_name: str | None = None, memory_writer=None):
    """确认后仍执行道路距离、Reviewer 和开放时间核验，再生成最终行程。"""
    g = StateGraph(TravelPlanState)
    g.add_node("route_distance_check", route_distance_check_node)
    g.add_node("risk_gate", route_risk_gate_node)
    g.add_node("reviewer",       make_reviewer_node(model_name))
    g.add_node("planner",        make_planner_node(model_name))
    g.add_node("time_check",     make_time_check_node(model_name))
    g.add_node("meal_enrichment", make_meal_enrichment_node(model_name))
    g.add_node("spot_tips",      make_spot_tips_node(model_name))
    g.add_node("finalize",       make_finalize_node(memory_writer))
    g.add_edge(START,            "route_distance_check")
    g.add_edge("route_distance_check", "risk_gate")
    g.add_conditional_edges(
        "risk_gate", route_after_risk_gate,
        {"reviewer": "reviewer", "time_check": "time_check",
         "meal_search": "meal_enrichment", "spot_tips": "spot_tips"},
    )
    g.add_conditional_edges(
        "reviewer", route_after_review,
        {"planner": "planner", "time_check": "time_check"},
    )
    g.add_edge("planner", "route_distance_check")
    g.add_conditional_edges(
        "time_check", route_after_time_check,
        {"planner": "planner", "reviewer": "reviewer", "meal_search": "meal_enrichment", "spot_tips": "spot_tips"},
    )
    g.add_edge("meal_enrichment", "finalize")
    g.add_edge("spot_tips",      "finalize")
    g.add_edge("finalize",   END)
    return g.compile()


async def run_modification_stream(
    checkpoint: dict[str, Any],
    modification_notes: str,
    memory_writer=None,
    **overrides: Any,
) -> AsyncIterator[dict[str, Any]]:
    """修改模式流式执行。

    从 checkpoint 恢复 planner 状态，注入用户修改意见，运行迷你图。
    若 planner 有顾虑（modification_concern 非空），yield modification_warning 事件后停止
    （由调用方存 pending 状态，等待用户确认后调 run_confirm_stream 续跑）。
    """
    app = build_modification_graph(overrides.get("model_name"), memory_writer)
    init = TravelPlanState(
        query=checkpoint.get("query", "修改行程"),
        route=checkpoint.get("route", []),
        pois=checkpoint.get("pois", []),
        planner_reviewer_dialogue=checkpoint.get("planner_reviewer_dialogue", []),
        destination=checkpoint.get("destination"),
        travel_start_date=checkpoint.get("travel_start_date"),
        travel_end_date=checkpoint.get("travel_end_date"),
        days=checkpoint.get("days", 0),
        attraction_preference=checkpoint.get("attraction_preference"),
        food_preference=checkpoint.get("food_preference"),
        habit_preference=checkpoint.get("habit_preference"),
        max_walking_km=checkpoint.get("max_walking_km"),
        rain_indoor_priority=bool(checkpoint.get("rain_indoor_priority", False)),
        weather_forecast=checkpoint.get("weather_forecast", []),
        weather_note=checkpoint.get("weather_note"),
        route_distance_legs=checkpoint.get("route_distance_legs", []),
        route_distance_mode=checkpoint.get("route_distance_mode"),
        route_distance_note=checkpoint.get("route_distance_note"),
        route_risk_flags=checkpoint.get("route_risk_flags", []),
        route_risk_score=int(checkpoint.get("route_risk_score", 0) or 0),
        review_required=bool(checkpoint.get("review_required", True)),
        time_check_required=bool(checkpoint.get("time_check_required", True)),
        review_skipped=bool(checkpoint.get("review_skipped", False)),
        max_per_day=checkpoint.get("max_per_day", 3),
        route_modify_opinion=f"【用户修改意见】{modification_notes}",
        max_review_rounds=2,
        **{k: v for k, v in overrides.items()
           if k not in ("model_name", "max_per_day", "route_modify_opinion", "max_review_rounds")},
    )
    config = {"recursion_limit": 2 * (2 + 1) + 2 * init.max_time_check_rounds + 14}

    acc: dict[str, Any] = init.model_dump()
    planner_done = False

    async for event in app.astream_events(init, config=config, version="v2"):
        ev_type = event.get("event")
        node = event.get("name")
        if node not in _NODE_LABELS:
            continue
        if ev_type == "on_chain_start":
            yield _stage_event(node, acc, {})
            continue
        if ev_type != "on_chain_end":
            continue
        upd = (event.get("data") or {}).get("output")
        if not isinstance(upd, dict):
            continue

        # planner 完成后检查顾虑（仅第 1 轮：直接响应用户修改意见时才暂停）
        if node == "planner":
            # 必须在 acc.update(upd) 之前推事件：
            # _stage_event 对 planner 的轮次计算假设 acc 是节点运行前的状态
            # （review_round 尚未递增），update 之后再算会多加 1
            planner_done = True
            yield {"type": "stage_summary", "node": node, "summary": _stage_summary(node, acc, upd)}
            acc.update(upd)
            concern = acc.get("modification_concern") or ""
            if concern and acc.get("review_round") == 1:
                # 有顾虑：构造 pending_state 供调用方存 DB，然后停止
                pending_state = {
                    "route": acc.get("route", []),
                    "pois":  init.pois,
                    "planner_reviewer_dialogue": acc.get("planner_reviewer_dialogue", []),
                    "destination": init.destination,
                    "travel_start_date": str(init.travel_start_date or ""),
                    "travel_end_date":   str(init.travel_end_date or ""),
                    "days": init.days,
                    "attraction_preference": init.attraction_preference,
                    "food_preference":       init.food_preference,
                    "habit_preference":      init.habit_preference,
                    "max_walking_km":        init.max_walking_km,
                    "rain_indoor_priority":  init.rain_indoor_priority,
                    "weather_forecast": init.weather_forecast,
                    "weather_note":     init.weather_note,
                    "route_distance_legs": acc.get("route_distance_legs", []),
                    "route_distance_mode": acc.get("route_distance_mode"),
                    "route_distance_note": acc.get("route_distance_note"),
                    "max_per_day":      init.max_per_day,
                    "query":            init.query,
                }
                yield {
                    "type": "modification_warning",
                    "concern": concern,
                    "pending_state": pending_state,  # 由 main.py 存 DB 并替换为 pending_id
                }
                return
            continue  # 无顾虑，继续

        yield {"type": "stage_summary", "node": node, "summary": _stage_summary(node, acc, upd)}
        acc.update(upd)

    if not planner_done:
        # planner 事件未触发（不应出现）
        yield {"type": "result", "success": False, "missing_fields": ["规划失败，请重试"], "history": [], "plan": None}
        return

    final = TravelPlanState(**acc)
    success = final.final_plan is not None
    yield {
        "type": "result",
        "success": success,
        "missing_fields": [],
        "history": final.history,
        "plan": final.final_plan if success else None,
    }


async def run_confirm_stream(
    pending_state: dict[str, Any],
    memory_writer=None,
    **overrides: Any,
) -> AsyncIterator[dict[str, Any]]:
    """用户确认后，从 pending_state 续跑 meal_search → finalize。"""
    app = build_confirm_graph(overrides.get("model_name"), memory_writer)
    init = TravelPlanState(
        query=pending_state.get("query", "修改行程"),
        route=pending_state.get("route", []),
        pois=pending_state.get("pois", []),
        planner_reviewer_dialogue=pending_state.get("planner_reviewer_dialogue", []),
        destination=pending_state.get("destination"),
        travel_start_date=pending_state.get("travel_start_date"),
        travel_end_date=pending_state.get("travel_end_date"),
        days=pending_state.get("days", 0),
        attraction_preference=pending_state.get("attraction_preference"),
        food_preference=pending_state.get("food_preference"),
        habit_preference=pending_state.get("habit_preference"),
        max_walking_km=pending_state.get("max_walking_km"),
        rain_indoor_priority=bool(pending_state.get("rain_indoor_priority", False)),
        weather_forecast=pending_state.get("weather_forecast", []),
        weather_note=pending_state.get("weather_note"),
        route_distance_legs=pending_state.get("route_distance_legs", []),
        route_distance_mode=pending_state.get("route_distance_mode"),
        route_distance_note=pending_state.get("route_distance_note"),
        max_per_day=pending_state.get("max_per_day", 3),
        **{k: v for k, v in overrides.items() if k not in ("model_name", "max_per_day")},
    )
    config = {"recursion_limit": 2 * (init.max_review_rounds + 1) + 2 * init.max_time_check_rounds + 14}

    acc: dict[str, Any] = init.model_dump()
    async for event in app.astream_events(init, config=config, version="v2"):
        ev_type = event.get("event")
        node = event.get("name")
        if node not in _NODE_LABELS:
            continue
        if ev_type == "on_chain_start":
            yield _stage_event(node, acc, {})
            continue
        if ev_type != "on_chain_end":
            continue
        upd = (event.get("data") or {}).get("output")
        if not isinstance(upd, dict):
            continue
        yield {"type": "stage_summary", "node": node, "summary": _stage_summary(node, acc, upd)}
        acc.update(upd)

    final = TravelPlanState(**acc)
    success = final.final_plan is not None
    yield {
        "type": "result",
        "success": success,
        "missing_fields": [],
        "history": final.history,
        "plan": final.final_plan if success else None,
    }


async def run_stream(
    query: str,
    profile_hint: str = "",
    memory_writer=None,
    user_id: str | None = None,
    **overrides: Any,
) -> AsyncIterator[dict[str, Any]]:
    """分阶段流式执行流水线，逐节点 yield 进度事件，末尾 yield 最终结果。

    事件类型：
      {"type": "stage",  "node", "label", ...}   每个节点完成时
      {"type": "result", "success", "missing_fields", "history", "plan", "plan_id"}  全部结束

    与 `run()` 共用 build_graph / config，仅改为 astream_events 事件源；
    旧 invoke 路径不受影响。
    """
    app = build_graph(
        overrides.get("model_name"),
        profile_hint=profile_hint,
        memory_writer=memory_writer,
        user_id=user_id,
    )
    init = TravelPlanState(query=query, profile_hint=profile_hint or None, **overrides)
    # 主循环 planner⇄reviewer 最多 (max_review_rounds+1) 对节点；
    # 时间修正 planner⇄time_check 最多 max_time_check_rounds 对节点；
    # 其余非循环节点（intent/query_rewrite/attraction_search/meal_search/meal_recommend/spot_tips/finalize）+ 缓冲
    config = {"recursion_limit":
        2 * (init.max_review_rounds + 1) + 2 * init.max_time_check_rounds + 10}

    acc: dict[str, Any] = init.model_dump()
    async for event in app.astream_events(init, config=config, version="v2"):
        ev_type = event.get("event")
        node    = event.get("name")
        if node not in _NODE_LABELS:
            continue
        if ev_type == "on_chain_start":
            # 节点开始时立即推送进度，避免用户等待长时间后才看到第一条进度
            yield _stage_event(node, acc, {})
        elif ev_type == "on_chain_end":
            # 节点完成后更新累积状态（不再重复推送 stage，避免前端出现重复条目）
            upd = (event.get("data") or {}).get("output")
            if isinstance(upd, dict):
                yield {"type": "stage_summary", "node": node, "summary": _stage_summary(node, acc, upd)}
                acc.update(upd)

    final = TravelPlanState(**acc)
    success = not bool(final.missing_fields) and final.final_plan is not None
    yield {
        "type": "result",
        "success": success,
        "missing_fields": final.missing_fields,
        "history": final.history,
        "plan": final.final_plan if success else None,
    }
