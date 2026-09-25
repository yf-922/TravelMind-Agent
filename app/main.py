"""FastAPI 应用入口。

路由：
  POST /api/plan/stream — SSE 流式规划（含多轮续接、修改规划、记忆注入）
  POST /api/plan        — 同步规划（向后兼容）
  POST /api/auth/*      — 注册 / 登录
  GET  /api/history     — 历史行程列表
  GET  /api/history/:id — 历史行程详情
  GET  /api/health      — 健康检查
  GET  /api/config      — 前端高德 JS Key
  GET  /                — 前端首页
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import asyncio
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core.auth import decode_token
from app.core.agent_runs import agent_runs, observe_agent_events
from app.core.database import get_conn, init_db
from app.core.env import load_local_env
from app.core.memory import (
    format_profile_for_prompt,
    get_user_profile,
    load_itinerary,
    save_itinerary,
    save_pending_modification,
    set_user_profile,
)
from app.core.semantic_memory import (
    format_semantic_memories,
    run_semantic_memory_update,
    search_user_memories,
)
from app.core.thread_store import thread_store
from app.core.cache import redis_status
from app.core.observability import prometheus_text, request_finished, request_started
from app.planning.graph import run_modification_stream
from app.planning.graph import run_stream as run_plan_stream
from app.planning.helpers import amap_key, merge_verified_poi, missing_explicit_places
from app.planning.profile_updater import run_profile_update_agent
from app.providers.amap.poi import ATTRACTION_TYPE, poi_to_spot, search_city_pois
from app.api.auth_routes import router as auth_router
from app.api.history_routes import router as history_router
from app.api.profile_routes import router as profile_router
from app.api.plan_routes import router as plan_router
from app.api.sweep_routes import router as sweep_router

load_local_env()
init_db()
logger = logging.getLogger(__name__)

# ─── 应用 ────────────────────────────────────────────────────

app = FastAPI(title="AI 旅游规划助手", version="0.1.0")


@app.middleware("http")
async def request_observability(request: Request, call_next):
    """Attach a request id and record latency without exposing prompt contents."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    request_started()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        request_finished(request.url.path, status_code, (time.perf_counter() - started) * 1000)
        if 'response' in locals():
            response.headers["X-Request-ID"] = request_id

app.include_router(auth_router)
app.include_router(history_router)
app.include_router(profile_router)
app.include_router(plan_router)
app.include_router(sweep_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:8765").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Agent-Run-ID"],
)


# ─── 辅助：从 Authorization header 提取 user_id（强制登录）────

def _get_required_user_id(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录后再使用规划功能")
    user_id = decode_token(auth[7:])
    if not user_id:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    return user_id


async def _encode_observed_sse(
    source: AsyncIterator[dict[str, Any]], run_id: str
) -> AsyncIterator[str]:
    async for event in observe_agent_events(source, run_id):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ─── API 路由 ─────────────────────────────────────────────────

class PlanRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    max_per_day: int = Field(default=5, ge=1, le=12)
    min_rating: float = Field(default=4.5, ge=0, le=5)
    max_spots: int = Field(default=30, ge=1, le=100)
    max_review_rounds: int = Field(default=3, ge=0, le=8)
    # 多轮续接
    thread_id: Optional[str] = None
    # 修改规划
    plan_id: Optional[str] = None
    modification_notes: Optional[str] = None
    # A place selected from the manual POI search. The server verifies it again
    # before adding it to the checkpoint candidate pool for local replanning.
    selected_poi_name: Optional[str] = None


@app.get("/api/health")
def health():
    """Liveness/readiness probe with dependency status for deployment checks."""
    db_status = "ok"
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception:
        db_status = "error"
    cache = redis_status()
    overall = "ok" if db_status == "ok" else "degraded"
    return {"status": overall, "database": db_status, "redis": cache["status"]}


@app.get("/api/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus-compatible counters for local demos and deployment probes."""
    return prometheus_text()


@app.get("/api/agent-runs/{run_id}")
def agent_run_trace(run_id: str, request: Request):
    """Return a privacy-safe node trace to the user who owns the run."""
    user_id = _get_required_user_id(request)
    trace = agent_runs.get_owned(run_id, user_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="执行记录不存在或已过期")
    return trace


@app.get("/api/config")
def config():
    """暴露前端地图所需的高德 JS API 密钥（不含敏感 REST Key）。"""
    return {
        "amap_js_key":           os.getenv("AMAP_JS_KEY", ""),
        "amap_js_security_code": os.getenv("AMAP_JS_SECURITY_CODE", ""),
    }




@app.post("/api/plan/stream")
async def create_plan_stream(req: PlanRequest, request: Request):
    """分阶段 SSE：逐节点推送进度，末帧推送完整 plan。

    扩展能力：
    - thread_id: 多轮续接（missing_fields 后补充信息）
    - plan_id + modification_notes: 修改已有行程（走 checkpoint 迷你图）
    - Authorization: 登录用户自动保存行程 + 记忆提取 + checkpoint 存储
    """
    user_id = _get_required_user_id(request)

    # ── 1. 多轮续接：合并原始 query ──────────────────────────
    if req.thread_id:
        original = thread_store.get(req.thread_id)
        if original:
            query = f"{original}，{req.query}"
            thread_store.delete(req.thread_id)
        else:
            query = req.query
    else:
        query = req.query

    # ── 2. 基础 overrides ────────────────────────────────────
    parent_plan_id = req.plan_id
    overrides = dict(
        max_per_day=req.max_per_day,
        min_rating=req.min_rating,
        max_spots=req.max_spots,
        max_review_rounds=req.max_review_rounds,
    )

    # ── 3. memory_writer closure（供 finalize 节点写入） ──────
    saved_plan_id: list[str] = []  # 用列表传引用

    def memory_writer(final_plan: dict, state) -> None:
        planner_checkpoint = {
            "route": state.route,
            "pois":  state.pois,
            "planner_reviewer_dialogue": state.planner_reviewer_dialogue,
            "destination": str(state.destination or ""),
            "travel_start_date": str(state.travel_start_date or ""),
            "travel_end_date":   str(state.travel_end_date or ""),
            "days": state.days,
            "attraction_preference": state.attraction_preference,
            "food_preference":       state.food_preference,
            "habit_preference":      state.habit_preference,
            "max_walking_km":        state.max_walking_km,
            "rain_indoor_priority":  state.rain_indoor_priority,
            "weather_forecast": state.weather_forecast,
            "weather_note":     state.weather_note,
            "route_distance_legs": state.route_distance_legs,
            "route_distance_mode": state.route_distance_mode,
            "route_distance_note": state.route_distance_note,
            "route_risk_flags": state.route_risk_flags,
            "route_risk_score": state.route_risk_score,
            "review_required": state.review_required,
            "time_check_required": state.time_check_required,
            "review_skipped": state.review_skipped,
            "max_per_day":      state.max_per_day,
            "query":            state.query,
        }
        with get_conn() as conn:
            pid = save_itinerary(
                user_id, final_plan, query, conn,
                parent_id=parent_plan_id,
                modification_notes=req.modification_notes,
                planner_state=planner_checkpoint,
            )
            # visited_destinations 由代码维护（客观事实，不交给 LLM）
            if state.destination:
                profile = get_user_profile(user_id, conn)
                dests = profile.get("visited_destinations", [])
                if state.destination not in dests:
                    dests = ([state.destination] + dests)[:20]
                    set_user_profile(user_id, {**profile, "visited_destinations": dests}, conn)
        saved_plan_id.append(pid)

    # ── 4. 修改模式：走 checkpoint 迷你图 ────────────────────
    if req.plan_id and req.modification_notes:
        with get_conn() as conn:
            data = load_itinerary(req.plan_id, conn)
        checkpoint = data["planner_state"] if data else None

        if checkpoint:
            modification_notes = req.modification_notes
            # 对带引号的明确景点名做保守的候选池覆盖检测。普通偏好描述不触发
            # 外部搜索，避免把模糊语义误当成 POI；用户明确选择的景点仍走下方
            # selected_poi_name 的严格核验路径。
            if not req.selected_poi_name:
                missing_places = missing_explicit_places(
                    modification_notes, list(checkpoint.get("pois") or [])
                )
                for requested_name in missing_places[:3]:
                    destination = str(checkpoint.get("destination") or "").strip()
                    if not destination:
                        logger.warning("[modification] skip targeted POI lookup without destination name=%s", requested_name)
                        break
                    try:
                        raw_pois = search_city_pois(
                            destination, amap_key(), keywords=requested_name,
                            types=ATTRACTION_TYPE, offset=8,
                        )
                    except RuntimeError as exc:
                        logger.warning("[modification] targeted POI lookup failed name=%s error=%s", requested_name, exc)
                        continue
                    verified = next(
                        (poi_to_spot(raw) for raw in raw_pois
                         if str(raw.get("name") or "").strip() == requested_name),
                        None,
                    )
                    if verified:
                        checkpoint = {
                            **checkpoint,
                            "pois": merge_verified_poi(list(checkpoint.get("pois") or []), verified),
                        }
                        modification_notes += (
                            f"\n【系统已核验新增景点】{verified['name']}，已加入候选池并参与本轮重规划。"
                        )
            if req.selected_poi_name:
                poi_name = req.selected_poi_name.strip()
                destination = str(checkpoint.get("destination") or "").strip()
                if not poi_name or len(poi_name) > 100:
                    raise HTTPException(status_code=400, detail="景点名称不能为空或超过 100 个字符")
                if not destination:
                    raise HTTPException(status_code=400, detail="原行程缺少目的地，无法核验新增景点")
                try:
                    raw_pois = search_city_pois(
                        destination,
                        amap_key(),
                        keywords=poi_name,
                        types=ATTRACTION_TYPE,
                        offset=8,
                    )
                except RuntimeError as exc:
                    raise HTTPException(status_code=502, detail=f"新增景点核验失败：{exc}") from exc

                verified = next(
                    (poi_to_spot(raw) for raw in raw_pois if str(raw.get("name") or "").strip() == poi_name),
                    None,
                )
                if not verified:
                    raise HTTPException(
                        status_code=404,
                        detail=f"未能在 {destination} 核验到景点“{poi_name}”，请从搜索结果中重新选择",
                    )
                checkpoint = {
                    **checkpoint,
                    "pois": merge_verified_poi(list(checkpoint.get("pois") or []), verified),
                }
                modification_notes = (
                    f"{modification_notes}\n"
                    f"【用户已确认新增景点】{verified['name']}。该景点已由高德核验并加入候选池，"
                    "必须安排进最终行程；请据此重新平衡当天顺序、时间和周边餐饮。"
                )

            async def gen_modification():
                async for ev in run_modification_stream(
                    checkpoint,
                    modification_notes,
                    memory_writer=memory_writer,
                    **overrides,
                ):
                    if ev.get("type") == "modification_warning":
                        # 存 pending 状态到 DB，用 pending_id 替换 pending_state
                        pending_state = ev.pop("pending_state", {})
                        if user_id:
                            with get_conn() as conn:
                                pid = save_pending_modification(user_id, pending_state, conn)
                            ev["pending_id"] = pid
                        ev["parent_plan_id"] = parent_plan_id
                    elif ev.get("type") == "result" and ev.get("success") and saved_plan_id:
                        ev["plan_id"] = saved_plan_id[0]
                    yield ev

            run_id = agent_runs.start(
                user_id, "modification", getattr(request.state, "request_id", None)
            )

            return StreamingResponse(
                _encode_observed_sse(gen_modification(), run_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                    "X-Agent-Run-ID": run_id,
                },
            )
        # 无 checkpoint（旧数据）→ 降级走普通流程，带 modification_notes
        overrides["modification_notes"] = req.modification_notes
        if parent_plan_id:
            overrides["parent_plan_id"] = parent_plan_id

    # ── 5. 记忆注入：读取用户历史偏好 ────────────────────────
    profile_hint = ""
    if user_id:
        with get_conn() as conn:
            profile = get_user_profile(user_id, conn)
        structured_hint = format_profile_for_prompt(profile)
        semantic_hint = ""
        # Chroma may download its embedding model on first use. Never perform
        # that blocking operation on FastAPI's event loop: otherwise a single
        # planning request can make login and health checks appear frozen.
        try:
            memories = await asyncio.wait_for(
                asyncio.to_thread(search_user_memories, user_id, query),
                timeout=min(5.0, max(1.0, float(os.getenv("MEMORY_LOOKUP_TIMEOUT_SECONDS", "2.5")))),
            )
            semantic_hint = format_semantic_memories(memories)
        except TimeoutError:
            logger.warning("[semantic_memory] lookup timed out; continuing without semantic memory")
        except (ValueError, TypeError):
            logger.warning("[semantic_memory] invalid lookup timeout configuration; skipping lookup")
        except Exception:  # noqa: BLE001
            logger.warning("[semantic_memory] lookup failed; continuing without semantic memory", exc_info=True)
        profile_hint = "\n".join(part for part in (structured_hint, semantic_hint) if part)

    # ── 6. 普通规划流 ─────────────────────────────────────────
    async def gen():
        async for ev in run_plan_stream(
            query,
            profile_hint=profile_hint,
            memory_writer=memory_writer,
            user_id=user_id,
            **overrides,
        ):
            # missing_fields 时追加 thread_id 供前端续接
            if ev.get("type") == "result" and not ev.get("success") and ev.get("missing_fields"):
                ev["thread_id"] = thread_store.create(query)
            # 成功时追加 plan_id，并触发异步画像更新（只看 raw query，不看改写后的）
            if ev.get("type") == "result" and ev.get("success") and saved_plan_id:
                ev["plan_id"] = saved_plan_id[0]
                asyncio.create_task(
                    run_profile_update_agent(user_id, query, overrides.get("model_name"))
                )
                asyncio.create_task(
                    run_semantic_memory_update(user_id, query, overrides.get("model_name"))
                )
            yield ev

    run_id = agent_runs.start(user_id, "plan", getattr(request.state, "request_id", None))

    return StreamingResponse(
        _encode_observed_sse(gen(), run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Agent-Run-ID": run_id,
        },
    )


# ─── 静态文件（前端）—— 必须在 API 路由之后挂载 ─────────────

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/")
def index():
    return FileResponse(_FRONTEND_DIR / "index.html")


@app.get("/history")
def history_page():
    return FileResponse(_FRONTEND_DIR / "history.html")


@app.get("/profile")
def profile_page():
    return FileResponse(_FRONTEND_DIR / "profile.html")


app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")
