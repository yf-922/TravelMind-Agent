"""Bound optional tips independently of the main planning lifecycle."""

import asyncio
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from langchain_core.runnables import RunnableLambda

from app.core.agent_runs import node_timeout_seconds


def tips_timeout_seconds():
    try:
        configured = float(os.getenv("SPOT_TIPS_TIMEOUT_SECONDS", "15"))
    except ValueError:
        configured = 15.0
    return min(max(configured, 0.1), max(0.1, node_timeout_seconds("spot_tips") - 0.5))


async def run_spot_tips(node, state):
    # A cancelled sync worker can finish later. It receives a private copy and
    # its late result is discarded, never merged into the finalized itinerary.
    local = state.model_copy(deep=True)
    work = node(local) if inspect.iscoroutinefunction(node) else asyncio.to_thread(node, local)
    try:
        return await asyncio.wait_for(work, timeout=tips_timeout_seconds())
    except Exception:
        return {"spot_tips": {}, "spot_guides": {}, "spot_tips_status": "degraded"}


def bounded_spot_tips(node):
    def sync_tips(state):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optional-tips")
        context = copy_context()
        future = executor.submit(context.run, node, state.model_copy(deep=True))
        try:
            return future.result(timeout=tips_timeout_seconds())
        except Exception:
            return {"spot_tips": {}, "spot_guides": {}, "spot_tips_status": "degraded"}
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    async def spot_tips(state):
        return await run_spot_tips(node, state)
    return RunnableLambda(sync_tips, afunc=spot_tips)
