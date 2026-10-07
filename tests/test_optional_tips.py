import asyncio
import threading

from app.planning import enrichment
from app.planning.schemas import TravelPlanState


def test_hung_sync_tips_returns_without_waiting_for_worker_and_keeps_route(monkeypatch):
    monkeypatch.setattr(enrichment, "tips_timeout_seconds", lambda: 0.01)
    release = threading.Event()
    state = TravelPlanState(query="trip", route=[{"day": 1, "spots": [{"name": "Museum"}]}])
    def hung(local):
        release.wait(2)
        local.route.clear()
        return {"spot_tips": {"Museum": "late"}}
    async def test():
        try:
            result = await enrichment.run_spot_tips(hung, state)
            assert not release.is_set()
            assert result["spot_tips_status"] == "degraded"
            assert result["spot_tips"] == {}
        finally:
            release.set()
    asyncio.run(test())
    assert state.route[0]["spots"][0]["name"] == "Museum"


def test_optional_tips_failure_still_allows_graph_finalization(monkeypatch):
    from langgraph.graph import StateGraph, START, END
    from app.planning.graph import _stage_summary
    monkeypatch.setattr(enrichment, "tips_timeout_seconds", lambda: 0.01)
    async def slow(state):
        await asyncio.sleep(1)
    graph = StateGraph(TravelPlanState)
    graph.add_node("spot_tips", enrichment.bounded_spot_tips(slow))
    graph.add_node("finalize", lambda s: {"final_plan": {"route": s.route, "tips_status": s.spot_tips_status}})
    graph.add_edge(START, "spot_tips")
    graph.add_edge("spot_tips", "finalize")
    graph.add_edge("finalize", END)
    async def run():
        return await graph.compile().ainvoke(TravelPlanState(query="trip", route=[{"day": 1, "spots": []}]))
    result = asyncio.run(run())
    assert result["final_plan"]["tips_status"] == "degraded"
    assert result["final_plan"]["route"] == [{"day": 1, "spots": []}]
    assert "已跳过" in _stage_summary("spot_tips", {}, result)


def test_tips_deadline_is_shorter_than_observer_deadline(monkeypatch):
    monkeypatch.setenv("SPOT_TIPS_TIMEOUT_SECONDS", "15")
    monkeypatch.setattr(enrichment, "node_timeout_seconds", lambda node: 5)
    assert enrichment.tips_timeout_seconds() == 4.5
