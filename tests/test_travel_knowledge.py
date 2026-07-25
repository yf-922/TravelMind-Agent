from app.core import travel_knowledge
from app.planning import nodes
from app.planning.schemas import DayRoute, SpotPlan, TravelPlanState, TravelRoute


def test_rag_loads_three_version_controlled_knowledge_documents():
    documents = travel_knowledge.load_documents()

    assert len(documents) >= 3
    assert {item["source"] for item in documents} >= {
        "chongqing_visitor_guide",
        "transport_and_pacing",
        "responsible_trip_checklist",
    }


def test_rag_chunks_preserve_readable_paragraph_context():
    chunks = travel_knowledge._split_text("第一段完整句子。\n\n第二段完整句子。\n\n第三段完整句子。", size=16)

    assert chunks
    assert all(not chunk.startswith("整句") for chunk in chunks)


def test_search_docs_returns_source_labels(monkeypatch):
    monkeypatch.setattr(travel_knowledge, "search_travel_knowledge", lambda query, limit: [{
        "source": "transport_and_pacing", "chunk_id": "transport_and_pacing-0", "text": "不能假设每段都步行。",
    }])

    result = travel_knowledge.search_docs("能否每段都走路")

    assert "[source: transport_and_pacing#transport_and_pacing-0]" in result
    assert "不能假设每段都步行" in result


def test_rag_acceptance_set_retrieves_expected_source_in_top_1():
    travel_knowledge.build_index(rebuild=True)
    cases = [
        ("重庆带老人下雨怎么安排", "chongqing_visitor_guide"),
        ("每段交通能都走路吗", "transport_and_pacing"),
        ("资料没有门票价格怎么办", "responsible_trip_checklist"),
        ("远郊地点搜索不到怎么办", "chongqing_visitor_guide"),
        ("一天安排几个景点合适", "transport_and_pacing"),
    ]

    for query, expected_source in cases:
        rows = travel_knowledge.search_travel_knowledge(query)
        assert rows, query
        assert rows[0]["source"] == expected_source, query


def test_planner_agent_automatically_receives_rag_tool_result(monkeypatch):
    captured = {}

    class FakePlannerModel:
        def invoke(self, messages):
            captured["prompt"] = messages[-1][1]
            return TravelRoute(
                reasoning="Use the verified candidate.",
                days=[DayRoute(day=1, theme="test", spots=[SpotPlan(
                    name="Museum", period="morning", start_time="09:00", end_time="11:00"
                )])],
                notes="[source: transport_and_pacing#transport_and_pacing-0]",
            )

    source = {
        "source": "transport_and_pacing",
        "chunk_id": "transport_and_pacing-0",
        "text": "Do not assume every route can be walked.",
        "distance": 0.1,
    }
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *args, **kwargs: FakePlannerModel())
    monkeypatch.setattr(nodes, "search_travel_knowledge", lambda *args, **kwargs: [source])

    update = nodes.make_planner_node(None)(TravelPlanState(
        query="Plan a relaxed day", destination="Chongqing", days=1,
        pois=[{"name": "Museum", "location": {"lat": 1, "lng": 1}}],
    ))

    assert "[source: transport_and_pacing#transport_and_pacing-0]" in captured["prompt"]
    assert update["rag_sources"] == [source]
