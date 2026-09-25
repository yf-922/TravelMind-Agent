from pydantic import BaseModel

from app.llm import factory
from app.llm.router import ModelRouter, route_model
from app.planning import nodes


def test_router_keeps_primary_model_when_disabled():
    router = ModelRouter(primary_model="strong", small_model="cheap", enabled=False)

    decision = router.choose("query_rewrite", estimated_input_tokens=100)

    assert decision.model == "strong"
    assert decision.task_class == "short"
    assert decision.routing_enabled is False


def test_router_sends_short_tasks_to_small_model_and_planning_to_primary():
    router = ModelRouter(primary_model="strong", small_model="cheap", enabled=True)

    rewrite = router.choose("query_rewrite", estimated_input_tokens=100)
    planner = router.choose("planner", estimated_input_tokens=3000)

    assert rewrite.model == "cheap"
    assert rewrite.reason.startswith("bounded task")
    assert planner.model == "strong"
    assert planner.task_class == "complex"


def test_route_model_preserves_explicit_primary_override_when_disabled(monkeypatch):
    monkeypatch.setenv("LLM_ROUTING_ENABLED", "0")
    monkeypatch.setenv("LLM_SMALL_MODEL", "cheap")

    decision = route_model("query_rewrite", primary_model="explicit")

    assert decision.model == "explicit"
    assert decision.routing_enabled is False


def test_router_default_provider_matches_factory_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-primary")
    monkeypatch.setenv("LLM_ROUTING_ENABLED", "0")

    assert route_model("planner").model == "deepseek-primary"


def test_factory_applies_enabled_routing_without_calling_provider(monkeypatch):
    class Output(BaseModel):
        value: str

    selected: list[str | None] = []
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_MODEL", "strong")
    monkeypatch.setenv("LLM_SMALL_MODEL", "cheap")
    monkeypatch.setenv("LLM_ROUTING_ENABLED", "1")
    monkeypatch.setattr(
        "app.llm.deepseek.build_structured_deepseek",
        lambda schema, *, model, temperature: selected.append(model) or object(),
    )

    factory.build_structured_llm(Output, task_type="query_rewrite")
    factory.build_structured_llm(Output, task_type="planner")

    assert selected == ["cheap", "strong"]


def test_production_nodes_declare_task_types(monkeypatch):
    task_types: list[str | None] = []

    def fake_builder(*args, **kwargs):
        task_types.append(kwargs.get("task_type"))
        return object()

    monkeypatch.setattr(nodes, "build_structured_llm", fake_builder)
    nodes.make_query_rewrite_node(None, None)
    nodes.make_intent_node(None)
    nodes.make_planner_node(None)
    nodes.make_reviewer_node(None)
    nodes.make_time_check_node(None)
    nodes.make_meal_recommend_node(None)
    nodes.make_spot_tips_node(None)

    assert task_types == [
        "query_rewrite", "intent", "planner", "reviewer",
        "time_check", "meal_recommend", "spot_tips",
    ]
