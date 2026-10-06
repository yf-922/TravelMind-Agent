from app.planning import nodes
from app.planning.schemas import MealSlotPlan, TravelPlanState, TravelRoute


def test_joint_planner_keeps_route_after_name_canonicalization(monkeypatch):
    class FakeLLM:
        pass

    monkeypatch.setattr(nodes, "build_structured_llm", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(nodes, "invoke_structured", lambda *args, **kwargs: TravelRoute(
        reasoning="use the verified candidate",
        days=[{"day": 1, "theme": "culture", "spots": [{
            "name": "Musem", "period": "morning", "start_time": "10:00", "end_time": "11:00",
        }]}],
        meal_slots=[MealSlotPlan(meal="lunch", restaurant_name="Verified Restaurant",
                                 start_time="12:00", end_time="13:00")],
        notes="joint",
    ))
    state = TravelPlanState(query="trip", destination="Nanjing", days=1,
                            pois=[{"name": "Museum"}],
                            main_meal_candidates=[{"name": "Verified Restaurant"}])
    update = nodes.make_joint_planner_node(None)(state)
    assert update["route"][0]["spots"][0]["name"] == "Museum"
    assert update["meal_slots"][0]["restaurant_name"] == "Verified Restaurant"


def test_joint_meal_risk_gate_rejects_unknown_and_overlapping_restaurant():
    state = TravelPlanState(query="trip", days=1,
                            route=[{"day": 1, "spots": [{
                                "name": "Museum", "start_time": "12:30", "end_time": "13:30",
                            }]}],
                            main_meal_candidates=[{"name": "Known", "open_time": "11:00-14:00"}],
                            meal_slots=[{"day": 1, "restaurant_name": "Unknown",
                                          "start_time": "12:00", "end_time": "13:00"}])
    flags = nodes._main_meal_risk_flags(state)
    assert "meal_unknown_restaurant" in flags
    assert "meal_overlap" in flags


def test_supervisor_production_nodes_include_joint_meal_stage(monkeypatch):
    import app.planning.nodes as production
    from app.multi_agent_core.runtime import production_nodes

    monkeypatch.setattr(production, "build_structured_llm", lambda *args, **kwargs: object())
    nodes_map = production_nodes()
    assert "main_meal_search" in nodes_map
    assert nodes_map["planner"] is not None
    assert "main_meal_candidates" in __import__("app.multi_agent_core.runtime", fromlist=["INPUTS"]).INPUTS["planner"]
