from app.planning.nodes import _build_day_budget, _build_travel_leg


def test_travel_leg_recommends_modes_by_distance():
    walk = _build_travel_leg(0.8, "A", "B")
    assert walk["mode"] == "walk"
    assert walk["estimated_cost"] == 0

    transit = _build_travel_leg(6.0, "A", "B")
    assert transit["mode"] == "transit"
    assert transit["duration_min"] > 0
    assert transit["estimated_cost"] >= 2

    car = _build_travel_leg(20.0, "A", "B")
    assert car["mode"] == "taxi_or_car"
    assert "租车" in car["instruction"]


def test_day_budget_separates_known_and_unknown_costs():
    timeline = [
        {"type": "attraction", "name": "收费景点", "cost": "80"},
        {
            "type": "lunch",
            "name": "午餐店",
            "cost": "45元",
            "travel_from_prev": {"estimated_cost": 4},
        },
        {
            "type": "attraction",
            "name": "价格未知景点",
            "cost": None,
            "travel_from_prev": {"estimated_cost": 0},
        },
    ]
    budget = _build_day_budget(timeline)
    assert budget["ticket_known"] == 80
    assert budget["meal_known"] == 45
    assert budget["transport_estimated"] == 4
    assert budget["known_subtotal"] == 129
    assert "价格未知景点门票" in budget["unknown_items"]
