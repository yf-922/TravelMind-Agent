from app.planning.nodes import _build_day_budget, _build_travel_leg, enrich_plan_ticket_budget
from app.providers.tickets.catalog import enrich_attraction_ticket, lookup_ticket_price

from datetime import date


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


def test_official_ticket_catalog_handles_season_and_free_admission():
    assert lookup_ticket_price("颐和园", date(2026, 7, 27))["price"] == 30
    assert lookup_ticket_price("颐和园", date(2026, 12, 1))["price"] == 20
    assert lookup_ticket_price("颐和园博物馆", date(2026, 7, 27))["price"] == 20
    assert lookup_ticket_price("紫竹院公园", date(2026, 7, 27))["price"] == 0
    assert lookup_ticket_price("名称相似但未核验的公园", date(2026, 7, 27)) is None


def test_free_attraction_is_counted_as_zero_instead_of_unknown():
    item = enrich_attraction_ticket(
        {"type": "attraction", "name": "紫竹院公园", "cost": None},
        date(2026, 7, 27),
    )
    budget = _build_day_budget([item])
    assert budget["ticket_known"] == 0
    assert budget["unknown_items"] == []
    assert item["ticket_info"]["source_url"].startswith("https://")


def test_existing_plan_ticket_budget_can_be_backfilled():
    plan = {
        "days": [{
            "date": "2026-07-27",
            "timeline": [
                {"type": "attraction", "name": "颐和园", "cost": None},
                {"type": "attraction", "name": "颐和园博物馆", "cost": None},
                {"type": "attraction", "name": "紫竹院公园", "cost": None},
            ],
        }],
    }

    enrich_plan_ticket_budget(plan)

    assert plan["budget_summary"]["ticket_known"] == 50
    assert plan["budget_summary"]["unknown_items"] == []
    assert all("ticket_info" in item for item in plan["days"][0]["timeline"])
