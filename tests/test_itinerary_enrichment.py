from datetime import date

from app.planning import nodes
from app.planning.nodes import _build_day_budget, _build_travel_leg, enrich_plan_ticket_budget
from app.providers import pricing
from app.providers.tickets.live import _parse_official_page


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


def test_official_ticket_page_parser_handles_season_and_free_admission():
    source = ("官网", "https://example.com")
    palace_text = "每年4月1日至10月31日为旺季，大门票60元/人；每年11月1日至次年3月31日为淡季，大门票40元/人"
    assert _parse_official_page(palace_text, "palace_museum", date(2026, 7, 27), *source)["price"] == 60
    assert _parse_official_page(palace_text, "palace_museum", date(2026, 12, 1), *source)["price"] == 40
    assert _parse_official_page("门票价格：免费", "free_page", None, *source)["price"] == 0
    assert _parse_official_page("门票价格: 门票价格：2元；半价：1元", "jingshan", None, *source)["price"] == 2


def test_free_attraction_is_counted_as_zero_instead_of_unknown():
    ticket = {
        "price": 0, "price_label": "实时查询·免费", "source_name": "官网",
        "source_url": "https://example.com", "verified_at": "2026-07-27",
        "live_query": True,
    }
    item = pricing.resolve_attraction_price(
        {"type": "attraction", "name": "紫竹院公园", "cost": None},
        "2026-07-27", live_ticket=ticket, live_lookup_done=True,
    )
    budget = _build_day_budget([item])
    assert budget["ticket_known"] == 0
    assert budget["unknown_items"] == []
    assert item["ticket_info"]["source_url"].startswith("https://")


def test_existing_plan_ticket_budget_can_be_backfilled(monkeypatch):
    prices = {"颐和园": 30, "颐和园博物馆": 20, "紫竹院公园": 0}
    monkeypatch.setattr(nodes, "lookup_live_ticket_prices", lambda _requests: {})
    monkeypatch.setattr(
        nodes,
        "resolve_attraction_price",
        lambda item, *_args, **_kwargs: {
            **item, "cost": prices[item["name"]],
            "ticket_info": {"live_query": True},
        },
    )
    monkeypatch.setattr(nodes, "enrich_restaurant_prices", lambda candidates, _city: candidates)
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


def test_meal_estimate_is_visible_but_separate_from_known_cost():
    budget = _build_day_budget([
        {"type": "lunch", "name": "实时有价餐厅", "cost": 60,
         "cost_info": {"estimate": False}},
        {"type": "dinner", "name": "缺价餐厅", "cost": 80,
         "cost_info": {"estimate": True}},
    ])
    assert budget["meal_known"] == 60
    assert budget["meal_estimated"] == 80
    assert budget["estimated_subtotal"] == 140
    assert budget["unknown_items"] == []
