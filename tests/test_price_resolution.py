from app.providers import pricing
from app.providers.tickets import live


def test_missing_restaurant_cost_uses_nearby_median(monkeypatch):
    candidates = [
        {"name": "餐厅A", "cost": "40", "category": "中餐"},
        {"name": "餐厅B", "cost": "80", "category": "中餐"},
        {"name": "餐厅C", "cost": None, "category": "中餐"},
    ]

    result = pricing.enrich_restaurant_prices(candidates, "北京")

    assert result[2]["cost"] == 60
    assert result[2]["cost_info"]["estimate"] is True
    assert result[2]["cost_info"]["source_name"] == "本次同区域候选餐厅人均价中位数"
    assert result[2]["cost_info"]["low"] < 60 < result[2]["cost_info"]["high"]


def test_live_attraction_price_uses_current_amap_response(monkeypatch):
    monkeypatch.setattr(pricing, "lookup_live_ticket_price", lambda *_args: None)

    result = pricing.resolve_attraction_price(
        {"type": "attraction", "name": "测试景点", "cost": "35元"},
        "2026-07-27", "测试市",
    )

    assert result["cost"] == 35
    assert result["ticket_info"]["pricing_basis"] == "amap_live"
    assert result["ticket_info"]["source_name"] == "高德地点实时查询"


def test_ticket_lookup_fetches_official_page_every_time(monkeypatch):
    calls = []
    page = "旺季：4月1日-10月31日 票务服务 门票 30元/张 淡季：11月1日-3月31日 门票 20元/张"
    monkeypatch.setattr(live, "http_get_text", lambda url, **_kwargs: calls.append(url) or page)

    first = live.lookup_live_ticket_price("颐和园", "2026-07-27")
    second = live.lookup_live_ticket_price("颐和园", "2026-07-27")

    assert first["price"] == second["price"] == 30
    assert first["live_query"] is True
    assert len(calls) == 2
