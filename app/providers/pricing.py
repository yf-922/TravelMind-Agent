"""景点与餐饮价格解析：门票实时网页查询，餐厅使用本次候选数据估算。"""

from __future__ import annotations

import re
import statistics
from datetime import date
from typing import Any

from app.providers.tickets.live import lookup_live_ticket_price


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    price = float(match.group())
    return price if 0 <= price <= 5000 else None


def resolve_attraction_price(
    item: dict[str, Any], visit_date: date | str | None, city: str = "",
    live_ticket: dict[str, Any] | None = None, *, live_lookup_done: bool = False,
) -> dict[str, Any]:
    """使用本次在线查询结果；失败后仅接受本次高德响应，不读历史价格。"""
    del city  # 城市保留在接口中，便于后续接入城市级官方票务平台。
    ticket = live_ticket
    if not live_lookup_done:
        ticket = lookup_live_ticket_price(str(item.get("name") or ""), visit_date)
    if ticket:
        return {**item, "cost": ticket["price"], "ticket_info": ticket}

    live_price = _number(item.get("cost"))
    if live_price is not None:
        return {
            **item, "cost": live_price,
            "ticket_info": {
                "price": live_price, "price_label": f"本次高德参考价 ¥{live_price:g}",
                "source_name": "高德地点实时查询", "source_url": None,
                "verified_at": date.today().isoformat(), "live_query": True,
                "pricing_basis": "amap_live", "note": "第三方地点数据，购票前请到景区官方渠道复核。",
            },
        }
    return item


def _category_baseline(category: str, city: str) -> float:
    text = str(category or "")
    baseline = 70
    rules = [
        (("小吃", "快餐", "面", "粉", "饺子", "包子"), 35),
        (("咖啡", "茶馆", "甜品", "饮品"), 45),
        (("火锅", "烤肉", "烧烤"), 100),
        (("日本", "韩国", "东南亚"), 110),
        (("西餐", "牛排"), 150),
        (("海鲜", "鱼翅", "燕鲍翅"), 180),
    ]
    for keywords, value in rules:
        if any(keyword in text for keyword in keywords):
            baseline = value
            break
    factor = 1.15 if any(x in str(city) for x in ("北京", "上海", "深圳", "广州", "杭州")) else 1.0
    return round(baseline * factor / 5) * 5


def enrich_restaurant_prices(candidates: list[dict[str, Any]], city: str = "") -> list[dict[str, Any]]:
    """只使用本次高德候选：有价用实时值，无价用本批候选中位数估算。"""
    enriched = [dict(candidate) for candidate in candidates]
    live_prices = [
        price for candidate in enriched
        if (price := _number(candidate.get("cost"))) is not None
    ]
    observed_date = date.today().isoformat()
    for candidate in enriched:
        price = _number(candidate.get("cost"))
        if price is not None:
            candidate["cost"] = price
            candidate["cost_info"] = {
                "estimate": False, "source_name": "本次高德餐饮查询",
                "observed_at": observed_date, "price_label": f"高德人均 ¥{price:g}",
            }
            continue

        estimate = statistics.median(live_prices) if len(live_prices) >= 2 else _category_baseline(
            str(candidate.get("category") or candidate.get("keytag") or ""), city,
        )
        estimate = round(float(estimate) / 5) * 5
        low = max(10, round(estimate * 0.75 / 5) * 5)
        high = max(low, round(estimate * 1.3 / 5) * 5)
        method = "本次同区域候选餐厅人均价中位数" if len(live_prices) >= 2 else "本次查询缺少价格样本，采用城市与品类基线"
        candidate["cost"] = estimate
        candidate["cost_info"] = {
            "estimate": True, "source_name": method, "observed_at": observed_date,
            "low": low, "high": high,
            "price_label": f"估算人均 ¥{estimate:g}（约 ¥{low:g}–{high:g}）",
        }
    return enriched
