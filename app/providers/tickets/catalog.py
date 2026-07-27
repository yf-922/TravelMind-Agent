"""少量高可信景点票价目录。

票价只来自景区官网或政府旅游信息页，并保留来源与核验日期。
这里不使用模型猜测票价，也不在规划主链路中抓取任意搜索结果。
"""

from __future__ import annotations

from datetime import date
from typing import Any


VERIFIED_AT = "2026-07-27"


def _normalized(name: str) -> str:
    return "".join(str(name or "").split()).strip()


def lookup_ticket_price(name: str, visit_date: date | str | None = None) -> dict[str, Any] | None:
    """按景点规范名返回成人基础票价；没有可信资料时返回 None。"""
    normalized = _normalized(name)
    if isinstance(visit_date, str):
        try:
            visit_date = date.fromisoformat(visit_date)
        except ValueError:
            visit_date = None

    if normalized == "颐和园":
        peak_season = visit_date is None or 4 <= visit_date.month <= 10
        price = 30 if peak_season else 20
        season = "旺季（4月1日—10月31日）" if peak_season else "淡季（11月1日—3月31日）"
        return {
            "price": price,
            "price_label": f"{season}成人大门票 ¥{price}",
            "ticket_type": "成人大门票",
            "pricing_basis": "official_seasonal",
            "source_name": "颐和园官网·票务服务",
            "source_url": "https://www.summerpalace.net.cn/visit.html",
            "verified_at": VERIFIED_AT,
            "note": "联票旺季60元、淡季50元；本预算按各景点单买基础票计算。",
        }

    if normalized in {"颐和园博物馆", "颐和园文昌院", "文昌院"}:
        return {
            "price": 20,
            "price_label": "成人园中园票 ¥20",
            "ticket_type": "成人园中园票",
            "pricing_basis": "official_fixed",
            "source_name": "颐和园官网·票务服务",
            "source_url": "https://www.summerpalace.net.cn/visit.html",
            "verified_at": VERIFIED_AT,
            "note": "颐和园联票已包含该项目；若购买联票，请勿与本项重复计算。",
        }

    if normalized in {"紫竹院公园", "紫竹院"}:
        return {
            "price": 0,
            "price_label": "免费开放",
            "ticket_type": "免费入园",
            "pricing_basis": "government_tourism_page",
            "source_name": "北京旅游网·紫竹院公园",
            "source_url": "https://s.visitbeijing.com.cn/attraction/101455",
            "verified_at": VERIFIED_AT,
            "note": "公园门票免费；园内临时活动或其他消费不包含在内。",
        }

    return None


def enrich_attraction_ticket(item: dict[str, Any], visit_date: date | str | None = None) -> dict[str, Any]:
    """用可追溯目录补全门票，同时保留来源供前端展示。"""
    ticket = lookup_ticket_price(str(item.get("name") or ""), visit_date)
    if not ticket:
        return item
    enriched = dict(item)
    enriched["cost"] = ticket["price"]
    enriched["ticket_info"] = ticket
    return enriched
