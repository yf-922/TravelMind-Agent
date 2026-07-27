"""基于本次高德 POI 查询的连锁酒店推荐；不保存或伪造房价。"""

from __future__ import annotations

import re
from typing import Any

from app.providers.amap.poi import poi_to_spot, search_city_pois


CHAIN_KEYWORDS = ("全季", "汉庭", "如家", "7天", "锦江之星", "亚朵")
HOTEL_TYPES = "住宿服务"


def _price(value: Any) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


def recommend_chain_hotel(city: str, api_key: str) -> dict[str, Any] | None:
    """查询本次高德结果并按可见价格优先、连锁品牌次之排序。"""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for keyword in ("全季酒店", "汉庭酒店", "如家酒店", "7天酒店", "亚朵酒店"):
        try:
            raw_pois = search_city_pois(city, api_key, keywords=keyword, types=HOTEL_TYPES, offset=8)
        except RuntimeError:
            continue
        for raw in raw_pois:
            item = poi_to_spot(raw)
            if not item or not item.get("name") or item["name"] in seen:
                continue
            if not any(chain in str(item["name"]) for chain in CHAIN_KEYWORDS):
                continue
            seen.add(item["name"])
            item["nightly_price"] = _price(item.get("cost"))
            candidates.append(item)

    if not candidates:
        return None
    # 有本次高德价格的候选始终排在前面；同价时保持更常见的全季/汉庭优先。
    brand_rank = {"全季": 0, "汉庭": 1, "如家": 2, "7天": 3, "锦江之星": 4, "亚朵": 5}
    def rank(item: dict[str, Any]) -> tuple[float, int, str]:
        price = item.get("nightly_price")
        brand = min((value for key, value in brand_rank.items() if key in str(item.get("name"))), default=99)
        return (float(price) if price is not None else float("inf"), brand, str(item.get("name")))
    return min(candidates, key=rank)
