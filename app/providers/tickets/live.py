"""规划时实时查询门票网页，不读取历史票价或本地知识库。"""

from __future__ import annotations

import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

from app.core.http import http_get_text
from app.providers.tickets.sources import OFFICIAL_TICKET_SOURCES


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _page_text(raw_html: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw_html)
    return re.sub(r"\s+", " ", " ".join(parser.parts))


def _visit_date(value: date | str | None) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _result(price: float, label: str, source_name: str, source_url: str, note: str) -> dict[str, Any]:
    queried_at = datetime.now(timezone.utc).isoformat()
    return {
        "price": price, "price_label": label, "ticket_type": "成人基础票",
        "pricing_basis": "live_web_query", "source_name": source_name,
        "source_url": source_url, "verified_at": queried_at[:10],
        "queried_at": queried_at, "live_query": True, "note": note,
    }


def _parse_official_page(
    text: str, parser_name: str, visit_date: date | None,
    source_name: str, source_url: str,
) -> dict[str, Any] | None:
    peak = visit_date is None or 4 <= visit_date.month <= 10
    if parser_name == "summer_palace":
        match = re.search(rf"{'旺季' if peak else '淡季'}.{{0,100}}?门票\s*(\d+)元/张", text)
        if match:
            price = float(match.group(1))
            return _result(price, f"官网实时查询·{'旺季' if peak else '淡季'}成人大门票 ¥{price:g}", source_name, source_url, "联票与园中园另计，以本次官网页面为准。")
    elif parser_name == "summer_palace_museum":
        match = re.search(r"颐和园博物馆\s*(\d+)元/张", text)
        if match:
            price = float(match.group(1))
            return _result(price, f"官网实时查询·园中园票 ¥{price:g}", source_name, source_url, "颐和园联票已包含该项目，购买联票时不要重复计算。")
    elif parser_name == "palace_museum":
        season = "旺季" if peak else "淡季"
        pattern = r"4月1日至10月31日为旺季，大门票(\d+)元/人" if peak else r"11月1日至次年3月31日为淡季，大门票(\d+)元/人"
        match = re.search(pattern, text)
        if match:
            price = float(match.group(1))
            return _result(price, f"官网实时查询·{season}成人大门票 ¥{price:g}", source_name, source_url, "珍宝馆、钟表馆等附加项目另计，请提前预约。")
    elif parser_name == "jingshan":
        match = re.search(r"门票价格[：:]?\s*门票价格[：:]\s*(\d+)元", text)
        if match:
            price = float(match.group(1))
            return _result(price, f"实时查询·成人日常门票 ¥{price:g}", source_name, source_url, "牡丹花节等活动期间页面显示可能执行活动票价，请出行前复核。")
    elif parser_name == "tiantan":
        season = "旺季" if peak else "淡季"
        # 官网页面、公告或其嵌入票务文案均可命中；只接受明确的成人大门票字段。
        patterns = (
            rf"{season}.{{0,100}}?(?:大门票|门票).{{0,24}}?(\d+(?:\.\d+)?)\s*元",
            rf"(?:大门票|门票).{{0,80}}?{season}.{{0,48}}?(\d+(?:\.\d+)?)\s*元",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                price = float(match.group(1))
                return _result(price, f"官网实时查询·{season}成人大门票 ¥{price:g}", source_name, source_url, "联票及祈年殿、回音壁等园内项目可能另计，以本次官网页面为准。")
    elif parser_name == "gongwangfu":
        match = re.search(r"(?:成人(?:票)?|全价票|门票).{0,80}?(\d+(?:\.\d+)?)\s*元", text)
        if match:
            price = float(match.group(1))
            return _result(price, f"官网实时查询·成人门票 ¥{price:g}", source_name, source_url, "讲解、特展等服务可能另计，请以预约购票页为准。")
    elif parser_name == "free_page" and re.search(r"门票价格.{0,12}免费|免费开放|无需门票", text):
        return _result(0, "实时查询·免费开放", source_name, source_url, "收费场馆、游船、活动和其他消费另计。")
    return None


def _parse_search_snippet(name: str, raw_xml: str) -> dict[str, Any] | None:
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError:
        return None
    candidates: list[tuple[int, str, str]] = []
    trusted = ("gov.cn", "visitbeijing.com.cn", "dpm.org.cn", "summerpalace.net.cn")
    for item in root.findall(".//item"):
        url = str(item.findtext("link") or "")
        text = _page_text(html.unescape(str(item.findtext("title") or "") + " " + str(item.findtext("description") or "")))
        if name not in text or "门票" not in text:
            continue
        domain = urlparse(url).netloc.lower()
        priority = 0 if any(domain.endswith(suffix) for suffix in trusted) else 1
        candidates.append((priority, url, text))
    for _, url, text in sorted(candidates):
        context = re.search(r"(?:门票(?:价格)?|票价).{0,80}?(免费|(?:仅需)?\s*(\d+(?:\.\d+)?)\s*元)", text)
        if not context:
            continue
        domain = urlparse(url).netloc or "网页来源"
        if "免费" in context.group(1):
            return _result(0, "实时网页查询·免费", f"实时搜索：{domain}", url, "这是本次在线搜索结果，进入收费场馆或参加活动可能另收费。")
        price = float(context.group(2))
        return _result(price, f"实时网页查询·成人参考票 ¥{price:g}", f"实时搜索：{domain}", url, "非官网统一接口数据，购票前请点击来源复核。")
    return None


def lookup_live_ticket_price(name: str, visit_date: date | str | None = None) -> dict[str, Any] | None:
    """每次调用都访问在线来源；不读取或写入任何票价缓存。"""
    normalized = "".join(str(name or "").split())
    source = OFFICIAL_TICKET_SOURCES.get(normalized)
    if source:
        source_name, source_url, parser_name = source
        try:
            page = http_get_text(source_url, timeout=6, retries=1)
            parsed = _parse_official_page(_page_text(page), parser_name, _visit_date(visit_date), source_name, source_url)
            if parsed:
                return parsed
        except RuntimeError:
            pass

    query = urllib.parse.urlencode({"format": "rss", "q": f"{name} 门票 价格"})
    try:
        rss = http_get_text(f"https://www.bing.com/search?{query}", timeout=6, retries=1)
    except RuntimeError:
        return None
    return _parse_search_snippet(normalized, rss)


def lookup_live_ticket_prices(
    requests: list[tuple[str, date | str | None]], max_workers: int = 6,
) -> dict[tuple[str, str], dict[str, Any] | None]:
    """并发查询一份行程的门票，避免逐景点串行拖慢规划。"""
    from concurrent.futures import ThreadPoolExecutor

    unique = {
        (name, value.isoformat() if isinstance(value, date) else str(value or "")): value
        for name, value in requests if name
    }

    def query(entry: tuple[tuple[str, str], date | str | None]):
        key, original_date = entry
        return key, lookup_live_ticket_price(key[0], original_date)

    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique))) as executor:
        return dict(executor.map(query, unique.items()))
