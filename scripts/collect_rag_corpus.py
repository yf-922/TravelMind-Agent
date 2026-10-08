"""Bounded official-source snapshot collection, never model-written facts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "knowledge" / "benchmark_v1"
TERMS = ("参观", "预约", "交通", "游览", "景区", "博物", "古城", "旅游提示", "出游", "旅游攻略", "服务指南", "旅游景点")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paragraphs(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select("script,style,nav,header,footer,noscript"):
        tag.decompose()
    root = soup.select_one("article, .TRS_Editor, .trs_editor, #zoom, #article, .article-content, .content-text") or soup.body or soup
    result = []
    for tag in root.find_all(["p", "li"]):
        text = " ".join(tag.stripped_strings)
        if len(text) >= 50 and not tag.find(["p", "li"]) and not any(x in text for x in ("网站标识码", "ICP备", "主办单位")):
            result.append(text)
    return list(dict.fromkeys(result))


def split_paragraph(text, size=420):
    # Keep sentence boundaries; an unusually long sentence remains intact.
    sentences = re.findall(r"[^。！？!?]+[。！？!?]?", text)
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > size:
            yield current
            current = ""
        current += sentence
    if current:
        yield current


def collect(max_pages):
    date = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    seeds = json.loads((OUT / "seeds.json").read_text(encoding="utf-8"))
    pages, chunks, attempts, seen_text = [], [], [], set()
    for seed in seeds:
        queue = [(seed["url"], "landing", 0)]
        seen_urls = set()
        used = 0
        while queue and used < max_pages:
            url, link_title, depth = queue.pop(0)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            used += 1
            print(f"fetch {seed['city']} {used}/{max_pages} {url}", flush=True)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "TravelMind research snapshot/1.0"})
                with urllib.request.urlopen(request, timeout=6) as response:
                    if "text/html" not in response.headers.get("Content-Type", ""):
                        attempts.append({"url": url, "status": "unsupported_content_type"})
                        continue
                    raw = response.read(3_000_000)
                    final_url = response.url
                soup = BeautifulSoup(raw, "html.parser")
                html = str(soup)
                title = soup.title.get_text(" ", strip=True) if soup.title else link_title
                attempts.append({"url": url, "status": "ok", "bytes": len(raw)})
            except Exception as exc:
                attempts.append({"url": url, "status": "failed", "error": type(exc).__name__})
                continue
            if depth < 2:
                for a in soup.find_all("a", href=True):
                    label = a.get_text(" ", strip=True)
                    base = soup.find("base", href=True)
                    child = urllib.parse.urljoin(urllib.parse.urljoin(final_url, base["href"]) if base else final_url, a["href"]).split("#")[0]
                    if not re.search(r"\.(?:docx?|xlsx?|pdf|zip)(?:\?|$)", child, re.I) and any(term in label for term in TERMS) and urllib.parse.urlparse(child).netloc == urllib.parse.urlparse(final_url).netloc and child.startswith("https://"):
                        queue.append((child, label, depth + 1))
            texts = paragraphs(html)
            if not texts:
                continue
            page_id = digest(final_url)[:16]
            snapshot = OUT / "snapshots" / (page_id + ".txt")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text("\n\n".join(texts), encoding="utf-8")
            page = {**seed, "url": final_url, "title": title, "collected_at": date,
                    "content_hash": digest("\n\n".join(texts)), "snapshot": str(snapshot.relative_to(ROOT))}
            pages.append(page)
            for paragraph in texts:
                # Only retain passages containing actionable visitor content,
                # not government procurement/news navigation boilerplate.
                if not any(t in paragraph for t in ("预约", "参观", "游客", "游览", "接驳", "入口", "门票", "公交", "乘车", "博物馆", "博物院", "雨天", "亲子")):
                    continue
                if any(t in paragraph for t in ("招租", "招标", "天然气", "代表团", "调研", "采购", "成交", "挂牌", "预算单位", "领导班子")):
                    continue
                group = digest(paragraph)
                for part in split_paragraph(paragraph):
                    h = digest(part)
                    if h in seen_text:
                        continue
                    seen_text.add(h)
                    chunks.append({"chunk_id": "official-" + digest(final_url + "\n" + part)[:24],
                                   "text": part, "source": page_id, "url": final_url,
                                   "city": seed["city"], "topic": link_title if link_title != "landing" else title,
                                   "collected_at": date, "content_hash": h, "source_type": seed["source_type"],
                                   "evidence_group": group[:24], "indoor": "unknown", "review_status": "pending",
                                   "fact_validity": "snapshot_not_live"})
    chunks.sort(key=lambda x: x["chunk_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"version": "benchmark-v1", "collected_at": date, "pages": pages, "attempts": attempts,
                "chunk_count": len(chunks), "corpus_fingerprint": digest(json.dumps(chunks, sort_keys=True, ensure_ascii=False)),
                "boundary": "Extracted official snapshots, pending source/content review. Not live ticket/weather/inventory facts."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"chunks": len(chunks), "pages": len(pages), "cities": sorted({x['city'] for x in chunks})}, ensure_ascii=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-pages-per-city", type=int, default=10)
    p.add_argument("--allow-external-calls", action="store_true")
    a = p.parse_args()
    if not a.allow_external_calls:
        p.error("pass --allow-external-calls for bounded public website retrieval")
    if not 1 <= a.max_pages_per_city <= 30:
        p.error("max pages per city must be 1..30")
    collect(a.max_pages_per_city)
