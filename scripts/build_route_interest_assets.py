"""Reproducible interest-selection assets; all labels remain review drafts.

Official quotes are accepted only if present in a saved successful response.
Unavailable sources are reported, never filled with model-generated facts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.rag_protocol import fingerprint

OUT = ROOT / "knowledge/route_interest_v1"
# city, official entity aliases, URL, verbatim stable fact quote, interest
SOURCES = [
    ("上海", ["上海迪士尼度假区"], "https://www.shanghaidisneyresort.com/zh-cn/",
     "游乐项目 娱乐演出 迪士尼朋友", "主题乐园游乐与角色娱乐"),
    ("三亚", ["三亚南山文化旅游区"], "https://www.nanshan.com/nanshan/byjd.html",
     "南山寺 海上观音 不二法门 三十三观音堂", "佛教建筑与观音文化"),
    ("南京", ["玄武湖景区", "玄武湖"], "https://www.xuanwuhu.net/mhls/mhls1.aspx",
     "六朝时期为皇家园林湖泊，明代为保存黄册的国家档案馆", "皇家园林与城市历史"),
    ("南京", ["总统府", "南京总统府"], "https://www.njztf.cn/",
     "孙中山与南京临时政府", "近代史与临时政府"),
    ("上海", ["上海自然博物馆"], "https://www.snhm.org.cn/cszl/zggk.htm",
     "上海自然博物馆的展示以“自然•人•和谐”为主题，以“演化”为主线", "生命演化与自然科学"),
    ("上海", ["上海动物园"], "https://www.shanghaizoo.cn/",
     "大熊猫 华南虎 大猩猩 金丝猴 火烈鸟 蓝黄金刚鹦鹉 小熊猫 斑嘴环企鹅 亚洲象 长颈鹿", "活体动物观察"),
    ("上海", ["上海海洋水族馆"], "https://www.sh-aquarium.com/",
     "主题展区 中国区 南美洲区 澳大利亚区 非洲区 东南亚区 冷水区 极地区 海岸区 深海区", "水生生物与海洋科普"),
    ("三亚", ["天涯海角游览区"], "https://www.aitianya.cn/",
     "这就是著名的“天涯”石，是景区的标志和象征", "海滨石刻与文化"),
    ("三亚", ["南山大小洞天旅游区", "大小洞天"], "https://www.sanyapark.com/",
     "寻踪自然博物馆 漫游旅拍基地 摩崖石刻", "摩崖石刻与自然博物馆"),
]
POOLS = {
    "南京": ["南京博物院", "玄武湖景区", "总统府", "中国科举博物馆(江南贡院)", "中山陵景区", "南京欢乐谷"],
    "上海": ["上海博物馆东馆", "上海自然博物馆", "上海动物园", "上海海洋水族馆", "上海豫园", "上海迪士尼度假区"],
    "三亚": ["蜈支洲岛旅游风景区", "天涯海角游览区", "南山大小洞天旅游区", "三亚西岛海洋文化旅游区", "鹿回头风景区", "三亚南山文化旅游区"],
}
# Each group is confined to one split; no synonymous query crosses the split.
QUERIES = {
    "南京": [
        ("dev", "古代文明", "想了解江苏古代文明和水乡文化", ["南京博物院"], []),
        ("dev", "近代史", "想看孙中山与南京临时政府的史料", ["总统府"], []),
        ("dev", "近代史", "想了解孙中山时期的政府，不去欢乐谷", ["总统府"], ["南京欢乐谷"]),
        ("dev", "排除娱乐", "想了解江苏古代文明，不要去欢乐谷", ["南京博物院"], ["南京欢乐谷"]),
        ("dev", "多兴趣", "想了解古代文明，也想看近代政府历史，尽量兼顾", ["南京博物院", "总统府"], []),
        ("test", "制度史", "对黄册国家档案馆的历史感兴趣", ["玄武湖景区"], []),
        ("test", "制度史", "以前喜欢游乐园，这次只想看皇家园林湖泊，不要欢乐谷", ["玄武湖景区"], ["南京欢乐谷"]),
        ("test", "制度史", "想了解保存黄册的湖泊历史，不要套用杭州西湖的资料", ["玄武湖景区"], []),
        ("test", "证据不足", "想看明天新增的沉浸式VR特展，没证据就说不知道", [], []),
        ("test", "模糊兴趣", "想看那个馆的神秘展厅，没有明确证据不要猜", [], []),
    ],
    "上海": [
        ("dev", "生命演化", "想了解生命演化的科学展示", ["上海自然博物馆"], []),
        ("dev", "动物观察", "想观察大熊猫与长颈鹿等活体动物", ["上海动物园"], []),
        ("dev", "动物观察", "想看华南虎和大熊猫，不要把标本当成活体动物", ["上海动物园"], []),
        ("dev", "排除乐园", "想学习自然演化，不去迪士尼", ["上海自然博物馆"], ["上海迪士尼度假区"]),
        ("dev", "生物组合", "想观察活体动物，也想了解生命演化，尽量兼顾", ["上海动物园", "上海自然博物馆"], []),
        ("test", "数字体验", "想看古代文明探索宫和数字馆，不要混淆馆区", ["上海博物馆东馆"], []),
        ("test", "海洋科普", "想看深海展区与水生生物，不去迪士尼", ["上海海洋水族馆"], ["上海迪士尼度假区"]),
        ("test", "馆区区分", "想看上博东馆专项参观项目，不是人民广场馆", ["上海博物馆东馆"], []),
        ("test", "未知新展", "想看明天首次开幕的机器人新展，缺少资料别承诺", [], []),
        ("test", "跨城干扰", "想看江苏古代文明专门展，只有上海候选时不要把南京资料套过来", [], []),
    ],
    "三亚": [
        ("dev", "海上运动", "想体验潜水与摩托艇，具体当天运营再核验", ["蜈支洲岛旅游风景区"], []),
        ("dev", "海上运动", "想体验风洞项目，不要把项目包含在门票里当作事实", ["蜈支洲岛旅游风景区"], []),
        ("dev", "海上运动", "喜欢潜水和环岛观光，不要把西岛与蜈支洲岛混淆", ["蜈支洲岛旅游风景区"], ["三亚西岛海洋文化旅游区"]),
        ("dev", "海上运动", "想了解蜈支洲岛海陆运动，不要南山宗教主题", ["蜈支洲岛旅游风景区"], ["三亚南山文化旅游区"]),
        ("dev", "海上运动", "想体验摩托艇并兼顾环岛观光，当天运营另外确认", ["蜈支洲岛旅游风景区"], []),
        ("test", "文化组合", "想比较天涯标志石与洞天摩崖石刻，两者尽量兼顾", ["天涯海角游览区", "南山大小洞天旅游区"], []),
        ("test", "摄影资源", "想看洞天旅拍基地与摩崖石刻，不要海上运动", ["南山大小洞天旅游区"], []),
        ("test", "标志石刻", "对天涯标志石有兴趣，不去蜈支洲岛", ["天涯海角游览区"], ["蜈支洲岛旅游风景区"]),
        ("test", "未知活动", "想参加明天新增的夜间珊瑚课程，没有证据不要说有", [], []),
        ("test", "未知名额", "只考虑明天肯定还有潜水名额的景点，知识快照不算证明", [], []),
    ],
}


def collect_sources(directory: Path):
    import httpx
    from bs4 import BeautifulSoup
    records, facts = [], []
    for city, entities, url, quote, topic in SOURCES:
        source = "interest_" + hashlib.sha256(url.encode()).hexdigest()[:16]
        record = {"url": url, "entities": entities, "city": city, "source": source}
        try:
            response = httpx.get(url, timeout=20, follow_redirects=True)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for tag in soup(["script", "style"]): tag.decompose()
            text = " ".join(soup.get_text(" ", strip=True).split())
            normalized_quote = " ".join(quote.split())
            if normalized_quote not in text:
                raise ValueError("verbatim fact not found; needs source review")
            digest = hashlib.sha256(text.encode()).hexdigest()
            (directory / "snapshots").mkdir(parents=True, exist_ok=True)
            (directory / "snapshots" / (source + ".txt")).write_text(text, encoding="utf-8")
            record.update(status="verified_quote", snapshot_hash=digest,
                          collected_at=datetime.now(timezone.utc).isoformat())
            facts.append({**record, "topic": topic, "knowledge_type": "stable_planning_fact",
                          "evidence_id": source, "validity": "official_snapshot_not_live_guarantee",
                          "text": normalized_quote, "content_hash": hashlib.sha256(normalized_quote.encode()).hexdigest(),
                          "review_status": "pending"})
        except Exception as exc:
            record.update(status="unavailable", reason=type(exc).__name__ + ": " + str(exc)[:180])
        records.append(record)
    return facts, records


def make_cases(pools, facts):
    cases = []
    for city, queries in QUERIES.items():
        for index, (split, group, interest, acceptable, forbidden) in enumerate(queries):
            sources = [f["source"] for f in facts if f.get("city") == city and set(f.get("entities", [])) & set(acceptable)]
            cases.append({"id": f"interest-{city}-{index+1:02}", "split": split,
                          "intent_group": f"{city}-{group}", "query": f"{city}一日游，每天最多两个不同景点。{interest}。",
                          "destination": city, "days": 1, "max_per_day": 2,
                          "travel_start_date": "2026-10-15", "travel_end_date": "2026-10-15",
                          "pois": pools[city], "acceptable_pois": acceptable, "forbidden_pois": forbidden,
                          "required_evidence": sources, "answerable": bool(acceptable),
                          "annotation_status": "draft", "annotation_version": "route-interest-v1",
                          "annotator": None, "reviewed_at": None, "change_log": [],
                          "reference_note": "Acceptable sets are draft labels, not unique optimal routes; missing knowledge is not proof of absence."})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prior_cases = json.loads((args.out/'cases.json').read_text(encoding='utf-8')) if (args.out/'cases.json').exists() else []
    if any(c.get('annotation_status') == 'human_reviewed' for c in prior_cases):
        parser.error('reviewed version must not be regenerated; create a new --out version')
    seeds = json.loads((ROOT / "knowledge/travel/planning_facts.json").read_text(encoding="utf-8"))
    if args.collect:
        extra, records = collect_sources(args.out)
        if (args.out/'facts.json').exists():
            saved = json.loads((args.out/'facts.json').read_text(encoding='utf-8'))
            captured = {f['source'] for f in extra}
            extra += [f for f in saved if f['source'].startswith('interest_') and f['source'] not in captured]
    elif (args.out / 'facts.json').exists():
        extra = [f for f in json.loads((args.out / 'facts.json').read_text(encoding='utf-8')) if f['source'].startswith('interest_')]
        records = json.loads((args.out / 'manifest.json').read_text(encoding='utf-8'))['source_collection']
    else:
        extra, records = [], []
    seeds = [s for s in seeds if not s["source"].startswith("interest_")]
    facts = seeds + extra
    inventory = {}
    for path in sorted((ROOT / "tests/eval/fixtures").glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        for poi in fixture.get("pois", []):
            inventory[(fixture.get("destination"), poi["name"])] = {**poi, "provenance": {"fixture": path.name, "kind": "saved_real_amap_response"}}
    pools = {city: [inventory.get((city, name), {"name": name, "provenance": {"kind": "official_entity_without_amap_coordinates"}})
                    for name in names] for city, names in POOLS.items()}
    for c in prior_cases:
        for poi in c['pois']:
            if poi.get('location'):
                for pool_poi in pools[c['destination']]:
                    if pool_poi['name'] == poi['name']: pool_poi.update(poi)
    cases = make_cases(pools, facts)
    for c in cases:
        c['route_distance_mode'] = 'drive'
        c['weather_forecast'] = [{'date': '2026-10-15', 'dayweather': '晴', 'nightweather': '晴', 'is_bad': False}]
        c['weather_note'] = '固定晴天评测场景，非真实天气预报；两组共享'
    manifest = {"version": "route-interest-v1", "source_collection": records,
                "candidates_per_city": {city: len(pool) for city, pool in pools.items()},
                "evidenced_candidates_per_city": {city: sum(any(set(f.get('entities', [])) & {p['name']} for f in facts if f.get('city') == city) for p in pool) for city, pool in pools.items()},
                "fact_fingerprint": fingerprint(facts), "case_fingerprint": fingerprint(cases),
                "label_status": "draft_not_human_confirmed", "splits": dict(Counter(c['split'] for c in cases)),
                "production_default_changed": False}
    if (args.out/'maps.json').exists():
        manifest['map_fingerprint'] = fingerprint(json.loads((args.out/'maps.json').read_text(encoding='utf-8')))
    for filename, value in [("facts.json", facts), ("cases.json", cases), ("manifest.json", manifest)]:
        (args.out / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__": main()
