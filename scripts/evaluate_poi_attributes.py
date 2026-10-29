"""Exploratory real-model, paired final-route attribute comparison (dry-run default)."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.env import load_local_env
from app.evaluation.route_interest import route_key, run_final_route
from app.planning import nodes
from app.planning.poi_attributes import CATALOG, enrich_pois, load_catalog
from app.planning.schemas import SpotTipsResult, TravelPlanState

ASSETS = ROOT / "knowledge/poi_attributes_v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True).encode()).hexdigest()


def ratio(n, d):
    return {"numerator": n, "denominator": d, "percent": 100*n/d if d else None}


def build_cases():
    source = json.loads((ROOT / "knowledge/route_interest_v1/cases.json").read_text(encoding="utf-8"))
    profiles = [
        ("nanjing-rain", "南京", "下雨，只去室内景点，每天最多1个景点，至少游览90分钟。", ["南京博物院"], [], True, False),
        ("nanjing-imperial-exam", "南京", "想看科举制度和古代教育史，不想去现代游乐园。每天最多1个景点，至少游览90分钟。", ["中国科举博物馆(江南贡院)"], ["南京欢乐谷"], False, False),
        ("nanjing-lake", "南京", "晴天想看湖泊园林，不想看室内展览，每天最多1个景点，至少游览90分钟。", ["玄武湖景区"], ["南京博物院"], False, False),
        ("shanghai-rain", "上海", "下雨，只去室内景点，每天最多1个景点，至少游览90分钟。", ["上海自然博物馆", "上海海洋水族馆", "上海博物馆东馆"], [], True, False),
        ("shanghai-evolution", "上海", "想看生命演化和动物标本，不看活体水族或动物园。每天最多1个景点，至少游览90分钟。", ["上海自然博物馆"], ["上海海洋水族馆", "上海动物园"], False, False),
        ("shanghai-garden-rain-conflict", "上海", "下雨，只去室内，必须游览上海豫园，禁止换点、禁止补查，每天最多1个景点。", [], [], True, True),
        ("shanghai-elder", "上海", "带老人，尽量避开步道不平、台阶和石桥较多的园林，优先看自然或海洋科普展览。每天最多1个景点，至少游览90分钟。没有证据不要保证全程无障碍。", ["上海自然博物馆", "上海海洋水族馆"], ["上海豫园"], False, False),
        ("sanya-rain-evidence-shortage", "三亚", "下雨，只去室内景点，至少一个，不允许将海岛、山顶公园或整个混合景区当成纯室内。禁止补查，每天最多1个景点。", [], [], True, True),
        ("sanya-village", "三亚", "喜欢渔村生活和海岛生态，不想看山顶公园和寺庙。每天最多1个景点，至少游览90分钟。", ["三亚西岛海洋文化旅游区"], ["鹿回头风景区", "三亚南山文化旅游区"], False, False),
        ("sanya-coastal-rocks", "三亚", "晴天想看海滨奇石和摩崖石刻，不看室内展陈。每天最多1个景点，至少游览90分钟。", ["天涯海角游览区", "南山大小洞天旅游区"], [], False, False),
    ]
    cases = []
    for ident, city, request, acceptable, forbidden, rain, reject in profiles:
        pool = deepcopy(next(c for c in source if c["destination"] == city)["pois"])
        for p in pool:
            # Current production parser leaves category-derived environment unknown.
            # Apply this same normalization in both arms; keep old hint as diagnostic.
            p["indoor_hint"] = p.get("indoor")
            p.update(indoor=None, indoor_source="unknown")
            p.pop("indoor_scope", None)
            p.pop("planning_attributes", None)
        cases.append({"id": ident, "destination": city, "query": f"{city}2026年10月31日一日游。"+request,
            "attraction_preference": request, "days": 1, "max_per_day": 1,
            "travel_start_date": "2026-10-31", "travel_end_date": "2026-10-31",
            "pois": pool, "rain_indoor_priority": rain, "candidate_pool_status": "insufficient",
            "weather_forecast": [{"date": "2026-10-31", "is_bad": rain, "day_weather": "雨" if rain else "晴", "night_weather": "阴", "day_temp": "24", "night_temp": "20"}],
            "rubric": {"acceptable": acceptable, "excluded": forbidden, "min_count": 1,
                "max_count": 1, "min_visit": 0 if reject else 90, "indoor_only": rain,
                "must_reject": reject, "required": ["上海豫园"] if ident == "shanghai-garden-rain-conflict" else []},
            "annotation_status": "author_draft_not_human_verified"})
    return cases


def validate_cases(cases, maps):
    if len(cases) != 10 or len({c["id"] for c in cases}) != 10:
        raise ValueError("Expected ten distinct exploratory cases")
    for c in cases:
        names = {p["name"] for p in c["pois"]}
        if not set(c["rubric"]["acceptable"] + c["rubric"]["excluded"]) <= names:
            raise ValueError("Rubric refers outside candidate pool")
        for a in c["pois"]:
            for b in c["pois"]:
                if a["name"] != b["name"] and route_key(a["location"], b["location"], "drive") not in maps:
                    raise ValueError("Incomplete saved road replay")


def grade(final, case, error=None):
    """Score delivered timeline against SAME reference in BOTH arms.

    Gold/reference attributes are never inferred from treatment state. This is
    an author-draft diagnostic, not independently human-verified correctness.
    """
    plan = final.final_plan or {}
    route = [{"day": d["day"], "spots": [s for s in d.get("timeline", []) if s.get("type") == "attraction"]}
             for d in plan.get("days", [])]
    spots = [s for d in route for s in d["spots"]]
    names = [s["name"] for s in spots]
    pool = {p["name"]: p for p in case["pois"]}
    reference = {p["name"]: p for p in enrich_pois(case["pois"], case["destination"], use_attributes=True, today=date(2026, 10, 30))}
    rubric, faults = case["rubric"], []
    if not spots:
        faults.append("empty_route")
    if len(set(names)) != len(names):
        faults.append("duplicate")
    if any(n not in pool for n in names):
        faults.append("outside_pool")
    if len(names) > rubric["max_count"]:
        faults.append("pace_count")
    if set(names) & set(rubric["excluded"]):
        faults.append("excluded_preference")
    if rubric["acceptable"] and not set(names) & set(rubric["acceptable"]):
        faults.append("interest_mismatch")
    if not set(rubric["required"]) <= set(names):
        faults.append("required_poi_missing")
    if rubric["indoor_only"] and any(reference.get(n, {}).get("indoor") is not True for n in names):
        faults.append("indoor_unknown_or_mixed_or_outdoor")
    for s in spots:
        try:
            if nodes._clock_minutes(s["end_time"]) - nodes._clock_minutes(s["start_time"]) < rubric["min_visit"]:
                faults.append("visit_too_short")
        except (KeyError, ValueError, TypeError):
            faults.append("invalid_time")
    # Same production rule check with reference POIs prevents changing the scorer
    # simply by adding treatment-side booleans. Additional interest rubric above
    # is independent of production Reviewer decisions.
    scored = final.model_copy(update={"pois": list(reference.values()), "route": route})
    flags = set(nodes._route_risk_flags(scored)) - {"opening_time_unknown", "user_modification", "long_road_leg", "preference_review"}
    faults.extend(sorted(flags))
    approved = plan.get("approved") is True
    reasons = plan.get("route_issues", [])
    rejection = not error and bool(plan) and not approved and any("无法生成满足" in str(v) or "未生成可用路线" in str(v) for v in reasons)
    success = not error and approved and not faults
    return {"selected": names, "approved": approved, "task_success": success,
        "safe_explained_rejection": rejection, "expected_rejection": rubric["must_reject"],
        "expected_outcome_correct": rejection if rubric["must_reject"] else success,
        "violations": sorted(set(faults)), "unsafe_release": approved and bool(faults),
        "rework": max(0, final.review_round-1), "user_messages": reasons}


def summarize(rows):
    arms = {}
    for arm in ("without_attributes", "with_attributes"):
        subset = [r for r in rows if r["arm"] == arm]
        feasible = [r for r in subset if not r["metrics"]["expected_rejection"]]
        negative = [r for r in subset if r["metrics"]["expected_rejection"]]
        measured = [r for r in subset if r["usage_known"]]
        arms[arm] = {"task_success_non_rejection": ratio(sum(r["metrics"]["task_success"] for r in feasible), len(feasible)),
            "expected_outcome": ratio(sum(r["metrics"]["expected_outcome_correct"] for r in subset), len(subset)),
            "safe_rejection_negative": ratio(sum(r["metrics"]["safe_explained_rejection"] for r in negative), len(negative)),
            "unsafe_release": ratio(sum(r["metrics"]["unsafe_release"] for r in subset), len(subset)),
            "execution_failures": ratio(sum(bool(r["error"]) for r in subset), len(subset)),
            "harness_interruptions": ratio(sum(r["error"] == "HarnessInterrupted" for r in subset), len(subset)),
            "mean_rework": sum(r["metrics"]["rework"] for r in subset)/len(subset) if subset else None,
            "mean_elapsed_ms": (sum(r["elapsed_ms"] for r in subset if r["elapsed_ms"] is not None)/sum(r["elapsed_ms"] is not None for r in subset)) if any(r["elapsed_ms"] is not None for r in subset) else None,
            "measured_usage_trials": len(measured),
            "actual_input_tokens_known_only": sum(r["input_tokens"] for r in measured),
            "actual_output_tokens_known_only": sum(r["output_tokens"] for r in measured),
            "model_calls": sum(r["model_calls"] for r in subset)}
    pairs = {}
    for r in rows:
        pairs.setdefault((r["case_id"], r["repetition"]), {})[r["arm"]] = r
    changes = []
    for key, pair in pairs.items():
        if len(pair) == 2:
            a,b = pair["without_attributes"]["metrics"],pair["with_attributes"]["metrics"]
            changes.append({"case_id": key[0], "repetition": key[1], "before": a["selected"], "after": b["selected"],
                "before_correct": a["expected_outcome_correct"], "after_correct": b["expected_outcome_correct"],
                "quality_comparable": not pair["without_attributes"]["error"] and not pair["with_attributes"]["error"],
                "selection_changed": set(a["selected"]) != set(b["selected"])})
    return {"arms": arms, "paired_trials": len(changes),
        "quality_comparable_pairs": sum(c["quality_comparable"] for c in changes),
        "improved_pairs": sum(c["quality_comparable"] and not c["before_correct"] and c["after_correct"] for c in changes),
        "regressed_pairs": sum(c["quality_comparable"] and c["before_correct"] and not c["after_correct"] for c in changes), "changes": changes}


class Meter:
    def __init__(self, out, model, max_calls, budget_db=None):
        self.out, self.model, self.max_calls = out, model, max_calls
        self.budget_db = budget_db
        self.calls, self.current, self.consecutive_failures = [], {}, 0
        from app.llm.factory import build_chat_llm
        # All paid attempts recorded; turn off SDK retries and bound output/time.
        self.chat = build_chat_llm(model=model, temperature=0.3).model_copy(update={
            "max_tokens": 4096, "max_retries": 0, "request_timeout": 60})
        # model_copy does not reconstruct SDK clients created by validators.
        # Clone the actual transports too, otherwise their retries remain 2.
        sync = self.chat.root_client.with_options(max_retries=0, timeout=60)
        async_client = self.chat.root_async_client.with_options(max_retries=0, timeout=60)
        self.chat = self.chat.model_copy(update={"root_client": sync, "root_async_client": async_client,
            "client": sync.chat.completions, "async_client": async_client.chat.completions})
        if self.chat.root_client.max_retries != 0 or self.chat.root_async_client.max_retries != 0:
            raise ValueError("SDK retry guard not applied")

    def append(self, path, row):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
            f.flush()
            os.fsync(f.fileno())

    def factory(self, schema, **kwargs):
        if schema is SpotTipsResult:
            class NoTips:
                def invoke(self, messages):
                    return SpotTipsResult(reasoning="paired optional tips disabled", tips=[])
            return NoTips()
        meter = self
        runnable = self.chat.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)
        class Structured:
            model_name = meter.model
            def invoke(self, messages):
                if len(meter.calls) >= meter.max_calls:
                    raise ValueError("Experiment model-call budget exhausted")
                call_id = reserve_call(meter.budget_db, meter.max_calls) if meter.budget_db else len(meter.calls)+1
                row = {**meter.current, "call_id": call_id, "schema": schema.__name__,
                       "input_tokens": None, "output_tokens": None, "error": None}
                meter.calls.append(row)
                meter.append(meter.out/"calls.jsonl", {**row, "event": "started", "messages": messages})
                start = time.perf_counter()
                try:
                    result = runnable.invoke(messages)
                    raw = result.get("raw")
                    usage = getattr(raw, "usage_metadata", None) or {}
                    row.update(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                               response_id=getattr(raw, "id", None))
                    if result.get("parsing_error") or result.get("parsed") is None:
                        raise ValueError("Structured model response invalid")
                    row["parsed"] = result["parsed"].model_dump(mode="json")
                    meter.consecutive_failures = 0
                    if meter.budget_db: update_provider_failure(meter.budget_db, False)
                    return result["parsed"]
                except Exception as exc:
                    row["error"] = type(exc).__name__
                    meter.consecutive_failures += 1
                    if meter.budget_db: update_provider_failure(meter.budget_db, True)
                    # Avoid retrying paid calls inside production helper. A failed
                    # attempt remains failed; never silently replace its trial.
                    raise ValueError("Metered model attempt failed: "+type(exc).__name__) from None
                finally:
                    row["elapsed_ms"] = (time.perf_counter()-start)*1000
                    meter.append(meter.out/"calls.jsonl", {**row, "event": "completed"})
        Structured.schema = schema
        return Structured()


def reserve_call(database, limit):
    """Cross-process atomic reservation, before any paid attempt."""
    with sqlite3.connect(database, timeout=30) as db:
        db.execute("BEGIN IMMEDIATE")
        used, failures = db.execute("SELECT used, failures FROM budget").fetchone()
        if used >= limit or failures >= 2:
            raise ValueError("Shared call budget exhausted or provider failures")
        db.execute("UPDATE budget SET used=?", (used+1,))
        return used+1


def update_provider_failure(database, failed):
    with sqlite3.connect(database, timeout=30) as db:
        db.execute("BEGIN IMMEDIATE")
        if failed: db.execute("UPDATE budget SET failures=failures+1")
        else: db.execute("UPDATE budget SET failures=0")


def run_trial(case, rep, arm, model, meter, maps):
    meter.current = {"case_id":case["id"],"repetition":rep,"arm":arm}
    started, offset, error = time.perf_counter(), len(meter.calls), None
    final = TravelPlanState(**{k:v for k,v in case.items() if k in TravelPlanState.model_fields})
    diagnostics = {}
    try:
        with patch.dict(os.environ, {"POI_ATTRIBUTES_ENABLED":"1" if arm=="with_attributes" else "0"}):
            final, diagnostics = run_final_route(case, model, lambda *a: [], meter.factory, maps)
        if diagnostics["map_misses"]:
            raise ValueError("Unexpected missing road replay")
    except Exception as exc:
        error = type(exc).__name__
    calls = meter.calls[offset:]
    known = all(type(c["input_tokens"]) is int and type(c["output_tokens"]) is int for c in calls) and bool(calls)
    row = {**meter.current,"metrics":grade(final,case,error),"error":error,"model_calls":len(calls),
        "usage_known":known,"input_tokens":sum(c["input_tokens"] for c in calls) if known else None,
        "output_tokens":sum(c["output_tokens"] for c in calls) if known else None,
        "elapsed_ms":(time.perf_counter()-started)*1000,"diagnostics":diagnostics,
        "final_state":final.model_dump(mode="json")}
    meter.append(meter.out/"trials.jsonl", row)
    return row


def isolated_trial(job):
    case, rep, arm, model, maps, out, database, limit = job
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    meter = Meter(out, model, limit, database)
    return run_trial(case, rep, arm, model, meter, maps), meter.calls


def run_parallel(jobs, workers, meter, rows, database):
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from multiprocessing import get_context
    # Each process exclusively owns its monkeypatches, environment and logs.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        pending, index, stopped = {}, 0, False
        while pending or index < len(jobs):
            with sqlite3.connect(database) as db:
                used, failures = db.execute("SELECT used, failures FROM budget").fetchone()
            stopped = stopped or used >= meter.max_calls or failures >= 2
            stopped = stopped or (meter.out/"STOP").exists()
            while not stopped and len(pending) < workers and index < len(jobs):
                future = executor.submit(isolated_trial, jobs[index])
                pending[future] = jobs[index]
                index += 1
            if not pending: break
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                row, calls = future.result()
                rows.append(row)
                meter.calls.extend(calls)
                meter.append(meter.out/"trials.jsonl", row)
                # Child logs remain original; parent copy enables normal resume.
                for call in calls: meter.append(meter.out/"calls.jsonl", {**call,"event":"completed"})
                print(json.dumps({k:row[k] for k in ("case_id","repetition","arm","error")},ensure_ascii=True),flush=True)
        if stopped and (meter.out/"STOP").exists(): return "stopped_by_operator_at_trial_boundary"
        return "stopped_call_budget_or_provider_failures" if stopped and index < len(jobs) else "completed"


def summarize_existing(out, repetitions=1):
    """No client creation or paid calls. Keep cancelled extra work auditable."""
    cases = json.loads((ASSETS/"cases.json").read_text(encoding="utf-8"))
    trial_paths=[out/"trials.jsonl"] + sorted((out/"workers").glob("*/trials.jsonl"))
    call_paths=[out/"calls.jsonl"] + sorted((out/"workers").glob("*/calls.jsonl"))
    rows, calls={}, {}
    for path in trial_paths:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row=json.loads(line)
                rows[(row["case_id"],row["repetition"],row["arm"])]=row
    for path in call_paths:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                call=json.loads(line)
                old=calls.get(call["call_id"],{})
                if old.get("event") != "completed" or call["event"] == "completed":
                    calls[call["call_id"]]=call
    for call in calls.values():
        key=(call["case_id"],call["repetition"],call["arm"])
        if key not in rows:
            case=next(c for c in cases if c["id"]==key[0])
            final=TravelPlanState(query=case["query"])
            rows[key]={"case_id":key[0],"repetition":key[1],"arm":key[2],"error":"HarnessInterrupted",
                "metrics":grade(final,case,"HarnessInterrupted"),"model_calls":sum((c["case_id"],c["repetition"],c["arm"])==key for c in calls.values()),
                "usage_known":False,"input_tokens":None,"output_tokens":None,"elapsed_ms":None,
                "diagnostics":{"partial_attempts_preserved":True},"final_state":{}}
    selected=[r for r in rows.values() if r["repetition"] < repetitions]
    extra=[r for r in rows.values() if r["repetition"] >= repetitions]
    versions=[json.loads(line) for line in (out/"protocol_versions.jsonl").read_text(encoding="utf-8").splitlines()]
    measured=[c for c in calls.values() if type(c.get("input_tokens")) is int and type(c.get("output_tokens")) is int]
    report={"protocol":"poi-attributes-route-pilot-v1","status":"small_pilot_closed_at_user_request",
        "model":"gpt-6.1-sol","max_calls":240,"attempted_calls":len(calls),"planned_trials":len(cases)*2*repetitions,
        "completed_trials":len(selected),"cost":"unpriced","fingerprints":versions[-1]["fingerprints"],
        "summary":summarize(selected),"results":selected,"cancelled_expansion_results":extra,"transport_audit":versions,
        "metering":{"recorded_model_invocations":len(calls),"finished_invocations":sum(c["event"]=="completed" for c in calls.values()),
            "usage_known_invocations":len(measured),"actual_input_tokens_known_only":sum(c["input_tokens"] for c in measured),
            "actual_output_tokens_known_only":sum(c["output_tokens"] for c in measured),
            "invocations_with_missing_usage":len(calls)-len(measured),"initial_http_retry_count":"unknown_for_first_26_invocations"},
        "ledger_hashes":{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in trial_paths+call_paths if p.exists()}}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"report.md").write_text(render(report),encoding="utf-8")
    print(json.dumps({"summary":report["summary"],"metering":report["metering"],"extra_trials":len(extra)},ensure_ascii=True),flush=True)
    return report


def render(report):
    def fmt(v):
        return f'{v["numerator"]}/{v["denominator"]} ({v["percent"]:.2f}%)' if v["denominator"] else "未运行"
    lines = ["# 景点属性对最终路线的影响：探索性真实模型对照", "",
        "同一候选池、天气、模型、生产联合 Planner→道路核验→审核/返工→时间核查→最终输出，仅切换属性开关。道路使用已保存的真实响应，RAG 关闭，贴士分支两组固定为空，不计入模型质量。",
        "10 条作者预设请求，南京/上海/三亚，包含兴趣、雨天、老人避台阶偏好及两条证据不足/冲突拒绝对照。标签为作者草稿，未人工确认；不是正式冻结测试，不给泛化显著性结论。评分读取最终时间线，不以说明文字变长作为改善。每例仅安排一个景点，不能用本实验证明多景点动线或通勤质量提升。",
        "两组评分使用同一份独立参考属性，不能因实验组多了 indoor=true 就直接改分；硬规则检查部分复用生产代码，不能视为独立人工质量判定。室内标签只适用于有证据的展厅参观，不能保证排队、庭院或通勤避雨。",
        "所有真实调用及失败保留账本，费用 unpriced；前26次调用的底层HTTP重试次数不可统计，随后修复实际SDK客户端关闭重试。Token 仅统计响应实报用量，缺失不会补零。重复运行非独立新请求。切换重试保护及并发时的中断保留，不能算作属性收益。", "",
        "| 指标 | 无属性 | 有属性 |", "|---|---:|---:|"]
    for key,label in (("task_success_non_rejection","正常请求任务成功（含中断）"),("expected_outcome","含拒绝对照的预期结果正确（含中断）"),("safe_rejection_negative","负例明确安全拒绝"),("unsafe_release","不合格路线放行"),("execution_failures","执行失败"),("harness_interruptions","其中实验工具中断")):
        lines.append(f'| {label} | {fmt(report["summary"]["arms"]["without_attributes"][key])} | {fmt(report["summary"]["arms"]["with_attributes"][key])} |')
    lines += ["", f'记录配对 {report["summary"]["paired_trials"]} 组，其中两组均无执行错误、可比较质量的配对 {report["summary"]["quality_comparable_pairs"]} 组；改善 {report["summary"]["improved_pairs"]} 组，退化 {report["summary"]["regressed_pairs"]} 组。', "",
        "| 请求 | 重复 | 组别 | 最终选点 | 预期结果正确 | 返工 | 问题 |", "|---|---:|---|---|---|---:|---|"]
    for r in report["results"]:
        m=r["metrics"]
        lines.append(f'| {r["case_id"]} | {r["repetition"]} | {r["arm"]} | {", ".join(m["selected"])} | {m["expected_outcome_correct"]} | {m["rework"]} | {", ".join(m["violations"])} {r["error"] or ""} |')
    lines += ["", "## 用量、耗时与指纹", "", "```json", json.dumps({k:report[k] for k in ("protocol","status","model","max_calls","attempted_calls","summary","fingerprints")},ensure_ascii=False,indent=2), "```", "",
        "复现：`python scripts/evaluate_poi_attributes.py` 先校验并预览；`python scripts/evaluate_poi_attributes.py --run --repetitions 2 --max-calls 240 --out data/poi_attributes_v1/new_run` 运行真实模型。新目录防止覆盖账本。复现相同条件不保证模型逐字一致。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1, choices=(1,2,3))
    parser.add_argument("--max-calls", type=int, default=240)
    parser.add_argument("--resume", action="store_true", help="Keep every completed/aborted attempt; never rerun a recorded trial")
    parser.add_argument("--workers", type=int, default=4, choices=(1,2,4,8))
    parser.add_argument("--summarize-only", action="store_true", help="Read existing ledgers, no model calls")
    parser.add_argument("--out", type=Path, default=ROOT/"data/poi_attributes_v1/pilot_20261030")
    args = parser.parse_args()
    if args.summarize_only:
        summarize_existing(args.out,args.repetitions)
        return 0
    load_local_env()
    model = os.getenv("OPENAI_MODEL") or os.getenv("AI_MODEL")
    cases = json.loads((ASSETS/"cases.json").read_text(encoding="utf-8"))
    maps = json.loads((ROOT/"knowledge/route_interest_v1/maps.json").read_text(encoding="utf-8"))["routes"]
    load_catalog()
    validate_cases(cases, maps)
    print(json.dumps({"mode":"real_model" if args.run else "dry_run", "cases":len(cases), "final_route_trials":len(cases)*2*args.repetitions,
        "max_calls":args.max_calls,"max_output_tokens_per_call":4096,"model":model,"cost":"unpriced","real_map_calls":0},ensure_ascii=True),flush=True)
    if not args.run:
        return 0
    if not model:
        raise ValueError("Configured model required")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    meter, rows = Meter(args.out, model, args.max_calls), []
    if args.resume:
        events = [json.loads(line) for line in (args.out/"calls.jsonl").read_text(encoding="utf-8").splitlines()]
        # Recover child calls if a parent died before copying their records.
        for path in sorted((args.out/"workers").glob("*/calls.jsonl")):
            events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        by_id = {r["call_id"]:r for r in events}
        meter.calls = [by_id[i] for i in sorted(by_id)]
        rows = [json.loads(line) for line in (args.out/"trials.jsonl").read_text(encoding="utf-8").splitlines()]
        root_keys = {(r["case_id"],r["repetition"],r["arm"]) for r in rows}
        for path in sorted((args.out/"workers").glob("*/trials.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                r=json.loads(line)
                key=(r["case_id"],r["repetition"],r["arm"])
                if key not in root_keys:
                    rows.append(r)
                    root_keys.add(key)
                    meter.append(args.out/"trials.jsonl",r)
        complete = {(r["case_id"],r["repetition"],r["arm"]) for r in rows}
        interrupted = {}
        for call in meter.calls:
            key = (call["case_id"],call["repetition"],call["arm"])
            if key not in complete:
                interrupted.setdefault(key, []).append(call)
        for (ident,rep,arm), calls in interrupted.items():
            case = next(c for c in cases if c["id"] == ident)
            final = TravelPlanState(query=case["query"])
            row = {"case_id":ident,"repetition":rep,"arm":arm,"error":"HarnessInterrupted",
                "metrics":grade(final,case,"HarnessInterrupted"),"model_calls":len(calls),"usage_known":False,
                "input_tokens":None,"output_tokens":None,"elapsed_ms":0,"diagnostics":{"partial_attempts_preserved":True},
                "final_state":final.model_dump(mode="json")}
            rows.append(row)
            meter.append(args.out/"trials.jsonl",row)
    completed = {(r["case_id"],r["repetition"],r["arm"]) for r in rows}
    files = ["app/planning/nodes.py","app/planning/poi_attributes.py","app/planning/helpers.py","app/planning/constraints.py","app/evaluation/route_interest.py","scripts/evaluate_poi_attributes.py"]
    binding = {"cases":fingerprint(cases),"maps":fingerprint(maps),"attributes":fingerprint(json.loads(CATALOG.read_text(encoding="utf-8"))),
        "code":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    meter.append(args.out/"protocol_versions.jsonl", {"fingerprints":binding,"sdk_max_retries":0,"resumed_calls":len(meter.calls),
        "note":"Initial 26 invocations preceded SDK transport fix; internal HTTP retry counts unavailable. Aborted trial retained, not rerun." if args.resume else "strict transport retry guard active from start"})
    status = "completed"
    try:
        if args.workers > 1:
            database = args.out/"call_budget.sqlite3"
            with sqlite3.connect(database) as db:
                db.execute("CREATE TABLE IF NOT EXISTS budget (used INTEGER, failures INTEGER)")
                if not db.execute("SELECT 1 FROM budget").fetchone():
                    db.execute("INSERT INTO budget VALUES (?,0)",(max((c["call_id"] for c in meter.calls),default=0),))
            jobs=[]
            for rep in range(args.repetitions):
                for case in cases:
                    for arm in (("without_attributes","with_attributes") if rep%2==0 else ("with_attributes","without_attributes")):
                        if (case["id"],rep,arm) not in completed:
                            out=args.out/"workers"/f'{case["id"]}_{rep}_{arm}'
                            jobs.append((case,rep,arm,model,maps,str(out),str(database),args.max_calls))
            status=run_parallel(jobs,args.workers,meter,rows,database)
        for rep in range(args.repetitions):
            if args.workers > 1: break
            for case in cases:
                arms = ("without_attributes","with_attributes") if rep % 2 == 0 else ("with_attributes","without_attributes")
                for arm in arms:
                    if (case["id"],rep,arm) in completed:
                        continue
                    if len(meter.calls) >= args.max_calls or meter.consecutive_failures >= 2:
                        status="stopped_call_budget_or_provider_failures"
                        break
                    row = run_trial(case,rep,arm,model,meter,maps)
                    rows.append(row)
                    print(json.dumps({**meter.current,"correct":row["metrics"]["expected_outcome_correct"],"selected":row["metrics"]["selected"],"calls":len(meter.calls),"error":row["error"]},ensure_ascii=True),flush=True)
                if status!="completed":break
            if status!="completed":break
    except BaseException:
        status = "interrupted_or_harness_failure"
        raise
    finally:
        report={"protocol":"poi-attributes-route-pilot-v1","status":status,"model":model,"max_calls":args.max_calls,
            "attempted_calls":len(meter.calls),"planned_trials":len(cases)*2*args.repetitions,"completed_trials":len(rows),
            "cost":"unpriced","workers":args.workers,"fingerprints":binding,"summary":summarize(rows),"results":rows}
        report["transport_audit"] = [json.loads(line) for line in (args.out/"protocol_versions.jsonl").read_text(encoding="utf-8").splitlines()]
        (args.out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        (args.out/"report.md").write_text(render(report),encoding="utf-8")
    return 0 if status=="completed" and all(not r["error"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
