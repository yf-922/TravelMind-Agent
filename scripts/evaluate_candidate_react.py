"""Paired, fail-closed offline ablation of the production bounded candidate loop.

Run: python scripts/evaluate_candidate_react.py
No credentials, network, model download, or paid model calls are needed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import socket
import sys
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import END, START, StateGraph
from app.planning import candidate_react as cr
from app.planning.schemas import TravelPlanState
from app.providers.amap import poi as provider

ASSETS = ROOT / "tests/eval/candidate_react_v3"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "percent": 100 * numerator / denominator if denominator else None}


def build_cases_for_freeze(snapshot):
    """Construct controlled deficits, NOT a representative user-query sample.

    This generator is not called by evaluate(): evaluation reads frozen cases.
    Gold requirements are declared here, independently of production coverage().
    """
    cases = []
    for ci, (city, data) in enumerate(snapshot["cities"].items()):
        # Bootstrap labels are historical name heuristics, not new verification.
        indoor = [p for p in data["bootstrap"] if p.get("indoor") is True]
        outdoor = [p for p in data["bootstrap"] if p.get("indoor") is False]
        for i, target_count in enumerate((1, 2, 1)):
            targets = data["museum_rows"][i * 2:i * 2 + target_count]
            names = [p["name"] for p in targets]
            base = [indoor[0]] + outdoor[:3]
            base = [p for p in base if p["name"] not in names]
            cases.append({"id": f"named-{ci+1}-{i+1}", "group": "named",
                "city": city, "query": city + "一日游，" + "，".join('想去“'+n+'”' for n in names),
                "days": 1, "base": base, "required_names": names,
                "min_count": 4, "min_indoor": 0, "response_mode": "normal"})
        for i, (days, only, initial_indoor) in enumerate(((1, False, 0), (1, True, 1), (2, False, 1))):
            target = days * 2 + max(2, days)
            cases.append({"id": f"indoor-{ci+1}-{i+1}", "group": "indoor",
                "city": city, "query": city + ("只去室内" if only else "雨天游，室内优先"),
                "days": days, "base": indoor[:initial_indoor] + outdoor[:target-initial_indoor],
                "required_names": [], "min_count": target,
                "min_indoor": target if only else 3, "rain": True,
                "response_mode": "normal"})
    cities = list(snapshot["cities"])
    for i, (days, nbase) in enumerate(((1, 1), (1, 2), (2, 2), (2, 3), (1, 3))):
        city = cities[i % 3]
        data = snapshot["cities"][city]
        indoor = [p for p in data["bootstrap"] if p.get("indoor") is True]
        outdoor = [p for p in data["bootstrap"] if p.get("indoor") is False]
        cases.append({"id": f"count-{i+1}", "group": "count", "city": city,
            "query": city + "历史与自然结合，候选去重筛选后数量不足，需要补充", "days": days,
            "base": (indoor[:1] + outdoor)[:nbase], "required_names": [],
            "min_count": days * 2 + max(2, days), "min_indoor": 0,
            "response_mode": "duplicates" if i in (1, 4) else "normal"})
    # These are deliberate negative/guard probes, excluded from feasible coverage.
    for ident, mode, query, used in [
        ("nonexistent-repeat", "empty", '想去“霍格沃茨魔法学校”', 0),
        ("near-name-reject", "near_name", '想去“南京博物院魔法分馆”', 0),
        ("round-limit", "empty", "雨天只去室内", 0),
        ("retry-budget", "timeout", "只去室内", 0),
        ("shared-budget", "timeout", "只去室内", 3),
    ]:
        data = snapshot["cities"]["南京"]
        cases.append({"id": ident, "group": "boundary", "city": "南京", "query": query,
            "days": 1, "base": [p for p in data["bootstrap"] if p.get("indoor") is False][:4],
            "required_names": ["霍格沃茨魔法学校"] if ident == "nonexistent-repeat" else
                              (["南京博物院魔法分馆"] if ident == "near-name-reject" else []),
            "min_count": 4, "min_indoor": 0 if "想去" in query else 4,
            "initial_api_calls": used, "response_mode": mode})
    return cases


def validate_assets(snapshot, cases):
    if Counter(c["group"] for c in cases) != {"named": 9, "indoor": 9, "count": 5, "boundary": 5}:
        raise ValueError("Expected 9 named + 9 indoor + 5 count + 5 boundary cases")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case ID")
    for c in cases:
        if c["city"] not in snapshot["cities"]:
            raise ValueError("Unknown city")
        if len({p["name"] for p in c["base"]}) != len(c["base"]):
            raise ValueError("Duplicate bootstrap POI")
        known_base = {p["name"]: p for p in snapshot["cities"][c["city"]]["bootstrap"]}
        if any(p != known_base.get(p["name"]) for p in c["base"]):
            raise ValueError("Bootstrap POI differs from frozen provenance")
        if c["group"] == "named":
            available = {p["name"] for p in snapshot["cities"][c["city"]]["museum_rows"]}
            if not set(c["required_names"]) <= available:
                raise ValueError("Named target lacks captured POI evidence")
        if not 0 <= c.get("initial_api_calls", 0) <= 4:
            raise ValueError("Invalid shared HTTP budget")
    if not snapshot.get("sources") or any(len(s.get("sha256", "")) != 64 for s in snapshot["sources"]):
        raise ValueError("Missing source provenance")


def score(case, pool):
    """Independent case-level prerequisite scorer; no call to cr.coverage()."""
    unique = {p["name"]: p for p in pool}
    missing_names = sorted(set(case["required_names"]) - unique.keys())
    indoor = sum(p.get("indoor") is True for p in unique.values())
    hinted = sum(p.get("indoor") is True or p.get("indoor_hint") is True for p in unique.values())
    blockers = []
    if missing_names:
        blockers.append("missing_required_poi")
    if len(unique) < case["min_count"]:
        blockers.append("insufficient_count")
    if indoor < case["min_indoor"]:
        blockers.append("indoor_not_established")
    return {"pass": not blockers, "blockers": blockers, "missing_names": missing_names,
            "count": len(unique), "indoor": indoor, "indoor_with_hints": hinted,
            "hint_only_diagnostic_pass": not missing_names and len(unique) >= case["min_count"]
                                         and hinted >= case["min_indoor"]}


def offline_decision(observation, case):
    """Deterministic observation-driven policy double, not LLM-quality evaluation.

    Never reads gold pass/fail labels or response rows. Boundary probes deliberately
    repeat or vary queries to challenge server guards.
    """
    gaps = observation["active_gaps"]
    names = gaps["missing_named_pois"]
    if names:
        actions = [{"action": "search_user_requested_poi", "query": n} for n in names[:2]]
        if case["id"] in ("nonexistent-repeat", "near-name-reject"):
            actions = actions * 2
    elif case["id"] == "round-limit":
        r = len(observation["tried_keys"])
        actions = [{"action": "search_indoor", "query": q} for q in
                   (("博物馆", "美术馆") if not r else ("展览馆", "科技馆"))]
    elif case["response_mode"] == "timeout":
        actions = [{"action": "search_indoor", "query": q} for q in ("博物馆", "美术馆")]
    else:
        already = bool(observation["tried_keys"])
        actions = [{"action": "search_indoor", "query": "美术馆" if already else "博物馆"}]
    return cr.CandidateDecision.model_validate({"actions": actions})


class Replay:
    def __init__(self, case, snapshot):
        self.case = case
        self.rows = snapshot["cities"][case["city"]]["museum_rows"]
        self.http = []
        self.mapping_errors = []

    def get(self, url, *, attempts):
        q = parse_qs(urlsplit(url).query, keep_blank_values=True)
        city, keyword, types = (q[k][0] for k in ("city", "keywords", "types"))
        if city != self.case["city"] or attempts != 1:
            self.mapping_errors.append("Unmapped city or transport attempts")
            raise AssertionError("Unmapped HTTP request")
        mode = self.case["response_mode"]
        record = {"city": city, "query": keyword, "types": types}
        self.http.append(record)
        if mode == "timeout":
            record["outcome"] = "injected_timeout"
            raise RuntimeError("offline injected connection timeout")
        if mode == "near_name":
            rows = self.rows[:1]  # A real, differently named museum; never a fake POI.
        elif mode == "empty":
            rows = []
        elif types == "":
            rows = [p for p in self.rows if p["name"] == keyword]
        elif types == "140100" and keyword in cr.INDOOR_QUERIES:
            rows = self.rows
        else:
            self.mapping_errors.append(f"Unmapped query: {keyword}/{types}")
            raise AssertionError("No controlled response mapping for query")
        if mode == "duplicates":
            rows = rows + rows[:3] + rows[:1]
        record.update(outcome="success", raw_count=len(rows))
        return {"status": "1", "pois": deepcopy(rows)}


def run_pair(case, snapshot):
    initial = TravelPlanState(query=case["query"], destination=case["city"], days=case["days"],
        pois=deepcopy(case["base"]), candidate_api_budget=4, candidate_max_rounds=2,
        candidate_api_calls=case.get("initial_api_calls", 0), candidate_min_per_day=2,
        rain_indoor_priority=case.get("rain", False),
        weather_forecast=[{"is_bad": case.get("rain", False)}])
    baseline = score(case, initial.pois)
    base_digest = digest(initial.model_dump(mode="json"))
    replay, observations = Replay(case, snapshot), []
    issued_keys, repeat_opportunities = set(), []
    violations = []

    def forbidden(*args, **kwargs):
        violations.append("external_or_unmapped_call")
        raise AssertionError("Offline benchmark forbids network/model calls")

    def invoke(llm, messages, **kwargs):
        obs = json.loads(messages[1][1])
        observations.append(obs)
        decision = offline_decision(obs, case)
        # Independent action-side denominator, not derived from production errors.
        for action in decision.actions:
            if action.action == "stop":
                continue
            types = "" if action.action == "search_user_requested_poi" else (
                "140100" if action.action == "search_indoor" else "110000")
            key = (case["city"], action.query.strip(), types)
            if key in issued_keys:
                repeat_opportunities.append(key)
            issued_keys.add(key)
        return decision

    # Execute actual production decision wrapper, search, validator and graph edges.
    graph = StateGraph(TravelPlanState)
    graph.add_node("candidate_react", cr.make_candidate_react_node())
    graph.add_node("search", cr.candidate_search_node)
    graph.add_node("validate", cr.candidate_validator_node)
    graph.add_edge(START, "candidate_react")
    graph.add_edge("candidate_react", "search")
    graph.add_edge("search", "validate")
    graph.add_conditional_edges("validate", cr.route_after_candidate_validation,
                               {"candidate_react": "candidate_react", "planner": END})
    compiled = graph.compile()
    error = None
    try:
        with patch.object(cr, "build_structured_llm", return_value=object()), \
             patch.object(cr, "invoke_structured", side_effect=invoke), \
             patch.object(cr, "get_cached", return_value=None), patch.object(cr, "set_cached"), \
             patch.object(cr, "amap_key", return_value="offline-placeholder"), \
             patch.object(provider, "http_get_json", side_effect=replay.get), \
             patch.object(provider.time, "sleep"), \
             patch.object(socket.socket, "connect", side_effect=forbidden), \
             patch.object(socket, "create_connection", side_effect=forbidden):
            result = compiled.invoke(initial.model_copy(deep=True), {"recursion_limit": 12})
        final = TravelPlanState.model_validate(result)
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        final = initial
    if violations:
        raise AssertionError("Offline isolation violated")
    if replay.mapping_errors:
        raise AssertionError("Replay mapping defect: " + str(replay.mapping_errors))
    after = score(case, final.pois)
    if error:
        after["pass"] = False
        after["blockers"].append("execution_failed")
    # Guard against production exception handlers accidentally hiding mapping defects.
    accounted_attempts = final.candidate_api_calls - initial.candidate_api_calls
    if len(replay.http) != accounted_attempts and not error:
        raise AssertionError("Provider attempt accounting mismatch")
    # A graph error can discard a state update, never observed HTTP attempts.
    attempts = len(replay.http)
    total_attempts = initial.candidate_api_calls + attempts
    rounds = len(observations) if error else final.candidate_search_round
    stop_reason = ("execution_error" if error else "coverage_ready" if final.candidate_pool_status == "ready"
                   else "shared_http_budget_exhausted" if total_attempts >= 4
                   else "round_limit" if rounds >= 2 else "policy_stop")
    final_names = [p["name"] for p in final.pois]
    known = {p["name"] for p in case["base"]} | {p["name"] for p in replay.rows}
    errors = Counter(t.get("error_code") for t in final.candidate_search_trace if t.get("error_code"))
    # Diagnostic upper bound: all admissible captured rows, ignoring action/HTTP
    # limits, and (optimistically, not officially) trusting indoor category hints.
    optimistic = {p["name"]: deepcopy(p) for p in case["base"]}
    for row in replay.rows:
        p = provider.poi_to_spot(row)
        if not p or (p.get("rating") is not None and p["rating"] < initial.min_rating
                     and p["name"] not in case["required_names"]):
            continue
        if p.get("indoor_hint") is True:
            p["indoor"] = True
        optimistic.setdefault(p["name"], p)
    return {"id": case["id"], "group": case["group"], "city": case["city"],
        "query": case["query"], "initial_state_sha256": base_digest,
        "initial_state_unchanged": digest(initial.model_dump(mode="json")) == base_digest,
        "requirements": {k: case[k] for k in ("required_names", "min_count", "min_indoor")},
        "baseline": baseline, "bounded_react": after, "error": error,
        "initial_names": [p["name"] for p in case["base"]], "final_names": final_names,
        "status": final.candidate_pool_status, "production_missing": final.candidate_missing_coverage,
        "rounds": rounds, "supplemental_http_attempts": attempts,
        "initial_http_attempts": initial.candidate_api_calls, "total_http_attempts": total_attempts,
        "stop_reason": stop_reason,
        "trace": final.candidate_search_trace, "http": replay.http, "observations": observations,
        "duplicate_blocks": errors["REPEATED_QUERY"], "search_failures": errors["SEARCH_FAILED"],
        "repeat_opportunities": len(repeat_opportunities),
        "optimistic_fixture_upper_bound": score(case, list(optimistic.values())),
        "budget_guard_errors": errors["BUDGET_EXHAUSTED"],
        "unique_names": len(final_names) == len(set(final_names)),
        "provenance_ok": set(final_names) <= known,
        "guard_pass": not error and rounds <= 2 and total_attempts <= 4
                      and final.candidate_pool_status == "insufficient" and not after["pass"]}


def summarize(records):
    deficits = [r for r in records if r["group"] != "boundary"]
    boundaries = [r for r in records if r["group"] == "boundary"]
    def paired(rows):
        return {arm: fraction(sum(r[arm]["pass"] for r in rows), len(rows))
                for arm in ("baseline", "bounded_react")}
    return {"all_cases": len(records), "non_boundary_coverage": paired(deficits),
        "by_group": {g: paired([r for r in records if r["group"] == g]) for g in ("named", "indoor", "count")},
        "raw_coverage_including_deliberate_negatives": paired(records),
        "baseline_prerequisite_blocked": fraction(sum(not r["baseline"]["pass"] for r in deficits), len(deficits)),
        "after_prerequisite_blocked": fraction(sum(not r["bounded_react"]["pass"] for r in deficits), len(deficits)),
        "blocker_counts": {arm: dict(Counter(b for r in deficits for b in r[arm]["blockers"]))
                           for arm in ("baseline", "bounded_react")},
        "fixture_limited_cases_even_trusting_hints": [r["id"] for r in deficits if not r["optimistic_fixture_upper_bound"]["pass"]],
        "optimistically_fixture_satisfiable": paired([r for r in deficits if r["optimistic_fixture_upper_bound"]["pass"]]),
        "boundary_safe_incomplete": fraction(sum(r["guard_pass"] for r in boundaries), len(boundaries)),
        "budget_and_round_compliance": fraction(sum(r["rounds"] <= 2 and r["total_http_attempts"] <= 4 for r in records), len(records)),
        "fictional_or_near_name_accepted": fraction(sum(r["bounded_react"]["pass"] for r in boundaries if r["requirements"]["required_names"]), 2),
        "duplicate_query_blocks": sum(r["duplicate_blocks"] for r in records),
        "repeat_query_guard": fraction(sum(r["duplicate_blocks"] for r in records), sum(r["repeat_opportunities"] for r in records)),
        "mean_rounds": sum(r["rounds"] for r in records) / len(records),
        "mean_supplemental_http_attempts": sum(r["supplemental_http_attempts"] for r in records) / len(records),
        "total_rounds": sum(r["rounds"] for r in records),
        "total_supplemental_http_attempts": sum(r["supplemental_http_attempts"] for r in records),
        "max_rounds": max(r["rounds"] for r in records), "max_total_http_attempts": max(r["total_http_attempts"] for r in records),
        "execution_failures": fraction(sum(r["error"] is not None for r in records), len(records)),
        "hint_only_diagnostic": fraction(sum(r["bounded_react"]["hint_only_diagnostic_pass"] for r in deficits), len(deficits))}


def evaluate(assets=ASSETS):
    snapshot = json.loads((assets / "poi_snapshot.json").read_text(encoding="utf-8"))
    cases = json.loads((assets / "cases.json").read_text(encoding="utf-8"))
    validate_assets(snapshot, cases)
    records = [run_pair(c, snapshot) for c in cases]
    config = {"round_limit": 2, "shared_http_limit": 4, "cache": "disabled",
              "candidate_min_per_day": 2, "min_rating": 4.5, "model": "deterministic_policy_double",
              "indoor_policy": "type whitelist + outdoor-name veto; scoped amap_type_verified; no LLM labeling"}
    return {"protocol": "candidate-react-controlled-v3", "scope": "post-bootstrap controlled deficits; offline mechanism ablation",
        "annotation_status": "controlled requirements, not human-verified real-user labels",
        "config": config, "environment": {"python": sys.version.split()[0],
            "langgraph": version("langgraph"), "pydantic": version("pydantic")},
        "fingerprints": {"snapshot": digest(snapshot), "cases": digest(cases), "config": digest(config),
            "code": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                     ("app/planning/candidate_react.py", "app/planning/schemas.py", "app/planning/helpers.py",
                      "app/providers/amap/poi.py", "scripts/evaluate_candidate_react.py")}},
        "summary": summarize(records), "cases": records}


def markdown(report):
    s = report["summary"]
    def fmt(v):
        return f'{v["numerator"]}/{v["denominator"]} ({v["percent"]:.2f}%)' if v["denominator"] else "N/A (0/0)"
    lines = ["# 候选池与有界 ReAct：离线配对消融", "",
        "## 测量范围与限制", "",
        "28 个主动构造的检索后缺口：9 点名、9 室内、5 数量、5 边界；覆盖南京、上海、三亚。不是随机真实用户样本，不报告置信区间或线上成功率。",
        "两组从完全相同的固定基础候选池出发，Baseline 不补查；实验组运行生产 LangGraph 决策包装器→搜索→校验→条件边。LLM 决策由读取当前缺口的确定性策略替身代替，因此不验证真实 LLM 的工具选择能力。",
        "POI 来自已有真实高德响应快照，保留原名称、ID、坐标、类型与评分。点名查询按同名过滤原始记录，分类查询返回冻结博物馆记录；这是受控 HTTP 响应映射，不是这些具体查询的实测响应或实时高德召回率。空结果、超时、重复行、近名干扰为显式注入。",
        "v3 保留旧基础池与独立通过标准。新增 POI 通过生产分类白名单取得带范围的 amap_type_verified 标签，室外名称词根否决；标签表示高德建筑场馆分类，不是现场全程避雨证明。旧 v2 资产与报告保留。named-3-3 将误分类的绿盾门窗替换为同一三亚原始快照中的三亚自然博物馆，不更改原始 POI 字段；快照记录 curations。",
        "生产基础检索本身已按请求选择查询词；本实验刻意选取其后仍有缺口的状态，不代表完整基础检索平均表现。4 次 HTTP 是共享预算：多数场景起点为 0（前置开销未录制），另有起点已用 3 次的共享预算探针；报告同时保留初始、补查、总次数，不声称全流程实测最多 4 次。",
        "候选前提独立评分：点名集合全部出现、去重数量达标、明确室内数量达标，三项同时满足才通过。生产 validator 状态另外记录；不执行 Planner，不将前提满足冒充最终路线可规划/成功率。主分母为 23 个非边界受控缺口，不称为全都可满足；其中可能存在固定响应容量不足，另报诊断子集。",
        "无真实 LLM、外网或付费调用；禁用缓存、退避等待。HTTP 重试与预算使用生产 Provider 实现。不存在景点只在受控响应中设为空，不声称现实中同名地点一定不存在。", "",
        "## 结果", "", "| 场景 | 仅基础候选 | + 有界补查 |", "|---|---:|---:|"]
    for g, label in (("named", "点名缺口"), ("indoor", "室内缺口"), ("count", "数量缺口")):
        lines.append(f'| {label} | {fmt(s["by_group"][g]["baseline"])} | {fmt(s["by_group"][g]["bounded_react"])} |')
    lines += [f'| 非边界缺口合计 | {fmt(s["non_boundary_coverage"]["baseline"])} | {fmt(s["non_boundary_coverage"]["bounded_react"])} |', "",
        f'- 基础候选前提受阻：{fmt(s["baseline_prerequisite_blocked"])}；补查后仍受阻：{fmt(s["after_prerequisite_blocked"])}。',
        f'- 边界安全退出并保留不足：{fmt(s["boundary_safe_incomplete"])}；轮次/预算合规：{fmt(s["budget_and_round_compliance"])}。',
        f'- 虚构/近名错误接纳：{fmt(s["fictional_or_near_name_accepted"])}；重复查询拦截：{fmt(s["repeat_query_guard"])}（决策侧独立计算重复动作数作分母，核对实际 HTTP 未重复）。',
        f'- 补查共 {s["total_rounds"]} 轮，均值 {s["mean_rounds"]:.4f} 轮/单；HTTP 尝试共 {s["total_supplemental_http_attempts"]} 次，均值 {s["mean_supplemental_http_attempts"]:.4f} 次/单；最大 {s["max_rounds"]} 轮、总 HTTP {s["max_total_http_attempts"]} 次。',
        f'- 执行异常：{fmt(s["execution_failures"])}，失败请求不剔除。',
        f'- 若误把 indoor_hint 当作确定室内，诊断得分会变成 {fmt(s["hint_only_diagnostic"])}；该数不是正式通过率，不能用于宣称室内安全。', "",
        f'- 即使忽略动作预算并把全部类型提示当作室内，固定响应仍不足的场景：{", ".join(s["fixture_limited_cases_even_trusting_hints"])}。未将这些失败悄悄删除。',
        f'- 在上述乐观假设下固定响应容量足够的诊断子集：基础 {fmt(s["optimistically_fixture_satisfiable"]["baseline"])}，补查 {fmt(s["optimistically_fixture_satisfiable"]["bounded_react"])}；最终评分仍遵守正式 indoor，不用提示替代事实。', "",
        "## 逐例结果", "", "| ID | 需求 | 基础→补查 | 轮次/HTTP补查/初始 | 状态与未满足前提 |", "|---|---|---|---|---|"]
    for r in report["cases"]:
        lines.append(f'| {r["id"]} | {r["query"]} | {int(r["baseline"]["pass"])}→{int(r["bounded_react"]["pass"])} | {r["rounds"]}/{r["supplemental_http_attempts"]}/{r["initial_http_attempts"]} | {r["status"]}: {", ".join(r["bounded_react"]["blockers"])} |')
    lines += ["", "## 结论与下一步", "",
        "点名补查和数量补查在固定响应下能够修复部分候选前提，证明补查工具链有实际作用；该结果不能证明真实模型每次能选对动作，也不能证明最终行程成功。",
        "室内组由 0/9 提升至 7/9。仍失败的 indoor-3-1、indoor-3-2 与 count-3 受三亚冻结响应中评分过滤后可用候选容量限制。没有降低评分阈值、删除失败请求或伪造额外 POI。",
        "零分基线是选样条件：本集专门挑选基础池存在缺口的状态；不应写成基础检索总体通过率 0%，也不应把 5 条边界的安全退出率当成全系统可靠性。",
        "本次修改生产解析、缺口计算、模型输入和服务端动作约束。该配对只隔离是否执行补查，不能单独归因于提示词或模型智能；没有真实 Token 节省率或最终路线质量数据。", "",
        "## 复现与审计", "", "```powershell",
        "python scripts/evaluate_candidate_react.py",
        "python -m pytest -q tests/test_candidate_react.py tests/test_candidate_react_benchmark.py",
        "python scripts/validate_eval_assets.py", "python -m pytest -q", "```", "",
        "逐请求 JSON：`docs/candidate_react_ablation_v3.json`，包含两组评分、缺口、候选名称、决策观察、搜索追踪、HTTP 尝试和停止状态。测试与报告均使用版本化快照，复现不依赖被忽略的 data/。没有测量 Token、网络耗时或真实模型质量。", "",
        "指纹（SHA256；数据按排序键规范化 JSON）：", "", "```json",
        json.dumps(report["fingerprints"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=ROOT / "docs/candidate_react_ablation_v3.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/candidate_react_ablation_v3.md")
    args = parser.parse_args()
    report = evaluate()
    for path, content in ((args.json, json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
                          (args.markdown, markdown(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=True, indent=2))
    return 1 if report["summary"]["execution_failures"]["numerator"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
