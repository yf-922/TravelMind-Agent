"""可复现的 Planner 质量闭环消融评测。

该脚本不调用外部 LLM/API，目的不是宣称真实模型质量，而是量化三种
编排策略在同一组结构化负例上的硬约束收益和额外调用成本：

* planner_only：只生成，不做独立审核；
* planner_reviewer：增加 Reviewer，修复候选池越界和重复景点；
* planner_reviewer_time_check：再增加 Time Check，修复开放时间冲突。

真实 LLM 评测仍应使用 tests/eval/run_eval.py；本脚本只验证“为什么要加
审核节点”的可解释基线，避免把 deterministic fixture 结果包装成线上效果。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AblationCase:
    case_id: str
    candidates: tuple[str, ...]
    route: tuple[dict[str, Any], ...]
    opening: dict[str, tuple[int, int]]


MODES = ("planner_only", "planner_reviewer", "planner_reviewer_time_check")


def default_cases() -> list[AblationCase]:
    """Return a stratified, deterministic contract set with mixed outcomes.

    The first version used only three hand-written cases, which made the
    percentages look artificially neat.  Keep the named smoke cases for
    backwards compatibility, then add repeated-but-not-identical boundary
    cases and a few irreparable drafts.  This is still an offline contract
    benchmark, not a claim about live model accuracy.
    """
    cases = [
        AblationCase(
            "unknown_and_duplicate",
            ("Museum", "Park"),
            (
                {"name": "Museum", "start_min": 600, "end_min": 720},
                {"name": "Unknown", "start_min": 780, "end_min": 840},
                {"name": "Museum", "start_min": 900, "end_min": 960},
            ),
            {"Museum": (540, 1080), "Park": (540, 1080)},
        ),
        AblationCase(
            "opening_time_conflict",
            ("Museum", "Park"),
            (
                {"name": "Museum", "start_min": 480, "end_min": 1140},
                {"name": "Park", "start_min": 780, "end_min": 900},
            ),
            {"Museum": (540, 1080), "Park": (540, 1080)},
        ),
        AblationCase(
            "already_valid",
            ("Museum", "Park"),
            (
                {"name": "Museum", "start_min": 600, "end_min": 720},
                {"name": "Park", "start_min": 780, "end_min": 900},
            ),
            {"Museum": (540, 1080), "Park": (540, 1080)},
        ),
    ]
    # 36 additional cases: clean controls, candidate-pool faults, duplicate
    # faults, opening-hours faults, and drafts that no deterministic repair can
    # safely approve.  The values vary enough to avoid a three-case toy demo.
    for index in range(8):
        start = 540 + index * 15
        cases.append(AblationCase(
            f"clean_control_{index:02d}", ("Museum", "Park", "Gallery"),
            ({"name": "Museum", "start_min": start, "end_min": start + 45},
             {"name": "Park", "start_min": start + 90, "end_min": start + 150}),
            {"Museum": (540, 1080), "Park": (540, 1080), "Gallery": (540, 1080)},
        ))
    for index in range(8):
        cases.append(AblationCase(
            f"candidate_pool_fault_{index:02d}", ("Museum", "Park"),
            ({"name": "Museum", "start_min": 600, "end_min": 660},
             {"name": f"Unknown-{index}", "start_min": 720, "end_min": 780}),
            {"Museum": (540, 1080), "Park": (540, 1080)},
        ))
    for index in range(8):
        cases.append(AblationCase(
            f"duplicate_fault_{index:02d}", ("Museum", "Park"),
            ({"name": "Museum", "start_min": 600, "end_min": 660},
             {"name": "Museum", "start_min": 720, "end_min": 780}),
            {"Museum": (540, 1080), "Park": (540, 1080)},
        ))
    for index in range(8):
        opening = 540 + index * 10
        cases.append(AblationCase(
            f"opening_fault_{index:02d}", ("Museum", "Park"),
            ({"name": "Museum", "start_min": opening - 90, "end_min": opening - 30},
             {"name": "Park", "start_min": 780, "end_min": 840}),
            {"Museum": (opening, opening + 480), "Park": (540, 1080)},
        ))
    for index in range(4):
        cases.append(AblationCase(
            f"irreparable_empty_after_review_{index:02d}", ("Museum",),
            ({"name": f"Unknown-only-{index}", "start_min": 600, "end_min": 660},),
            {"Museum": (540, 1080)},
        ))
    return cases


def _review_route(route: list[dict[str, Any]], candidates: set[str]) -> list[dict[str, Any]]:
    """Deterministic Reviewer repair: remove unknown and repeated POIs."""
    seen: set[str] = set()
    repaired: list[dict[str, Any]] = []
    for item in route:
        name = str(item.get("name") or "")
        if name not in candidates or name in seen:
            continue
        seen.add(name)
        repaired.append(dict(item))
    return repaired


def _time_check_route(route: list[dict[str, Any]], opening: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    """Clamp each visit into its known opening interval, preserving order."""
    repaired: list[dict[str, Any]] = []
    for item in route:
        result = dict(item)
        bounds = opening.get(str(item.get("name") or ""))
        if bounds:
            open_min, close_min = bounds
            start = max(int(item.get("start_min", open_min)), open_min)
            end = min(int(item.get("end_min", close_min)), close_min)
            if end <= start:
                end = min(close_min, start + 60)
            result.update(start_min=start, end_min=end)
        repaired.append(result)
    return repaired


def _checks(route: list[dict[str, Any]], case: AblationCase) -> dict[str, bool]:
    names = [str(item.get("name") or "") for item in route]
    has_route = bool(route)
    closed_pool = has_route and all(name in set(case.candidates) for name in names)
    no_duplicate = len(names) == len(set(names))
    time_valid = all(
        str(item.get("name") or "") not in case.opening
        or (
            case.opening[str(item.get("name") or "")][0] <= int(item.get("start_min", -1))
            and int(item.get("end_min", -1)) <= case.opening[str(item.get("name") or "")][1]
            and int(item.get("end_min", -1)) > int(item.get("start_min", -1))
        )
        for item in route
    )
    return {
        "closed_pool": closed_pool,
        "has_route": has_route,
        "no_duplicate": no_duplicate,
        "time_valid": time_valid,
        "overall": has_route and closed_pool and no_duplicate and time_valid,
    }


def run_ablation(cases: list[AblationCase] | None = None) -> dict[str, Any]:
    """Evaluate all modes and return per-mode metrics plus case transcripts."""
    cases = cases or default_cases()
    records: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for case in cases:
        for mode in MODES:
            route = [dict(item) for item in case.route]
            calls = 1  # planner
            if mode != "planner_only":
                route = _review_route(route, set(case.candidates))
                calls += 1
            if mode == "planner_reviewer_time_check":
                route = _time_check_route(route, case.opening)
                calls += 1
            records[mode].append({
                "case_id": case.case_id,
                "checks": _checks(route, case),
                "calls": calls,
                "route": route,
            })

    metrics: dict[str, dict[str, Any]] = {}
    for mode, rows in records.items():
        n = len(rows) or 1
        metrics[mode] = {
            "cases": len(rows),
            "overall_pass_rate": round(sum(row["checks"]["overall"] for row in rows) / n, 3),
            "closed_pool_pass_rate": round(sum(row["checks"]["closed_pool"] for row in rows) / n, 3),
            "no_duplicate_pass_rate": round(sum(row["checks"]["no_duplicate"] for row in rows) / n, 3),
            "time_valid_pass_rate": round(sum(row["checks"]["time_valid"] for row in rows) / n, 3),
            "avg_calls": round(sum(row["calls"] for row in rows) / n, 3),
        }
    return {"scope": "deterministic_contract_ablation", "metrics": metrics, "records": records}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Planner 编排消融评测（离线结构契约）",
        "",
        "> 仅用于比较编排策略在固定负例上的硬约束表现，不代表真实 LLM 质量。",
        "",
        "| 策略 | 总体通过率 | 候选池 | 无重复 | 时间合法 | 平均调用节点 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, metric in report["metrics"].items():
        lines.append(
            f"| {mode} | {metric['overall_pass_rate']:.0%} | "
            f"{metric['closed_pool_pass_rate']:.0%} | {metric['no_duplicate_pass_rate']:.0%} | "
            f"{metric['time_valid_pass_rate']:.0%} | {metric['avg_calls']:.1f} |"
        )
    lines.extend([
        "",
        "结论：增加独立 Reviewer 能修复候选池越界和重复景点；再增加 Time Check 才能覆盖开放时间冲突。代价是每次任务增加对应节点调用。",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Planner ablation")
    parser.add_argument("--json", type=Path, help="write JSON report")
    parser.add_argument("--out", type=Path, help="write Markdown report")
    args = parser.parse_args()
    report = run_ablation()
    print(render_markdown(report))
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out:
        args.out.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
