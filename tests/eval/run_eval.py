"""评估入口：加载 fixtures → 每例跑 k 次 planner⇄reviewer → 打分 → 聚合 → 报告。

用法：
    python -m tests.eval.run_eval --dry-run
    python -m tests.eval.run_eval --k 1 --max-cases 1 --max-llm-calls 30 --allow-external-calls
    python -m tests.eval.run_eval --only nanjing-3d-sunny --k 1 --max-llm-calls 30 --allow-external-calls
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.eval.graders.code_graders import grade_code
from tests.eval.graders.llm_judge import judge_plan
from tests.eval.graders.reviewer_reliability import planner_rebuttal, reviewer_reliability
from tests.eval.harness import load_fixtures, run_planner_reviewer_loop
from tests.eval.report import aggregate_case, render_report
from app.core.eval_safety import (
    estimate_planner_reviewer_calls,
    require_call_budget,
    require_external_calls,
)
from app.core.llm_usage import snapshot as llm_usage_snapshot
from app.core.llm_usage import usage_delta

TRANSCRIPT_DIR = Path(__file__).resolve().parent / "transcripts"


def _failed_trial(exc: Exception) -> dict[str, Any]:
    """Represent provider/runtime failures as failed trials, never missing data."""
    detail = f"执行失败：{type(exc).__name__}"
    results = {
        key: {"passed": False, "detail": detail}
        for key in (
            "g1_closed_pool", "g2_time_check", "g4_structure", "g5_coverage",
            "g6_weather", "g7_convergence", "g8_time_check_efficiency",
            "g9_walking_distance", "g10_habit_constraints",
        )
    }
    return {
        "code": {"results": results, "objective_pass": False},
        "judge": {"scores": {}, "avg": 0.0, "error": detail},
        "reliability": {"false_approval": False, "false_rejection": False},
        "rebuttal": {"transitions": 0, "rebuttal_rate": 0.0,
                     "ignore_rate": 0.0, "rebutted": False, "health": "n/a"},
        "overall_pass": False,
        "rounds": 0,
        "error": detail,
        "_state": {"error": detail},
    }


def _run_trial(fx: dict[str, Any], use_judge: bool) -> dict[str, Any]:
    state = run_planner_reviewer_loop(fx)
    code = grade_code(state, fx)
    objective_pass = code["objective_pass"]
    converged = code["results"]["g7_convergence"]["passed"]

    judge = {"scores": {}, "avg": 0.0}
    rebuttal: dict[str, Any] = {"transitions": 0, "rebuttal_rate": 0.0,
                                "ignore_rate": 0.0, "rebutted": False, "health": "n/a"}
    if use_judge:
        judge = judge_plan(state, fx)
        rebuttal = planner_rebuttal(state, objective_pass)

    reliability = reviewer_reliability(state, objective_pass)
    return {
        "code": code,
        "judge": judge,
        "reliability": reliability,
        "rebuttal": rebuttal,
        # 整体通过 = 客观硬约束全过且已收敛(G7)
        "overall_pass": objective_pass and converged,
        "rounds": state.review_round,
        "_state": {
            "route": state.route,
            "approved": state.approved,
            "review_round": state.review_round,
            "dialogue": state.planner_reviewer_dialogue,
            "reviewer_issues": state.reviewer_issues,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Planner⇄Reviewer 评估")
    ap.add_argument("--k", type=int, default=5, help="每个用例重复次数（默认 5）")
    ap.add_argument("--only", type=str, default=None, help="只跑指定用例 id")
    ap.add_argument("--no-judge", action="store_true", help="跳过 LLM 评委与反驳分析")
    ap.add_argument("--out", type=str, default=None, help="Markdown 报告输出路径")
    ap.add_argument("--json-out", type=str, default=None, help="结构化执行报告输出路径")
    ap.add_argument("--max-cases", type=int, default=None, help="最多执行前 N 个用例")
    ap.add_argument("--max-llm-calls", type=int, default=None,
                    help="本次批准的 LLM 调用上限（在线执行必填）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只输出预算预估，不调用 LLM")
    ap.add_argument("--verbose-preflight", action="store_true",
                    help="预检时输出逐用例预算")
    ap.add_argument(
        "--allow-external-calls",
        action="store_true",
        help="确认调用真实 LLM 并消耗 API 额度",
    )
    args = ap.parse_args()
    if not 1 <= args.k <= 10:
        ap.error("--k must be between 1 and 10")
    if args.max_cases is not None and not 1 <= args.max_cases <= 100:
        ap.error("--max-cases must be between 1 and 100")

    fixtures = load_fixtures(only=args.only)
    input_source = "generated_fixtures"
    if not fixtures and args.dry_run:
        from tests.eval.generate_fixtures import SPECS

        fixtures = [spec for spec in SPECS if not args.only or spec["id"] == args.only]
        input_source = "manifest_only"
    if args.max_cases is not None:
        fixtures = fixtures[:args.max_cases]
    if not fixtures:
        print("未找到 fixture，请先在 tests/eval/fixtures/ 放置用例 JSON。")
        return 1

    budget = estimate_planner_reviewer_calls(
        fixtures,
        trials_per_case=args.k,
        use_judge=not args.no_judge,
        include_time_check=True,
    )
    budget["input_source"] = input_source
    budget["execution_ready"] = input_source == "generated_fixtures"
    public_budget = budget if args.verbose_preflight else {
        key: value for key, value in budget.items() if key != "per_case"
    }
    print(json.dumps({"preflight": public_budget}, ensure_ascii=False))
    if args.dry_run:
        return 0
    require_external_calls(
        ap,
        allowed=args.allow_external_calls,
        operation="online planner/reviewer evaluation",
    )
    require_call_budget(
        ap,
        estimated_upper_bound=budget["llm_calls_upper_bound"],
        maximum=args.max_llm_calls,
    )

    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    use_judge = not args.no_judge
    case_aggs: list[dict[str, Any]] = []
    usage_before = llm_usage_snapshot()

    for fx in fixtures:
        cid, tier = fx["id"], fx.get("tier", "capability")
        print(f"▶ {cid}（{tier}）跑 {args.k} 次…")
        trials: list[dict[str, Any]] = []
        for i in range(args.k):
            try:
                t = _run_trial(fx, use_judge)
            except Exception as exc:  # noqa: BLE001
                print(f"  trial {i + 1} 失败：{exc}")
                traceback.print_exc()
                trials.append(_failed_trial(exc))
                continue
            trials.append(t)
            flag = "✅" if t["overall_pass"] else "❌"
            print(f"  trial {i + 1}: {flag} 客观={t['code']['objective_pass']} "
                  f"收敛={t['rounds']}轮 反驳率={t['rebuttal'].get('rebuttal_rate')}")
        # 落盘最后一次 trial 的 transcript，便于人工读「失败是否公平」
        (TRANSCRIPT_DIR / f"{cid}.json").write_text(
            json.dumps([t["_state"] for t in trials], ensure_ascii=False, indent=2),
            encoding="utf-8")
        case_aggs.append(aggregate_case(cid, tier, trials))

    actual_usage = usage_delta(usage_before, llm_usage_snapshot())
    execution = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": (os.getenv("LLM_PROVIDER") or "openai").strip().lower(),
        "preflight": budget,
        "actual_llm_usage": actual_usage,
        "completed_cases": len(case_aggs),
    }
    report = render_report(case_aggs)
    report += (
        "\n## 执行元数据\n\n"
        f"- 预算调用上限：{budget['llm_calls_upper_bound']}\n"
        f"- 实际 LLM 调用：{actual_usage['calls_total']}\n"
        f"- 输入 / 输出 Token：{actual_usage['input_tokens']} / {actual_usage['output_tokens']}\n"
        f"- Provider usage 精确率：{actual_usage['usage_exact_ratio']:.1%}\n"
        f"- 估算成本（USD）：{actual_usage['estimated_cost_usd']:.6f}"
        f"（价格环境变量{'已配置' if actual_usage['cost_rates_configured'] else '未配置，当前金额不可用'}）\n"
    )
    print("\n" + report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\n报告已写入 {args.out}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"execution": execution, "cases": case_aggs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"结构化报告已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
