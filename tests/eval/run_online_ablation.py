"""Run a fair real-LLM ablation on identical frozen travel fixtures.

This command compares one-shot Planner, Planner+Reviewer, and the full
Planner+Reviewer+Time Check workflow. External calls are opt-in and guarded by
a conservative provider-attempt budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.eval_safety import (
    estimate_online_ablation_calls,
    require_call_budget,
    require_external_calls,
)
from app.core.llm_usage import snapshot as llm_usage_snapshot
from app.core.llm_usage import usage_delta
from tests.eval.graders.code_graders import grade_code
from tests.eval.graders.llm_judge import judge_plan
from tests.eval.graders.reviewer_reliability import planner_rebuttal, reviewer_reliability
from tests.eval.harness import EVALUATION_MODES, load_fixtures, run_evaluation_loop
from tests.eval.report import JUDGE_KEYS, _mean
from tests.eval.run_eval import _failed_trial


def run_trial(fx: dict[str, Any], mode: str, use_judge: bool) -> dict[str, Any]:
    state = run_evaluation_loop(fx, mode)  # type: ignore[arg-type]
    code = grade_code(state, fx)
    objective_pass = code["objective_pass"]
    if mode == "planner_only":
        converged = True
    elif mode == "planner_reviewer":
        converged = bool(state.approved) and state.review_round <= state.max_review_rounds
    else:
        converged = code["results"]["g7_convergence"]["passed"]
    judge = judge_plan(state, fx) if use_judge else {"scores": {}, "avg": 0.0}
    if mode == "planner_only":
        reliability = {
            "applicable": False, "false_approval": False, "false_rejection": False,
        }
        rebuttal = {"transitions": 0, "rebuttal_rate": 0.0, "ignore_rate": 0.0}
    else:
        reliability = {
            "applicable": True,
            **reviewer_reliability(state, objective_pass),
        }
        rebuttal = (
            planner_rebuttal(state, objective_pass)
            if use_judge else
            {"transitions": 0, "rebuttal_rate": 0.0, "ignore_rate": 0.0}
        )
    return {
        "code": code,
        "judge": judge,
        "overall_pass": objective_pass and converged,
        "rounds": state.review_round,
        "node_calls": _node_call_counts(state, mode),
        "reliability": reliability,
        "rebuttal": rebuttal,
        "error": None,
        "_state": {
            "route": state.route,
            "approved": state.approved,
            "dialogue": state.planner_reviewer_dialogue,
            "time_violations": state.time_violations,
        },
    }


def _node_call_counts(state: Any, mode: str) -> dict[str, int]:
    dialogue = list(state.planner_reviewer_dialogue or [])
    return {
        "planner": sum(1 for line in dialogue if "] Planner" in line),
        "reviewer": sum(1 for line in dialogue if "] Reviewer" in line),
        "time_check": (
            int(state.time_check_round or 0)
            if mode == "planner_reviewer_time_check" else 0
        ),
    }


def _failed_ablation_trial(exc: Exception) -> dict[str, Any]:
    trial = _failed_trial(exc)
    return {
        "code": trial["code"],
        "judge": trial["judge"],
        "overall_pass": False,
        "rounds": 0,
        "node_calls": {"planner": 0, "reviewer": 0, "time_check": 0},
        "reliability": {"applicable": False, "false_approval": False, "false_rejection": False},
        "rebuttal": {"transitions": 0, "rebuttal_rate": 0.0, "ignore_rate": 0.0},
        "error": trial["error"],
        "_state": {"error": trial["error"]},
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return round(ordered[index], 3)


def aggregate_mode(
    mode: str,
    trials: list[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    passes = sum(bool(trial["overall_pass"]) for trial in trials)
    by_case: dict[str, list[bool]] = {}
    for trial in trials:
        by_case.setdefault(str(trial.get("case_id") or "unknown"), []).append(
            bool(trial["overall_pass"])
        )
    judge_scores = {
        key: _mean([
            float(trial["judge"]["scores"][key])
            for trial in trials
            if key in trial["judge"].get("scores", {})
        ])
        for key in JUDGE_KEYS
    }
    node_calls = {
        node: sum(int(trial["node_calls"].get(node, 0)) for trial in trials)
        for node in ("planner", "reviewer", "time_check")
    }
    runs = len(trials)
    applicable_reliability = [
        trial["reliability"] for trial in trials
        if trial.get("reliability", {}).get("applicable")
    ]
    elapsed = [float(trial.get("elapsed_ms") or 0) for trial in trials]
    return {
        "mode": mode,
        "runs": runs,
        "passed_runs": passes,
        "pass_rate": round(passes / runs, 3) if runs else 0.0,
        "pass_at_k_rate": _mean([
            1.0 if any(outcomes) else 0.0 for outcomes in by_case.values()
        ]),
        "pass_pow_k_rate": _mean([
            1.0 if all(outcomes) else 0.0 for outcomes in by_case.values()
        ]),
        "failed_runs": sum(1 for trial in trials if trial.get("error")),
        "judge_avg": judge_scores,
        "judge_overall": _mean([value for value in judge_scores.values() if value]),
        "review_rounds_mean": _mean([float(trial["rounds"]) for trial in trials]),
        "false_approval_rate": _mean([
            1.0 if item["false_approval"] else 0.0 for item in applicable_reliability
        ]),
        "false_rejection_rate": _mean([
            1.0 if item["false_rejection"] else 0.0 for item in applicable_reliability
        ]),
        "rebuttal_rate": _mean([
            float(trial.get("rebuttal", {}).get("rebuttal_rate", 0.0)) for trial in trials
        ]),
        "ignore_rate": _mean([
            float(trial.get("rebuttal", {}).get("ignore_rate", 0.0)) for trial in trials
        ]),
        "node_calls": node_calls,
        "node_calls_per_run": round(sum(node_calls.values()) / runs, 3) if runs else 0.0,
        "llm_usage": usage,
        "tokens_per_run": round(float(usage.get("total_tokens", 0)) / runs, 1) if runs else 0.0,
        "latency_ms_per_run": round(float(usage.get("latency_sum_ms", 0)) / runs, 3) if runs else 0.0,
        "end_to_end_latency_ms": {
            "p50": _percentile(elapsed, 0.50),
            "p95": _percentile(elapsed, 0.95),
            "max": round(max(elapsed), 3) if elapsed else 0.0,
        },
    }


def render_report(results: list[dict[str, Any]], budget: dict[str, Any]) -> str:
    lines = [
        "# Online Agent Ablation",
        "",
        "> Same fixtures and provider configuration across modes. Provider failures count as failed runs.",
        "",
        "| Mode | Runs | Pass | pass@k | pass^k | Judge | False approve | Node calls/run | Tokens/run | E2E P50/P95 | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['mode']} | {row['runs']} | {row['pass_rate']:.1%} | "
            f"{row['pass_at_k_rate']:.1%} | {row['pass_pow_k_rate']:.1%} | "
            f"{row['judge_overall']:.2f} | {row['false_approval_rate']:.1%} | "
            f"{row['node_calls_per_run']:.2f} | {row['tokens_per_run']:.1f} | "
            f"{row['end_to_end_latency_ms']['p50']:.1f}/{row['end_to_end_latency_ms']['p95']:.1f} ms | "
            f"{row['failed_runs']} |"
        )
    lines.extend([
        "",
        f"- Provider-attempt budget ceiling: {budget['llm_calls_upper_bound']}",
        "- Token counts remain labelled estimates when structured provider responses omit usage metadata.",
        "- This report does not claim statistical stability unless each case is repeated at least five times.",
        "",
    ])
    return "\n".join(lines)


def _configuration() -> dict[str, Any]:
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    model_key = {
        "grok": "GROK_MODEL",
        "deepseek": "DEEPSEEK_MODEL",
        "doubao": "DOUBAO_MODEL",
        "openai": "OPENAI_MODEL",
    }.get(provider, "OPENAI_MODEL")
    fingerprint = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    for relative in ("app/planning/nodes.py", "app/planning/prompts.py"):
        fingerprint.update((root / relative).read_bytes())
    return {
        "provider": provider,
        "model": os.getenv(model_key, "unknown"),
        "prompt_code_fingerprint": fingerprint.hexdigest()[:16],
        "python": platform.python_version(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="repeats per case and mode")
    parser.add_argument("--only", help="run one fixture id")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--modes", nargs="+", choices=EVALUATION_MODES,
        default=list(EVALUATION_MODES),
    )
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-external-calls", action="store_true")
    parser.add_argument("--max-llm-calls", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not 1 <= args.k <= 10:
        parser.error("--k must be between 1 and 10")

    fixtures = load_fixtures(args.only)
    if args.max_cases is not None:
        fixtures = fixtures[:args.max_cases]
    if not fixtures:
        parser.error("no matching fixture")
    budget = estimate_online_ablation_calls(
        fixtures,
        trials_per_case=args.k,
        modes=args.modes,
        use_judge=not args.no_judge,
    )
    print(json.dumps({"preflight": budget}, ensure_ascii=False))
    if args.dry_run:
        return 0
    require_external_calls(
        parser, allowed=args.allow_external_calls, operation="online agent ablation",
    )
    require_call_budget(
        parser,
        estimated_upper_bound=budget["llm_calls_upper_bound"],
        maximum=args.max_llm_calls,
    )

    results: list[dict[str, Any]] = []
    raw: dict[str, list[dict[str, Any]]] = {}
    for mode in args.modes:
        mode_trials: list[dict[str, Any]] = []
        before = llm_usage_snapshot()
        for fx in fixtures:
            for trial_no in range(1, args.k + 1):
                started = time.perf_counter()
                try:
                    trial = run_trial(fx, mode, not args.no_judge)
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    trial = _failed_ablation_trial(exc)
                trial["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
                trial["case_id"] = fx["id"]
                trial["trial"] = trial_no
                mode_trials.append(trial)
        usage = usage_delta(before, llm_usage_snapshot())
        raw[mode] = mode_trials
        results.append(aggregate_mode(mode, mode_trials, usage))

    execution = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": _configuration(),
        "budget": budget,
        "cases": [fx["id"] for fx in fixtures],
        "repeats": args.k,
    }
    report = render_report(results, budget)
    print("\n" + report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {"execution": execution, "results": results, "trials": raw},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
