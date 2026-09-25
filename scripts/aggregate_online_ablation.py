"""Aggregate comparable online-ablation JSON runs without hiding failures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def aggregate(paths: list[Path]) -> dict[str, Any]:
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    configurations = [document["execution"]["configuration"] for document in documents]
    comparable_keys = ("provider", "model", "prompt_code_fingerprint")
    fingerprints = {
        tuple(configuration.get(key) for key in comparable_keys)
        for configuration in configurations
    }
    if len(fingerprints) != 1:
        raise ValueError("input reports do not share provider, model and prompt fingerprint")

    trials_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cases: list[str] = []
    for document in documents:
        cases.extend(str(case) for case in document["execution"].get("cases", []))
        for mode, trials in document.get("trials", {}).items():
            trials_by_mode[mode].extend(trials)

    modes: list[dict[str, Any]] = []
    for mode, trials in trials_by_mode.items():
        runs = len(trials)
        passed = sum(bool(trial.get("overall_pass")) for trial in trials)
        judge = [float(trial.get("judge", {}).get("avg") or 0) for trial in trials]
        elapsed = [float(trial.get("elapsed_ms") or 0) for trial in trials]
        modes.append({
            "mode": mode,
            "runs": runs,
            "passed": passed,
            "pass_rate": round(passed / runs, 3) if runs else 0.0,
            "judge_mean": round(mean(judge), 3) if judge else 0.0,
            "estimated_tokens_total": sum(
                int(trial.get("usage", {}).get("total_tokens") or 0) for trial in trials
            ),
            "elapsed_mean_ms": round(mean(elapsed), 3) if elapsed else 0.0,
            "failed_runs": sum(bool(trial.get("error")) for trial in trials),
            "node_calls_mean": round(mean(
                sum(int(value) for value in trial.get("node_calls", {}).values())
                for trial in trials
            ), 3) if trials else 0.0,
        })

    # Usage is aggregated per mode in each source report, not copied into each
    # raw trial. Reconstruct only mode totals from the source summaries.
    summary_by_mode: dict[str, dict[str, float]] = defaultdict(lambda: {
        "tokens": 0.0, "latency": 0.0, "calls": 0.0,
    })
    for document in documents:
        for row in document.get("results", []):
            summary = summary_by_mode[str(row["mode"])]
            usage = row.get("llm_usage") or {}
            summary["tokens"] += float(usage.get("total_tokens") or 0)
            summary["latency"] += float(usage.get("latency_sum_ms") or 0)
            summary["calls"] += float(usage.get("calls_total") or 0)
    for row in modes:
        usage = summary_by_mode[row["mode"]]
        row["estimated_tokens_total"] = int(usage["tokens"])
        row["estimated_tokens_mean"] = round(usage["tokens"] / row["runs"], 1)
        row["provider_latency_mean_ms"] = round(usage["latency"] / row["runs"], 3)
        row["llm_attempts"] = int(usage["calls"])

    order = {
        "planner_only": 0,
        "planner_reviewer": 1,
        "planner_reviewer_time_check": 2,
    }
    modes.sort(key=lambda row: order.get(row["mode"], 99))
    baseline = next((row for row in modes if row["mode"] == "planner_only"), None)
    full = next((row for row in modes if row["mode"] == "planner_reviewer_time_check"), None)
    conclusion = "insufficient_data"
    if baseline and full:
        if full["pass_rate"] > baseline["pass_rate"] or full["judge_mean"] > baseline["judge_mean"]:
            conclusion = "quality_gain_with_cost_tradeoff"
        elif full["pass_rate"] == baseline["pass_rate"] and full["judge_mean"] <= baseline["judge_mean"]:
            conclusion = "no_observed_multi_agent_quality_gain"
    return {
        "measurement_type": "real-provider-online-ablation",
        "configuration": configurations[0],
        "cases": list(dict.fromkeys(cases)),
        "repeats_per_case": 1,
        "statistically_stable": False,
        "modes": modes,
        "conclusion": conclusion,
        "boundary": (
            "One run per case. Token values are estimates because provider usage metadata "
            "was unavailable; no statistical significance is claimed."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real Single-Agent vs Multi-Agent Ablation",
        "",
        f"> Provider: {report['configuration']['provider']} / {report['configuration']['model']}; "
        f"prompt fingerprint: `{report['configuration']['prompt_code_fingerprint']}`.",
        "",
        "| Mode | Runs | Pass | Judge | LLM attempts | Est. tokens/run | E2E mean | Provider latency/run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["modes"]:
        lines.append(
            f"| {row['mode']} | {row['runs']} | {row['pass_rate']:.1%} | "
            f"{row['judge_mean']:.2f} | {row['llm_attempts']} | "
            f"{row['estimated_tokens_mean']:.1f} | {row['elapsed_mean_ms'] / 1000:.1f}s | "
            f"{row['provider_latency_mean_ms'] / 1000:.1f}s |"
        )
    lines.extend([
        "",
        f"- Cases: {', '.join(report['cases'])}",
        f"- Machine conclusion: `{report['conclusion']}`.",
        "- All provider/runtime failures remain in the denominator.",
        f"- Boundary: {report['boundary']}",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.inputs)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"conclusion": report["conclusion"], "modes": report["modes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
