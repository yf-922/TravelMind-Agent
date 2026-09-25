"""Evaluate TravelMind RAG retrieval without model or external API calls."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import travel_knowledge  # noqa: E402

CITATION_RE = re.compile(r"\[source:\s*([^#\]\s]+)#([^\]\s]+)\]")


def validate_citations(text: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = {(str(row["source"]), str(row["chunk_id"])) for row in retrieved}
    citations = CITATION_RE.findall(text or "")
    valid = [citation for citation in citations if citation in allowed]
    return {
        "citations": len(citations),
        "valid_citations": len(valid),
        "validity_rate": round(len(valid) / len(citations), 3) if citations else 0.0,
        "has_citation": bool(citations),
    }


def evaluate(cases: list[dict[str, Any]], top_k: int = 3, retriever: str = "keyword") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    effective_modes: set[str] = set()
    for case in cases:
        retrieved = travel_knowledge.search_travel_knowledge(case["query"], limit=top_k, mode=retriever)
        ranked_sources = [str(row["source"]) for row in retrieved]
        expected = set(case["expected_sources"])
        generation_only = case.get("expected_behavior") in {"abstain", "clarify"}
        answerable = bool(expected)
        first_rank = next((index + 1 for index, source in enumerate(ranked_sources) if source in expected), None)
        rendered = "\n".join(
            f"[source: {row['source']}#{row['chunk_id']}]\n{row['text']}" for row in retrieved
        )
        citation = validate_citations(rendered, retrieved)
        effective_mode = (retrieved[0].get("retrieval_mode") if retrieved else retriever)
        effective_modes.add(str(effective_mode))
        rows.append({
            "id": case["id"],
            "difficulty": case["difficulty"],
            "expected_sources": sorted(expected),
            "retrieved_sources": ranked_sources,
            "retrieval_mode": effective_mode,
            "answerable": answerable,
            "expected_behavior": case.get("expected_behavior", "answer" if answerable else "empty_retrieval"),
            "missing_fact": case.get("missing_fact"),
            "hit_at_1": first_rank == 1,
            "hit_at_k": first_rank is not None,
            "correct_rejection": not answerable and not generation_only and not ranked_sources,
            "false_positive": not answerable and not generation_only and bool(ranked_sources),
            "case_passed": None if generation_only else (first_rank is not None if answerable else not ranked_sources),
            "reciprocal_rank": round(1 / first_rank, 3) if first_rank else 0.0,
            "citation_render_integrity": citation["validity_rate"],
        })

    count = len(rows)
    answerable_rows = [row for row in rows if row["answerable"]]
    no_answer_rows = [row for row in rows if row["expected_behavior"] == "empty_retrieval"]
    generation_only_rows = [row for row in rows if row["case_passed"] is None]
    graded_rows = [row for row in rows if row["case_passed"] is not None]
    answered_rows = [row for row in rows if row["retrieved_sources"]]
    difficulty = {
        label: {"total": len(subset), "graded": sum(row["case_passed"] is not None for row in subset),
                "passed": sum(row["case_passed"] is True for row in subset)}
        for label in sorted(Counter(row["difficulty"] for row in rows))
        for subset in [[row for row in rows if row["difficulty"] == label]]
    }
    return {
        "schema_version": 1,
        "measurement_type": "deterministic-offline-retrieval",
        "retriever_requested": retriever,
        "retriever": next(iter(effective_modes)) if len(effective_modes) == 1 else "mixed",
        "top_k": top_k,
        "case_count": count,
        "answerable_cases": len(answerable_rows),
        "no_answer_cases": len(no_answer_rows),
        "generation_only_cases": len(generation_only_rows),
        "by_difficulty": difficulty,
        "metrics": {
            "hit_at_1": round(sum(row["hit_at_1"] for row in answerable_rows) / len(answerable_rows), 3) if answerable_rows else 0.0,
            "recall_at_k": round(sum(row["hit_at_k"] for row in answerable_rows) / len(answerable_rows), 3) if answerable_rows else 0.0,
            "mrr": round(mean(row["reciprocal_rank"] for row in answerable_rows), 3) if answerable_rows else 0.0,
            "no_answer_accuracy": round(sum(row["correct_rejection"] for row in no_answer_rows) / len(no_answer_rows), 3) if no_answer_rows else 0.0,
            "false_positive_rate": round(sum(row["false_positive"] for row in no_answer_rows) / len(no_answer_rows), 3) if no_answer_rows else 0.0,
            "case_accuracy": round(sum(row["case_passed"] for row in graded_rows) / len(graded_rows), 3) if graded_rows else None,
            "generation_only_context_rate": round(sum(bool(row["retrieved_sources"]) for row in generation_only_rows) / len(generation_only_rows), 3) if generation_only_rows else None,
            "no_result_rate": round(sum(not row["retrieved_sources"] for row in rows) / count, 3) if count else 0.0,
            "citation_render_integrity": round(mean(row["citation_render_integrity"] for row in answered_rows), 3) if answered_rows else 0.0,
        },
        "cases": rows,
        "boundary": "No LLM generation is graded. In-domain unknowns need generated-answer abstention or clarification checks and are ungraded here. Citation render integrity checks the evaluator's own formatting, not model faithfulness.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Offline RAG Evaluation",
        "",
        f"> Deterministic {report['retriever']} retrieval only. This is not an online LLM answer-quality score.",
        "",
        f"- Cases: {report['case_count']}",
        f"- Answerable / empty-retrieval / generation-only: {report['answerable_cases']} / {report['no_answer_cases']} / {report['generation_only_cases']}",
        f"- Hit@1: {metrics['hit_at_1']:.1%}",
        f"- Recall@{report['top_k']}: {metrics['recall_at_k']:.1%}",
        f"- MRR: {metrics['mrr']:.3f}",
        f"- Empty-retrieval negative accuracy: {metrics['no_answer_accuracy']:.1%}",
        f"- False-positive rate: {metrics['false_positive_rate']:.1%}",
        f"- Graded case accuracy: {metrics['case_accuracy']:.1%}" if metrics['case_accuracy'] is not None else "- Graded case accuracy: not measured",
        f"- In-domain unknowns receiving context: {metrics['generation_only_context_rate']:.1%}" if metrics['generation_only_context_rate'] is not None else "- In-domain unknowns receiving context: not measured",
        f"- No-result rate: {metrics['no_result_rate']:.1%}",
        f"- Citation render integrity (not answer faithfulness): {metrics['citation_render_integrity']:.1%}",
        "- By difficulty: " + ", ".join(
            f"{name} {row['passed']}/{row['graded']} graded" if row["graded"] else f"{name} {row['total']} ungraded"
            for name, row in report["by_difficulty"].items()
        ),
        "",
        "| Case | Difficulty | Expected | Retrieved | Pass | RR |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['id']} | {row['difficulty']} | {', '.join(row['expected_sources'])} | "
            f"{', '.join(row['retrieved_sources']) or '-'} | {'ungraded' if row['case_passed'] is None else ('yes' if row['case_passed'] else 'no')} | "
            f"{row['reciprocal_rank']:.3f} |"
        )
    lines.extend(["", f"Boundary: {report['boundary']}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation" / "rag_golden_set.json")
    parser.add_argument("--include-hard-negatives", action="store_true", help="Append in-domain unknown questions; answer abstention remains ungraded without LLM output")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retriever",
        choices=("keyword", "hybrid", "auto"),
        default="keyword",
        help="Retrieval mode. keyword is deterministic and disables vector-store dependency.",
    )
    parser.add_argument("--json-out", type=Path, default=ROOT / "evaluation" / "rag_report.json")
    parser.add_argument("--markdown-out", type=Path, default=ROOT / "evaluation" / "rag_report.md")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 5:
        parser.error("--top-k must be between 1 and 5")

    # Keep the default evaluation dependency-free. Explicit hybrid/auto runs
    # may use an already configured local Chroma index, but are never required
    # for the deterministic gate below.
    if args.retriever == "keyword":
        os.environ["TRAVEL_KNOWLEDGE_ENABLED"] = "0"
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.include_hard_negatives:
        cases.extend(json.loads((ROOT / "evaluation" / "rag_hard_negative_set.json").read_text(encoding="utf-8")))
    report = evaluate(cases, args.top_k, args.retriever)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["metrics"], sort_keys=True))
    return 0 if (
        report["metrics"]["recall_at_k"] >= 0.90
        and report["metrics"]["no_answer_accuracy"] >= 0.90
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
