"""Evaluate user and Agent memory isolation without external services."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import semantic_memory  # noqa: E402
from app.multi_agent_core.memory import SQLiteAgentMemoryStore  # noqa: E402


def _score(query: str, text: str) -> int:
    return len({char for char in query if char.strip()} & set(text))


class DeterministicCollection:
    """Chroma-compatible test double that enforces and records metadata filters."""

    def __init__(self, memories: list[dict[str, str]]):
        self.memories = memories
        self.where_filters: list[dict[str, str]] = []

    def query(self, query_texts, n_results, where, include):
        self.where_filters.append(dict(where))
        owner = where.get("user_id")
        rows = [row for row in self.memories if row["user_id"] == owner]
        rows.sort(key=lambda row: -_score(query_texts[0], row["text"]))
        return {"documents": [[row["text"] for row in rows[:n_results]]]}


def evaluate_semantic_memory(data: dict, top_k: int = 2) -> dict:
    store = DeterministicCollection(data["memories"])
    owner_by_text = {row["text"]: row["user_id"] for row in data["memories"]}
    cases = []
    leakage_count = 0
    with patch.object(semantic_memory, "_collection", return_value=store):
        for case in data["queries"]:
            retrieved = semantic_memory.search_user_memories(case["user_id"], case["query"], top_k)
            leaked = [text for text in retrieved if owner_by_text.get(text) != case["user_id"]]
            leakage_count += len(leaked)
            cases.append({
                "id": case["id"],
                "user_id": case["user_id"],
                "hit_at_k": case["expected"] in retrieved,
                "returned": len(retrieved),
                "leaked_items": len(leaked),
            })
    count = len(cases)
    return {
        "case_count": count,
        "hit_at_k": round(sum(row["hit_at_k"] for row in cases) / count, 3) if count else 0.0,
        "cross_user_leakage_rate": round(leakage_count / sum(row["returned"] for row in cases), 3)
        if sum(row["returned"] for row in cases) else 0.0,
        "metadata_filter_applied_rate": round(
            sum(bool(item.get("user_id")) for item in store.where_filters) / len(store.where_filters), 3
        ) if store.where_filters else 0.0,
        "cases": cases,
    }


def evaluate_agent_memory() -> dict:
    with tempfile.TemporaryDirectory(prefix="travelmind-memory-eval-") as temp_dir:
        store = SQLiteAgentMemoryStore(Path(temp_dir) / "agent_memory.db")
        entries = {
            ("session-a", "planner_agent"): {"role": "assistant", "content": "planner private"},
            ("session-a", "reviewer_agent"): {"role": "assistant", "content": "reviewer private"},
            ("session-b", "planner_agent"): {"role": "assistant", "content": "other session"},
        }
        for (session_id, agent_name), entry in entries.items():
            store.append(session_id, agent_name, entry)

        leakage_checks = []
        for key, expected in entries.items():
            loaded = store.load(*key)
            leakage_checks.append(loaded == [expected])
        empty_boundaries = [
            store.load("session-a", "intent_agent") == [],
            store.load("session-c", "planner_agent") == [],
        ]
    return {
        "boundaries_checked": len(leakage_checks) + len(empty_boundaries),
        "isolation_pass_rate": round(
            sum(leakage_checks + empty_boundaries) / (len(leakage_checks) + len(empty_boundaries)), 3
        ),
    }


def render_markdown(report: dict) -> str:
    semantic = report["semantic_memory"]
    agent = report["agent_memory"]
    return "\n".join([
        "# Offline Memory Evaluation",
        "",
        "> Uses a deterministic Chroma-compatible retrieval double and temporary real SQLite Agent storage.",
        "",
        f"- Semantic cases: {semantic['case_count']}",
        f"- Semantic Hit@{report['top_k']}: {semantic['hit_at_k']:.1%}",
        f"- Cross-user leakage rate: {semantic['cross_user_leakage_rate']:.1%}",
        f"- User metadata filter applied: {semantic['metadata_filter_applied_rate']:.1%}",
        f"- SQLite Agent/session boundaries: {agent['boundaries_checked']}",
        f"- SQLite isolation pass rate: {agent['isolation_pass_rate']:.1%}",
        "",
        "Boundary: this validates production filtering and SQLite isolation contracts, not Chroma embedding quality.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation" / "memory_golden_set.json")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--json-out", type=Path, default=ROOT / "evaluation" / "memory_report.json")
    parser.add_argument("--markdown-out", type=Path, default=ROOT / "evaluation" / "memory_report.md")
    args = parser.parse_args()
    data = json.loads(args.cases.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "measurement_type": "deterministic-memory-isolation",
        "top_k": args.top_k,
        "semantic_memory": evaluate_semantic_memory(data, args.top_k),
        "agent_memory": evaluate_agent_memory(),
    }
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "semantic_hit_at_k": report["semantic_memory"]["hit_at_k"],
        "cross_user_leakage_rate": report["semantic_memory"]["cross_user_leakage_rate"],
        "agent_isolation_pass_rate": report["agent_memory"]["isolation_pass_rate"],
    }, sort_keys=True))
    passed = (
        report["semantic_memory"]["hit_at_k"] >= 0.90
        and report["semantic_memory"]["cross_user_leakage_rate"] == 0
        and report["agent_memory"]["isolation_pass_rate"] == 1
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
