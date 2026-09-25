"""Low-cardinality metrics for deterministic audit routing."""

from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Any

_lock = Lock()
_decisions: Counter[str] = Counter()
_flags: Counter[str] = Counter()


def record(decision: str, flags: list[str]) -> None:
    with _lock:
        _decisions[decision] += 1
        _flags.update(flags)


def snapshot() -> dict[str, Any]:
    with _lock:
        return {"decisions": dict(_decisions), "flags": dict(_flags)}


def prometheus_lines() -> list[str]:
    data = snapshot()
    lines = [
        "# HELP travelmind_audit_gate_decisions_total Deterministic audit routing decisions.",
        "# TYPE travelmind_audit_gate_decisions_total counter",
    ]
    for decision, count in sorted(data["decisions"].items()):
        lines.append(f'travelmind_audit_gate_decisions_total{{decision="{decision}"}} {count}')
    lines.extend([
        "# HELP travelmind_audit_gate_flags_total Deterministic risk flags observed.",
        "# TYPE travelmind_audit_gate_flags_total counter",
    ])
    for flag, count in sorted(data["flags"].items()):
        lines.append(f'travelmind_audit_gate_flags_total{{flag="{flag}"}} {count}')
    return lines


def reset_for_tests() -> None:
    with _lock:
        _decisions.clear()
        _flags.clear()
