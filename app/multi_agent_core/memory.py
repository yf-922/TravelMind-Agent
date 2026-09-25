"""Session-scoped, Agent-isolated memory stores for the orchestration core."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol


MemoryEntry = dict[str, str]


class AgentMemoryStore(Protocol):
    def load(self, session_id: str, agent_name: str) -> list[MemoryEntry]: ...
    def append(self, session_id: str, agent_name: str, entry: MemoryEntry) -> None: ...


class InMemoryAgentMemoryStore:
    """Useful for unit tests and a process-local demo."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], list[MemoryEntry]] = {}

    def load(self, session_id: str, agent_name: str) -> list[MemoryEntry]:
        return list(self._data.get((session_id, agent_name), []))

    def append(self, session_id: str, agent_name: str, entry: MemoryEntry) -> None:
        self._data.setdefault((session_id, agent_name), []).append(dict(entry))


class SQLiteAgentMemoryStore:
    """Durable memory isolated by session and Agent name."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else Path(__file__).resolve().parents[2] / "data" / "multi_agent_memory.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memories (
                    session_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    PRIMARY KEY (session_id, agent_name, position)
                )
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), timeout=5)

    def load(self, session_id: str, agent_name: str) -> list[MemoryEntry]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content FROM agent_memories WHERE session_id=? AND agent_name=? ORDER BY position",
                (session_id, agent_name),
            ).fetchall()
        return [{"role": str(role), "content": str(content)} for role, content in rows]

    def append(self, session_id: str, agent_name: str, entry: MemoryEntry) -> None:
        with closing(self._connect()) as conn:
            position = conn.execute(
                "SELECT COUNT(*) FROM agent_memories WHERE session_id=? AND agent_name=?",
                (session_id, agent_name),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO agent_memories(session_id, agent_name, position, role, content) VALUES (?,?,?,?,?)",
                (session_id, agent_name, position, entry["role"], entry["content"]),
            )
            conn.commit()
