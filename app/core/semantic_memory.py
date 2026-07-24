"""Local, user-isolated semantic memory for travel preferences.

The editable SQLite profile remains the source of truth. This module only stores
stable context that is useful when a later request is phrased differently.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

from app.llm.factory import build_structured_llm
from app.planning.helpers import invoke_structured
from app.planning.schemas import SemanticMemoryExtraction

logger = logging.getLogger(__name__)

_MEMORY_DIR = Path(__file__).resolve().parents[2] / "data" / "semantic_memory"
_COLLECTION = "travel_preferences"
_MAX_MEMORY_LENGTH = 240

_EXTRACTION_SYSTEM = """You extract durable travel preferences for a personalized travel planner.
Only save facts explicitly stated by the user that can help a future trip, such as
mobility needs, food restrictions, transport preferences, budget style, companions,
or a request to avoid repeated destinations. Do not save dates, one-off scheduling
details, guesses, private identifiers, or a complete itinerary. Return [] if there
is no durable preference. Each fact must be concise Chinese and written as a user preference."""


def _enabled() -> bool:
    return os.getenv("SEMANTIC_MEMORY_ENABLED", "1").strip().lower() not in {"0", "false", "no"}


def _collection():
    """Open Chroma lazily so a missing optional dependency never blocks planning."""
    if not _enabled():
        return None
    try:
        import chromadb

        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(_MEMORY_DIR))
        return client.get_or_create_collection(name=_COLLECTION)
    except Exception:
        logger.warning("[semantic_memory] vector store unavailable", exc_info=True)
        return None


def search_user_memories(user_id: str, query: str, limit: int = 3) -> list[str]:
    """Return only memories owned by this user; failure degrades to no memory."""
    if not user_id or not query.strip():
        return []
    collection = _collection()
    if collection is None:
        return []
    try:
        result = collection.query(
            query_texts=[query],
            n_results=max(1, min(limit, 5)),
            where={"user_id": user_id},
            include=["documents"],
        )
        docs = (result.get("documents") or [[]])[0] or []
        return [str(doc) for doc in docs if str(doc).strip()][:limit]
    except Exception:
        logger.warning("[semantic_memory] retrieval failed user=%s", user_id, exc_info=True)
        return []


def format_semantic_memories(memories: list[str]) -> str:
    if not memories:
        return ""
    return "AI 长期记忆（仅作参考，当前用户明确需求优先）：" + "；".join(memories)


def _clean_memories(memories: list[str]) -> list[str]:
    seen: set[str] = set()
    clean: list[str] = []
    for memory in memories:
        item = str(memory).strip().replace("\n", " ")[:_MAX_MEMORY_LENGTH]
        if item and item not in seen:
            seen.add(item)
            clean.append(item)
    return clean[:3]


def _sync_store_user_memories(user_id: str, raw_query: str, model_name: str | None) -> None:
    collection = _collection()
    if collection is None or not user_id or not raw_query.strip():
        return
    llm = build_structured_llm(SemanticMemoryExtraction, model=model_name, temperature=0)
    extracted = invoke_structured(llm, [("system", _EXTRACTION_SYSTEM), ("human", raw_query)])
    memories = _clean_memories(extracted.memories)
    if not memories:
        return
    collection.upsert(
        ids=[str(uuid.uuid4()) for _ in memories],
        documents=memories,
        metadatas=[{"user_id": user_id, "source": "user_request"} for _ in memories],
    )
    logger.info("[semantic_memory] stored=%d user=%s", len(memories), user_id)


async def run_semantic_memory_update(user_id: str, raw_query: str, model_name: str | None = None) -> None:
    """Keep extraction off the request path; a failure must not affect planning."""
    try:
        await asyncio.to_thread(_sync_store_user_memories, user_id, raw_query, model_name)
    except Exception:
        logger.warning("[semantic_memory] update failed user=%s", user_id, exc_info=True)
