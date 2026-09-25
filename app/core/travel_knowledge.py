"""Destination knowledge RAG used as a read-only tool by the Planner Agent."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "knowledge" / "travel"
_STORE_DIR = _ROOT / "data" / "travel_knowledge"
_COLLECTION = "travel_knowledge_v1"
_CHUNK_SIZE = 420
_CHUNK_OVERLAP = 80


def _keyword_score(query: str, text: str) -> float:
    """Deterministic Chinese reranker using character and bigram overlap.

    Character-only overlap overweights generic words such as ``安排`` and
    ``景点``. Bigrams retain short Chinese phrases such as ``开放时间`` and
    ``历史偏好`` without requiring a tokenizer.
    """
    normalized_query = "".join(char.lower() for char in query if char.isalnum())
    normalized_text = "".join(char.lower() for char in text if char.isalnum())
    query_chars = set(normalized_query)
    if not query_chars:
        return 0.0
    unigram = len(query_chars & set(normalized_text)) / len(query_chars)
    query_bigrams = {normalized_query[index:index + 2] for index in range(len(normalized_query) - 1)}
    if not query_bigrams:
        return unigram
    text_bigrams = {normalized_text[index:index + 2] for index in range(len(normalized_text) - 1)}
    bigram = len(query_bigrams & text_bigrams) / len(query_bigrams)
    return 0.25 * unigram + 0.75 * bigram


def _collection():
    """Open the persistent Chroma collection lazily."""
    if os.getenv("TRAVEL_KNOWLEDGE_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        return None
    try:
        import chromadb

        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(_STORE_DIR))
        return client.get_or_create_collection(name=_COLLECTION)
    except Exception:
        logger.warning("[travel_rag] vector store unavailable", exc_info=True)
        return None


def _split_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split by paragraph first, then preserve a small overlap for context."""
    paragraphs = [part.strip() for part in text.replace("\r\n", "\n").split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > size:
            chunks.append(current)
            # Keep whole paragraph context instead of cutting a sentence mid-word.
            previous = current.split("\n\n")[-1]
            current = f"{previous}\n\n{paragraph}"
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def load_documents() -> list[dict[str, str]]:
    """Load the version-controlled Markdown knowledge sources."""
    documents: list[dict[str, str]] = []
    for path in sorted(_SOURCE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({"source": path.stem, "text": text})
    return documents


def build_index(rebuild: bool = False) -> int:
    """Load, split, embed, and persist all project travel documents in Chroma."""
    collection = _collection()
    if collection is None:
        return 0
    if rebuild:
        # Delete through Chroma instead of removing the SQLite directory. The
        # latter fails on Windows when the running API process holds the file.
        existing_ids = collection.get(include=[]).get("ids") or []
        if existing_ids:
            collection.delete(ids=existing_ids)
    if not rebuild and collection.count() > 0:
        return collection.count()

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    documents = load_documents()
    for document in documents:
        for index, chunk in enumerate(_split_text(document["text"])):
            chunk_id = f"{document['source']}-{index}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({"source": document["source"], "chunk_id": chunk_id})
    if ids:
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    logger.info("[travel_rag] indexed=%d chunks from %d documents", len(ids), len(documents))
    return len(ids)


def _all_chunks() -> list[dict[str, str]]:
    """Return the deterministic local chunk inventory used by lexical recall."""
    chunks: list[dict[str, str]] = []
    for document in load_documents():
        for index, chunk in enumerate(_split_text(document["text"])):
            chunks.append({
                "source": document["source"],
                "chunk_id": f"{document['source']}-{index}",
                "text": chunk,
            })
    return chunks


def _keyword_candidates(query: str, candidate_limit: int = 10) -> list[dict[str, Any]]:
    """Independently retrieve lexical candidates before score fusion."""
    rows: list[dict[str, Any]] = []
    minimum_score = float(os.getenv("RAG_MIN_KEYWORD_SCORE", "0.25"))
    for chunk in _all_chunks():
        score = _keyword_score(query, chunk["text"])
        if score < minimum_score:
            continue
        rows.append({**chunk, "keyword_score": score})
    rows.sort(key=lambda row: (-row["keyword_score"], row["chunk_id"]))
    return rows[:max(1, candidate_limit)]


def _rrf_fuse(
    vector_rows: list[dict[str, Any]],
    keyword_rows: list[dict[str, Any]],
    limit: int = 3,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse independently ranked channels with Reciprocal Rank Fusion.

    The function is pure and deliberately keeps provenance (ranks, channels,
    source and chunk id) so retrieval decisions can be inspected in traces.
    """
    fused: dict[str, dict[str, Any]] = {}
    for channel, rows in (("vector", vector_rows), ("keyword", keyword_rows)):
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row.get("chunk_id") or "unknown")
            item = fused.setdefault(chunk_id, {
                "source": str(row.get("source") or "unknown"),
                "chunk_id": chunk_id,
                "text": str(row.get("text") or ""),
                "distance": row.get("distance"),
                "keyword_score": row.get("keyword_score", 0.0),
                "rrf_score": 0.0,
                "vector_rank": None,
                "keyword_rank": None,
                "retrieval_channels": [],
            })
            item["rrf_score"] += 1.0 / (rrf_k + rank)
            item[f"{channel}_rank"] = rank
            if channel not in item["retrieval_channels"]:
                item["retrieval_channels"].append(channel)
            # Keyword scoring is calculated independently, so prefer that
            # value when the vector channel happened to arrive first.
            if row.get("keyword_score") is not None:
                item["keyword_score"] = max(float(item["keyword_score"] or 0.0), float(row["keyword_score"]))
            if item["text"] == "" and row.get("text"):
                item["text"] = str(row["text"])
            if item["distance"] is None and row.get("distance") is not None:
                item["distance"] = row["distance"]

    for item in fused.values():
        item["rrf_score"] = round(item["rrf_score"], 6)
        item["retrieval_channels"] = sorted(item["retrieval_channels"])
    return sorted(
        fused.values(),
        # RRF scores that differ only in the fourth decimal are effectively a
        # tie for this small local corpus; use lexical relevance and then the
        # semantic distance as deterministic tie-breakers.
        key=lambda row: (
            -round(row["rrf_score"], 4),
            -float(row["keyword_score"] or 0.0),
            float(row["distance"] or 0.0),
            row["chunk_id"],
        ),
    )[:max(1, limit)]


def search_travel_knowledge(query: str, limit: int = 3, mode: str = "auto") -> list[dict[str, Any]]:
    """RAG tool with independent lexical/vector recall and optional RRF fusion.

    ``auto`` uses hybrid retrieval when Chroma is available and falls back to
    keyword-only retrieval otherwise. ``keyword`` is useful for deterministic
    offline evaluation; ``hybrid`` explicitly opts into vector recall and
    still degrades safely when Chroma is unavailable.
    """
    if not query.strip():
        return []
    if mode not in {"auto", "hybrid", "keyword"}:
        raise ValueError("mode must be one of: auto, hybrid, keyword")
    requested_limit = max(1, min(limit, 5))
    keyword_rows = _keyword_candidates(query, candidate_limit=requested_limit * 2)
    if mode == "keyword":
        for rank, row in enumerate(keyword_rows[:requested_limit], start=1):
            row.update({
                "distance": None,
                "vector_rank": None,
                "keyword_rank": rank,
                "rrf_score": None,
                "retrieval_channels": ["keyword"],
                "retrieval_mode": "keyword",
            })
        return keyword_rows[:requested_limit]

    collection = _collection()
    if collection is None:
        # Dependency-free fallback for first boot/offline containers. It avoids
        # Chroma's embedding-model download while preserving source labels.
        for rank, row in enumerate(keyword_rows[:requested_limit], start=1):
            row.update({
                "distance": None,
                "vector_rank": None,
                "keyword_rank": rank,
                "rrf_score": None,
                "retrieval_channels": ["keyword"],
                "retrieval_mode": "keyword",
            })
        return keyword_rows[:requested_limit]
    try:
        build_index()
        result = collection.query(
            query_texts=[query],
            n_results=min(10, requested_limit * 2),
            include=["documents", "metadatas", "distances"],
        )
        docs = (result.get("documents") or [[]])[0] or []
        metas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []
        vector_rows: list[dict[str, Any]] = []
        max_distance = float(os.getenv("RAG_MAX_DISTANCE", "1.8"))
        for text, meta, distance in zip(docs, metas, distances):
            if distance is not None and float(distance) > max_distance:
                continue
            metadata = meta or {}
            vector_rows.append({
                "source": str(metadata.get("source") or "unknown"),
                "chunk_id": str(metadata.get("chunk_id") or "unknown"),
                "text": str(text),
                "distance": round(float(distance), 4) if distance is not None else None,
                "keyword_score": _keyword_score(query, str(text)),
            })
        vector_rows = vector_rows[:requested_limit * 2]
        if not keyword_rows:
            # A nearest neighbour always exists, even for an out-of-domain
            # query. Without any lexical support, keep only unusually strong
            # semantic matches instead of forcing an unrelated answer.
            vector_only_limit = float(os.getenv("RAG_MAX_VECTOR_ONLY_DISTANCE", "0.9"))
            vector_rows = [
                row for row in vector_rows
                if row.get("distance") is not None and float(row["distance"]) <= vector_only_limit
            ]
        fused = _rrf_fuse(vector_rows, keyword_rows, limit=requested_limit)
        for row in fused:
            row["retrieval_mode"] = "hybrid"
        return fused
    except Exception:
        logger.warning("[travel_rag] retrieval failed", exc_info=True)
        return []


def search_docs(query: str, limit: int = 3) -> str:
    """Agent tool wrapper returning Top-K passages with mandatory source labels."""
    rows = search_travel_knowledge(query, limit=limit)
    if not rows:
        return "No relevant travel-knowledge passages were found."
    return "\n\n".join(f"[source: {row['source']}#{row['chunk_id']}]\n{row['text']}" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TravelMind RAG knowledge base")
    parser.add_argument("--rebuild", action="store_true", help="Discard the local Chroma index before rebuilding")
    args = parser.parse_args()
    print(f"Indexed {build_index(rebuild=args.rebuild)} chunks.")


if __name__ == "__main__":
    main()
