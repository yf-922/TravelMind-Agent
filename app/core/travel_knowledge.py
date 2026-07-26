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
    """Small deterministic reranker for Chinese short queries after vector recall."""
    terms = {char for char in query if char.strip() and char not in "，。！？、：；（）()"}
    if not terms:
        return 0.0
    return len(terms & set(text)) / len(terms)


def _collection():
    """Open the persistent Chroma collection lazily."""
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


def search_travel_knowledge(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """RAG tool: return source-labelled, relevant passages for a travel query."""
    if not query.strip():
        return []
    collection = _collection()
    if collection is None:
        return []
    try:
        build_index()
        result = collection.query(
            query_texts=[query],
            n_results=max(1, min(limit, 5)),
            include=["documents", "metadatas", "distances"],
        )
        docs = (result.get("documents") or [[]])[0] or []
        metas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []
        rows: list[dict[str, Any]] = []
        max_distance = float(os.getenv("RAG_MAX_DISTANCE", "1.8"))
        for text, meta, distance in zip(docs, metas, distances):
            if distance is not None and float(distance) > max_distance:
                continue
            metadata = meta or {}
            rows.append({
                "source": str(metadata.get("source") or "unknown"),
                "chunk_id": str(metadata.get("chunk_id") or "unknown"),
                "text": str(text),
                "distance": round(float(distance), 4) if distance is not None else None,
                "keyword_score": _keyword_score(query, str(text)),
            })
        # Hybrid retrieval: Chroma supplies semantic candidates; lexical overlap
        # reranks concise Chinese questions whose important terms are explicit.
        return sorted(rows, key=lambda row: (-row["keyword_score"], row["distance"] or 0.0))
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
