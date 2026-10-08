"""Versioned retrieval benchmark independent from the production RAG path.

The benchmark deliberately fails closed for ``vector``/``hybrid``. A keyword
fallback is useful in production, but would make a hybrid experiment look
successful when the embedding service was never available.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "knowledge" / "benchmark_v1"
INDEX_DIR = ROOT / "data" / "travel_knowledge_benchmark_v1"


def load_chunks() -> list[dict[str, Any]]:
    rows = json.loads((CORPUS_DIR / "chunks.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("benchmark corpus is empty")
    return rows


def corpus_fingerprint(rows: list[dict[str, Any]] | None = None) -> str:
    rows = rows or load_chunks()
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _keyword(query: str, rows: list[dict[str, Any]], limit: int = 20):
    from app.core.travel_knowledge import _keyword_score
    ranked = [{**row, "score": _keyword_score(query, row["text"])} for row in rows]
    return sorted(ranked, key=lambda x: (-x["score"], x["chunk_id"]))[:limit]


def _bm25(query: str, rows: list[dict[str, Any]], limit: int = 20):
    tokenized = [list(jieba.cut(row["text"], cut_all=False)) for row in rows]
    scores = BM25Okapi(tokenized).get_scores(list(jieba.cut(query, cut_all=False)))
    ranked = [{**row, "score": float(score)} for row, score in zip(rows, scores)]
    return sorted(ranked, key=lambda x: (-x["score"], x["chunk_id"]))[:limit]


def _collection(rows: list[dict[str, Any]], rebuild: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    name = "benchmark_" + corpus_fingerprint(rows)[:16]
    collection = client.get_or_create_collection(name=name, metadata={"corpus_fingerprint": corpus_fingerprint(rows)})
    if rebuild or collection.count() != len(rows):
        old = collection.get(include=[]).get("ids") or []
        if old:
            collection.delete(ids=old)
        collection.upsert(
            ids=[row["chunk_id"] for row in rows],
            documents=[row["text"] for row in rows],
            metadatas=[{k: str(row.get(k, "")) for k in ("chunk_id", "source", "url", "city", "topic", "source_type", "evidence_group")} for row in rows],
        )
    return collection


def _vector(query: str, rows: list[dict[str, Any]], limit: int = 20):
    collection = _collection(rows)
    result = collection.query(query_texts=[query], n_results=min(limit, len(rows)), include=["documents", "metadatas", "distances"])
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    by_id = {row["chunk_id"]: row for row in rows}
    output = []
    for text, meta, distance in zip(docs, metas, distances):
        chunk_id = str((meta or {}).get("chunk_id") or "")
        # Chroma metadata is intentionally supplemented by the versioned rows.
        row = dict(by_id.get(chunk_id) or {"chunk_id": chunk_id, "text": text, **(meta or {})})
        row.update({"distance": float(distance) if distance is not None else None})
        output.append(row)
    if not output:
        raise RuntimeError("vector retrieval returned no rows")
    return output


def rrf(vector_rows, lexical_rows, limit: int, c: int = 60):
    fused = {}
    for channel, rows in (("vector", vector_rows), ("lexical", lexical_rows)):
        for rank, row in enumerate(rows, 1):
            item = fused.setdefault(row["chunk_id"], {**row, "rrf_score": 0.0, "channels": [], "vector_rank": None, "lexical_rank": None})
            item["rrf_score"] += 1.0 / (c + rank)
            item[f"{channel}_rank"] = rank
            if channel not in item["channels"]:
                item["channels"].append(channel)
    return sorted(fused.values(), key=lambda x: (-x["rrf_score"], x["chunk_id"]))[:limit]


def search(query: str, mode: str, limit: int, rrf_c: int = 60, rows=None):
    rows = rows or load_chunks()
    limit = max(1, min(int(limit), len(rows)))
    if mode == "keyword":
        return _keyword(query, rows, limit)
    if mode == "bm25":
        return _bm25(query, rows, limit)
    if mode == "vector":
        return _vector(query, rows, limit)
    if mode in {"hybrid", "keyword_rrf", "legacy_hybrid"}:
        # Explicit hybrid is strict: vector failures are surfaced to the report.
        vector = _vector(query, rows, 20)
        lexical = _bm25(query, rows, 20) if mode == "hybrid" else _keyword(query, rows, 20)
        if mode == "legacy_hybrid":
            from app.core.travel_knowledge import _rrf_fuse
            lexical = [{**x, "keyword_score": x["score"]} for x in lexical]
            return _rrf_fuse(vector, lexical, limit, rrf_c)
        return rrf(vector, lexical, limit, rrf_c)
    raise ValueError(f"unknown retrieval mode: {mode}")
