"""Versioned retrieval benchmark independent from the production RAG path.

The benchmark deliberately fails closed for ``vector``/``hybrid``. A keyword
fallback is useful in production, but would make a hybrid experiment look
successful when the embedding service was never available.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import importlib.metadata
from functools import lru_cache
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "knowledge" / "benchmark_v1"
INDEX_DIR = ROOT / "data" / "travel_knowledge_benchmark_v1"
MODEL = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def embedding_config():
    return {"model": MODEL, "revision": REVISION, "distance": "cosine",
            "normalize": True, "query_prefix": QUERY_PREFIX, "max_tokens": 512}


def lexical_config():
    dictionary = Path(jieba.__file__).parent / 'dict.txt'
    return {'name':'jieba','version':importlib.metadata.version('jieba'),'cut_all':False,
            'dictionary_sha256':hashlib.sha256(dictionary.read_bytes()).hexdigest(),
            'bm25_version':importlib.metadata.version('rank-bm25'),'non_word_tokens_removed':True}


@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    model.max_seq_length = 512
    return model


def token_count(text):
    return len(embedding_model().tokenizer.encode(text, add_special_tokens=True))


def index_fingerprint(rows, config=None):
    payload = {"corpus": corpus_fingerprint(rows), "embedding": config or embedding_config()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_rows(rows):
    ids = [row["chunk_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("corpus must be nonempty with unique chunk IDs")
    if any(not row.get("text", "").strip() for row in rows):
        raise ValueError("empty chunk text")


def load_chunks() -> list[dict[str, Any]]:
    rows = json.loads((CORPUS_DIR / "chunks.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("benchmark corpus is empty")
    return rows


def corpus_fingerprint(rows: list[dict[str, Any]] | None = None) -> str:
    rows = load_chunks() if rows is None else rows
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _keyword(query: str, rows: list[dict[str, Any]], limit: int = 20):
    from app.core.travel_knowledge import _keyword_score
    ranked = [{**row, "score": _keyword_score(query, row["text"])} for row in rows]
    return sorted(ranked, key=lambda x: (-x["score"], x["chunk_id"]))[:limit]


def lexical_tokens(text):
    return [t for t in jieba.cut(text, cut_all=False) if re.search(r"\w", t)]


@lru_cache(maxsize=8)
def _bm25_index(serialized):
    texts = json.loads(serialized)
    return BM25Okapi([lexical_tokens(text) or ["__empty__"] for text in texts])


def _bm25(query: str, rows: list[dict[str, Any]], limit: int = 20):
    index = _bm25_index(json.dumps([r["text"] for r in rows], ensure_ascii=False))
    scores = index.get_scores(lexical_tokens(query))
    ranked = [{**row, "score": float(score)} for row, score in zip(rows, scores)]
    return sorted(ranked, key=lambda x: (-x["score"], x["chunk_id"]))[:limit]


def _collection(rows: list[dict[str, Any]], rebuild: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    name = "bge_" + index_fingerprint(rows)[:24]
    collection = client.get_or_create_collection(name=name, embedding_function=None,
        metadata={"corpus_fingerprint": corpus_fingerprint(rows), "index_fingerprint": index_fingerprint(rows), "hnsw:space": "cosine"})
    expected = {r["chunk_id"]: r["text"] for r in rows}
    stored = collection.get(include=["documents"])
    actual = dict(zip(stored["ids"], stored.get("documents") or []))
    if rebuild or actual != expected:
        if any(token_count(row["text"]) > 512 for row in rows):
            raise ValueError("chunk exceeds embedding tokenizer budget; rechunk instead of truncating")
        old = collection.get(include=[]).get("ids") or []
        if old:
            collection.delete(ids=old)
        collection.upsert(
            ids=[row["chunk_id"] for row in rows],
            documents=[row["text"] for row in rows],
            embeddings=embedding_model().encode([r["text"] for r in rows], normalize_embeddings=True).tolist(),
            metadatas=[{k: str(row.get(k, "")) for k in ("chunk_id", "source", "url", "city", "topic", "source_type", "evidence_group")} for row in rows],
        )
    return collection


def _vector(query: str, rows: list[dict[str, Any]], limit: int = 20):
    collection = _collection(rows)
    if token_count(QUERY_PREFIX + query) > 512:
        raise ValueError("query exceeds embedding token budget")
    embedding = embedding_model().encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=embedding, n_results=min(limit, len(rows)), include=["documents", "metadatas", "distances"])
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    by_id = {row["chunk_id"]: row for row in rows}
    output = []
    for text, meta, distance in zip(docs, metas, distances):
        chunk_id = str((meta or {}).get("chunk_id") or "")
        # Chroma metadata is intentionally supplemented by the versioned rows.
        if chunk_id not in by_id:
            raise RuntimeError("vector index returned an unknown chunk ID")
        row = dict(by_id[chunk_id])
        row.update({"distance": float(distance) if distance is not None else None})
        output.append(row)
    if not output:
        raise RuntimeError("vector retrieval returned no rows")
    return output


def rrf(vector_rows, lexical_rows, limit: int, c: int = 60):
    if c <= 0:
        raise ValueError("rrf c must be positive")
    fused = {}
    for channel, rows in (("vector", vector_rows), ("lexical", lexical_rows)):
        seen = set()
        rank = 0
        for row in rows:
            if row["chunk_id"] in seen:
                continue
            seen.add(row["chunk_id"])
            rank += 1
            item = fused.setdefault(row["chunk_id"], {**row, "rrf_score": 0.0, "channels": [], "vector_rank": None, "lexical_rank": None})
            item["rrf_score"] += 1.0 / (c + rank)
            item[f"{channel}_rank"] = rank
            if channel not in item["channels"]:
                item["channels"].append(channel)
    return sorted(fused.values(), key=lambda x: (-x["rrf_score"], x["chunk_id"]))[:limit]


def search(query: str, mode: str, limit: int, rrf_c: int = 60, rows=None, filters=None):
    rows = load_chunks() if rows is None else rows
    validate_rows(rows)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty")
    if limit < 1:
        raise ValueError("limit must be positive")
    if filters:
        rows = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
        if not rows:
            return []
    limit = max(1, min(int(limit), len(rows)))
    if mode == "keyword":
        return _keyword(query, rows, 20)[:limit]
    if mode == "bm25":
        return _bm25(query, rows, 20)[:limit]
    if mode == "vector":
        return _vector(query, rows, 20)[:limit]
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
