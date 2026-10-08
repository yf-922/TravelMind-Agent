"""Annotation, freeze and corpus invariants shared by evaluation commands."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

CITIES = ("南京", "上海", "重庆", "丽江", "三亚", "景德镇")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_corpus(rows, require_quota=False):
    errors = []
    ids = [r.get("chunk_id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate chunk IDs")
    hashes = [r.get("content_hash") for r in rows]
    if len(hashes) != len(set(hashes)):
        errors.append("duplicate content hashes")
    required = {"chunk_id", "text", "url", "city", "topic", "collected_at", "content_hash", "source_type"}
    for row in rows:
        if required - row.keys():
            errors.append(f"{row.get('chunk_id')}: missing provenance")
        if hashlib.sha256(row.get("text", "").encode()).hexdigest() != row.get("content_hash"):
            errors.append(f"{row.get('chunk_id')}: content hash mismatch")
    counts = Counter(r.get("city") for r in rows)
    if require_quota:
        if not 120 <= len(rows) <= 180:
            errors.append("corpus must contain 120..180 chunks")
        errors.extend(f"{city}: fewer than 20 chunks" for city in CITIES if counts[city] < 20)
    return errors


def validate_annotations(cases, rows, require_review=False, require_quota=True):
    errors = []
    ids = [c.get("id") for c in cases]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case IDs")
    groups = {}
    chunks = {r["chunk_id"] for r in rows}
    for case in cases:
        groups.setdefault(case.get("intent_group"), set()).add(case.get("split"))
        labels = case.get("relevant_chunks", [])
        if len({x['chunk_id'] for x in labels}) != len(labels):
            errors.append(f"{case['id']}: duplicate relevance labels")
        if any(x['chunk_id'] not in chunks or x.get('relevance') not in (0, 1, 2) for x in labels):
            errors.append(f"{case['id']}: invalid relevance labels")
        if case.get('type') == 'answerable' and not any(x.get('relevance',0)>0 for x in labels):
            errors.append(f"{case['id']}: answerable without evidence")
        if require_review and (case.get("annotation_status") != "human_reviewed" or not case.get("annotator") or not case.get("reviewed_at") or not case.get("change_log")):
            errors.append(f"{case['id']}: missing signed human review")
    if any(len(splits) != 1 for splits in groups.values()):
        errors.append("intent group leaks across splits")
    if require_quota:
        if Counter(c.get('type') for c in cases) != Counter(answerable=90, in_corpus_missing=15, out_of_corpus=15):
            errors.append("expected 90 answerable + 15 missing + 15 outside")
        if Counter(c.get('split') for c in cases) != Counter(dev=60, test=60):
            errors.append("expected dev60/test60")
    return errors


def freeze_payload(cases, rows, config):
    from app.evaluation.rag_retrieval import embedding_config
    errors = validate_corpus(rows, True) + validate_annotations(cases, rows, True)
    if errors:
        raise ValueError("; ".join(errors))
    return {"schema_version": 1, "corpus": fingerprint(rows), "annotations": fingerprint(cases),
            "configuration": fingerprint(config), "config": config, "embedding": embedding_config(), "status": "frozen"}


def verify_freeze(path: Path, cases, rows, config):
    frozen = json.loads(path.read_text(encoding="utf-8"))
    expected = freeze_payload(cases, rows, config)
    if frozen != expected:
        raise ValueError("frozen corpus, annotations or configuration changed")
    return frozen
