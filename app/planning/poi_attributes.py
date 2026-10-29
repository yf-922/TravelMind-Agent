"""Evidence-backed, opt-in POI attributes; never classify from names alone.

Catalog entries are author-extracted drafts with checked quotations, not signed
human annotations. They describe visit scope, not live safety/booking guarantees.
No model or HTTP request is made on the planning path.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "knowledge/poi_attributes_v1/catalog.json"
SOURCE = "official_poi_attributes_v1"
logger = logging.getLogger(__name__)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str
    collected_at: date
    quote: str = Field(min_length=5, max_length=1200)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = Field(min_length=1, max_length=160)
    evidence: Evidence


class Attributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    city: str
    names: list[str] = Field(min_length=1, max_length=5)
    poi_id: str | None = None
    location: dict[str, float]
    environment: Literal["indoor_main", "outdoor_main", "mixed", "unknown"]
    environment_evidence: Evidence | None = None
    scope: str = Field(min_length=1, max_length=200)
    themes: list[Claim] = Field(default_factory=list, max_length=4)
    cautions: list[Claim] = Field(default_factory=list, max_length=3)
    max_age_days: int = Field(default=180, ge=1, le=365)
    review_status: Literal["author_extracted_quote_checked"]


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def enabled():
    return os.getenv("POI_ATTRIBUTES_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _check_evidence(evidence):
    path = (ROOT / evidence.snapshot).resolve()
    if not path.is_relative_to((ROOT / "knowledge").resolve()):
        raise ValueError("Attribute evidence must stay inside knowledge/")
    if urlsplit(evidence.url).scheme != "https" or not urlsplit(evidence.url).hostname:
        raise ValueError("Attribute evidence requires an HTTPS source")
    text = path.read_text(encoding="utf-8")
    if text_hash(text) != evidence.sha256 or evidence.quote not in text:
        raise ValueError("Attribute evidence hash/quotation mismatch")


@lru_cache(maxsize=8)
def _read_catalog(path, modified_ns):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = [Attributes.model_validate(r) for r in data["records"]]
    keys, ids = set(), set()
    for record in records:
        if record.id in ids:
            raise ValueError("Duplicate attribute ID")
        ids.add(record.id)
        if not all(math.isfinite(record.location[k]) for k in ("lng", "lat")):
            raise ValueError("Invalid attribute anchor")
        if not (-180 <= record.location["lng"] <= 180 and -90 <= record.location["lat"] <= 90):
            raise ValueError("Invalid attribute coordinates")
        if record.environment != "unknown" and record.environment_evidence is None:
            raise ValueError("Environment label requires evidence")
        for name in record.names:
            key = (record.city, name)
            if key in keys:
                raise ValueError("Ambiguous city/branch attribute binding")
            keys.add(key)
        evidence = [c.evidence for c in record.themes + record.cautions]
        if record.environment_evidence:
            evidence.append(record.environment_evidence)
        for item in evidence:
            _check_evidence(item)
    return records


def load_catalog(path=CATALOG):
    path = Path(path)
    records = _read_catalog(str(path.resolve()), path.stat().st_mtime_ns)
    # Cached schema parsing must not hide changed/corrupted evidence files.
    for record in records:
        for evidence in ([record.environment_evidence] if record.environment_evidence else []) + [c.evidence for c in record.themes + record.cautions]:
            _check_evidence(evidence)
    return records


def _fresh(evidence, record, today):
    return 0 <= (today - evidence.collected_at).days <= record.max_age_days


def _matches(record, city, poi):
    if record.city != city or poi.get("name") not in record.names:
        return False
    identity = poi.get("poi_id") or poi.get("id")
    if identity and record.poi_id:
        return identity == record.poi_id
    # Old snapshots lack IDs: require exact city+name AND nearby coordinates.
    loc = poi.get("location") or {}
    try:
        lat, lng = float(loc["lat"]), float(loc["lng"])
        if not math.isfinite(lat) or not math.isfinite(lng):
            return False
        dy = (lat - record.location["lat"]) * 111.2
        dx = (lng - record.location["lng"]) * 111.2 * math.cos(math.radians(lat))
        return math.hypot(dx, dy) <= 0.2
    except (KeyError, TypeError, ValueError):
        return False


def enrich_pois(pois, city, *, use_attributes=None, catalog=None, today=None):
    """Idempotent overlay; preserve candidate identity, ordering and live facts.

    Clear our previous overlay first, including on expiry, failure or branch
    mismatch. A refreshed request cannot retain an old indoor certification.
    """
    output = deepcopy(pois)
    for poi in output:
        poi.pop("planning_attributes", None)
        if poi.get("indoor_source") == SOURCE:
            poi.update(indoor=None, indoor_source="unknown")
            poi.pop("indoor_scope", None)
    if not (enabled() if use_attributes is None else use_attributes):
        return output
    try:
        records = catalog if catalog is not None else load_catalog()
    except (OSError, ValueError, KeyError):
        logger.warning("POI attribute catalog unavailable; stale overlays cleared")
        return output
    today = today or date.today()
    for poi in output:
        record = next((r for r in records if _matches(r, city, poi)), None)
        if record is None:
            continue
        env = record.environment
        env_ok = record.environment_evidence is not None and _fresh(record.environment_evidence, record, today)
        themes = [c.model_dump(mode="json") for c in record.themes if _fresh(c.evidence, record, today)]
        cautions = [c.model_dump(mode="json") for c in record.cautions if _fresh(c.evidence, record, today)]
        if not env_ok and not themes and not cautions:
            continue
        poi["planning_attributes"] = {"id": record.id, "environment": env if env_ok else "unknown",
            "scope": record.scope, "themes": themes, "cautions": cautions,
            "environment_evidence": record.environment_evidence.model_dump(mode="json") if env_ok else None,
            "review_status": record.review_status}
        if env_ok:
            poi.update(indoor={"indoor_main": True, "outdoor_main": False}.get(env),
                       indoor_source=SOURCE, indoor_scope=record.scope)
    return output


def attribute_description(poi):
    data = poi.get("planning_attributes")
    if not data:
        return ""
    facts = []
    for key, label in (("themes", "主题"), ("cautions", "注意")):
        for claim in data.get(key, []):
            evidence = claim["evidence"]
            facts.append(f"{label}={claim['value']} [依据:{evidence['url']}；采集:{evidence['collected_at']}；原文:{evidence['quote']}]")
    evidence = data.get("environment_evidence")
    if evidence:
        facts.insert(0, f"环境={data['environment']}；适用范围={data['scope']} [依据:{evidence['url']}；采集:{evidence['collected_at']}；原文:{evidence['quote']}]")
    return "；".join(facts)
