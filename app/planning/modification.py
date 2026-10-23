"""Conservative candidate refresh shared by the modification API and offline probes."""
from __future__ import annotations

import logging
from typing import Any, Callable

from app.planning.helpers import amap_key, merge_verified_poi, missing_explicit_places
from app.providers.amap.poi import ATTRACTION_TYPE, poi_to_spot, search_city_pois

logger = logging.getLogger(__name__)


def refresh_explicit_modification_candidates(
    checkpoint: dict[str, Any], notes: str, *,
    lookup: Callable = search_city_pois, key_provider: Callable = amap_key,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Look up at most three missing quoted names; preserve the original checkpoint.

    Unquoted category requests remain the Planner's responsibility. A provider
    result is merged only after exact-name matching and coordinate validation.
    Trace entries describe lookup outcomes, not low-level HTTP retry counts.
    """
    trace = []
    missing = missing_explicit_places(notes, list(checkpoint.get('pois') or []))
    for requested_name in missing[:3]:
        destination = str(checkpoint.get('destination') or '').strip()
        if not destination:
            trace.append({'name': requested_name, 'status': 'missing_destination'})
            break
        try:
            raw_pois = lookup(destination, key_provider(), keywords=requested_name,
                              types=ATTRACTION_TYPE, offset=8)
        except RuntimeError as exc:
            logger.warning('[modification] targeted POI lookup failed name=%s error=%s', requested_name, exc)
            trace.append({'name': requested_name, 'status': 'provider_failed'})
            continue
        verified = next(
            (poi_to_spot(raw) for raw in raw_pois
             if str(raw.get('name') or '').strip() == requested_name), None,
        )
        if verified:
            checkpoint = {**checkpoint, 'pois': merge_verified_poi(
                list(checkpoint.get('pois') or []), verified)}
            notes += f"\n【系统已核验新增景点】{verified['name']}，已加入候选池并参与本轮重规划。"
        trace.append({'name': requested_name, 'status': 'added' if verified else 'not_verified'})
    return checkpoint, notes, trace
