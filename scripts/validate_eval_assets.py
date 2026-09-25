"""Validate the 30-case evaluation manifest without calling external services."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.eval.generate_fixtures import FIXTURES_DIR, SPECS
from app.core.eval_safety import estimate_planner_reviewer_calls

REQUIRED_SPEC_FIELDS = {
    "id", "tier", "destination", "days", "start", "pref", "habit",
    "weather", "pool", "day_temp", "night_temp", "outdoor_max", "query",
}
REQUIRED_FIXTURE_FIELDS = {
    "id", "tier", "destination", "query", "travel_start_date",
    "travel_end_date", "days", "pois", "weather_forecast", "expectations", "provenance",
}
ALLOWED_WEATHER = {"sunny", "single_rain", "all_rain", "beyond"}
ALLOWED_POOLS = {"full", "outdoor_only", "top_4", "top_5", "tight_hours"}


def validate_manifest(specs: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [str(spec.get("id") or "") for spec in specs]
    if len(specs) != 30:
        errors.append(f"expected 30 specs, found {len(specs)}")
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate ids: {duplicates}")
    for index, spec in enumerate(specs):
        missing = sorted(REQUIRED_SPEC_FIELDS - spec.keys())
        prefix = str(spec.get("id") or f"index-{index}")
        if missing:
            errors.append(f"{prefix}: missing fields {missing}")
        try:
            date.fromisoformat(str(spec.get("start") or ""))
        except ValueError:
            errors.append(f"{prefix}: invalid start date")
        if not isinstance(spec.get("days"), int) or not 1 <= spec["days"] <= 14:
            errors.append(f"{prefix}: days must be between 1 and 14")
        if spec.get("weather") not in ALLOWED_WEATHER:
            errors.append(f"{prefix}: unsupported weather scenario")
        if spec.get("pool") not in ALLOWED_POOLS:
            errors.append(f"{prefix}: unsupported pool scenario")
    if len({spec.get("destination") for spec in specs}) < 4:
        errors.append("manifest must cover at least four destinations")
    if not any(spec.get("pool") != "full" for spec in specs):
        errors.append("manifest has no constrained-pool negative case")
    if not any(spec.get("weather") == "all_rain" for spec in specs):
        errors.append("manifest has no all-rain case")
    return errors


def inspect_fixtures(fixtures_dir: Path, expected_ids: set[str]) -> dict[str, Any]:
    found_ids: set[str] = set()
    errors: list[str] = []
    captured_at_values: set[str] = set()
    poi_sources: set[str] = set()
    total_poi_rows = 0
    for path in sorted(fixtures_dir.glob("*.json")) if fixtures_dir.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: invalid JSON ({type(exc).__name__})")
            continue
        missing = sorted(REQUIRED_FIXTURE_FIELDS - payload.keys())
        fixture_id = str(payload.get("id") or path.stem)
        if missing:
            errors.append(f"{path.name}: missing fields {missing}")
        if fixture_id != path.stem:
            errors.append(f"{path.name}: id does not match filename")
        provenance = payload.get("provenance") or {}
        if provenance.get("poi_source") != "AMap Web Service Place Text API v3":
            errors.append(f"{path.name}: POI provenance is missing or unsupported")
        if not provenance.get("poi_captured_at"):
            errors.append(f"{path.name}: POI capture timestamp is missing")
        else:
            try:
                datetime.fromisoformat(str(provenance["poi_captured_at"]))
                captured_at_values.add(str(provenance["poi_captured_at"]))
            except ValueError:
                errors.append(f"{path.name}: invalid POI capture timestamp")
        if provenance.get("poi_source"):
            poi_sources.add(str(provenance["poi_source"]))
        pois = payload.get("pois") or []
        total_poi_rows += len(pois)
        if not pois:
            errors.append(f"{path.name}: POI pool is empty")
        names: set[str] = set()
        for index, poi in enumerate(pois):
            name = str(poi.get("name") or "").strip()
            if not name:
                errors.append(f"{path.name}: POI {index} has no name")
            if name in names:
                errors.append(f"{path.name}: duplicate POI name {name}")
            names.add(name)
            location = poi.get("location")
            if not isinstance(location, dict) or not all(
                isinstance(location.get(axis), (int, float)) for axis in ("lat", "lng")
            ):
                errors.append(f"{path.name}: POI {index} has invalid coordinates")
            if not isinstance(poi.get("indoor"), bool):
                errors.append(f"{path.name}: POI {index} has no boolean indoor label")
            rating = poi.get("rating")
            threshold = float(provenance.get("poi_rating_threshold", payload.get("min_rating", 4.5)))
            if not isinstance(rating, (int, float)) or float(rating) < threshold:
                errors.append(f"{path.name}: POI {index} is below the declared rating threshold")
        found_ids.add(fixture_id)
    return {
        "found": len(found_ids & expected_ids),
        "expected": len(expected_ids),
        "missing_ids": sorted(expected_ids - found_ids),
        "unexpected_ids": sorted(found_ids - expected_ids),
        "errors": errors,
        "execution_ready": found_ids == expected_ids and not errors,
        "provenance": {
            "poi_sources": sorted(poi_sources),
            "capture_batches": len(captured_at_values),
            "captured_at": sorted(captured_at_values),
            "total_poi_rows_across_cases": total_poi_rows,
            "weather_source": "synthetic frozen evaluation scenario",
        },
    }


def build_report(fixtures_dir: Path = FIXTURES_DIR) -> dict[str, Any]:
    manifest_errors = validate_manifest(SPECS)
    expected_ids = {str(spec["id"]) for spec in SPECS}
    return {
        "manifest": {
            "valid": not manifest_errors,
            "cases": len(SPECS),
            "errors": manifest_errors,
            "tiers": dict(sorted(Counter(str(spec["tier"]) for spec in SPECS).items())),
            "destinations": dict(sorted(Counter(str(spec["destination"]) for spec in SPECS).items())),
            "weather_scenarios": dict(sorted(Counter(str(spec["weather"]) for spec in SPECS).items())),
            "pool_scenarios": dict(sorted(Counter(str(spec["pool"]) for spec in SPECS).items())),
        },
        "fixtures": inspect_fixtures(fixtures_dir, expected_ids),
        "online_eval_budget": {
            "one_trial_without_judge_provider_attempts": estimate_planner_reviewer_calls(
                SPECS, trials_per_case=1, use_judge=False, include_time_check=True
            )["provider_attempts_upper_bound"],
            "one_trial_with_judge_provider_attempts": estimate_planner_reviewer_calls(
                SPECS, trials_per_case=1, use_judge=True, include_time_check=True
            )["provider_attempts_upper_bound"],
            "five_trials_with_judge_provider_attempts": estimate_planner_reviewer_calls(
                SPECS, trials_per_case=5, use_judge=True, include_time_check=True
            )["provider_attempts_upper_bound"],
            "type": "conservative_upper_bound",
        },
        "validator_external_calls_made": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-fixtures",
        action="store_true",
        help="fail unless all generated fixture JSON files are present and valid",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report to a file")
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, ensure_ascii=True, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["manifest"]["valid"]:
        return 1
    if args.require_fixtures and not report["fixtures"]["execution_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
