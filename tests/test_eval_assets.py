import json

from scripts.validate_eval_assets import build_report, inspect_fixtures, validate_manifest
from tests.eval.generate_fixtures import SPECS, estimate_amap_attempts, fetch_balanced_city_spots


def test_30_case_manifest_is_valid_and_has_negative_slices():
    assert validate_manifest(SPECS) == []
    report = build_report()
    assert report["manifest"]["cases"] == 30
    assert report["manifest"]["weather_scenarios"]["all_rain"] > 0
    assert report["manifest"]["pool_scenarios"]["full"] < 30
    assert report["online_eval_budget"]["five_trials_with_judge_provider_attempts"] == 6750
    assert report["validator_external_calls_made"] is False


def test_fixture_inspection_reports_missing_and_schema_errors(tmp_path):
    (tmp_path / "case-a.json").write_text(
        json.dumps({"id": "case-a"}), encoding="utf-8"
    )

    result = inspect_fixtures(tmp_path, {"case-a", "case-b"})

    assert result["found"] == 1
    assert result["missing_ids"] == ["case-b"]
    assert result["execution_ready"] is False
    assert "missing fields" in result["errors"][0]


def test_fixture_generation_budget_counts_searches_and_retries():
    assert estimate_amap_attempts(5) == 60


def test_balanced_fixture_fetch_uses_all_channels_and_round_robin(monkeypatch):
    calls = []

    def fake_search(city, key, *, keywords):
        calls.append(keywords)
        suffix = keywords.removeprefix(city)
        return [
            {"name": f"{suffix}-{index}", "location": f"118.{index},32.{index}",
             "biz_ext": {"rating": "4.8"}}
            for index in range(3)
        ]

    monkeypatch.setattr("tests.eval.generate_fixtures.search_attraction_pois", fake_search)
    rows = fetch_balanced_city_spots("南京", "key", max_spots=6)

    assert calls == ["南京必去景点", "南京热门景区", "南京博物馆"]
    assert [row["name"] for row in rows] == [
        "必去景点-0", "热门景区-0", "博物馆-0",
        "必去景点-1", "热门景区-1", "博物馆-1",
    ]
