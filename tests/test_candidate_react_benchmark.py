"""Independent scoring, replay accounting, negative probes and asset contracts."""
from copy import deepcopy
import json

import pytest

from scripts import evaluate_candidate_react as bench
from app.planning import candidate_react as cr
from app.planning.schemas import TravelPlanState


@pytest.fixture(scope="module")
def assets():
    snapshot = json.loads((bench.ASSETS / "poi_snapshot.json").read_text(encoding="utf-8"))
    cases = json.loads((bench.ASSETS / "cases.json").read_text(encoding="utf-8"))
    return snapshot, cases


@pytest.fixture(scope="module")
def report():
    return bench.evaluate()


def test_frozen_manifest_and_capture_provenance(assets):
    snapshot, cases = assets
    bench.validate_assets(snapshot, cases)
    assert len(snapshot["sources"]) == 6
    assert cases == bench.build_cases_for_freeze(snapshot)
    assert all(s.get("captured_at") or s["provenance"].get("poi_captured_at") for s in snapshot["sources"])


@pytest.mark.parametrize("field,value", [("city", "不存在城市"), ("initial_api_calls", 5)])
def test_reject_corrupt_case_config(assets, field, value):
    snapshot, cases = assets
    corrupt = deepcopy(cases)
    corrupt[0][field] = value
    with pytest.raises(ValueError):
        bench.validate_assets(snapshot, corrupt)


def test_reject_missing_named_source_and_modified_base(assets):
    snapshot, cases = assets
    corrupt = deepcopy(cases)
    corrupt[0]["required_names"] = ["无证据的名称"]
    with pytest.raises(ValueError, match="evidence"):
        bench.validate_assets(snapshot, corrupt)
    corrupt = deepcopy(cases)
    corrupt[0]["base"][0]["location"]["lng"] = 0
    with pytest.raises(ValueError, match="provenance"):
        bench.validate_assets(snapshot, corrupt)


@pytest.mark.parametrize("index", range(28))
def test_paired_invariants_and_real_provider_accounting(report, index):
    row = report["cases"][index]
    assert row["error"] is None
    assert row["initial_state_unchanged"]
    assert row["unique_names"] and row["provenance_ok"]
    assert row["rounds"] <= 2
    assert row["total_http_attempts"] <= 4
    assert row["total_http_attempts"] == row["initial_http_attempts"] + len(row["http"])
    assert len(row["http"]) == row["supplemental_http_attempts"]
    assert row["duplicate_blocks"] == row["repeat_opportunities"]
    if row["group"] == "boundary":
        assert row["guard_pass"]
    if row["group"] == "named":
        assert set(row["requirements"]["required_names"]) <= set(row["final_names"])


def test_score_is_independent_of_production_validator(monkeypatch):
    monkeypatch.setattr(cr, "coverage", lambda *a: (_ for _ in ()).throw(AssertionError("coupled scorer")))
    case = {"required_names": ["A"], "min_count": 2, "min_indoor": 1}
    pool = [{"name": "A", "indoor": True}, {"name": "B", "indoor": False}]
    assert bench.score(case, pool)["pass"]
    assert not bench.score(case, pool[:1])["pass"]
    assert not bench.score(case, [{"name": "A", "indoor_hint": True}, pool[1]])["pass"]


def test_duplicate_names_and_unknown_indoor_do_not_inflate_quality():
    case = {"required_names": [], "min_count": 2, "min_indoor": 1}
    pool = [{"name": "A", "indoor_hint": True}, {"name": "A", "indoor_hint": True}]
    score = bench.score(case, pool)
    assert score["count"] == 1 and score["indoor"] == 0
    assert not score["pass"]


def test_type_whitelist_repairs_indoor_gaps_without_llm_labels(report):
    rows = [r for r in report["cases"] if r["group"] == "indoor"]
    for row in rows:
        assert row["bounded_react"]["indoor"] >= row["baseline"]["indoor"]
    assert sum(r["bounded_react"]["pass"] for r in rows) == 7
    assert {r["id"] for r in rows if not r["bounded_react"]["pass"]} == {"indoor-3-1","indoor-3-2"}


def test_nonexistent_and_near_name_cannot_pollute_pool(report):
    rows = {r["id"]: r for r in report["cases"]}
    for ident in ("nonexistent-repeat", "near-name-reject"):
        row = rows[ident]
        assert row["final_names"] == row["initial_names"]
        assert row["bounded_react"]["missing_names"] == row["requirements"]["required_names"]
        assert row["supplemental_http_attempts"] == 2
        assert row["duplicate_blocks"] == row["repeat_opportunities"] == 0
        assert len({h["query"] for h in row["http"]}) == 2
    assert rows["near-name-reject"]["http"][0]["raw_count"] == 1


def test_round_limit_and_retry_budget_are_independent(report):
    rows = {r["id"]: r for r in report["cases"]}
    assert rows["round-limit"]["rounds"] == 2
    assert rows["round-limit"]["supplemental_http_attempts"] == 4
    assert rows["retry-budget"]["rounds"] == 1
    assert rows["retry-budget"]["supplemental_http_attempts"] == 4
    assert rows["retry-budget"]["search_failures"] == 2
    assert all(h["outcome"] == "injected_timeout" for h in rows["retry-budget"]["http"])
    assert rows["shared-budget"]["initial_http_attempts"] == 3
    assert rows["shared-budget"]["supplemental_http_attempts"] == 1
    assert rows["shared-budget"]["total_http_attempts"] == 4


def test_previously_searched_empty_query_is_not_retried(assets, monkeypatch):
    snapshot, cases = assets
    case = next(c for c in cases if c["id"] == "nonexistent-repeat")
    replay = bench.Replay(case, snapshot)
    from app.providers.amap import poi as provider
    monkeypatch.setattr(cr, "get_cached", lambda *a: None)
    monkeypatch.setattr(cr, "amap_key", lambda: "offline")
    monkeypatch.setattr(provider, "http_get_json", replay.get)
    state = TravelPlanState(query=case["query"], destination=case["city"],
        candidate_search_actions=[{"action": "search_user_requested_poi", "query": case["required_names"][0]}])
    state = state.model_copy(update=cr.candidate_search_node(state))
    # Production bootstrap resets only round number, preserving trace and budget.
    state.candidate_search_round = 0
    state = state.model_copy(update=cr.candidate_search_node(state))
    assert state.candidate_api_calls == 1 and len(replay.http) == 1
    assert state.candidate_search_trace[-1]["error_code"] == "REPEATED_QUERY"


def test_unmapped_replay_cannot_be_hidden_by_production_exception_handler(assets, monkeypatch):
    snapshot, cases = assets
    monkeypatch.setattr(bench.Replay, "get", lambda self, url, **kw: (
        self.mapping_errors.append("injected missing mapping") or {"status":"1","pois":[]}))
    # Deliberately bypass action policy for this transport-mapping fault probe.
    monkeypatch.setattr(cr, "valid_actions", lambda state, actions: actions)
    with pytest.raises(AssertionError, match="mapping defect"):
        bench.run_pair(cases[0], snapshot)


def test_graph_execution_failure_stays_in_denominator(assets, monkeypatch):
    snapshot, cases = assets
    monkeypatch.setattr(cr, "candidate_validator_node", lambda *a: (_ for _ in ()).throw(RuntimeError("injected graph error")))
    record = bench.run_pair(cases[0], snapshot)
    assert "injected graph error" in record["error"]
    assert not record["bounded_react"]["pass"]
    assert record["supplemental_http_attempts"] == len(record["http"]) == 1
    assert record["stop_reason"] == "execution_error"
    summary = bench.summarize([record])
    assert summary["execution_failures"] == bench.fraction(1, 1)
    assert summary["non_boundary_coverage"]["bounded_react"] == bench.fraction(0, 1)


def test_score_counters_do_not_drop_failed_rows_or_conflate_guard_success(report):
    summary = report["summary"]
    assert summary["non_boundary_coverage"]["bounded_react"]["denominator"] == 23
    assert summary["boundary_safe_incomplete"]["denominator"] == 5
    # Named gaps now use the deterministic exact/core queue: no duplicate
    # model proposal is exercised here. The direct guard probe above remains.
    assert summary["repeat_query_guard"] == bench.fraction(0, 0)
    assert summary["baseline_prerequisite_blocked"]["numerator"] == 23
    assert sum(summary["blocker_counts"]["baseline"].values()) == 23
    assert set(summary["fixture_limited_cases_even_trusting_hints"]) == {"indoor-3-1", "indoor-3-2", "count-3"}


def test_determinism(assets, report):
    snapshot, cases = assets
    for index in (0, 10, 20, 23, 26):
        assert bench.run_pair(cases[index], snapshot) == report["cases"][index]


def test_empty_denominator_is_not_reported_as_perfect():
    assert bench.fraction(0, 0) == {"numerator": 0, "denominator": 0, "percent": None}
