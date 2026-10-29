"""Evidence and identity contracts for the opt-in attribute overlay."""
from copy import deepcopy
from datetime import date
import json

import pytest

from app.planning import poi_attributes as pa
from app.planning.schemas import TravelPlanState
from scripts import evaluate_poi_attributes as evaluation

TODAY = date(2026, 10, 30)


@pytest.fixture
def museum():
    record = next(r for r in pa.load_catalog() if r.names[0] == "南京博物院")
    return record, {"name": record.names[0], "location": dict(record.location),
                    "indoor": None, "ticket_price": 17, "opening_hours": "09:00-17:00"}


def enrich(poi, record, **kw):
    return pa.enrich_pois([poi], record.city, catalog=[record], use_attributes=True,
                          today=kw.pop("today", TODAY), **kw)[0]


def test_evidence_overlay_preserves_live_facts_and_input(museum):
    record, poi = museum
    before = deepcopy(poi)
    result = enrich(poi, record)
    assert poi == before
    assert result["indoor"] is True
    for field in ("name", "location", "ticket_price", "opening_hours"):
        assert result[field] == poi[field]
    assert "不含" in result["indoor_scope"]
    assert "https://" in pa.attribute_description(result)
    assert "原文:" in pa.attribute_description(result)


@pytest.mark.parametrize("change", ["other_city", "other_branch", "far", "no_coordinates", "nan"])
def test_no_fuzzy_or_cross_branch_match(museum, change):
    record, poi = museum
    city = record.city
    if change == "other_city": city = "上海"
    elif change == "other_branch": poi["name"] += "分馆"
    elif change == "far": poi["location"]["lng"] += 1
    elif change == "no_coordinates": poi.pop("location")
    else: poi["location"]["lat"] = float("nan")
    result = pa.enrich_pois([poi], city, catalog=[record], use_attributes=True, today=TODAY)
    assert "planning_attributes" not in result[0]


def test_conflicting_known_id_cannot_use_coordinate_fallback(museum):
    record, poi = museum
    record = record.model_copy(update={"poi_id": "expected"})
    poi["poi_id"] = "different"
    assert "planning_attributes" not in enrich(poi, record)
    poi["poi_id"] = "expected"
    assert enrich(poi, record)["indoor"] is True


@pytest.mark.parametrize("reason", ["disabled", "expired", "future", "mismatch", "unavailable"])
def test_stale_overlay_cleared(museum, monkeypatch, reason):
    record, poi = museum
    enriched = enrich(poi, record)
    kwargs = {"catalog": [record], "use_attributes": True, "today": TODAY}
    if reason == "disabled": kwargs["use_attributes"] = False
    elif reason == "expired": kwargs["today"] = date(2028, 1, 1)
    elif reason == "future": kwargs["today"] = date(2020, 1, 1)
    elif reason == "mismatch": enriched["name"] += "另一馆区"
    else:
        kwargs.pop("catalog")
        def fail(): raise ValueError("corrupt catalog")
        monkeypatch.setattr(pa, "load_catalog", fail)
    result = pa.enrich_pois([enriched], record.city, **kwargs)[0]
    assert "planning_attributes" not in result
    assert result["indoor"] is None
    assert "indoor_scope" not in result


def test_mixed_is_never_indoor_certification():
    record = next(r for r in pa.load_catalog() if r.names[0] == "上海豫园")
    poi = {"name": record.names[0], "location": record.location, "indoor": True}
    assert enrich(poi, record)["indoor"] is None


def test_unknown_does_not_invent_environment():
    record = next(r for r in pa.load_catalog() if r.names[0] == "中山陵景区")
    result = enrich({"name": record.names[0], "location": record.location, "indoor": None}, record)
    assert result["indoor"] is None
    assert result["planning_attributes"]["environment"] == "unknown"


@pytest.mark.parametrize("failure", ["quote", "hash", "path"])
def test_evidence_rejected(failure, museum):
    record, _ = museum
    evidence = record.environment_evidence.model_copy()
    if failure == "quote": evidence.quote = "a nonexistent source statement"
    elif failure == "hash": evidence.sha256 = "0" * 64
    else: evidence.snapshot = "../escape.txt"
    with pytest.raises(ValueError): pa._check_evidence(evidence)


def test_cached_catalog_does_not_hide_evidence_corruption(monkeypatch, museum):
    pa.load_catalog()
    def corrupt(evidence): raise ValueError("changed snapshot")
    monkeypatch.setattr(pa, "_check_evidence", corrupt)
    with pytest.raises(ValueError): pa.load_catalog()


def test_default_disabled_and_idempotent(museum, monkeypatch):
    record, poi = museum
    monkeypatch.delenv("POI_ATTRIBUTES_ENABLED", raising=False)
    assert pa.enrich_pois([poi], record.city) == [poi]
    result = enrich(poi, record)
    assert enrich(result, record) == result


def test_case_assets_and_saved_maps():
    cases = json.loads((evaluation.ASSETS / "cases.json").read_text(encoding="utf-8"))
    maps = json.loads((pa.ROOT / "knowledge/route_interest_v1/maps.json").read_text(encoding="utf-8"))["routes"]
    evaluation.validate_cases(cases, maps)
    assert cases == evaluation.build_cases()
    assert sum(c["rubric"]["must_reject"] for c in cases) == 2
    assert all(c["annotation_status"] == "author_draft_not_human_verified" for c in cases)


def test_scorer_uses_final_timeline_not_notes():
    case = evaluation.build_cases()[0]
    final = TravelPlanState(query=case["query"], final_plan={"approved": True, "days": [], "notes": "南京博物院 09:00-11:00"})
    result = evaluation.grade(final, case)
    assert not result["task_success"]
    assert result["selected"] == []
    assert "empty_route" in result["violations"]


def test_failed_execution_never_counts_as_safe_rejection():
    case = next(c for c in evaluation.build_cases() if c["rubric"]["must_reject"])
    final = TravelPlanState(query=case["query"], final_plan={"approved": False, "route_issues": ["无法生成满足要求的路线"]})
    assert evaluation.grade(final, case)["expected_outcome_correct"]
    assert not evaluation.grade(final, case, "TimeoutError")["expected_outcome_correct"]


def test_failed_trials_and_missing_tokens_retained_in_denominator():
    case = evaluation.build_cases()[0]
    row = {"arm": "with_attributes", "case_id": case["id"], "repetition": 0,
           "metrics": evaluation.grade(TravelPlanState(query=case["query"]), case, "TimeoutError"),
           "error": "TimeoutError", "usage_known": False, "elapsed_ms": 10, "model_calls": 1}
    summary = evaluation.summarize([row])["arms"]["with_attributes"]
    assert summary["expected_outcome"]["denominator"] == 1
    assert summary["execution_failures"]["numerator"] == 1
    assert summary["measured_usage_trials"] == 0


def test_joint_planner_passes_attribute_evidence_to_prompt_and_shared_state(museum, monkeypatch):
    from app.planning import nodes
    from app.planning.schemas import TravelRoute
    record, poi = museum
    monkeypatch.setenv("POI_ATTRIBUTES_ENABLED", "1")
    monkeypatch.setattr(nodes, "build_structured_llm", lambda *a, **kw: object())
    messages = []
    def invoke(llm, prompt):
        messages.extend(prompt)
        return TravelRoute(reasoning="test", days=[])
    planner = nodes.make_joint_planner_node("offline", knowledge_lookup=lambda *a: [], invoke_fn=invoke)
    result = planner(TravelPlanState(query="只去室内", destination=record.city, pois=[poi]))
    assert result["pois"][0]["indoor"] is True
    assert record.environment_evidence.url in str(messages)
    assert "planning_attributes" not in poi


def test_candidate_decision_refreshes_before_gap_check(museum, monkeypatch):
    from app.planning import candidate_react as cr
    record, poi = museum
    monkeypatch.setenv("POI_ATTRIBUTES_ENABLED", "1")
    observed = []
    def coverage(state):
        observed.extend(state.pois)
        return {}, []
    monkeypatch.setattr(cr, "coverage", coverage)
    result = cr.make_candidate_react_node()(TravelPlanState(query="只去室内", destination=record.city, pois=[poi]))
    assert observed[0]["indoor"] is True
    assert result["pois"] == observed
    assert result["candidate_search_actions"] == [{"action": "stop"}]


def test_meter_updates_actual_sdk_transports_and_enforces_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app.llm import factory
    from app.planning.schemas import TravelRoute
    class SDK:
        def __init__(self, retries=2):
            self.max_retries = retries
            self.chat = SimpleNamespace(completions=object())
        def with_options(self, **kw):
            assert kw["timeout"] == 60
            return SDK(kw["max_retries"])
    class Chat:
        root_client, root_async_client = SDK(), SDK()
        def model_copy(self, *, update):
            clone = Chat()
            clone.__dict__.update(self.__dict__)
            clone.__dict__.update(update)
            return clone
        def with_structured_output(self, schema, **kw):
            assert self.root_client.max_retries == self.root_async_client.max_retries == 0
            return SimpleNamespace(invoke=lambda messages: {
                "raw":SimpleNamespace(usage_metadata=None, id="fake"),
                "parsed":TravelRoute(reasoning="offline", days=[]), "parsing_error":None})
    monkeypatch.setattr(factory, "build_chat_llm", lambda **kw: Chat())
    meter = evaluation.Meter(tmp_path, "offline", 1)
    runnable = meter.factory(TravelRoute)
    runnable.invoke([("human", "test")])
    assert meter.calls[0]["input_tokens"] is None
    assert meter.calls[0]["output_tokens"] is None
    with pytest.raises(ValueError, match="budget"):
        runnable.invoke([("human", "test")])
    assert len(meter.calls) == 1


def test_concurrent_budget_reservations_are_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import sqlite3
    database = tmp_path/"budget.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE budget (used INTEGER, failures INTEGER)")
        db.execute("INSERT INTO budget VALUES (59,0)")
    def attempt(_):
        try: return evaluation.reserve_call(database, 64)
        except ValueError: return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(attempt, range(20)))
    assert sorted(i for i in ids if i is not None) == [60,61,62,63,64]
    evaluation.update_provider_failure(database, True)
    evaluation.update_provider_failure(database, True)
    with pytest.raises(ValueError): evaluation.reserve_call(database, 100)


def test_interrupted_pair_never_counted_as_attribute_improvement():
    case = evaluation.build_cases()[0]
    bad = evaluation.grade(TravelPlanState(query=case["query"]),case,"HarnessInterrupted")
    good = {**bad,"expected_outcome_correct":True,"task_success":True}
    common = {"case_id":case["id"],"repetition":0,"usage_known":False,"elapsed_ms":0,"model_calls":1}
    rows = [{**common,"arm":"without_attributes","metrics":bad,"error":"HarnessInterrupted"},
            {**common,"arm":"with_attributes","metrics":good,"error":None}]
    summary = evaluation.summarize(rows)
    assert summary["paired_trials"] == 1
    assert summary["quality_comparable_pairs"] == 0
    assert summary["improved_pairs"] == 0
    assert summary["arms"]["without_attributes"]["expected_outcome"]["denominator"] == 1
