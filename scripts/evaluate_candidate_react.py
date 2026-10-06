"""Offline candidate-loop contract benchmark with deterministic tool/model doubles."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.planning import candidate_react as cr
from app.planning.schemas import TravelPlanState


def evaluate():
    records = []
    for name, query, fail in [
        ("outdoor", "喜欢户外", False), ("rain", "喜欢户外，雨天室内优先", False),
        ("indoor_only", "只去室内", False), ("explicit", '想去"指定博物馆"', False),
        ("provider_failure", "旅行", True), ("unknown_category", "旅行", False),
    ]:
        state = TravelPlanState(query=query, destination="南京", days=1,
            rain_indoor_priority=name == "rain", weather_forecast=[{"is_bad": name == "rain"}])
        requested = name == "explicit"
        def search(*a, keywords, request_budget, **k):
            request_budget["remaining"] -= 1
            request_budget["used"] += 1
            if fail:
                raise RuntimeError("injected timeout")
            category = "未知类型" if name == "unknown_category" else keywords
            return [{"name": keywords if requested else ("地点" if name == "unknown_category" else keywords) + str(i), "type": category,
                     "location": "118,32", "biz_ext": {"rating": "5"}} for i in range(4)]
        with patch.object(cr, "get_cached", return_value=None), patch.object(cr, "set_cached"), \
             patch.object(cr, "amap_key", return_value="offline"), patch.object(cr, "search_city_pois", side_effect=search):
            for round_no in range(2):
                if requested:
                    action, keyword = "search_user_requested_poi", "指定博物馆"
                elif name == "indoor_only" or round_no == 1:
                    action, keyword = "search_indoor", "博物馆"
                else:
                    action, keyword = "search_outdoor", "公园"
                state.candidate_search_actions = [{"action": action, "query": keyword}]
                state = state.model_copy(update=cr.candidate_search_node(state))
                state = state.model_copy(update=cr.candidate_validator_node(state))
                if state.candidate_pool_status != "searching":
                    break
        records.append({"case": name, "status": state.candidate_pool_status,
            "coverage": state.candidate_coverage, "missing": state.candidate_missing_coverage,
            "rounds": state.candidate_search_round, "api_attempts": state.candidate_api_calls,
            "provenance_violations": sum(p.get("source") != "amap" for p in state.pois)})
    return {"scope": "offline deterministic policy/tool contracts; no real LLM quality measured",
            "cases": records, "ready_cases": sum(r["status"] == "ready" for r in records),
            "average_rounds": sum(r["rounds"] for r in records) / len(records),
            "total_api_attempts": sum(r["api_attempts"] for r in records),
            "insufficient_cases": sum(r["status"] == "insufficient" for r in records)}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
