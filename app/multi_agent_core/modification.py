"""Preference edits and bounded, provider-grounded candidate refresh."""

from pydantic import BaseModel, Field

from app.llm.factory import build_structured_llm
from app.planning.helpers import amap_key, invoke_structured
from app.providers.amap.poi import ATTRACTION_TYPE, poi_to_spot, search_city_pois


class ModificationAnalysis(BaseModel):
    attraction_preference: str | None = None
    food_preference: str | None = None
    habit_preference: str | None = None
    candidate_pool_sufficient: bool = Field(description="现有真实候选池是否足够满足本次景点修改；仅时间或餐饮修改通常不需补查")
    search_keywords: list[str] = Field(default_factory=list, max_length=3,
        description="候选不足时给出最多三个简短地图搜索关键词，不要整段用户问题")
    reasoning: str = Field(max_length=1200)


def make_modification_intent_node(model_name=None):
    llm = build_structured_llm(ModificationAnalysis, model=model_name, temperature=0,
                               task_type="query_rewrite")

    def analyze(state):
        import json
        result = invoke_structured(llm, [
            ("system", "分析旅行修改意见。只更新用户本次明确改变的偏好，未改变字段返回 null。"
             "现有候选足够则复用；新主题或新增类型没有足够候选时输出地图关键词。"
             "不要编造景点，不要把调时间、调餐饮等局部修改误判为需要重新搜索景点。"
             "只分析偏好与候选覆盖，不修改目的地、日期或路线。"),
            ("human", json.dumps({"modification": state.modification_notes,
                "destination": state.destination, "days": state.days,
                "preferences": {"attraction": state.attraction_preference,
                    "food": state.food_preference, "habit": state.habit_preference},
                "candidates": [{key: poi.get(key) for key in ("name", "type", "indoor")}
                               for poi in state.pois]}, ensure_ascii=False)),
        ])
        keywords = list(dict.fromkeys(word.strip()[:80] for word in result.search_keywords if word.strip()))
        if not result.candidate_pool_sufficient and not keywords:
            raise ValueError("Candidate pool is insufficient but no search keywords were supplied")
        update = {"modification_search_keywords": [] if result.candidate_pool_sufficient else keywords,
                  "modification_search_status": "not_required" if result.candidate_pool_sufficient else "pending",
                  "history": ["modification_intent: " + result.reasoning]}
        for field in ("attraction_preference", "food_preference", "habit_preference"):
            value = getattr(result, field)
            if value is not None:
                update[field] = value.strip()
        return update
    return analyze


def candidate_refresh_node(state):
    keywords = state.modification_search_keywords
    if not keywords:
        return {"modification_search_status": "not_required"}
    added = []
    failed = 0
    for keyword in keywords[:3]:
        try:
            raw = search_city_pois(state.destination or "", amap_key(), keywords=keyword,
                                   types=ATTRACTION_TYPE, offset=8)
            added.extend(spot for item in raw if (spot := poi_to_spot(item)))
        except Exception:
            failed += 1
    # Preserve provider facts from the old pool and deduplicate exact names.
    merged = {str(poi.get("name") or "").strip(): poi for poi in state.pois if poi.get("name")}
    for poi in added:
        merged.setdefault(str(poi["name"]).strip(), poi)
    status = ("failed" if failed == len(keywords[:3]) else "partial" if failed
              else "complete" if added else "empty")
    return {"pois": list(merged.values()), "modification_search_status": status,
            "history": [f"candidate_refresh: queries={len(keywords[:3])}, provider_results={len(added)}, status={status}"]}
