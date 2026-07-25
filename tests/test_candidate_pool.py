from app.planning.helpers import merge_verified_poi
from app.planning import nodes
from app.planning.schemas import TravelPlanState


def test_merge_verified_poi_puts_user_selected_place_first_and_removes_duplicate():
    existing = [
        {"name": "Museum", "location": {"lat": 1, "lng": 1}},
        {"name": "Old Town", "location": {"lat": 2, "lng": 2}},
    ]
    verified = {"name": "Museum", "location": {"lat": 3, "lng": 3}, "rating": 4.2}

    merged = merge_verified_poi(existing, verified)

    assert [item["name"] for item in merged] == ["Museum", "Old Town"]
    assert merged[0]["location"] == {"lat": 3, "lng": 3}


def test_merge_verified_poi_keeps_new_user_selected_place_even_without_rating():
    verified = {"name": "Xinglongao", "location": {"lat": 28.4, "lng": 109.6}}

    merged = merge_verified_poi([], verified)

    assert merged == [verified]


def test_attraction_search_expands_pool_with_targeted_user_query(monkeypatch):
    monkeypatch.setattr(nodes, "amap_key", lambda: "test-key")
    monkeypatch.setattr(nodes, "fetch_city_spots", lambda *args, **kwargs: [{
        "name": "Generic Park", "rating": 5.0, "location": {"lat": 1, "lng": 1},
    }])
    monkeypatch.setattr(nodes, "search_city_pois", lambda *args, **kwargs: [{
        "name": "Xinglongao", "location": "109.6,28.4", "address": "秀山兴隆坳",
    }])

    update = nodes.attraction_search_node(TravelPlanState(
        query="我想去重庆秀山兴隆坳", destination="重庆", min_rating=4.5,
    ))

    assert [spot["name"] for spot in update["pois"]] == ["Xinglongao", "Generic Park"]
    assert "定向扩展" in update["history"][-1]
