from app.planning.helpers import merge_verified_poi


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
