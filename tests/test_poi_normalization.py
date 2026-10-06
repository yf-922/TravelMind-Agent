import pytest

from app.providers.amap.poi import poi_to_spot


@pytest.mark.parametrize("cost", [[], {}, None, "[]", "null", "NaN", ""])
def test_missing_provider_prices_are_not_display_prices(cost):
    spot = poi_to_spot({"name": "Museum", "location": "118.8,32.0", "biz_ext": {"cost": cost}})
    assert spot["cost"] is None


def test_zero_price_and_category_source_are_preserved():
    spot = poi_to_spot({"name": "南京博物馆", "location": "118.8,32.0", "biz_ext": {"cost": 0}})
    assert spot["cost"] == "0"
    assert spot["indoor"] is True
    assert spot["indoor_source"] == "amap_category_inference"


def test_unknown_category_is_not_assumed_indoor():
    spot = poi_to_spot({"name": "南京夫子庙", "location": "118.8,32.0"})
    assert spot["indoor"] is None
