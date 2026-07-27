from app.providers import hotels


def test_recommend_chain_hotel_prefers_current_lower_price(monkeypatch):
    monkeypatch.setattr(hotels, "search_city_pois", lambda *_args, **kwargs: [
        {"name": "全季酒店测试店", "location": "116.1,39.1", "biz_ext": {"cost": "320"}},
        {"name": "汉庭酒店测试店", "location": "116.2,39.2", "biz_ext": {"cost": "260"}},
    ])
    result = hotels.recommend_chain_hotel("北京", "test-key")
    assert result is not None
    assert result["name"] == "汉庭酒店测试店"
    assert result["nightly_price"] == 260
