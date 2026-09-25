from app.providers.amap import direction


def test_transit_plan_extracts_line_stops_time_and_cost(monkeypatch):
    payload = {
        "status": "1",
        "route": {
            "transits": [{
                "cost": "5",
                "duration": "2460",
                "distance": "13200",
                "segments": [
                    {
                        "walking": {"distance": "420"},
                        "bus": {"buslines": [{
                            "name": "地铁2号线(内环)",
                            "departure_stop": {"name": "前门"},
                            "arrival_stop": {"name": "雍和宫"},
                        }]},
                    },
                    {"walking": {"distance": "180"}, "bus": {"buslines": []}},
                ],
            }]
        },
    }
    monkeypatch.setattr(direction, "_fetch_json", lambda url: payload)
    plan = direction.plan_transport_leg(
        {"lng": 116.4, "lat": 39.9}, {"lng": 116.42, "lat": 39.95},
        "北京", "fake-key", 6.0,
    )
    assert plan is not None
    assert plan["mode"] == "transit"
    assert plan["mode_label"] == "地铁/公交"
    assert plan["duration_min"] == 41
    assert plan["estimated_cost"] == 5
    assert "地铁2号线" in plan["instruction"]
    assert "前门上车" in plan["instruction"]
    assert "雍和宫下车" in plan["instruction"]


def test_direction_failure_returns_none_for_safe_fallback(monkeypatch):
    def fail(url):
        raise OSError("network unavailable")

    monkeypatch.setattr(direction, "_fetch_json", fail)
    plan = direction.plan_transport_leg(
        {"lng": 116.4, "lat": 39.9}, {"lng": 116.41, "lat": 39.91},
        "北京", "fake-key", 3.0,
    )
    assert plan is None


def test_direction_retries_once_after_temporary_empty_response(monkeypatch):
    calls = 0
    payload = {
        "status": "1",
        "route": {"transits": [{
            "cost": "3",
            "duration": "1200",
            "distance": "5000",
            "segments": [{"walking": {"distance": "100"}, "bus": {"buslines": [{
                "name": "地铁测试线",
                "departure_stop": {"name": "甲站"},
                "arrival_stop": {"name": "乙站"},
            }]}}],
        }]},
    }

    def fetch(_url):
        nonlocal calls
        calls += 1
        return {"status": "0"} if calls == 1 else payload

    monkeypatch.setattr(direction, "_fetch_json", fetch)
    monkeypatch.setattr(direction.time, "sleep", lambda _seconds: None)
    plan = direction.plan_transport_leg(
        {"lng": 120.1, "lat": 30.1}, {"lng": 120.2, "lat": 30.2},
        "杭州", "fake-key", 8.0,
    )

    assert calls == 2
    assert plan is not None
    assert plan["source"] == "amap"
    assert "地铁测试线" in plan["instruction"]


def test_route_distance_uses_actual_walking_endpoint(monkeypatch):
    direction._PLAN_CACHE.clear()
    payload = {
        "status": "1",
        "route": {"paths": [{"distance": "1800", "duration": "1500", "steps": []}]},
    }
    seen: list[str] = []

    def fetch(url):
        seen.append(url)
        return payload

    monkeypatch.setattr(direction, "_fetch_json", fetch)
    monkeypatch.setattr(direction.time, "sleep", lambda _seconds: None)
    plan = direction.plan_route_distance(
        {"lng": 118.78, "lat": 32.04},
        {"lng": 118.80, "lat": 32.05},
        "fake-key",
        mode="walk",
    )

    assert plan is not None
    assert plan["mode"] == "walk"
    assert plan["distance_km"] == 1.8
    assert "/v3/direction/walking?" in seen[0]


def test_route_distance_uses_actual_driving_endpoint(monkeypatch):
    direction._PLAN_CACHE.clear()
    monkeypatch.setattr(direction, "_fetch_json", lambda _url: {
        "status": "1",
        "route": {"paths": [{"distance": "26800", "duration": "2400"}]},
    })
    monkeypatch.setattr(direction.time, "sleep", lambda _seconds: None)
    plan = direction.plan_route_distance(
        {"lng": 118.78, "lat": 32.04},
        {"lng": 118.90, "lat": 32.10},
        "fake-key",
        mode="drive",
    )

    assert plan is not None
    assert plan["mode"] == "taxi_or_car"
    assert plan["distance_km"] == 26.8
