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
