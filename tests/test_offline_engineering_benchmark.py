from scripts.offline_engineering_benchmark import (
    benchmark_checkpoint_replan,
    benchmark_enrichment_fanout,
    benchmark_post_intent_fanout,
    benchmark_road_distance,
    percentile,
    render_markdown,
)


def test_percentile_uses_nearest_rank():
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.50) == 2.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0


def test_road_distance_benchmark_exercises_concurrent_production_node():
    result = benchmark_road_distance(runs=5, leg_count=4, delay_ms=20)

    assert result["pass"] is True
    assert result["p50_speedup"] >= 2.0
    assert result["measurement_type"] == "production-node-with-delayed-fake-provider"


def test_fanout_benchmark_exercises_compiled_production_graph():
    result = benchmark_post_intent_fanout(runs=5, delay_ms=30)

    assert result["pass"] is True
    assert result["p50_speedup"] >= 1.5
    assert result["parallel_nodes"] == ["query_rewrite", "weather_search"]


def test_enrichment_benchmark_exercises_compiled_production_graph():
    result = benchmark_enrichment_fanout(runs=5, delay_ms=30)

    assert result["pass"] is True
    assert result["p50_speedup"] >= 1.5
    assert result["parallel_nodes"] == ["meal_enrichment", "spot_tips"]


def test_checkpoint_benchmark_is_explicitly_simulated_and_reports_avoided_work():
    result = benchmark_checkpoint_replan(runs=5, unit_delay_ms=0.5)

    assert result["pass"] is True
    assert result["avoided_nodes"] == [
        "intent", "query_rewrite", "weather_search", "attraction_search",
    ]
    assert result["simulated_llm_input_tokens"]["avoided"] == 1500
    assert "not historical production data" in result["baseline_label"]

    fanout = {
        "parallel_nodes": ["a", "b"], "runs": 5, "fake_delay_ms_per_node": 1,
        "serial_baseline_ms": {"p50": 5, "p95": 6},
        "current_fanout_ms": {"p50": 3, "p95": 4},
        "p50_speedup": 1.7, "pass": True,
    }
    report = render_markdown({"post_intent_fanout": fanout, "enrichment_fanout": fanout, "road_distance": {
        "route_legs": 1, "runs": 5, "fake_provider_delay_ms_per_leg": 1,
        "serial_baseline_ms": {"p50": 5, "p95": 6},
        "current_concurrent_ms": {"p50": 2, "p95": 3},
        "p50_speedup": 2.5, "pass": True,
    }, "checkpoint_replan": result})
    assert "must not be presented as online quality data" in report
