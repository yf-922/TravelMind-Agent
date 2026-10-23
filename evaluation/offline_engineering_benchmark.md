# Offline Engineering Benchmark

> This report uses no real LLM or map API calls. It must not be presented as online quality data.

## Post-intent LangGraph fan-out

- Workload: query_rewrite, weather_search, main_meal_search, 30 repeated runs, 40.0 ms deterministic delay per node
- Serial baseline P50/P95: 134.407 / 139.727 ms
- Current graph P50/P95: 52.257 / 57.525 ms
- P50 speedup: 2.572x; threshold result: PASS

## Meal and spot-tip fan-out

- Workload: main_meal_output, spot_tips, 30 repeated runs, 40.0 ms deterministic delay per branch
- Serial baseline P50/P95: 90.101 / 93.772 ms
- Current graph P50/P95: 49.425 / 55.507 ms
- P50 speedup: 1.823x; threshold result: PASS

## Road-distance concurrency

- Workload: 8 route legs, 30 repeated runs, 20.0 ms deterministic provider delay per leg
- Serial baseline P50/P95: 164.498 / 168.125 ms
- Current concurrent node P50/P95: 25.275 / 27.782 ms
- P50 speedup: 6.508x; threshold result: PASS

## Checkpoint local replanning

- Actual saved checkpoints: 30; repeats per checkpoint: 3
- Full replay P50/P95: 0.7521 / 1.0277 ms
- Local replan P50/P95: 0.2955 / 0.5278 ms
- Timing scope: local serialization, production prompt assembly and checkpoint restore only.
- Avoided stages: intent, query_rewrite, weather_search, attraction_search
- Production prompt text tokens (cl100k_base): full 73858, local 51941; reduction 29.7%.
- Avoided front-node LLM invocations: 60 across 30 requests.
- Skipped external query categories per request: weather and POI; exact historical HTTP requests unavailable.
- Token boundary: text tokenization, not provider billing; schemas and wire framing excluded.
- External API/model timing trace: unavailable; no end-to-end latency claim.

## Interpretation boundary

The concurrency result measures the production node around a deterministic fake provider. It proves scheduling behavior, not real AMap network latency. The checkpoint result uses actual saved states and production prompt constructors with replayed parsed fields. Real end-to-end latency, exact token usage, answer quality and external-service failure rates require the online suite.
