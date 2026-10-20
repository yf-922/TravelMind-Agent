# Offline Engineering Benchmark

> This report uses no real LLM or map API calls. It must not be presented as online quality data.

## Post-intent LangGraph fan-out

- Workload: query_rewrite, weather_search, main_meal_search, 30 repeated runs, 40.0 ms deterministic delay per node
- Serial baseline P50/P95: 132.684 / 140.027 ms
- Current graph P50/P95: 50.707 / 60.712 ms
- P50 speedup: 2.617x; threshold result: PASS

## Meal and spot-tip fan-out

- Workload: main_meal_output, spot_tips, 30 repeated runs, 40.0 ms deterministic delay per branch
- Serial baseline P50/P95: 88.9 / 96.736 ms
- Current graph P50/P95: 48.519 / 58.046 ms
- P50 speedup: 1.832x; threshold result: PASS

## Road-distance concurrency

- Workload: 8 route legs, 30 repeated runs, 20.0 ms deterministic provider delay per leg
- Serial baseline P50/P95: 164.091 / 166.96 ms
- Current concurrent node P50/P95: 24.186 / 26.682 ms
- P50 speedup: 6.785x; threshold result: PASS

## Checkpoint local replanning

- Comparison boundary: simulated full replay; not historical production data
- Full replay P50/P95: 94.741 / 99.837 ms
- Local replan P50/P95: 64.591 / 65.784 ms
- Modeled P50 latency reduction: 31.8%
- Avoided stages: intent, query_rewrite, weather_search, attraction_search
- Simulated LLM input budget avoided: 1500 / 9900 (15.2%)
- Token boundary: fixed budget assumptions, not provider-reported token usage

## Interpretation boundary

The concurrency result measures the production node around a deterministic fake provider. It proves scheduling behavior, not real AMap network latency. The checkpoint result is a controlled stage-cost simulation because no runnable historical version exists. Real end-to-end latency, exact token usage, answer quality and external-service failure rates require the online suite.
