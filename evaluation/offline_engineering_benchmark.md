# Offline Engineering Benchmark

> This report uses no real LLM or map API calls. It must not be presented as online quality data.

## Post-intent LangGraph fan-out

- Workload: query_rewrite, weather_search, 10 repeated runs, 40.0 ms deterministic delay per node
- Serial baseline P50/P95: 81.163 / 81.853 ms
- Current graph P50/P95: 45.227 / 48.512 ms
- P50 speedup: 1.795x; threshold result: PASS

## Road-distance concurrency

- Workload: 8 route legs, 10 repeated runs, 20.0 ms deterministic provider delay per leg
- Serial baseline P50/P95: 164.339 / 167.026 ms
- Current concurrent node P50/P95: 22.247 / 23.138 ms
- P50 speedup: 7.387x; threshold result: PASS

## Checkpoint local replanning

- Comparison boundary: simulated full replay; not historical production data
- Full replay P50/P95: 100.237 / 102.205 ms
- Local replan P50/P95: 70.106 / 72.876 ms
- Modeled P50 latency reduction: 30.1%
- Avoided stages: intent, query_rewrite, weather_search, attraction_search
- Simulated LLM input budget avoided: 1500 / 9900 (15.2%)
- Token boundary: fixed budget assumptions, not provider-reported token usage

## Interpretation boundary

The concurrency result measures the production node around a deterministic fake provider. It proves scheduling behavior, not real AMap network latency. The checkpoint result is a controlled stage-cost simulation because no runnable historical version exists. Real end-to-end latency, exact token usage, answer quality and external-service failure rates require the online suite.
