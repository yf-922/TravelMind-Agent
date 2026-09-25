# Real Single-Agent vs Multi-Agent Ablation

> Provider: grok / grok-4.6; prompt fingerprint: `abdb00bf6b86cbd5`.

| Mode | Runs | Pass | Judge | LLM attempts | Est. tokens/run | E2E mean | Provider latency/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| planner_only | 3 | 100.0% | 4.47 | 6 | 3080.7 | 62.3s | 59.2s |
| planner_reviewer | 3 | 100.0% | 4.33 | 9 | 4762.3 | 70.4s | 69.3s |
| planner_reviewer_time_check | 3 | 100.0% | 4.27 | 16 | 8060.7 | 126.2s | 124.0s |

- Cases: nanjing-3d-sunny-history, nanjing-3d-singlerain-history, nanjing-3d-tighthours-negative
- Machine conclusion: `no_observed_multi_agent_quality_gain`.
- All provider/runtime failures remain in the denominator.
- Boundary: One run per case. Token values are estimates because provider usage metadata was unavailable; no statistical significance is claimed.
