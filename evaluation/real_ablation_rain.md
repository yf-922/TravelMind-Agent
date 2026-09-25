# Online Agent Ablation

> Same fixtures and provider configuration across modes. Provider failures count as failed runs.

| Mode | Runs | Pass | pass@k | pass^k | Judge | False approve | Node calls/run | Tokens/run | E2E P50/P95 | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| planner_only | 1 | 100.0% | 100.0% | 100.0% | 4.60 | 0.0% | 1.00 | 3135.0 | 53310.7/53310.7 ms | 0 |
| planner_reviewer | 1 | 100.0% | 100.0% | 100.0% | 4.40 | 0.0% | 2.00 | 4823.0 | 68330.0/68330.0 ms | 0 |
| planner_reviewer_time_check | 1 | 100.0% | 100.0% | 100.0% | 4.20 | 0.0% | 5.00 | 12351.0 | 190822.0/190822.0 ms | 0 |

- Provider-attempt budget ceiling: 81
- Token counts remain labelled estimates when structured provider responses omit usage metadata.
- This report does not claim statistical stability unless each case is repeated at least five times.
