# Multi-Agent Trajectory Evaluation

> Offline deterministic fixtures. No LLM, AMap, or other external API was called.

| Scenario | Result | Checks | Dispatches | Tool attempts |
|---|---:|---:|---:|---:|
| happy_path | PASS | 8/8 | 4 | 1 |
| reviewer_replan | PASS | 8/8 | 6 | 1 |
| transient_tool_retry | PASS | 8/8 | 5 | 2 |
| persistent_tool_failure | PASS | 8/8 | 3 | 2 |

- Overall: 4/4 scenarios passed.
- Failed checks: none.
- Free-text queries are represented only by presence, length, and a 12-character SHA-256 prefix.
- This suite proves orchestration contracts and failure handling, not semantic plan quality or online latency.
