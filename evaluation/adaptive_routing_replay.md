# Single vs Full vs Adaptive Audit Replay

> Saved real-provider outcomes and deterministic fault injection; not a new online A/B.

| Policy | Hard-constraint pass | Total latency | Mean latency |
|---|---:|---:|---:|
| single | 3/6 | 366.9s | 61.1s |
| full | 6/6 | 835.4s | 139.2s |
| adaptive | 6/6 | 654.9s | 109.1s |

- Adaptive latency reduction vs full: 21.6% on this selected replay.
- Natural-case estimated Token reduction: 61.8% (3 cases only).
- Isolated opening-hours fault estimated Tokens: time_check_only=6939, planner_reviewer_time_check=6658.
- Overall Token reduction: unavailable; old fault report does not attribute usage per mode.
- Offline routing matrix: 32/32 mode decisions correct; no LLM calls.

| Draft | Routed mode | Single | Full | Adaptive |
|---|---|---:|---:|---:|
| nanjing-3d-sunny-history | planner_only | pass | pass | pass |
| nanjing-3d-singlerain-history | planner_only | pass | pass | pass |
| nanjing-3d-tighthours-negative | planner_only | pass | pass | pass |
| injected:duplicate_poi | planner_reviewer_time_check | fail | pass | pass |
| injected:opening_hours | planner_reviewer_time_check | fail | pass | pass |
| injected:opening_hours_only | time_check_only | fail | pass | pass |

Boundary: Six selected saved drafts, including three injected faults; one provider run per route. Timing adds measured Planner and recovery runs, not a new end-to-end adaptive execution. Fault-mode tokens are not separately attributable in the old report, so no all-case Token saving is claimed. Fault prevalence and statistical quality stability cannot be inferred.
