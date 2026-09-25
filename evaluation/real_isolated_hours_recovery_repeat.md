# Real Multi-Agent Fault-Recovery Ablation

> Real LLM Reviewer/Planner/Time Check calls; deterministic faults injected into identical saved real-Planner drafts.
> Provider: grok / grok-4.6; prompt fingerprint: `1ca037d1d8cf58da`.

| Mode | Passed | Recovery rate | E2E total | Est. Tokens | LLM attempts |
|---|---:|---:|---:|---:|---:|
| time_check_only | 0/1 | 0.0% | 149.1s | 6202 | 5 |
| planner_reviewer_time_check | 1/1 | 100.0% | 67.1s | 6095 | 4 |

- Recorded LLM attempts: 9 (2 failed provider attempts).
- Estimated tokens across selected modes: 12297; exact usage ratio: 0.0%.
- Provider latency across both multi-Agent modes: 212.1s.
- Regrading only re-runs deterministic checks on saved routes; it does not call the provider.
Boundary: Both modes start from identical real-Planner drafts with deterministic injected faults. This measures recovery robustness, not natural-request quality or failure prevalence.
