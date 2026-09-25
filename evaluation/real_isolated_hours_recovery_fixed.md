# Real Multi-Agent Fault-Recovery Ablation

> Real LLM Reviewer/Planner/Time Check calls; deterministic faults injected into identical saved real-Planner drafts.
> Provider: grok / grok-4.6; prompt fingerprint: `056834d3a0543efd`.

| Mode | Passed | Recovery rate | E2E total | Est. Tokens | LLM attempts |
|---|---:|---:|---:|---:|---:|
| time_check_only | 1/1 | 100.0% | 84.9s | 6939 | 5 |
| planner_reviewer_time_check | 1/1 | 100.0% | 73.6s | 6658 | 4 |

- Recorded LLM attempts: 9 (0 failed provider attempts).
- Estimated tokens across selected modes: 13597; exact usage ratio: 0.0%.
- Provider latency across both multi-Agent modes: 151.8s.
- Regrading only re-runs deterministic checks on saved routes; it does not call the provider.
Boundary: Both modes start from identical real-Planner drafts with deterministic injected faults. This measures recovery robustness, not natural-request quality or failure prevalence.
