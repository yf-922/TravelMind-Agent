# Real Multi-Agent Fault-Recovery Ablation

> Real LLM Reviewer/Planner/Time Check calls; deterministic faults injected into identical saved real-Planner drafts.
> Provider: grok / grok-4.6; prompt fingerprint: `abdb00bf6b86cbd5`.

| Mode | Passed | Recovery rate | E2E total |
|---|---:|---:|---:|
| single_no_audit | 0/2 | 0.0% | 0.0s |
| planner_reviewer | 2/2 | 100.0% | 126.7s |
| planner_reviewer_time_check | 2/2 | 100.0% | 203.1s |

- Recorded LLM attempts: 15 (1 failed provider attempts).
- Estimated tokens across both multi-Agent modes: 27243; exact usage ratio: 0.0%.
- Provider latency across both multi-Agent modes: 323.0s.
- Regrading only re-runs deterministic checks on saved routes; it does not call the provider.
Boundary: Both modes start from identical real-Planner drafts with deterministic injected faults. This measures recovery robustness, not natural-request quality or failure prevalence.
