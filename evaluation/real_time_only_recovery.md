# Real Multi-Agent Fault-Recovery Ablation

> Real LLM Reviewer/Planner/Time Check calls; deterministic faults injected into identical saved real-Planner drafts.
> Provider: grok / grok-4.6; prompt fingerprint: `1ca037d1d8cf58da`.

| Mode | Passed | Recovery rate | E2E total |
|---|---:|---:|---:|
| time_check_only | 1/1 | 100.0% | 64.5s |

- Recorded LLM attempts: 3 (0 failed provider attempts).
- Estimated tokens across both multi-Agent modes: 4533; exact usage ratio: 0.0%.
- Provider latency across both multi-Agent modes: 61.5s.
- Regrading only re-runs deterministic checks on saved routes; it does not call the provider.
Boundary: Both modes start from identical real-Planner drafts with deterministic injected faults. This measures recovery robustness, not natural-request quality or failure prevalence.
