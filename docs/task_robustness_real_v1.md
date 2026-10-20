# 60-case real model evaluation: original baseline

Model: `grok-4.6`. Execution date: 2026-10-10 (Asia/Shanghai).
Original ledger: `data/eval/task_robustness_real_v1.json` (ignored runtime data).
60 requests have explicit outcomes: 57 returned final outputs, 2 timed out,
1 was interrupted with unknown usage. Interrupted calls were not repeated.

| Suite | Requests | Task completion | Explained refusal | Execution failure |
| --- | ---: | ---: | ---: | ---: |
| Task | 30 | 25 | 4 | 1 timeout |
| Map faults | 10 | 0 | 0 | 0 |
| Weather changes | 10 | 6 | 3 | 1 timeout |
| User modifications | 10 | 7 | 2 | 1 interrupted |

Original task success is 25/30 (83.3%). Original resilience outcome is 18/30
(60%): 13 recoveries and 5 explained refusals. These are the original rubric
results, not validated production rates. Map probes originally record 10
approved-but-nonpassing outputs; the rubric review below limits interpretation.

## Rubric audit

The evaluator introduced two asset defects, discovered after execution:

1. `task-三亚-excluded` asks for and excludes the same sole acceptable POI.
   This is not a feasible-task failure. Do not present the 30-case success rate
   as a clean feasible benchmark without fixing and independently confirming
   the cases. Do not silently remove or replace this row.
2. Map scenarios have a minimum-two rubric, while their prompts say only
   maximum-two. Five outputs fail only `required_count`; this does not establish
   an actual user-requirement violation. They are not evidence of map robustness
   success either, since choosing one POI can avoid exercising a road leg.

Five map outputs are confirmed to approve adjacent POIs without delivered road
duration evidence: map-01, map-02, map-04, map-07, map-10. This is a concrete
unverified-route release, independent of the defective count rubric.

## Resource use

282 attempted model calls: 279 completed, 2 failed, 1 interrupted.
Recorded actual input tokens: 513,368; output tokens: 211,786; total: 725,154.
Timeout and interrupted usage is unknown; the recorded total is not a complete
billing total. Monetary cost remains `unpriced`.

Mean measured task latency: 66.8 seconds. Mean resilience trial latency: 106.5
seconds (29 timed trials); weather/modification include both initial planning
and the subsequent change. These values are not online SLA or per-modification
latency, and the interrupted trial has no recoverable elapsed measurement.

## Scope and reproduction

Real production joint Planner/audit/finalize tail; upstream POI pools and road
responses are saved fixtures. RAG and optional tips are disabled. Weather and
modifications are injected between planning stages, not while a tool call is
in flight. Current adaptive review policy was measured without changing it.

```powershell
python scripts/evaluate_task_robustness.py --execute --suite all --max-calls 1000 --out data/eval/task_robustness_real_v1.json
python scripts/summarize_task_robustness.py data/eval/task_robustness_real_v1.json --out data/eval/task_robustness_real_v1_analysis.json
```

Completed ledger rows are reused rather than called again. The reconciliation
backup preserves the process interruption. Next priority: correct and predeclare
rubrics, then prevent approval with missing required road evidence; run any
post-fix comparison into a separate report, never overwrite this baseline.
