# Task success and resilience evaluation v1

## Assets and scoring

`app/evaluation/task_robustness.py` declares two independent 30-case suites.
Inputs reuse real candidate POIs and saved road responses for Nanjing, Shanghai
and Sanya. Opening/weather snapshots are experiment inputs, not current facts.
Labels are author drafts, not independently human-confirmed gold annotations.

Task suite: 10 requests per city: interest, exclusion, slow pace, late start,
family/required POI, rain, nature, time window, mixed interests, elderly travel.
Sanya has no indoor POIs in this pool; its feasible rain case is replaced by
a sunny coastal slow-visit case. No invented indoor candidates are added.

Task success = final output approved AND all hard checks pass AND all declared
soft criteria pass AND no execution error. Hard criteria include candidate
membership, uniqueness, opening/time consistency, required/excluded POIs,
weather requirements and delivered road evidence. Soft criteria include
acceptable-interest matches, visit duration and pace. Reviewer approval alone
is not the evaluator. Only final route POIs/times count, not improved notes.
The rubrics cover explicitly requested aspects, not universal subjective quality.

Resilience suite (10 each):

| Family | Cases |
| --- | --- |
| Map | timeout; rate limit; exception; empty; partial; all missing; missing duration; unverified walking; slow road; first lookup failure then recovery |
| Weather | rain; heavy rain; thunder; snow; unavailable; indoor requirement; slow pace in rain; sunshine after rain; mixed preferences; rain plus exclusion |
| Modification | exclude; required POI; late start; slow pace; indoor only; shortened time; replace; override history; impossible window; nonexistent required POI |

Weather and modification real trials first generate a route, then modify the
checkpoint and invoke the production modification graph. This is between-stage
session mutation, NOT cancellation of an in-flight request. Weather is supplied
at the checkpoint boundary, NOT a background live weather subscription.
Map faults are injected at the production route-provider boundary, not into
the POI search provider. Rate limit uses a provider exception, not HTTP transport.

Report task recovery and safe explained rejection separately. An empty route or
model error is not a successful refusal. Safe refusal counts toward resilience,
never task completion. Resilience = recovery OR safe explained refusal; include
the fraction of refusals so this cannot reward an always-rejecting application.
Original failures and unexecuted cases remain in the planned denominator.

## Execution

```powershell
python scripts/evaluate_task_robustness.py
python -m pytest -q tests/test_task_robustness.py
python scripts/validate_eval_assets.py
python -m pytest -q
```

The default executes 60 offline node/provider probes. Its success-rate fields
are null: crash-free injected nodes do not demonstrate real planning success.
JSON exports contain all 60 exact requests, rubrics and scenario expectations
under ignored `data/eval/`. Source definitions and tests are tracked.

After confirming an external-call budget, run a five-case real pilot:

```powershell
python scripts/evaluate_task_robustness.py --execute --suite task --limit 5 --max-calls 60 --out data/eval/task_real_pilot_v1.json
```

Real calls use the configured Grok model, record actual input/output usage and
elapsed time, and disable hidden application/provider retries. Optional tips
are explicitly skipped; fixed candidate pools and saved road responses isolate
the production joint Planner/audit/finalize tail. RAG is disabled here.
Current adaptive review policy is recorded in each final plan; forced auditing
is not silently substituted. These are not complete live-search online SLAs.

Run each complete suite to a NEW output path after the pilot. Reports bind
assets, model, evaluator and route implementation; changed code cannot silently
resume an old run. An interrupted in-flight call requires manual reconciliation.
Errors and failed requests are retained. Cost is `unpriced`; no invented CNY
cost. A failed actual-usage monitor stops the experiment. Paid trials are never
automatically rerun. Model runs require budget confirmation; offline results
cannot be presented as their substitute.
