# Travel Supervisor runtime

The authenticated `POST /api/plan/stream` endpoint accepts
`"engine": "supervisor"`; the default remains `langgraph`.

The runtime reuses production Intent, Query Rewrite, Weather, POI, Planner,
road-distance, risk-gate, Reviewer, Time Check, meal, tips and finalization nodes.
It emits the existing SSE stage/result events and uses request traces and timeouts.
Modification requests reuse the saved candidate pool and weather; explicitly
requested new places use the existing verified POI lookup before planning.

Role context is projected by field allow-lists. Planner, Reviewer and Time Check
receive their own persisted history with a 4,000-character context budget.
Planner and Reviewer consume that history in their production prompts. Time
Check receives it but currently does not include it in its prompt; no claim of
effective long-term memory for every role is made. Histories are frozen before
the first node executes, but this is not an atomic database snapshot across roles.
This is application-level context separation, not a process security sandbox.

## Verified

- Complete fixture flow and rejected-route attempt bounds.
- Role/session filtering, SQLite persistence and 100 concurrent memory appends.
- Returned in-memory entries cannot mutate stored entries.
- Authenticated API engine selection and missing-field continuation events.
- Modification reuses candidates and skips Intent/weather/POI prefetch.
- Full regression suite: 201 passed, one dependency deprecation warning.
- Browser planning page exposes `标准规划` and `协作规划`; selecting the latter sends `engine=supervisor`.
- Human confirmation preserves the Supervisor engine and verifies the accepted draft.
- Memory lookup/write errors degrade without terminating usable planning.
- Independent deterministic checks prevent model approval from overriding hard faults.
- Optional weather/rewrite/meal/tips exceptions have explicit fallback updates.
- Cancellation regression verifies late synchronous worker completion does not write memory.
- A real configured Provider Intent call returned a two-day destination request
  with no missing fields and one persisted role-memory entry. This is a node smoke,
  not a complete dual-engine quality comparison.

## Real-provider smoke evidence

- Frozen Nanjing one-day POI/weather input, real model and road calls, enrichment
  omitted: both runtimes approved and passed objective grading. LangGraph took
  33.935 seconds and Supervisor 33.846 seconds. One sequential trial per engine
  is not a latency benchmark or evidence that private memory improves quality.
- Initial tight-hours trial: LangGraph timed out at 150 seconds. Supervisor's
  report had a KeyError because the script assumed every terminal event included
  a checkpoint. Confirmation events now have a distinct outcome and regression
  tests; that earlier record must not be treated as proof of a runtime crash.
- Retest at 240 seconds: LangGraph still timed out with 10 recorded model calls.
  This case exposes convergence/latency risk rather than a guaranteed pass.
- Supervisor returned a confirmation event after 112.138 seconds / four model
  calls, not an approved final plan. Internal repair concerns incorrectly triggered
  confirmation; this is now restricted to the first genuine user-modification
  revision, with regression coverage. The fix still needs a real-provider retest.
- Token numbers in these reports are estimates; pricing is not configured.

## Remaining acceptance work

- Real Provider runs and independent quality grading against the LangGraph path.
- Same-session concurrent API runs, cancellation and partial parallel dependency failure.
- User-facing engine selection and browser workflow validation.

The above missing checks mean production parity is not yet established.
