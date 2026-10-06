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
This is application-level context separation, not a process security sandbox.

## Verified

- Complete fixture flow and rejected-route attempt bounds.
- Role/session filtering, SQLite persistence and 100 concurrent memory appends.
- Returned in-memory entries cannot mutate stored entries.
- Authenticated API engine selection and missing-field continuation events.
- Modification reuses candidates and skips Intent/weather/POI prefetch.
- Full regression suite: 194 passed, one dependency deprecation warning.
- Browser planning page exposes `标准规划` and `协作规划`; selecting the latter sends `engine=supervisor`.
- Human confirmation preserves the Supervisor engine and verifies the accepted draft.
- Memory lookup/write errors degrade without terminating usable planning.
- Independent deterministic checks prevent model approval from overriding hard faults.
- Optional weather/rewrite/meal/tips exceptions have explicit fallback updates.
- Cancellation regression verifies late synchronous worker completion does not write memory.
- A real configured Provider Intent call returned a two-day destination request
  with no missing fields and one persisted role-memory entry. This is a node smoke,
  not a complete dual-engine quality comparison.

## Remaining acceptance work

- Real Provider runs and independent quality grading against the LangGraph path.
- Same-session concurrent API runs, cancellation and partial parallel dependency failure.
- User-facing engine selection and browser workflow validation.

The above missing checks mean production parity is not yet established.
