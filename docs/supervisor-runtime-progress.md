# Travel Supervisor runtime

The authenticated `POST /api/plan/stream` endpoint accepts
`"engine": "supervisor"`; the default remains `langgraph`.

The runtime reuses production Intent, Query Rewrite, Weather, POI, Planner,
road-distance, risk-gate, Reviewer, Time Check, meal, tips and finalization nodes.
It emits the existing SSE stage/result events and uses request traces and timeouts.
Modification requests reuse the saved candidate pool and weather; explicitly
requested new places use the existing verified POI lookup before planning.
For preference changes, the Supervisor first checks whether the saved pool is
enough. If it is not, it performs at most three bounded keyword searches, merges
only provider-returned POIs, and turns failed or empty refreshes into a hard
review finding rather than allowing the Planner to claim the preference.

Role context is projected by field allow-lists. Planner, Reviewer and Time Check
receive their own persisted history with a 4,000-character context budget.
Planner, Reviewer and Time Check consume their own history in production prompts.
Time Check explicitly treats prior findings as reference and recomputes against
the current route. SQLite reads the last 32 entries per role in one transaction
snapshot on a worker thread, then applies the 4,000-character context budget.
The in-memory test implementation does not provide a cross-thread transaction.
This is application-level context separation, not a process security sandbox.

## Verified

- Complete fixture flow and rejected-route attempt bounds.
- Role/session filtering, SQLite persistence and 100 concurrent memory appends.
- Returned in-memory entries cannot mutate stored entries.
- Authenticated API engine selection and missing-field continuation events.
- Modification reuses candidates and skips Intent/weather/POI prefetch.
- Full regression suite: 270 passed, dependency deprecation warnings only.
- Five Node SSE client tests verify failed drafts cannot enter the successful-plan
  callback; missing-field continuation remains available. These run in CI.
- Enrichment inputs are projected so meal/tips nodes cannot read private contexts.
- Public execution logs survive role isolation without persisting private dialogue
  into the shared checkpoint.
- Saved/confirmed checkpoints retain walking/rain constraints and model/loop settings.
- Memory scopes include both owner and random trip session; modification and
  confirmation reuse the saved session, while unrelated trips are separated.
- Parallel stage completion is streamed per worker, rather than delayed until
  the slowest branch finishes. Failed audits skip meal/tips enrichment.
- Oversized role outputs retain bounded summaries rather than silently losing
  all memory. SQLite retains the latest 128 entries per session/role.
- Memory initialization failures fall back to request-local storage and expose
  `private_memory` as a degraded service.
- Runtime history/snapshot read and append failures also mark private memory
  degraded in the final result. The flag is request-local; a later successful
  request using the same store does not inherit an earlier failure.
- Four spawned OS processes concurrently append 80 entries to one SQLite
  session/role; reopening verifies every entry plus separate owner/role records.
  This verifies persistence contention, not multi-instance API coordination,
  load performance or same-trip modification conflict resolution.
- Confirmation warnings terminate trace normally as `awaiting_confirmation`,
  not `STREAM_ENDED_WITHOUT_RESULT`. Rejections report current risk flags and
  time/reviewer findings, recomputed after the final route revision.
- Concurrent API tests: same owner creates two distinct saved trip sessions;
  simultaneous same-trip runtime modifications read frozen role context and
  retain both writes. Providers are mocked; not a multi-process load test.
- Role output contracts reject cross-role writes and validate state types before
  persistence. Planner cannot approve or rewrite the pool; Reviewer cannot rewrite
  route; finalize cannot read private dialogue. All production role outputs have
  explicit owned fields. Query Rewrite receives original walking/rain constraints.
- Structured repair feedback retains the latest three revisions; current user
  modification is injected independently so Reviewer/Time Check cannot overwrite
  it. Role-private history is not used as a replacement for shared repair facts.
- Browser planning page exposes `标准规划` and `协作规划`; selecting the latter sends `engine=supervisor`.
- Human confirmation preserves the Supervisor engine and verifies the accepted draft.
- Memory lookup/write errors degrade without terminating usable planning.
- Chroma retrieval now exposes `ok`, `skipped` or `degraded` in the final service
  status. A vector-store outage remains non-fatal but is visible to the user;
  an empty result is distinct from an unavailable store. The API passes the
  status into both engines' final plans.
- A real in-memory Chroma collection test (deterministic local embedding) verifies
  metadata filtering for two users and no cross-user result; it does not claim
  production embedding quality or persistent-disk durability.
- Semantic preference IDs are stable per `(user_id, memory)` rather than random,
  so repeated expressions update the same Chroma record instead of multiplying
  duplicate recalls. IDs still differ across users.
- Authenticated users can delete only their own Chroma semantic-memory records;
  storage failure returns an explicit 503 rather than claiming deletion.
- API contract tests cover unauthenticated 401, storage failure 503 and owner
  propagation for semantic-memory deletion.
- Saved initial and confirmed-modification checkpoints now retain the semantic
  memory availability status, so later edits do not erase the fact that a
  request ran without personalized retrieval.
- Authenticated FastAPI replay now covers initial Supervisor save followed by a
  `plan_id` modification: it creates a child plan, preserves the memory session,
  emits `modification_intent -> planner -> route_distance_check`, and persists
  the updated preference. Empty serialized dates are normalized to `None` in
  modification and confirmation entrypoints; this fixed a real missing-terminal
  SSE failure for date-less trips.
- The replay now covers three generations: initial plan, a preference edit that
  refreshes and persists a verified POI, then a second local edit. The third
  checkpoint retains the refreshed POI and the same private-memory session.
- Supervisor now uses the shared joint meal planner path: meal candidate search
  runs before the first plan, meal slots are carried through route-distance and
  risk/reviewer inputs, and deterministic checks reject unknown restaurants,
  closed venues, malformed meal windows and attraction/meal overlap. A food
  preference edit refreshes meal candidates; a time-only edit keeps the saved
  pool.
- Main LangGraph and Supervisor fan out restaurant-candidate lookup with query
  rewrite and weather. The offline compiled-graph benchmark measures this
  three-way fan-out and the joint meal-output/tips fan-out using delayed fake
  nodes; it is not online latency data.
- Modification preference analysis and candidate refresh are covered: a new
  museum preference triggers provider search and exact-name deduplication;
  time/meal-only edits can reuse the frozen POI pool. Refresh status reaches
  Planner and the final checkpoint.
- A refresh with no provider result is a deliberate failed outcome carrying
  `candidate_refresh_unverified`; the old route cannot be reported as a
  successful fulfillment of the new preference.
- Multiple refresh keywords retain successful provider results as `partial`,
  but still trigger the hard review gate until the requested candidate set is
  verified complete.
- Modification-intent model failures now degrade to an explicit unverified
  refresh finding and a rejected draft, matching the main chain's non-fatal
  provider failure behavior without claiming preference compliance.
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
- Confirmation fix retest: Supervisor produced an approved result in 142.231
  seconds with six model calls, objective checks passed; convergence/efficiency
  graders did not pass (two planning rounds and two time checks). LangGraph timed
  out at 180 seconds. This run predated the subsequent modification-tag loop fix
  and atomic memory snapshot update. No statistical speedup claim.
- Token numbers in these reports are estimates; pricing is not configured.

## Remaining acceptance work

- Broader repeated Provider runs with independent grading against LangGraph;
  current evidence is single-case smoke, not quality/latency statistics.
- Multi-process concurrent deployment/load validation beyond threaded API and
  same-trip runtime regression coverage.
- Browser retest of quality fixes discovered during the authenticated full flow;
  API/provider retest is completed, not yet repeated through the final browser process.
- Generic natural-language modification compliance beyond the supported one-day
  explicit time-edit checks; multiday edits intentionally are not parsed by this rule.

The above missing checks mean production parity is not yet established.

## Latest private-memory audit verification

- Full regression: 243 pytest cases and five Node SSE-client cases passed.
  These are contract/regression checks, not a 100% real-request success claim.
- Production Reviewer and Time Check output contracts and role-local prompt
  inputs are covered with mocked structured-model calls.
- Real forced-audit run on `nanjing-3d-tighthours-negative`: Planner executed
  first, followed by road verification, risk gate, Reviewer and Time Check;
  one repair repeated those five stages. Final objective checks passed, with
  no remaining risk flags. Elapsed time was 129,600.34 ms, six model calls and
  11,619 estimated input-plus-output tokens (9,348 input / 2,271 output).
- Convergence and time-check efficiency grades failed: two audit rounds were
  needed. This is forced-audit mechanism verification over frozen POI/weather,
  not a natural-request quality trial, full API run or latency comparison.
- The earlier forced run entered with confirmation enabled and skipped the
  initial Planner. It is not evidence for normal planning; the script entry
  was corrected before the run reported above.
- Rejected production drafts now bypass live finalization/enrichment and expose
  `draft_only` plus unresolved findings. A regression checks both rejected and
  approved finalization paths.
- Evaluator supports engine selection and explicit audit forcing. Its provider
  attempt budget is a preflight estimate, not a runtime hard call limiter.

## Real modification and confirmation evidence

- Real initial plan, saved checkpoint, user time edit, owner confirmation and
  resaved history all executed. Local edit reused POI/weather, skipped external
  prefetch, changed route JSON and retained memory session. Planner history grew
  from one entry to two. Other user confirmation was denied with 404.
- Initial impossible-time trial revealed false success: Planner silently moved
  the requested 02:00-03:00 visit to daytime. Explicit supported time-edit rules
  now detect ignored requested start/window independently from opening hours.
- Real retest: initial plan approved in 73.592 seconds; impossible-time edit and
  confirmation returned `success=false` in 77.761 seconds, bounded replanning,
  no meal/tips calls and no saved replacement. This is a rejection success, not
  a completed travel-plan success. No generic natural-language compliance claim.
- Browser mode selector rendered and switched on desktop and 390px mobile.
  Mobile inspection found header overflow; browser tests have not yet covered
  authenticated submission/result rendering on the newest server process.
- Header overflow fixed with a two-row small-screen layout. Browser screenshot
  at 390x844 confirms scroll width 375px, login visible, no horizontal overflow.
  CSS/API/page resource versions bumped so existing browser caches receive fixes.

## Authenticated Browser Evidence

- Isolated localhost server, temporary account registration/login, selected
  Supervisor, submitted real request, observed run id/progress and final detailed
  route with map, transport, meals, hotel and history persistence. Browser backend
  uses production nodes; Chroma/background extraction are disabled.
- Quality defect discovered: indoor-only request selected an unknown/mixed site
  (Confucius Temple) and reported success. Do not count this as quality success.
- Added explicit indoor-only rule, museum-targeted retrieval, candidate prompt
  guidance and unknown/outdoor rejection. Classification from AMap type/name
  is labeled `amap_category_inference`, not official indoor confirmation.
- Real API retest completed/approved/saved in 65.281 seconds, four model calls,
  2,301 estimated tokens. One trial does not prove generic preference accuracy.
- Old provider `cost=[]` appeared as `¥[]/person`: normalization now removes
  empty/nonprice sentinels; frontend handles old saved data and keeps zero free.
- Output-contract real smoke: 62.413 seconds, approved/saved, selected Nanjing
  Yuhua Tea Museum, four model calls and 2,274 estimated tokens. All executed
  production outputs satisfied role contracts. Reviewer/Time Check were skipped;
  their contracts still require real high-risk retest (unit coverage is present).

## Authenticated full API evidence

- Real full production-node run on a two-spot request: 225.003 seconds, result
  incomplete, not saved. Two model attempts timed out and retried; 11 total model
  calls. All 12 distinct production nodes executed. This is retained as a failure,
  not excluded from evaluation.
- Subsequent one-indoor-spot request: 64.448 seconds, approved, saved history
  reloaded, walking/rain checkpoint fields present, other user denied with 403.
  Four model calls and 2,163 estimated tokens. Low-risk route skipped audit.
- Weather completed in 218 ms independently of rewrite (9,187 ms); meal and tips
  also had distinct finish events. Recorded overlap was 11,265 ms, not a claim
  of measured speedup versus a sequential control.
- Both runs used temporary application/private SQLite, disabled Chroma and
  background preference extraction, and ran through TestClient/ASGI rather than
  a browser over a deployed HTTP service. The successful run predates the final
  owner-plus-trip memory scope change, which has entrypoint regression coverage.
