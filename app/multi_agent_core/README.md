# Experiment 3: Multi-Agent Orchestration Core

This directory is an independently runnable core for the course's Experiment 3. It does not replace the existing FloatTrip LangGraph travel-planning workflow.

## Architecture

```text
User request
  -> Supervisor
  -> IntentAgent
  -> POIResearchAgent -- POI tool --> Amap / offline fixture
  -> PlannerAgent
  -> ReviewerAgent
  -> final itinerary

If the review fails: ReviewerAgent -> Supervisor -> PlannerAgent (one revision) -> ReviewerAgent
```

`Supervisor` owns only the current task trace (`dispatch_log`) and a centralized routing policy. Each worker owns a separate `system_prompt` and `private_memory`; the supervisor never passes one worker's private memory to another worker. Workers communicate only through `AgentMessage`.

The POI worker has an explicit `poi_search` allow-list. `ToolRegistry` rejects calls made by Agents without that permission. Worker model adapters are injectable: the production application can provide an LLM adapter while the deterministic default remains reproducible for tests.

## Message Schema

Every dispatch has these fields:

- `task_id`: the current travel-planning task.
- `task_type`: such as `intent_extract`, `poi_research`, or `itinerary_review`.
- `from` / `to`: message sender and receiver.
- `content`: structured task input or result.
- `status`: `pending`, `running`, `done`, `failed`, or `retrying`.
- `attempt`: `0` for the first route and `1` for the one permitted revision.
- `error_code`: populated on a terminal worker failure after the retry budget is exhausted.
- `trace_id` and `created_at`: log correlation and timestamp.

## Run

Use offline mode for a reproducible demonstration without any API key:

```powershell
.\.venv\Scripts\python.exe -m app.multi_agent_core.demo_run --offline --city Beijing --request "Plan a relaxed cultural day trip"
```

The output includes the structured dispatch log and each worker's private-memory entry count.

For a live POI lookup, configure `AMAP_API_KEY` in the ignored `.env.local` file and omit `--offline`:

```powershell
.\.venv\Scripts\python.exe -m app.multi_agent_core.demo_run --city Beijing --request "Plan a relaxed cultural day trip"
```

No key is stored in the source code or committed to Git.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_multi_agent_core.py -q
```

The tests verify four-worker dispatch, memory isolation, POI grounding, conditional revision, retry-to-failure handling, tool permission isolation, and model-adapter injection.

## Current Boundary

The default course demo uses deterministic role implementations so it is repeatable without an API key. `IntentAgent`, `PlannerAgent`, and `ReviewerAgent` now accept an injected model adapter; an adapter receives only that Agent's system prompt, private memory, and current message. The existing main FloatTrip workflow remains the production component that invokes the configured LLM. Migrating that UI path to this Supervisor is a separate integration step, not something this experiment core claims to have already done.
