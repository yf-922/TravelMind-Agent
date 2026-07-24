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

`Supervisor` owns only the current task trace (`dispatch_log`). Each worker owns a separate `system_prompt` and `private_memory`; the supervisor never passes one worker's private memory to another worker. Workers communicate only through `AgentMessage`.

## Message Schema

Every dispatch has these fields:

- `task_id`: the current travel-planning task.
- `task_type`: such as `intent_extract`, `poi_research`, or `itinerary_review`.
- `from` / `to`: message sender and receiver.
- `content`: structured task input or result.
- `status`: `pending`, `running`, `done`, `failed`, or `retrying`.
- `attempt`: `0` for the first route and `1` for the one permitted revision.
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

The tests verify four-worker dispatch, memory isolation, POI grounding, and the conditional revision route.

## Current Boundary

The course core intentionally uses deterministic role implementations so the offline demo and tests are repeatable. Their distinct system prompts and private memories are already modeled, but this module does not yet call an LLM directly. The existing main FloatTrip workflow remains the project component that invokes the configured LLM. The next extension is to inject an LLM client into `IntentAgent`, `PlannerAgent`, and `ReviewerAgent`, while retaining the same message schema, memory isolation, tool boundary, and offline tests.
