# Experiment 3 Verification Checklist

## One-command offline demo

```powershell
.\.venv\Scripts\python.exe -m app.multi_agent_core.demo_run --offline --city Beijing --request "Plan a relaxed cultural day trip"
```

Expected evidence:

- Four workers appear in the `[dispatch]` and `[result]` logs.
- Every worker prints its own private-memory count.
- The dispatch log contains structured `task_type`, `from`, `to`, `status`, `attempt`, and `trace_id` fields.
- A failed review can route to `planner_agent` once and then return to `reviewer_agent`.

## Automated checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_multi_agent_core.py -q
```

The tests cover independent Agent instances, private memory isolation, POI grounding, conditional routing, retry and terminal failure, tool permission rejection, and model adapter injection.

## Honest boundary for the presentation

The experiment core is a true isolated-instance multi-Agent implementation. The existing web product still runs its mature FloatTrip LangGraph workflow, whose nodes share `TravelPlanState`. It should be described as the production workflow plus this separately verifiable multi-Agent core until a future integration replaces the web graph's shared state with per-Agent state adapters.
