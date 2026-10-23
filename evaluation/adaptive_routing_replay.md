# Current Risk Gate: Evidence Audit

- Requests: 30 saved task requests.
- Current initial gate: review required 30; skipped 0.
- Observed historical audit calls: {'reviewer': 34}.
- Recorded provider audit Token: input 67892, output 21733, total 89625.
- Initial review skip opportunity: 0.0%.
- Full current Reviewer + TimeCheck call reduction: unavailable.
- Full current audit Token saving: unavailable.
- Initial Reviewer Token saving opportunity: 0.0; avoided Token 0.

## Why the old 61.8% is not a replacement metric

All current initial states requesting preferences enter semantic review. The historical ledger has no TimeCheck calls and no recorded adaptive arm. Do not invent TimeCheck tokens, reconstruct missing completions with a tokenizer, or present an offline routing decision as measured online savings.

## Per request

| Request | Review | TimeCheck | Flags |
|---|---|---|---|
| task-南京-interest | True | True | preference_review |
| task-南京-excluded | True | True | preference_review |
| task-南京-slow | True | True | preference_review |
| task-南京-late | True | True | commute_time_conflict, preference_review |
| task-南京-family | True | True | preference_review |
| task-南京-rain | True | True | preference_review |
| task-南京-nature | True | True | preference_review |
| task-南京-window | True | True | preference_review |
| task-南京-choice | True | True | preference_review |
| task-南京-elder | True | True | preference_review |
| task-上海-interest | True | True | preference_review |
| task-上海-excluded | True | True | preference_review |
| task-上海-slow | True | True | preference_review |
| task-上海-late | True | True | preference_review |
| task-上海-family | True | True | preference_review |
| task-上海-rain | True | True | preference_review |
| task-上海-nature | True | True | preference_review |
| task-上海-window | True | True | preference_review |
| task-上海-choice | True | True | preference_review |
| task-上海-elder | True | True | preference_review |
| task-三亚-interest | True | True | preference_review |
| task-三亚-excluded | True | True | preference_review |
| task-三亚-slow | True | True | preference_review |
| task-三亚-late | True | True | long_road_leg, commute_time_conflict, preference_review |
| task-三亚-family | True | True | preference_review |
| task-三亚-weather_window | True | True | preference_review |
| task-三亚-nature | True | True | long_road_leg, commute_time_conflict, preference_review |
| task-三亚-window | True | True | commute_time_conflict, preference_review |
| task-三亚-choice | True | True | long_road_leg, preference_review |
| task-三亚-elder | True | True | preference_review |

## Evidence gaps

- Historical experiment has no TimeCheck calls and no adaptive-policy arm.
- Intermediate repair states and per-round gate decisions were not recorded.
- Initial decisions are replayed with current code, not live production executions.
- Tokenizer counts cannot reconstruct unseen completions or provider billing usage.
