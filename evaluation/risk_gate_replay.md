# Risk-Gated Audit Policy Replay

> Deterministic replay over saved real-provider routes; no external calls.

- Natural-route audit skip rate: 100.0%
- Injected-fault escalation recall: 100.0%
- Constructed labelled cases: 49/49 pass; clean skip 100.0%, fault recall 100.0%
- Projected token reduction vs always-full audit: 61.8%
- Projected latency reduction vs always-full audit: 50.7%

| Input | Decision | Flags |
|---|---|---|
| nanjing-3d-sunny-history | skip | - |
| nanjing-3d-singlerain-history | skip | - |
| nanjing-3d-tighthours-negative | skip | - |
| injected:duplicate_poi | escalate | duplicate_poi |
| injected:opening_hours | escalate | habit_constraint, opening_time_conflict |
| constructed:sunny_indoor OK | skip | - |
| constructed:sunny_outdoor OK | skip | - |
| constructed:valid_two_spots OK | skip | - |
| constructed:no_forecast OK | skip | - |
| constructed:valid_walking_leg OK | skip | - |
| constructed:unknown_hours OK | skip | opening_time_unknown |
| constructed:rain_indoor OK | skip | - |
| constructed:rain_outdoor OK | escalate | weather_outdoor_conflict |
| constructed:duplicate_poi OK | escalate | duplicate_poi |
| constructed:unknown_poi OK | escalate | unknown_poi |
| constructed:missing_day OK | escalate | route_structure |
| constructed:empty_day OK | escalate | route_structure |
| constructed:duplicate_day_number OK | escalate | route_structure |
| constructed:overlap OK | escalate | route_structure |
| constructed:reverse_period OK | escalate | route_structure |
| constructed:missing_time OK | escalate | route_structure |
| constructed:closed_hours OK | escalate | opening_time_conflict |
| constructed:late_start OK | escalate | habit_constraint |
| constructed:slow_pace OK | escalate | habit_constraint |
| constructed:walking_too_far OK | escalate | walking_constraint |
| constructed:walking_leg_missing OK | escalate | walking_constraint |
| constructed:long_drive OK | escalate | long_road_leg |
| constructed:user_change OK | escalate | user_modification |
| constructed:compound_fault OK | escalate | duplicate_poi, unknown_poi |
| constructed:valid_evening OK | skip | - |
| constructed:too_many_per_day OK | escalate | route_structure |
| constructed:day_number_gap OK | escalate | route_structure |
| constructed:zero_length_visit OK | escalate | route_structure |
| constructed:period_missing OK | escalate | route_structure |
| constructed:opening_conflict_with_clean_structure OK | escalate | opening_time_conflict |
| constructed:unknown_hours_plus_duplicate OK | escalate | duplicate_poi, opening_time_unknown |
| constructed:transit_leg_not_walk OK | escalate | walking_constraint |
| constructed:walk_limit_exact_boundary OK | skip | - |
| constructed:drive_leg_below_threshold OK | skip | - |
| constructed:route_modify_opinion OK | escalate | user_modification |
| constructed:late_start_valid OK | skip | - |
| constructed:lijiang_all_rain_indoor OK | skip | - |
| constructed:lijiang_all_rain_outdoor OK | escalate | weather_outdoor_conflict |
| constructed:sanya_all_rain_outdoor OK | escalate | weather_outdoor_conflict |
| constructed:shanghai_all_rain_indoor OK | skip | - |
| constructed:shanghai_all_rain_outdoor OK | escalate | weather_outdoor_conflict |
| constructed:jingdezhen_open OK | skip | - |
| constructed:jingdezhen_closed OK | escalate | opening_time_conflict |
| constructed:lijiang_open OK | skip | - |
| constructed:lijiang_closed OK | escalate | opening_time_conflict |
| constructed:sanya_open OK | skip | - |
| constructed:sanya_closed OK | escalate | opening_time_conflict |
| constructed:shanghai_open OK | skip | - |
| constructed:shanghai_closed OK | escalate | opening_time_conflict |

Constructed category recall: duplicate_poi=100%, habit_constraint=100%, long_road_leg=100%, opening_time_conflict=100%, opening_time_unknown=100%, route_structure=100%, unknown_poi=100%, user_modification=100%, walking_constraint=100%, weather_outdoor_conflict=100%

Boundary: The natural/fault groups replay saved real-provider routes; constructed cases are hand-labelled routes over frozen real POI pools, not model generations. Savings are projections from the three-case online ablation, not a new online A/B.
