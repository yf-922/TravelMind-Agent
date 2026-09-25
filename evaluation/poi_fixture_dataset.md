# POI Fixture Dataset Card

## Snapshot

- Capture batch: `2026-09-14T15:04:00.960324+00:00`
- Cases: 30/30
- Destinations: 南京 8、上海 7、丽江 6、三亚 5、景德镇 4
- Scenario POI rows: 523（同一城市候选会被多个场景复用，不代表 523 个唯一景点）
- Source: AMap Web Service Place Text API v3
- Query channels per city: 必去景点、热门景区、博物馆
- Rating threshold: 4.5

The generator executes all three query channels and merges them round-robin by POI name. This prevents broad attraction queries from filling the pool before the museum channel is queried.

## Base Pools

| Destination | Rated POIs | Indoor heuristic | Outdoor heuristic |
|---|---:|---:|---:|
| 南京 | 14 | 3 | 11 |
| 上海 | 17 | 2 | 15 |
| 丽江 | 24 | 1 | 23 |
| 三亚 | 29 | 2 | 27 |
| 景德镇 | 14 | 2 | 12 |

Scenario filters then derive 24 full-pool cases, two outdoor-only cases, one tight-hours case, two Top-5 cases and one Top-4 case.

## Reproducibility And Boundaries

- POI names, ratings, opening text, coordinates and contact fields are frozen AMap results, not live facts at evaluation time.
- Weather is synthetic frozen test data because a live four-day forecast cannot reproduce future and historical scenarios.
- `indoor` is a deterministic name-keyword heuristic, not an AMap field or manually verified ground truth.
- The capture command allows at most 15 searches and 60 provider attempts including rate-limit retries. Cache hits mean this is an upper bound, not a claim about exact billed calls.
- Fixture readiness does not imply model quality. Planner/Reviewer/Judge evaluation is a separate opt-in run with its own LLM budget.

## Verification

```powershell
python scripts/validate_eval_assets.py --require-fixtures --out evaluation/eval_asset_report.json
python -m tests.eval.run_eval --dry-run
```

The validator checks 30 expected IDs, schema fields, source metadata, capture timestamps, non-empty pools, unique POI names, numeric coordinates, rating thresholds and boolean indoor labels without making network calls.
