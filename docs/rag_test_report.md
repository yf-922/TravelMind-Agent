# RAG Test Report

Date: 2026-07-25

## Scope

The Planner Agent uses `search_docs` through `search_travel_knowledge` before it creates an itinerary. The tool loads version-controlled Markdown files, splits them, uses Chroma to embed and persist chunks, retrieves Top-3 passages, and returns source-labelled evidence. Retrieved evidence is included in the Planner prompt and is returned in `final_plan.knowledge_sources`.

Knowledge sources:

1. `chongqing_visitor_guide`: Chongqing terrain, rain, pacing, evening crowds, and POI verification.
2. `transport_and_pacing`: travel modes, pacing, and opening-time handling.
3. `responsible_trip_checklist`: constraint priority, time-sensitive information, and unsupported-fact refusal.

## Test Cases

| Type | Query | Expected Top-1 source | Result |
| --- | --- | --- | --- |
| Fact | 重庆带老人下雨怎么安排 | `chongqing_visitor_guide` | Pass |
| Fact | 每段交通能都走路吗 | `transport_and_pacing` | Pass |
| Process | 资料没有门票价格怎么办 | `responsible_trip_checklist` | Pass |
| Process | 远郊地点搜索不到怎么办 | `chongqing_visitor_guide` | Pass |
| Comparison | 一天安排几个景点合适 | `transport_and_pacing` | Pass |

The automated acceptance set is implemented in `tests/test_travel_knowledge.py`. It validates the expected source in Top-1 for all five cases and verifies that `search_docs` includes the required `[source: ...]` citation label.

## Iteration Record

Initial test: English reference documents were queried in Chinese. Retrieval was biased toward the Chongqing document, and the transport and unsupported-price questions were not ranked first.

Fix: rewrote the three sources in Chinese, changed chunk overlap to keep a whole preceding paragraph, and added explicit hybrid retrieval. Chroma and the local lexical index now recall independently, then Reciprocal Rank Fusion (RRF) merges and de-duplicates candidates while retaining vector rank, keyword rank, channel provenance and source citation. When Chroma is unavailable, the tool reports a keyword-only fallback instead of claiming hybrid retrieval.

The checked-in offline report intentionally runs the deterministic `keyword` mode. It is a retrieval-contract baseline over three curated documents, not evidence that the hybrid mode improves online answer quality.

The current set has 30 cases: 24 answerable queries and 6 no-answer/out-of-domain queries. Keyword retrieval scored Hit@1 100%, Recall@3 100%, and no-answer accuracy 100%. Hybrid retrieval scored Hit@1 95.8%, Recall@3 100%, and no-answer accuracy 100%. The first hybrid run falsely returned a nearest neighbour for all no-answer queries; adding a lexical floor and a stricter vector-only distance gate fixed that regression. The honest conclusion remains that hybrid has not shown a quality advantage on this corpus.

Residual risk: the sources are project-curated guidance, not live authorities. The agent must not use this RAG corpus to state current ticket prices, real-time traffic, opening rules, or policy changes. Those claims still require live tools or official sources.

## Local latency note

On 2026-09-14, a 40-query local microbenchmark over this three-document corpus measured keyword-only retrieval at P50 0.422 ms (max 0.710 ms) and Chroma-backed hybrid retrieval at P50 180.310 ms (max 236.923 ms). This is a machine-local diagnostic, not an end-to-end service SLA. The difference is the local embedding/vector query cost; the mode is therefore explicit and can be switched to deterministic keyword fallback for offline or degraded operation.
