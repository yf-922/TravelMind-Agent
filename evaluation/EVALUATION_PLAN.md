# TravelMind Evaluation Plan

This plan turns the two referenced interview-experience posts into evidence that can be reproduced. Percentages from those posts are not reused as project results.

## 1. Evaluation layers

| Layer | Purpose | Data | Repeats | Cost |
|---|---|---|---:|---|
| Unit and contract | State transitions, memory isolation, constraints, fallback behavior | deterministic fixtures | every commit | free |
| Offline engineering benchmark | concurrency, stage avoidance, regression thresholds | delayed fake providers and declared stage-cost model | >= 5 | free |
| Golden-set online evaluation | final-plan quality, tool trajectory, stability, real latency and tokens | fixed 30-case set initially, then 50-80 | 5 per probabilistic case | real API cost |
| Failure and safety evaluation | timeout, partial tool failure, missing data, prompt injection and cross-user leakage | adversarial cases | 5 per case | mixed |

## 2. Golden-set composition

Start with 30 cases: 15 normal requests, 9 multi-step or modification requests, and 6 adversarial/error cases. Freeze prompts and expected constraints before execution. Every production-like failure becomes a named regression case.

Required slices include single/multi-day trips, walking limits, rain and outdoor conflicts, closed-day conflicts, explicit POIs absent from the initial pool, user modification with sufficient/insufficient checkpoint pool, duplicate attractions, provider timeout, one failed distance leg, malicious retrieved text, and two users with conflicting preferences.

## 3. Metrics and gates

| Dimension | Metrics | Initial gate |
|---|---|---|
| Final quality | deterministic constraint pass rate, judge score with evidence | all regression cases pass; report capability score without hiding failures |
| Agent trajectory | correct tool/node selection, invalid calls, review rounds | zero forbidden tool calls; bounded convergence |
| RAG and memory | Recall@K, citation correctness, preference hit rate, cross-user leakage | leakage = 0; other metrics reported by slice |
| Reliability | pass@5, pass^5, false approval/rejection, degraded completion | report both pass@5 and pass^5; no average-only claim |
| Performance | end-to-end P50/P95/max, per-node latency, HTTP latency | compare identical cases and configuration |
| Usage and cost | input/output/total tokens, exact-usage ratio, estimated cost | estimated values labelled; online report includes exact-usage ratio |
| Robustness | timeout recovery, partial route failure, injection resistance | no crash; degradation visible in structured output |

The initial gates are intentionally conservative. Quality thresholds should be set after the first honest baseline, then frozen before optimization.

## 4. Required comparisons

- Planner only vs planner + reviewer vs planner + reviewer + time-check, using the same deterministic contracts.
- Serial route-distance lookup vs the current concurrent node, using identical route legs and provider delay.
- Serial post-intent work vs the compiled LangGraph fan-out for query rewriting and weather lookup.
- Simulated full replay vs checkpoint local replan. This is labelled a reconstructed baseline because the repository has no historical Git revision.
- Memory off vs SQLite profile only vs SQLite + Chroma, using the same personalized requests.
- Full model vs routed smaller model only after correctness and exact Token collection are available.

Do not compare different prompts, datasets or provider conditions as if they were an ablation.

## 5. Current evidence and gaps

Current free evidence includes the full unit suite, focused Agent/RAG tests, the three-case deterministic reviewer ablation, and `offline_engineering_benchmark.py`. The three-case ablation verifies contracts only; it is too small to claim real planning quality.

Measured on 2026-09-14:

| Check | Result | Evidence boundary |
|---|---|---|
| Full test suite | 170 passed, 1 dependency deprecation warning | deterministic local tests |
| Multi-Agent tool trajectory | 4/4 scenarios, 8/8 checks each | deterministic offline tools; covers normal, Reviewer replan, transient retry and terminal failure; no semantic-quality claim |
| Planner-only / + reviewer / + time-check | 33% / 67% / 100% overall | three synthetic contract cases only |
| Query rewrite + weather fan-out | 80.920 ms serial vs 45.641 ms current P50, 1.773x | compiled production graph with delayed fake nodes |
| Meal enrichment + spot tips fan-out | 81.099 ms serial vs 45.710 ms current P50, 1.774x | compiled production graph with delayed fake nodes |
| Eight road-distance legs | 165.590 ms serial vs 22.345 ms current P50, 7.411x | production node with delayed fake provider |
| Checkpoint local replan | 31.9% modeled P50 reduction; 15.2% simulated input-token reduction | reconstructed stage-cost baseline, not historical or online data |
| RAG golden set | 30 cases: 24 answerable + 6 no-answer; keyword Hit@1/Recall@3/no-answer accuracy 100%/100%/100%; hybrid 95.8%/100%/100% | three-document retrieval set; generated-answer faithfulness is not yet graded |
| RAG local latency | keyword P50 0.422 ms vs Chroma+RRF P50 180.310 ms (40 queries) | local microbenchmark only; embedding/vector-store overhead, not end-to-end SLA |
| Degraded dependencies | map/meal-model/tips/time-check failures remain usable and are exposed in `degraded_services` | deterministic fault injection |
| Memory isolation | semantic Hit@2 100%, cross-user leakage 0%, SQLite Agent/session isolation 100% | deterministic retrieval double + real temporary SQLite storage |
| Run observability | total/node latency, three slowest executions, degraded/failed nodes, observed overlap and request-attributed LLM usage | privacy-safe in-memory traces; observed overlap is not graph critical path |

The 30-case fixture matrix and all 30 generated fixture files pass the asset validator. POIs were frozen from AMap Place Text API v3 in one capture batch; the checked-in cases contain 523 POI rows after scenario-specific filtering. Weather is synthetic and frozen for reproducibility, and indoor labels use a documented name-keyword heuristic. `tests.eval.run_eval` calls real planner/reviewer/time-check models even with `--no-judge`, so it still fails closed unless explicit external-call and numeric budgets are supplied. The full 30-case, five-trial run with Judge has a ceiling of 2,250 logical invocations and 6,750 provider attempts because structured calls may retry up to three times. Provider/runtime failures count as failed trials rather than being omitted from the denominator.

Real-provider smoke results on 2026-09-15 (Grok, one trial per case; these are diagnostics, not stability claims): `nanjing-3d-sunny-history` passed all deterministic gates with Judge 4.0/5 using 4 calls and 6,518 estimated tokens; `nanjing-3d-singlerain-history` passed with Judge 4.2/5 using 4 calls and 6,611 estimated tokens. The initial `nanjing-3d-tighthours-negative` run exposed repeated near-spelling of a POI plus a time-check timeout (failed after 4 review rounds, 11 attempts, about 240 seconds cumulative provider latency). After conservative candidate-name canonicalization, explicit late-start/slow-pace prompt constraints, and bounded transient-provider retries, the same case passed in one review round with Judge 4.2/5, 4 calls, 5,467 estimated tokens, and about 83 seconds cumulative provider latency. This before/after is a single-run observation and must not be presented as a statistically stable improvement. Provider token usage was unavailable after structured parsing, so token counts are explicitly estimates; price rates were not configured, so no cost claim is made.
 A later repeat of the regression case still passed, but had 3 transient timeout attempts before eventual success (9 recorded attempts and about 254 seconds cumulative provider latency). This is an unresolved provider-tail-latency risk; the retry change makes it observable and bounded, but does not claim to solve upstream instability.

Real single-vs-multi ablation on 2026-09-21 used Grok `grok-4.6`, one run each on three identical frozen inputs. Natural-request hard-constraint pass rate was 3/3 for every mode. Mean Judge score was 4.47 for Planner only, 4.33 for Planner+Reviewer and 4.27 for Planner+Reviewer+Time Check; estimated tokens per run were 3,081 / 4,762 / 8,061 and mean end-to-end latency was 62.3s / 70.4s / 126.2s. This does not show a natural-quality gain from always-on multi-Agent review. A separately labelled injected-fault recovery evaluation reused identical saved real-Planner drafts: no-audit recovered 0/2, real Planner+Reviewer recovered 2/2, and the full chain recovered 2/2. The two multi-Agent modes took 126.7s and 203.1s in total. This supports independent review as a robustness mechanism when a risky draft exists, not as evidence that every request benefits from more Agents. Both evaluations have only one run per case and make no statistical-stability claim.

The production graph now applies a deterministic risk gate after road-distance verification. Duplicate/unknown POIs, structural or habit violations, rain/outdoor conflicts, walking constraints, long road legs and user modifications escalate to Reviewer; known opening-hour conflicts escalate to Time Check. Unknown opening data is not guessed by another model: it is exposed as a `partial` service state for user verification. Low-risk drafts skip both model audits and expose that decision in `final_plan.audit_policy`. Offline replay over the saved real-provider outputs skipped 3/3 natural drafts and escalated 2/2 injected faults. Combining those decisions with the measured three-case online means projects 61.8% fewer estimated tokens and 50.7% lower latency than always running the full audit chain. These are policy-replay projections, not a new online A/B or production SLA.

The risk-gate regression matrix additionally uses 50 hand-labelled routes over frozen AMap candidate pools across five cities. It pairs legitimate and faulty schedules, and covers duplicate/unknown POIs, day numbering, time structure, opening hours, rain, habits, road-distance constraints, user changes and compound faults. This is deterministic policy coverage, not 50 new LLM-generated plans or a statistically representative online quality estimate. The matrix found a missing day-number check, which was fixed. Run `python scripts/evaluate_risk_gate.py` to regenerate per-case decisions and category recall; any mismatch exits nonzero.

The [Nowcoder evaluation walkthrough](https://www.nowcoder.com/discuss/928221310587502592) recommends a frozen golden set, basic/multi-step/adversarial slices, objective trajectory checks, repeated online trials and honest cost reporting; its [architecture companion](https://www.nowcoder.com/discuss/927156901626810368?sourceSSR=users) adds the single-vs-multi-Agent boundary and failure recovery. For this internship project, the next highest-value online test is repeated identical inputs with both quality and Token/latency reporting, not another framework integration. The RAG set now also has eight separately labelled in-domain unknown questions about live prices, queues, closures and transport. Retrieval cannot prove that a generated answer abstains or cites facts faithfully, so these questions remain explicitly ungraded until model outputs are captured and manually audited. `python scripts/evaluate_rag.py --include-hard-negatives --json-out evaluation/rag_hard_negative_report.json --markdown-out evaluation/rag_hard_negative_report.md` reports how often potentially misleading context is returned without counting it as an answer failure. The former citation-validity metric was only checking evaluator-rendered citations and is now labelled citation render integrity; it is not a model-quality result.

The adaptive-routing replay now compares five saved drafts plus an isolated opening-hours fault: single Planner, always-full audit, and risk-gated routing. The first isolated run exposed a real boundary failure: Time Check fixed the clock but a subsequent Planner revision introduced a duplicate POI. The production chain now re-runs the deterministic risk gate after Time Check and escalates to Reviewer when a new non-time risk appears. The fixed rerun passed 1/1 for both Time Check-only and full audit, but took 84.9s/6,939 estimated Tokens versus 73.6s/6,658 for the full chain because the fallback Reviewer was needed. This is intentionally evidence that adaptive routing is conditional: it saves cost only when the scoped check is sufficient, and must not be sold as universally cheaper. See `evaluation/real_isolated_hours_recovery_fixed.json` and `python scripts/evaluate_adaptive_routing.py`.

## 6. Internship-focused strengthening order

1. Complete the 30-case golden set and a one-run baseline, then repeat probabilistic cases five times.
2. Run the three-mode online ablation on a budgeted representative slice before making any multi-Agent quality claim.
3. Add generated-answer faithfulness/citation correctness to the expanded 30-case RAG retrieval set; memory leakage already has a deterministic isolation test but still needs a multi-user evaluation report.
4. Extend the completed offline tool-trajectory contracts to the production LangGraph online set, including node transitions and degraded dependency calls.
5. Extend the existing fault-injection and prompt-injection checks to the online golden set.
6. Use traces to explain latency and Token hotspots, then test model routing or context compression as measured ablations.
7. Treat MCP/A2A, fine-tuning and distributed runtime as later topics unless a real project requirement appears.

## 7. Commands

Free checks:

```powershell
python -m pytest -q
python -m tests.eval.ablation --json evaluation\ablation_report.json --out evaluation\ablation_report.md
python scripts\offline_engineering_benchmark.py --runs 10
python scripts\evaluate_rag.py
python scripts\evaluate_memory.py
python scripts\evaluate_risk_gate.py
python scripts\validate_eval_assets.py --out evaluation\eval_asset_report.json
python -m tests.eval.run_trajectory_eval
python -m tests.eval.run_eval --dry-run
python -m tests.eval.run_online_ablation --max-cases 10 --k 3 --dry-run
```

Online checks that incur real provider calls:

```powershell
python -m tests.eval.generate_fixtures --max-api-calls 60 --allow-external-calls
python -m tests.eval.run_eval --k 1 --max-cases 1 --max-llm-calls 45 --allow-external-calls
python -m tests.eval.run_online_ablation --only nanjing-3d-sunny-history --k 1 --max-llm-calls 81 --allow-external-calls
python scripts\benchmark_pipeline.py --runs 5 --allow-external-calls
```

Record model name, prompt version, dependency versions, date, region and price environment variables with every online result.
