# Real-model Reviewer ablation v2

Date: 2026-10-10. Model: `grok-4.6`. 60 cases, 3 arms, 180 recorded outcomes.
All arms share the actual initial joint Planner draft. Weather/modification
cases first generate a neutral draft, then share the changed-request draft.
Rules and rules+Reviewer both permit at most 3 repairs; only the latter invokes
Reviewer each round. TimeCheck is absent in ALL arms to isolate semantic review.
Rule repair reads application risk flags, never scoring gold labels.

Planner-only deliberately bypasses Finalize approval; otherwise the supposedly
unaudited baseline would secretly contain deterministic final checks. Other
arms retain the application final guard. Final POIs/times are scored using the
same rubric. This is a controlled production-node audit-tail experiment, not a
claim that the complete live graph or SSE modification API was evaluated.

## Predeclared assets

v2 corrects two v1 asset defects BEFORE any new call: map prompts explicitly
require at least two POIs, and the Sanya exclusion query no longer requires the
same sole acceptable POI that it excludes. v1 remains unchanged. Models,
prompts, maps, candidates and evaluator fingerprint are bound to the new report.

## Results

| Metric | Planner only | Rules + repair | Rules + Reviewer + repair |
| --- | ---: | ---: | ---: |
| Task success / 30 | 26 | 30 | 30 |
| Task unsafe release / 30 | 4 | 0 | 0 |
| Resilience success / 30 | 14 | 18 | 20 |
| Actual recovery / 30 | 14 | 15 | 15 |
| Explained refusal / 30 | 0 | 3 | 5 |
| Resilience unsafe release / 30 | 15 | 10 | 9 |
| Resilience execution errors | 1 | 2 | 1 |

Reviewer does NOT improve ordinary task success or the actual recovery count
on this batch. Its measured difference is safe refusal, not better selections.
Two rule-vs-Reviewer pairs differ:

* `robust-modification-10`: the required nonexistent POI is not satisfied.
  Application rules approve the alternative route; Reviewer refuses with an
  explanation. This is direct evidence of a semantic rejection benefit.
* `robust-map-08`: rules-arm repair times out, Reviewer arm safely refuses after
  3 repairs. This execution difference is NOT pure semantic-quality evidence.

Therefore 18/30 -> 20/30 alone must not be described as two proven semantic
repairs. Nine nonpassing releases remain in the Reviewer arm. Missing road
evidence is not reliably prevented by either strategy.

## Resources

196 paid attempts: 194 completed, 2 failed. Recorded usage: input 428,869,
output 195,661, total 624,530 tokens. Timeout usage unknown; cost `unpriced`.

Logical arm totals include shared drafts even though they were paid once:

| Arm | Recorded logical Token | Mean measured seconds | Repairs |
| --- | ---: | ---: | ---: |
| Planner only | 276,655 | 36.3 | 0 |
| Rules | 324,014 | 42.6 | 14 |
| Rules + Reviewer | 573,825 | 69.9 | 21 |

Mean timing excludes the shared-seed failure whose elapsed measurement was
not recovered. The analysis JSON is authoritative for precise means. Weather
and modification timings include both seed generations, not only modification.
Logical totals must not be summed as actual billable usage.

## Reproduction

```powershell
python scripts/evaluate_reviewer_ablation.py --execute --max-calls 1000
python -m scripts.summarize_reviewer_ablation data/eval/reviewer_ablation_real_v2.json --out data/eval/reviewer_ablation_real_v2_analysis.json
python -m pytest -q tests/test_reviewer_ablation.py
```

Runtime reports in ignored `data/` retain shared seeds, final plans, per-call
usage and errors. Completed arms are reused on resume. In-flight interrupted
calls cannot be automatically replayed. No production-default change was made.

## v3 controlled soft-constraint extension (pending human review)

The original v2 ledger above is immutable. v3 adds a separate audit-tail+experiment for cases that deterministic rules do not express well: elderly+mobility, heat exposure, aversion to commercial themes and avoidable+backtracking. It also includes one infeasible control and one clean control.
The initial routes are author-controlled injected drafts, so these results do
not estimate how often a natural Planner makes each mistake.

The composite reuses 25 unchanged natural task rows and replaces five rows+with the controlled challenges. It is therefore not a new 30-request natural+success rate. The four feasible defects are graded against an independent+predeclared rubric; a successful result requires the final route to change in+the route JSON, not merely in notes. The infeasible case is expected to be a+safe explained refusal, not a successful itinerary.

| Arm | Composite task success / 30 | Feasible challenge repairs / 4 |
| --- | ---: | ---: |
| Planner only | 21 (70.0%) | 0/4 |
| Rules + repair | 25 (83.3%) | 0/4 |
| Rules + Reviewer + repair | 29 (96.7%) | 4/4 |

The Reviewer arm made one actual parsing-failure call on the infeasible case;
the production fallback ultimately returned an unapproved, explained refusal.
This is safe fallback evidence, not a clean semantic rejection without+execution failure. The run used 20 paid attempts, 19 parsed successfully,+39,930 input and 21,184 output tokens, and `unpriced` cost. Requested output+cap was 4,096, while one provider usage record reported 4,288 output tokens;
the report records both rather than claiming strict provider enforcement.

Review every challenge in [the review packet](../evaluation/soft_review_v3.md)
and export/import the bound form before changing `pending_human_review` to+`verified_v2`:

```powershell
python scripts/review_soft_review_labels.py export
# edit data/eval/soft_review_v3_human_review.json and confirm each case
python scripts/review_soft_review_labels.py import
```
