# Tourism RAG Benchmark v2

## Status and boundaries

This is an evaluation path, not a production default change. Machine annotations
are proposals, not gold. Never quote draft retrieval scores as user accuracy.
Human review and a paid-model budget are required before final evaluation.

The curated corpus contains 128 chunks: Nanjing 23, Shanghai 20, Chongqing 23,
Lijiang 20, Sanya 21, Jingdezhen 21. It includes 28 explicitly project-authored
planning rules. These are advice, NOT attraction policy. The rest comes from
captured official/public tourism sources and local media; Lijiang has fewer
official visitor facts and more project rules, a material dataset limitation.

Raw snapshots and collection attempts are retained in the versioned source
directories. `benchmark_v2_curated/manifest.json` records input fingerprints,
excluded passages and near-duplicate aliases. Expired campaigns and visitor-
irrelevant government news are excluded heuristically; source review remains
necessary. All external facts are snapshots, not live ticket/weather inventory.
No uncertain indoor tag is converted to true.

The draft contains 90 answerable proposals, 15 missing-fact proposals and 15
out-of-corpus proposals. Dev/test each have 60 cases. Related wordings share an
intent group and cannot cross splits. Proposed relevance and reference facts
must be corrected by a human, especially generic and multi-evidence questions.

## Reproduction

Run from the repository root with Python 3.11 or 3.12:

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-rag-eval.txt
python scripts/validate_rag_assets.py --corpus knowledge/benchmark_v2_curated/chunks.json --cases evaluation/rag_benchmark_v2_draft.json --require-quota
python scripts/review_rag_annotations.py export --cases evaluation/rag_benchmark_v2_draft.json --corpus knowledge/benchmark_v2_curated/chunks.json --out evaluation/rag_v2_review_packet.json
```

Read each source, edit query/type/relevance/reference facts, then sign each case
with `annotator`, ISO `reviewed_at`, `annotation_status=human_reviewed`, a version
and change-log explanation. Export/import do not automatically sign drafts.
One reviewer is sufficient; do not claim multi-rater agreement.

```powershell
python scripts/review_rag_annotations.py import --cases evaluation/rag_v2_review_packet.json --corpus knowledge/benchmark_v2_curated/chunks.json --out evaluation/rag_benchmark_v2_reviewed.json
python scripts/build_rag_variants.py --corpus knowledge/benchmark_v2_curated/chunks.json --cases evaluation/rag_benchmark_v2_draft.json --out knowledge/benchmark_v2_variants
python scripts/sweep_rag_v2.py --cases evaluation/rag_benchmark_v2_draft.json --corpus knowledge/benchmark_v2_curated/chunks.json --out evaluation/rag_v2_dev_draft.json
```

For chunk comparisons run the sweep separately with `semantic.json` and
`semantic.cases.json`, then `token.json` and `token.cases.json`. Remapped labels
remain pending and must be reviewed. Both variants are derived from identical
source text. No silent tokenizer truncation; actual BGE tokenizer ceiling 512
including special tokens, semantic soft target 420 characters, overlap zero.
This is not evidence that character splitting is better than token splitting.

BM25 uses Jieba precision mode with non-word tokens removed and a cached index.
All modes use Top-20 per-channel candidates; k is a prefix. RRF uses unrounded
1/(c+rank), c=20/60/100. Legacy rounding is retained as its own baseline. The
embedding is BAAI/bge-small-zh-v1.5 revision
`7999e1d3359715c523056ef9478215996d62a620`, CPU, normalized vectors, cosine distance.
Its query instruction is recorded in every embedding configuration. Index names
include corpus and embedding configuration fingerprints; stored IDs/text are
validated. Explicit hybrid errors never masquerade as keyword success.

## Metrics and selection

Relevant chunks have grades 1 or 2, irrelevant 0. Recall and Precision count
unique IDs; MRR uses the first relevant rank. nDCG uses linear relevance gains
(not exponential gains), documented as formula v2. Fact coverage uses annotated
evidence groups; paragraph hash proposals are not yet semantic fact gold.

Main quality metrics include failed answerable requests as zero. Completed-only
quality and availability are separate. Missing/outside cases report retrieval,
empty and error rates; retrieval of a row is NOT itself hallucination. There is
no calibrated retrieval refusal threshold in v2, so unrelated Top-k results
should be visible rather than hidden.

Choose jointly on dev: smallest k within 0.02 of best Recall and nDCG; otherwise
highest Recall, then nDCG, then smaller k. Ties prefer better nDCG and c nearer
60. `draft_configuration_candidates` is diagnostic until human labels exist.
Paired bootstrap resamples intent groups with seed 20261008, 1000 iterations;
failed pairs score zero and both improvements and degradations are recorded.

```powershell
python scripts/benchmark_rag_latency.py --corpus knowledge/benchmark_v2_curated/chunks.json --query "上海博物馆怎么预约" --mode bm25 --out evaluation/rag_v2_bm25_latency.json
```

Cold timings include subprocess startup/import/model load/query. Warm timings
reuse the process and index after prewarm. Default 3 cold/10 warm trials. These
are local retrieval measurements, not end-to-end or online SLA. First BGE model
download/index construction must be performed before timing steady cold starts.

## Freeze and generation

After development selection, put only mode/k/rrf_c in a final config JSON.
`rag_v2_config_candidate.json` is an unmeasured candidate, not the selected best.

```powershell
python scripts/review_rag_annotations.py freeze --cases evaluation/rag_benchmark_v2_reviewed.json --corpus knowledge/benchmark_v2_curated/chunks.json --config evaluation/rag_v2_config_candidate.json --out evaluation/rag_v2_freeze.json
python scripts/evaluate_rag_v2.py --cases evaluation/rag_benchmark_v2_reviewed.json --corpus knowledge/benchmark_v2_curated/chunks.json --split test --mode hybrid --k 3 --rrf-c 60 --freeze evaluation/rag_v2_freeze.json --allow-vector --out evaluation/rag_v2_test.json
```

Any change to corpus, labels, final configuration or embedding configuration
invalidates the freeze. Do not use test outcomes to tune; create a new protocol
version with a fresh holdout if further tuning is necessary.

`evaluate_rag_generation.py` defaults to dry-run. Supply cases/corpus/config/
freeze, generator/judge model and endpoint, input/output price per million tokens,
and output path. It compares selected k and the two nearest sweep values with
three repetitions. Generator and Judge must differ in model or endpoint.
Use the higher provider prices for preflight; conservative byte-based ceilings
are not measured tokens. Execution additionally requires `--execute`, sufficient
`--max-llm-calls` and `--max-cost`, signed frozen labels and credentials named by
`--generator-key-env`/`--judge-key-env`. Provider retries are disabled.

Each result preserves supplied context, answer, Judge scores/raw output,
generator/Judge latency and actual provider usage (null if unavailable), errors
and pending human review. One trial per case at selected k plus all errors is
selected for human review. Judge scores are not human ground truth or retrieval
gain. No paid calls have been authorized by this implementation request.

## Verification and remaining gates

```powershell
python -m compileall -q app tests scripts
python scripts/validate_eval_assets.py
pytest -q
```

CI validates quotas and draft splits offline without downloading embeddings or
calling LLMs. Runtime indices under `data/` are never committed. Production
retrieval/chunk defaults remain unchanged until signed frozen-test evidence
supports a change. Source/content review, final human labels, frozen test and
budgeted generation/Judge are outstanding gates, not completed experiments.
