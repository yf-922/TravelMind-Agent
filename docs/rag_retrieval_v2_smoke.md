# RAG Retrieval Development Smoke

> Pending machine annotations: these are pipeline smoke scores, NOT human gold, test/generalization scores or resume-ready quality improvements.

Corpus: 38 extracted chunks; city counts {'上海': 16, '景德镇': 22}.
Draft queries: 106; split counts {'dev': 54, 'test': 52}.
Target not achieved: 120-180 reviewed chunks across six cities and 120 reviewed questions with a 60/60 split.
Current chunks include historical travel/transport snapshots. They must not be used as live weather, ticket prices or availability.
Development only. Frozen test NOT evaluated. Production corpus, index, retriever and Top-3 default unchanged.

| Retriever | k | Chunk Recall | Hit | MRR | nDCG (linear 0/1/2 gain) | Precision | Fact coverage | Context chars | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| keyword | 1 | 0.184 | 0.184 | 0.184 | 0.184 | 0.184 | 0.184 | 127 | 0 |
| keyword | 2 | 0.289 | 0.289 | 0.237 | 0.251 | 0.145 | 0.289 | 260 | 0 |
| keyword | 3 | 0.289 | 0.289 | 0.237 | 0.251 | 0.096 | 0.289 | 399 | 0 |
| keyword | 5 | 0.395 | 0.395 | 0.259 | 0.292 | 0.079 | 0.395 | 656 | 0 |
| keyword | 8 | 0.500 | 0.500 | 0.274 | 0.327 | 0.062 | 0.500 | 975 | 0 |
| keyword | 10 | 0.553 | 0.553 | 0.279 | 0.343 | 0.055 | 0.553 | 1198 | 0 |
| bm25 | 1 | 0.158 | 0.158 | 0.158 | 0.158 | 0.158 | 0.158 | 122 | 0 |
| bm25 | 2 | 0.237 | 0.237 | 0.197 | 0.208 | 0.118 | 0.237 | 250 | 0 |
| bm25 | 3 | 0.263 | 0.263 | 0.206 | 0.221 | 0.088 | 0.263 | 352 | 0 |
| bm25 | 5 | 0.395 | 0.395 | 0.235 | 0.274 | 0.079 | 0.395 | 570 | 0 |
| bm25 | 8 | 0.579 | 0.579 | 0.261 | 0.335 | 0.072 | 0.579 | 893 | 0 |
| bm25 | 10 | 0.632 | 0.632 | 0.267 | 0.351 | 0.063 | 0.632 | 1160 | 0 |
| vector | 1 | 0.053 | 0.053 | 0.053 | 0.053 | 0.053 | 0.053 | 97 | 0 |
| vector | 2 | 0.079 | 0.079 | 0.066 | 0.069 | 0.039 | 0.079 | 177 | 0 |
| vector | 3 | 0.132 | 0.132 | 0.083 | 0.096 | 0.044 | 0.132 | 260 | 0 |
| vector | 5 | 0.211 | 0.211 | 0.100 | 0.127 | 0.042 | 0.211 | 427 | 0 |
| vector | 8 | 0.368 | 0.368 | 0.123 | 0.180 | 0.046 | 0.368 | 708 | 0 |
| vector | 10 | 0.474 | 0.474 | 0.135 | 0.211 | 0.047 | 0.474 | 898 | 0 |
| hybrid | 1 | 0.105 | 0.105 | 0.105 | 0.105 | 0.105 | 0.105 | 112 | 0 |
| hybrid | 2 | 0.184 | 0.184 | 0.145 | 0.155 | 0.092 | 0.184 | 189 | 0 |
| hybrid | 3 | 0.263 | 0.263 | 0.171 | 0.195 | 0.088 | 0.263 | 271 | 0 |
| hybrid | 5 | 0.395 | 0.395 | 0.201 | 0.249 | 0.079 | 0.395 | 452 | 0 |
| hybrid | 8 | 0.553 | 0.553 | 0.222 | 0.300 | 0.069 | 0.553 | 772 | 0 |
| hybrid | 10 | 0.632 | 0.632 | 0.230 | 0.323 | 0.063 | 0.632 | 1012 | 0 |
| keyword_rrf | 1 | 0.105 | 0.105 | 0.105 | 0.105 | 0.105 | 0.105 | 116 | 0 |
| keyword_rrf | 2 | 0.132 | 0.132 | 0.118 | 0.122 | 0.066 | 0.132 | 212 | 0 |
| keyword_rrf | 3 | 0.237 | 0.237 | 0.154 | 0.174 | 0.079 | 0.237 | 291 | 0 |
| keyword_rrf | 5 | 0.263 | 0.263 | 0.160 | 0.186 | 0.053 | 0.263 | 491 | 0 |
| keyword_rrf | 8 | 0.474 | 0.474 | 0.189 | 0.255 | 0.059 | 0.474 | 801 | 0 |
| keyword_rrf | 10 | 0.605 | 0.605 | 0.203 | 0.294 | 0.061 | 0.605 | 1053 | 0 |
| legacy_hybrid | 1 | 0.105 | 0.105 | 0.105 | 0.105 | 0.105 | 0.105 | 117 | 0 |
| legacy_hybrid | 2 | 0.132 | 0.132 | 0.118 | 0.122 | 0.066 | 0.132 | 212 | 0 |
| legacy_hybrid | 3 | 0.237 | 0.237 | 0.154 | 0.174 | 0.079 | 0.237 | 298 | 0 |
| legacy_hybrid | 5 | 0.263 | 0.263 | 0.160 | 0.186 | 0.053 | 0.263 | 492 | 0 |
| legacy_hybrid | 8 | 0.526 | 0.526 | 0.197 | 0.272 | 0.066 | 0.526 | 818 | 0 |
| legacy_hybrid | 10 | 0.605 | 0.605 | 0.205 | 0.296 | 0.061 | 0.605 | 1059 | 0 |

## Interpretation

RRF = sum 1/(c + rank), no rounded score sorting. Both fusion channels retrieve up to 20 chunks; final k=1/2/3/5/8/10. Legacy rounded RRF is a separate control over the SAME corpus and embedding.
Recall is relevant chunk hits / all labeled relevant chunks. Hit is any relevant chunk. Fact coverage uses unique evidence groups. Precision uses actually returned passages; current runs return k passages.
No-answer retrieval is reported separately by type in JSON. Without a calibrated abstention threshold, top-k methods often return unrelated neighbors; that is not proof of generation refusal.
Machine questions were drafted from source topics. They are too generic and need manual rewrite/0-1-2 relevance review, cross-chunk and cross-city cases before conclusions.
All draft k candidates reach 10. Do NOT change the production default: inspect corpus/topic quality, query labels and Chinese embedding suitability first.
Independent annotation/review is still missing. Query-group bootstrap intervals are diagnostic, not confidence in real users or multi-annotator agreement.
Cold latency: first query per strategy includes setup; warm P50/P95 must be read separately, not sold as online latency. Context lengths are characters, not billed tokens.

## Generation Stage Dry Run

After review and retriever selection: 120 questions x 3 neighboring k values x 3 repeats = 1080 generation calls. One independent judge per answer adds 1080 calls (2160 logical calls before provider retries).
No generation/Judge calls executed. Monetary estimate cannot be supplied until model prices, prompt/context budget and actual usage are specified. Human spot checks remain required.

## Commands

```powershell
python -m pip install -r requirements-rag-eval.txt
python scripts/collect_rag_corpus.py --max-pages-per-city 10 --allow-external-calls
python scripts/draft_rag_questions.py
python scripts/sweep_rag_v2.py
python scripts/report_rag_v2.py
python -m pytest -q tests/test_rag_benchmark_v2.py
```

Collector reruns change the fingerprint and require fresh question annotation. Do not overwrite a frozen corpus/test set to improve a reported score.
