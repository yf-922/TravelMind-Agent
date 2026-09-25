# Offline Memory Evaluation

> Uses a deterministic Chroma-compatible retrieval double and temporary real SQLite Agent storage.

- Semantic cases: 6
- Semantic Hit@2: 100.0%
- Cross-user leakage rate: 0.0%
- User metadata filter applied: 100.0%
- SQLite Agent/session boundaries: 5
- SQLite isolation pass rate: 100.0%

Boundary: this validates production filtering and SQLite isolation contracts, not Chroma embedding quality.
