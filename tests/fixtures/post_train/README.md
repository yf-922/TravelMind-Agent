# Offline post-training regression fixtures

These versioned inputs make the offline test suite runnable from a clean
checkout. They are **regression projections, not complete evaluation releases**.
Production evaluation scripts still use their original `data/post_train/`
locations and strict frozen-protocol checks; tests do not change those defaults.

| Directory | Contents |
| --- | --- |
| `representative/` | All 50 original case states, rubrics and witnesses; unused Planner messages removed; 40 recorded witness-adjacent road edges; original negative-control audit |
| `holdout/` | All 50 original cases including messages for no-pool stripping checks; 40 recorded witness-adjacent road edges |
| `shared_dpo/` | First existing example per category in each split: 9 train and 9 validation rows; matching train export |

`provenance.json` records original source hashes and projection rules. Existing
captured POI facts and responses are retained; no model or map calls generated
these fixtures. Historical source hashes identify local originals, not a claim
that those complete originals ship in this repository.

The representative `FROZEN.json` checks only the projected asset bytes. It is
explicitly not the original evaluation manifest and cannot authorize a formal
model comparison. Tests point to this directory explicitly; runtime ledgers,
raw HTTP captures, credentials and user databases remain outside Git.

The DPO `*.ledger.jsonl` files contain JSON arrays (supported by the production
`read` helper), so multi-line records remain inspectable. They are not the full
training dataset. Non-empty fixtures, split disjointness, schema validity,
negative-label presence and export fidelity are checked by the tests.

Run the affected regression group:

```bash
python -m pytest -q tests/test_representative_eval50.py tests/test_planner_prompt_screen.py tests/test_controlled_online_holdout.py tests/test_no_pool_audit.py tests/test_shared_dpo.py
```
