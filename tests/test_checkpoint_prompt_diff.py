import json
from pathlib import Path

from scripts.audit_checkpoint_prompt_diff import classify

FIXTURES = Path(__file__).parent / "fixtures" / "checkpoint"
LEDGER = FIXTURES / "checkpoint_modifications_v2.json"
PROMPTS = FIXTURES / "checkpoint_modifications_v2_prompts.json"


def test_prompt_pairs_and_candidate_hashes_are_complete():
    prompts=json.loads(PROMPTS.read_text(encoding='utf8'))
    ledger=json.loads(LEDGER.read_text(encoding='utf8'))['checkpoint_replan']
    rows=ledger['rows']
    assert len(prompts)==len(rows)==120
    assert all(r['identical_candidate_pool'] for r in rows)
    assert all(set(p)=={'id','full','local'} for p in prompts)


def test_ledger_does_not_claim_model_binding_without_fields():
    ledger=json.loads(LEDGER.read_text(encoding='utf8'))['checkpoint_replan']
    assert 'model_or_checkpoint' not in ledger
    assert 'decoding' not in ledger
