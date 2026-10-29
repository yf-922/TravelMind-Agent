import json
import hashlib
from unittest.mock import patch
import pytest
from scripts.audit_sft_contract_targets import aggregate_duration_faults
from scripts.prepare_sft_v3_release import OUT, ROOT
from scripts.post_train_runtime import capture_messages
from scripts.prepare_sft_v2_release import read_rows


def test_aggregate_claim_checks_sum_not_individual_visit():
    plan = dict(days=[dict(spots=[dict(start_time='10:00',end_time='11:30'),
                                 dict(start_time='13:00',end_time='14:15')])],
                reasoning='',notes='共150分钟游玩时长：90分钟及75分钟。')
    assert aggregate_duration_faults(plan) == ['explanation_total_duration_mismatch']
    plan['notes'] = '合计游玩165分钟。'
    assert aggregate_duration_faults(plan) == []
    plan['notes'] = '首处参观共90分钟，第二处75分钟。'
    assert aggregate_duration_faults(plan) == []


def test_release_preserves_migration_provenance_and_validation():
    if not (OUT/'audit.json').exists():
        pytest.skip('Local versioned research release not present')
    rows=read_rows(OUT/'train.ledger.jsonl')
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    assert len(rows)==audit['train']==276
    assert audit['new_kept']==27 and audit['historical_migrated']==249
    assert 'synth-0127' not in {r['id'] for r in rows}
    # This is an immutable historical release. Pin its system prompt rather
    # than requiring future production prompt versions to equal the old one.
    historical_system=rows[0]['messages'][0]['content']
    assert hashlib.sha256(historical_system.encode()).hexdigest()=='123e43ffa9d5f43ddffc05b753b754afe0fdaee68c20c52d1740a57dab59636b'
    with patch('app.planning.nodes.JOINT_PLANNER_SYSTEM',historical_system):
        for r in rows:
            assert r['messages']==capture_messages(r['state'])
            if r['prompt_origin'].startswith('audited_contract_migration'):
                assert r['original_messages']
            assert not aggregate_duration_faults(r['chosen'])
    assert (OUT/'validation.ledger.jsonl').read_bytes()==(ROOT/'data/post_train/release_v1/validation.ledger.jsonl').read_bytes()
