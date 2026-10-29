import json
from pathlib import Path
import pytest
from scripts.prepare_migration_control import OUT, ROOT
from scripts.prepare_sft_v2_release import read_rows


def test_control_changes_only_membership_not_historical_rows():
    if not (OUT/'train.ledger.jsonl').exists():
        pytest.skip('Local research artifacts absent')
    source=read_rows(ROOT/'data/post_train/release_v3/train.ledger.jsonl')
    expected=[r for r in source if 'original_messages' in r]
    actual=read_rows(OUT/'train.ledger.jsonl')
    assert actual==expected and len(actual)==249
    assert not any(r['id'].startswith('newcity-contract-') for r in actual)
    audit=json.loads((OUT/'semantic_audit.json').read_text(encoding='utf-8'))
    assert audit['new_contract_route_conflicts']==0
    assert len(audit['preexisting_strict_start_issues'])==4
    assert (OUT/'validation.ledger.jsonl').read_bytes()==(ROOT/'data/post_train/release_v3/validation.ledger.jsonl').read_bytes()


def test_control_configuration_matches_v3_update_budget():
    import yaml
    configs=ROOT/'configs/post_train'
    a=yaml.safe_load((configs/'qwen25_7b_planner_sft_v3.yaml').read_text())
    b=yaml.safe_load((configs/'qwen25_7b_planner_sft_migration_control.yaml').read_text())
    assert a['max_steps']==b['max_steps']==48
    for key in a:
        if key not in {'output_dir','dataset_dir'}:
            assert a[key]==b[key], key
