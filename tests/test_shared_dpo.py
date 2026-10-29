"""Check the shared preference release's causal and split invariants."""
import json
from scripts.build_shared_dpo import OUT,read,digest,conv
from app.planning.schemas import TravelRoute

def test_chosen_is_valid_and_negatives_are_proven_distinct():
    for split in ['train','validation']:
        for row in read(OUT/f'{split}.ledger.jsonl'):
            TravelRoute.model_validate(row['chosen'])
            assert row['chosen_faults']==[]
            assert row['rejected_faults']
            assert digest(row['chosen'])!=digest(row['rejected'])
            assert max(row['tokens'])<=3584
            if row['category']=='time_format':
                assert any(e.startswith('schema_or_time:') for e in row['rejected_faults'])

def test_prompt_and_source_family_split_is_disjoint():
    tr=read(OUT/'train.ledger.jsonl');va=read(OUT/'validation.ledger.jsonl')
    assert not {digest(r['messages']) for r in tr}&{digest(r['messages']) for r in va}
    assert not {r['source_family'] for r in tr}&{r['source_family'] for r in va}
    for group in [tr,va]:
        assert len({digest(r['messages']) for r in group})==len(group)

def test_export_keeps_identical_production_message_content():
    rows=read(OUT/'train.ledger.jsonl')
    exported=read(OUT/'shared_dpo_train.json')
    for row,pair in zip(rows,exported):
        assert [m['content'] for m in row['messages']]==[m['value'] for m in pair['conversations']]
        assert json.loads(pair['chosen']['value'])==row['chosen']
        assert json.loads(pair['rejected']['value'])==row['rejected']
