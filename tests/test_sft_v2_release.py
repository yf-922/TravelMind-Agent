from scripts.prepare_sft_v2_release import explanation_faults
import json


def plan(reasoning):
    return {'reasoning': reasoning, 'notes': '', 'days': [{'day': 1, 'spots': [
        {'name': '甲', 'start_time': '09:00', 'end_time': '10:30'},
        {'name': '乙', 'start_time': '13:00', 'end_time': '15:00'}]}], 'selection_reasons': []}


def test_duration_ranges_bounds_and_transfers_are_not_exact_visit_claims():
    for text in ['每个景点游览90—120分钟。', '每个至少90分钟。', '每个安排超过60分钟。',
                 '每个景点之间预留40分钟。', '每个最多120分钟。']:
        assert explanation_faults(plan(text)) == []


def test_explicit_false_duration_claim_is_quarantined():
    for text in ['每个游览90分钟。', '每个至少100分钟。', '每个景点游览90—105分钟。',
                 '每个超过90分钟。', '每个至多100分钟。']:
        assert 'explanation_visit_duration_mismatch' in explanation_faults(plan(text))


def test_unselected_explanation_and_unseen_repair_context_are_quarantined():
    p = plan('根据上轮审核反馈修正。')
    p['selection_reasons'] = [{'name': '丙', 'reason': '必访', 'source_ids': []}]
    assert set(explanation_faults(p)) == {'unsupported_selection_reference', 'repair_context_in_round_zero_answer'}


def test_training_status_accepts_binding_and_keeps_dpo_disabled(tmp_path, monkeypatch):
    from scripts import launch_sft_v2_job as job
    (tmp_path/'logs').mkdir()
    monkeypatch.setattr(job, 'ROOT', tmp_path)
    job.status('sft_training', train=250, validation=30, dpo_auto_start=False)
    saved = json.loads((tmp_path/'logs/sft-v2-status.json').read_text())
    assert saved['train'] == 250 and saved['stage'] == 'sft_training'
    assert saved['dpo_auto_start'] is False
