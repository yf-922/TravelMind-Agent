"""The new gate must distinguish certified rejection from missed planning."""
import copy

from scripts.prepare_count_infeasible_ab import check, pair


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1]*100


def records():
    return pair({'id':'unit', 'state':{'destination':'测试城市', 'pois':[
        {'name':'候选甲', 'open_time':'09:00-18:00', 'indoor':True,
         'location':{'lng':120.0,'lat':30.0}}]}}, 'train', 0, Tokenizer())


def test_certified_empty_and_feasible_witness_pass():
    for row in records():
        assert check(row, row['chosen']) == []


def test_partial_route_cannot_pass_as_infeasible():
    impossible, feasible = records()
    assert 'unsafe_partial_schedule' in check(impossible, feasible['chosen'])


def test_unjustified_empty_answer_is_rejected():
    impossible, feasible = records()
    assert 'count_or_membership' in check(feasible, impossible['chosen'])


def test_tampered_certificate_and_missing_explanation_are_rejected():
    impossible, _ = records()
    tampered = copy.deepcopy(impossible)
    tampered['required_total'] = 3
    assert 'certificate_input_mismatch' in check(tampered, tampered['chosen'])
    wrong = copy.deepcopy(impossible['chosen'])
    wrong.update(reasoning='完成', notes='完成')
    assert 'not_explained' in check(impossible, wrong)
