import copy

from scripts.diagnose_sft_transfer import intervention, select_cases


def row():
    return {'messages': [{'role':'system','content':'system'}, {'role':'user','content':'original'}],
        'state': {'days':2,'query':'38°C高温','rain_indoor_priority':False,
            'pois':[{'name':'甲','indoor':True},{'name':'乙','indoor':False},{'name':'丙'}]}}


def test_diagnostic_interventions_are_isolated_and_do_not_modify_original():
    source = row()
    before = copy.deepcopy(source)
    assert intervention(source,'original') == source['messages']
    rules = intervention(source,'rules')
    metadata = intervention(source,'metadata')
    assert rules[0] == metadata[0] == source['messages'][0]
    assert '11:00-15:00' in rules[-1]['content']
    assert '甲：室内属性' not in rules[-1]['content']
    assert '甲：室内属性=明确室内' in metadata[-1]['content']
    assert '乙：室内属性=明确室外' in metadata[-1]['content']
    assert '丙：室内属性=未知' in metadata[-1]['content']
    assert '11:00-15:00' not in metadata[-1]['content']
    assert source == before
