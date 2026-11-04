import json
from scripts.compare_checkpoint_outputs import diff

def test_three_levels_and_categories():
    a={'days':[{'day':1,'spots':[{'name':'甲','start_time':'09:00','end_time':'10:00'}]}]}
    assert diff(a,a)['status']=='byte_equal'
    assert diff(a,{'days':[{'day':1,'spots':[{'name':'甲','start_time':'09:00','end_time':'10:00'}]}],'notes':'x'})['semantic_equal']
    assert diff(a,{'days':[{'day':1,'spots':[{'name':'甲','start_time':'10:30','end_time':'11:30'}]}]})['class']=='modified_part_response_difference'
    assert diff(a,{'days':[{'day':1,'spots':[{'name':'乙','start_time':'09:00','end_time':'10:00'}]}]})['class']=='quantity_or_order_change'
