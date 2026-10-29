"""Literal duration claims with local day scope; frozen scoring stays intact."""
import re


def duration_claim_faults(plan):
    by_day={d['day']:[(int(s['end_time'][:2])*60+int(s['end_time'][3:]))-
        (int(s['start_time'][:2])*60+int(s['start_time'][3:])) for s in d['spots']] for d in plan['days']}
    all_values=[v for vs in by_day.values() for v in vs]
    text=plan.get('reasoning','')+'。'+plan.get('notes','')
    prefix=r'(?:(?:均|各|连续|游览|游玩|参观|安排|停留|时长|为|了|达|不少于|不低于|至少|超过|大于|不超过|至多|最多|约|\s))*'
    pattern=r'(?:每个(?:景点)?|每处|各景点)('+prefix+r')(\d+)(?:\s*[—–~至-]\s*(\d+))?\s*分钟'
    faults=[]
    numbers={'一':1,'二':2,'两':2,'三':3}
    for match in re.finditer(pattern,text):
        context=re.split(r'[。；;【】\n]',text[:match.start()])[-1][-50:]
        scope=list(re.finditer(r'第\s*([123一二三])\s*天|首日|次日|最后一天|全程|每天|所有景点|全部景点',context))
        values=all_values
        if scope:
            last=scope[-1]
            if last.group(1):
                n=last.group(1);day=int(n) if n.isdigit() else numbers[n]
                values=by_day.get(day,[])
            elif last.group(0) in ['首日','次日','最后一天']:
                day={'首日':1,'次日':2,'最后一天':max(by_day,default=0)}[last.group(0)]
                values=by_day.get(day,[])
        modifier,n=match.group(1),int(match.group(2))
        if match.group(3):
            wrong=any(not n<=v<=int(match.group(3)) for v in values)
        elif any(t in modifier for t in ['至少','不少于','不低于']):
            wrong=any(v<n for v in values)
        elif any(t in modifier for t in ['不超过','至多','最多']):
            wrong=any(v>n for v in values)
        elif any(t in modifier for t in ['超过','大于']):
            wrong=any(v<=n for v in values)
        elif '约' in modifier:
            continue
        else:
            wrong=any(v!=n for v in values)
        if wrong:
            faults.append('explanation_visit_duration_mismatch')
    return sorted(set(faults))
