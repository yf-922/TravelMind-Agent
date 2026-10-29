"""Create a stratified, read-only human review packet; no model calls."""
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_sft_v2_release import read_rows

def kind(row):
    q=row['state']['query']
    if row['state'].get('rain_indoor_priority') or '暴雨' in q: return 'rain'
    if '高温' in q or '38°C' in q: return 'heat'
    return 'normal'

def main():
    folder=ROOT/'data/post_train/v2_large_pool_teacher_contract_v3'
    rows=[r for r in read_rows(folder/'synthesis.ledger.jsonl') if r.get('status')=='accepted']
    selected=[];seen=set()
    for city in sorted({r['state']['destination'] for r in rows}):
        city_rows=[r for r in rows if r['state']['destination']==city]
        for k in ['normal','heat','rain']:
            for r in sorted([x for x in city_rows if kind(x)==k],key=lambda x:x['id'])[:2]:
                if r['id'] not in seen:selected.append(r);seen.add(r['id'])
        for days in [1,2,3]:
            candidates=[r for r in city_rows if r['state']['days']==days]
            if candidates and candidates[0]['id'] not in seen:selected.append(candidates[0]);seen.add(candidates[0]['id'])
        repaired=next((r for r in city_rows if len(r.get('rounds',[]))>1),None)
        if repaired and repaired['id'] not in seen:selected.append(repaired);seen.add(repaired['id'])
    packet=[]
    for r in selected:
        packet.append(dict(id=r['id'],city=r['state']['destination'],days=r['state']['days'],weather=kind(r),
            repaired=len(r.get('rounds',[]))>1,query=r['state']['query'],candidate_count=len(r['state']['pois']),chosen=r['chosen'],
            review_questions=['景点是否全部来自候选池且无重复？','地理相邻、亲子/老人适配和待确认表述是否有输入依据？','时间、开放窗口、高温/室内边界和每日数量是否与最终路线一致？']))
    out=folder/'human_review_sample.json';out.write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'sample':len(packet),'source_accepted':len(rows),'path':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
