"""Compare full/local checkpoint outputs when output ledgers are available.

No model/provider calls. The current v2 benchmark stores prompts and timing,
not Planner JSON outputs, so the script reports that gap instead of inferring
semantic equivalence from token or candidate-pool equality.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'data/eval/checkpoint_modifications_v2.json'
PROMPTS=ROOT/'data/eval/checkpoint_modifications_v2_prompts.json'
OUT=ROOT/'data/eval/checkpoint_consistency_v1'

def normalize(x):
    if isinstance(x,dict): return {k:normalize(v) for k,v in sorted(x.items())}
    if isinstance(x,list): return [normalize(v) for v in x]
    if isinstance(x,str): return re.sub(r'\\s+',' ',x).strip()
    return x

def semantic_route(x):
    days=(x or {}).get('days',[]) if isinstance(x,dict) else []
    return [(d.get('day'),[(s.get('name'),s.get('start_time'),s.get('end_time')) for s in d.get('spots',[])]) for d in days]

def diff(full,local):
    if full is None or local is None:return {'status':'missing_output'}
    if full==local:return {'status':'byte_equal','normalized_json_equal':True,'semantic_equal':True,'class':None}
    nf,nl=normalize(full),normalize(local)
    if nf==nl:return {'status':'normalized_equal','normalized_json_equal':True,'semantic_equal':semantic_route(full)==semantic_route(local),'class':None}
    sf,sl=semantic_route(full),semantic_route(local)
    if sf==sl:return {'status':'semantic_equal','normalized_json_equal':False,'semantic_equal':True,'class':'unmodified_part_drift'}
    f_names=[n for _,spots in sf for n,_,_ in spots]; l_names=[n for _,spots in sl for n,_,_ in spots]
    if f_names==l_names:return {'status':'semantic_time_difference','normalized_json_equal':False,'semantic_equal':False,'class':'modified_part_response_difference'}
    if sorted(f_names)==sorted(l_names):return {'status':'semantic_order_difference','normalized_json_equal':False,'semantic_equal':False,'class':'quantity_or_order_change'}
    return {'status':'semantic_set_difference','normalized_json_equal':False,'semantic_equal':False,'class':'quantity_or_order_change'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');ap.add_argument('--full-output',type=Path);ap.add_argument('--local-output',type=Path);args=ap.parse_args()
    ledger=json.loads(LEDGER.read_text(encoding='utf8')); prompts=json.loads(PROMPTS.read_text(encoding='utf8'))
    rows=ledger['checkpoint_replan']['rows']; OUT.mkdir(parents=True,exist_ok=True)
    full={};local={}
    if args.full_output and args.full_output.exists(): full={r['id']:r for r in map(json.loads,args.full_output.read_text(encoding='utf8').splitlines())}
    if args.local_output and args.local_output.exists(): local={r['id']:r for r in map(json.loads,args.local_output.read_text(encoding='utf8').splitlines())}
    out=[]
    for r in rows:
        key=r['id']; f=full.get(key,{}).get('plan'); l=local.get(key,{}).get('plan'); result=diff(f,l)
        out.append({'id':key,'scenario':r['scenario'],'full_output_present':f is not None,'local_output_present':l is not None,**result})
    summary={'status':'ready' if full and local else 'outputs_missing_from_v2_ledger','probes':len(rows),'prompt_pairs':len(prompts),'full_outputs':sum(x['full_output_present'] for x in out),'local_outputs':sum(x['local_output_present'] for x in out),'semantic_metric':'semantic_equal: same ordered (day, POI name, start, end) route','categories':['unmodified_part_drift','modified_part_response_difference','quantity_or_order_change'],'results':out}
    (OUT/'consistency_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    md=['# Checkpoint 修改一致性准备','',f"状态：`{summary['status']}`",'',f"探针：{len(rows)}；Prompt 对：{len(prompts)}；全量输出：{summary['full_outputs']}；局部输出：{summary['local_outputs']}",'',
        '三级口径：字节级 JSON 文本一致；归一化 JSON 等价（对象键排序、空白归一）；行程语义等价（逐日景点名称及开始/结束时间顺序一致）。主指标预注册为行程语义等价。','',
        '不一致分类：未修改部分漂移、修改部分响应差异、数量或顺序变化。当前 v2 账本未保存两路 Planner JSON，因此不能从 Token、候选池哈希或耗时推断一致性。']
    (OUT/'consistency_summary.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    preview={k:summary[k] for k in ['status','probes','full_outputs','local_outputs']}
    preview['dry_run_probes']=out[:3]
    print(json.dumps(preview,ensure_ascii=False))

if __name__=='__main__':main()
