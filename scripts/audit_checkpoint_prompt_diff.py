"""B-0: offline diff audit for the 120 full/local checkpoint prompts."""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'data/eval/checkpoint_modifications_v2.json'
PROMPTS=ROOT/'data/eval/checkpoint_modifications_v2_prompts.json'
OUT=ROOT/'data/eval/checkpoint_prompt_diff_b0'

def canon(x):
    if isinstance(x,list): return [canon(v) for v in x]
    if isinstance(x,dict): return {k:canon(v) for k,v in sorted(x.items())}
    return re.sub(r'\\s+',' ',x).strip() if isinstance(x,str) else x

def text(messages):
    if isinstance(messages,dict): messages=[m for values in messages.values() for m in values]
    return '\n'.join(f"{m.get('role','')}:{m.get('content','')}" for m in messages)
def hash_obj(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def classify(full,local):
    # Full path contains front nodes; local intentionally only contains planner.
    f=text(full.get('planner',[])); l=text(local.get('planner',[]))
    def has(s,terms): return any(t in s for t in terms)
    diffs=[]
    def block(s,start,end):
        m=re.search(re.escape(start)+r'(.*?)'+re.escape(end),s,re.S); return re.sub(r'\s+',' ',m.group(1)).strip() if m else ''
    if block(f,'景点候选池：','候选覆盖状态：') != block(l,'景点候选池：','候选覆盖状态：'):
        diffs.append('candidate_pool')
    def lines(s,terms): return sorted(re.sub(r'\s+',' ',x).strip() for x in s.splitlines() if any(t in x for t in terms))
    if lines(f,['天气：','气温','晴/夜间','白天']) != lines(l,['天气：','气温','晴/夜间','白天']): diffs.append('weather_fields')
    if lines(f,['目的地：','天数：','景点偏好：','用餐偏好：','每天景点上限']) != lines(l,['目的地：','天数：','景点偏好：','用餐偏好：','每天景点上限']): diffs.append('intent_derived_fields')
    # Full replay starts from the pre-modification checkpoint, while local
    # replay carries the saved previous route and explicit feedback.
    if re.search(r'上一版路线（仅供修改参考）：(?:无|$)',f) and re.search(r'上一版路线（仅供修改参考）：(?!无)',l):
        diffs.append('previous_route_context')
    if ('本次修改：' in f and '本轮修改/审核意见：' in l) or ('本次修改：' in f and '【用户修改意见】' in l):
        diffs.append('modification_context')
    # Exact changed spans are recorded for audit; this category means the
    # prompt differs without fitting the named policy fields.
    if not diffs and f != l: diffs.append('other')
    return diffs or ['none']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    prompts=json.loads(PROMPTS.read_text(encoding='utf8')); ledger=json.loads(LEDGER.read_text(encoding='utf8'))['checkpoint_replan']; rows=ledger['rows']; byid={r['id']:r for r in rows}
    result=[]; totals=Counter(); pool=Counter(); levels=Counter()
    for p in prompts:
        f,l=p['full'],p['local']; rid=p['id']; row=byid[rid]
        ft=text(f);lt=text(l); fc=canon(f);lc=canon(l); cats=classify(f,l)
        byte=ft==lt; normalized=fc==lc
        # semantic field-level diff over the planner's message fields
        fs={m.get('role'):m.get('content','') for m in f.get('planner',[])}; ls={m.get('role'):m.get('content','') for m in l.get('planner',[])}
        semantic_same=fs==ls
        level='byte_equal' if byte else ('normalized_equal' if normalized else ('semantic_field_equal' if semantic_same else 'semantic_field_diff'))
        levels[level]+=1
        for c in cats: totals[c]+=1
        pool[rid]=row.get('candidate_pool_sha256')
        result.append({'id':rid,'scenario':row.get('scenario'),'checkpoint_case_id':row.get('checkpoint_case_id'),
            'byte_equal':byte,'normalized_equal':normalized,'semantic_planner_fields_equal':semantic_same,'level':level,
            'diff_categories':cats,'candidate_pool_sha256':row.get('candidate_pool_sha256'),
            'full_nodes':list(f),'local_nodes':list(l),'full_sha256':hashlib.sha256(ft.encode()).hexdigest(),'local_sha256':hashlib.sha256(lt.encode()).hexdigest(),
            'full_prompt_tokens':row.get('full_text_tokens'),'local_prompt_tokens':row.get('local_text_tokens')})
    assert len(result)==120
    assert len(set(pool.values()))>=1
    summary={'status':'B0_COMPLETE','probes':len(result),'levels':dict(levels),'diff_categories':dict(totals),
        'candidate_pool_hashes':len(set(pool.values())),'candidate_pool_hash_same_per_probe':all(x.get('identical_candidate_pool') for x in rows),
        'source_binding':ledger.get('source_binding'),'encoding':ledger.get('encoding'),'tokenizer_version':ledger.get('tokenizer_version'),
        'measurement_type':ledger.get('measurement_type'),'runs':ledger.get('runs'),'timed_runs_per_arm':ledger.get('timed_runs_per_arm'),
        'model_or_checkpoint':'not recorded in v2 ledger; must be supplied before B-1',
        'decoding':'not recorded in v2 ledger; must be supplied before B-1',
        'B1_gate':'freeze B-1 arms only after model/checkpoint and decoding are explicitly bound'}
    (args.out/'prompt_diff.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    md=['# B-0 Prompt diff audit','',f"- probes: {len(result)}",f"- normalized/semantic levels: `{dict(levels)}`",f"- candidate pool hashes observed: {len(set(pool.values()))}",f"- all row candidate pools identical within probe: {summary['candidate_pool_hash_same_per_probe']}",'',
        '## Difference categories', '', '|category|count|', '|---|---:|']+[f'|{k}|{v}|' for k,v in sorted(totals.items())]+['','## B-1 gate','', 'The v2 ledger binds the source fingerprint, tokenizer and local timing, but does not bind the Planner model/checkpoint or decoding parameters. B-1 must not start until those fields are explicitly recorded.']
    (args.out/'summary.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
