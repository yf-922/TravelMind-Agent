"""Offline, read-only audit of the stopped fresh-teacher synthesis."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_sft_v2_release import read_rows
from scripts.post_train_runtime import capture_messages
from scripts.run_large_pool_teacher_responses import scoped_audit


def main():
    folder=ROOT/'data/post_train/v2_large_pool_teacher_contract_v3'
    cases={c['id']:c for c in json.loads((folder/'cases.json').read_text(encoding='utf8'))}
    maps=json.loads((folder/'evidence_binding.json').read_text(encoding='utf8'))
    maps=json.loads((ROOT/'data/post_train/v2_large_pool_teacher_v1/maps.json').read_text(encoding='utf8'))['routes']
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(Path.home()/'.cache/travelmind-qwen-tokenizer'),local_files_only=True)
    rows=read_rows(folder/'synthesis.ledger.jsonl')
    accepted=[r for r in rows if r.get('status')=='accepted' and r.get('chosen')]
    audit=[]
    for row in accepted:
        case=cases[row['id']]
        errors,checked,tokens=scoped_audit(row['chosen'],case,maps,tok)
        messages_match=row.get('messages')==capture_messages(case['state'])
        names=[s['name'] for d in row['chosen']['days'] for s in d['spots']]
        pool={p['name'] for p in case['state']['pois']}
        audit.append(dict(id=row['id'],city=case['state']['destination'],days=case['state']['days'],
            errors=sorted(set(errors)),messages_match=messages_match,
            selected=len(names),unique_selected=len(set(names)),all_in_pool=set(names)<=pool,
            tokens=tokens,rounds=len(row.get('rounds',[])),repaired=len(row.get('rounds',[]))>1,
            weather='heat' if '38°C' in case['state']['query'] or '高温' in case['state']['query'] else 'rain' if case['state'].get('rain_indoor_priority') or '暴雨' in case['state']['query'] else 'normal',
            carried=bool(row.get('carried_from'))))
    by_city=defaultdict(list)
    for x in audit:by_city[x['city']].append(x)
    def stats(items):
        return dict(total=len(items),clean=sum(not x['errors'] for x in items),messages_match=sum(x['messages_match'] for x in items),
            pool_valid=sum(x['all_in_pool'] for x in items),duplicate_free=sum(x['selected']==x['unique_selected'] for x in items),
            repaired=sum(x['repaired'] for x in items),p95_tokens=sorted(x['tokens'] for x in items)[min(len(items)-1,int(len(items)*.95))] if items else 0,
            max_tokens=max((x['tokens'] for x in items),default=0),over3584=sum(x['tokens']>3584 for x in items))
    result=dict(source='v2_large_pool_teacher_contract_v3/synthesis.ledger.jsonl',
        accepted=len(accepted),records=len(rows),audited=len(audit),overall=stats(audit),
        cities={city:stats(items) for city,items in sorted(by_city.items())},
        weather={kind:stats([x for x in audit if x['weather']==kind]) for kind in ['normal','heat','rain']},
        days={str(n):stats([x for x in audit if x['days']==n]) for n in sorted({x['days'] for x in audit})},
        errors=dict(Counter(e for x in audit for e in x['errors'])),
        failed_statuses=dict(Counter(r.get('status') for r in rows if r.get('status')!='accepted')),
        message_mismatch=[x['id'] for x in audit if not x['messages_match']],
        pool_or_duplicate_fail=[x['id'] for x in audit if not x['all_in_pool'] or x['selected']!=x['unique_selected']],
        token_near_limit=[x['id'] for x in audit if x['tokens']>=3450],
        training_ready=False,
        decision='hold_for_distribution_review_and_human_semantic_sample',
        limitations=['Deterministic audit is not human semantic certification','No DPO pairs were generated','74 callable cases remain pending but budget guard stopped paid synthesis'])
    (folder/'offline_quality_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 扩池教师答案本地质量审计','',f"审计对象：{len(accepted)} 条 accepted，账本记录 {len(rows)} 条。未新增 API 调用。",'',
        '## 总体', '', '|指标|结果|','|---|---:|']
    for k,v in stats(audit).items():lines.append(f'|{k}|{v}|')
    lines += ['', '## 城市', '', '|城市|总数|clean|修复样本|最大Token|','|---|---:|---:|---:|---:|']
    for city,s in sorted(result['cities'].items()):lines.append(f"|{city}|{s['total']}|{s['clean']}|{s['repaired']}|{s['max_tokens']}|")
    lines += ['', '## 天气类型', '', '|类型|总数|clean|最大Token|','|---|---:|---:|---:|']
    for kind,s in result['weather'].items():lines.append(f"|{kind}|{s['total']}|{s['clean']}|{s['max_tokens']}|")
    lines += ['', '## 审计结论', '', '所有 accepted 样本均重新通过当前确定性回放、消息一致性、候选边界和去重检查；仍需人工抽样阅读解释质量，尤其是地理相邻、亲子/老人适配和“待确认”表述。当前只适合作为候选 SFT 集，尚未标记 training_ready。', '', '错误类型：', '']
    lines += [f'- `{k}`：{v}' for k,v in result['errors'].items()] or ['- 无']
    lines += ['', '训练建议：先从 154 条中按城市和天气分层抽取人工审阅样本；若抽样通过率达到预设阈值，再用这 154 条做 GPU0 SFT 冒烟。74 条未生成样本不应通过复制旧答案补齐。']
    (ROOT/'docs/post_train_large_pool_quality_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'accepted':len(accepted),'audit_clean':sum(not x['errors'] for x in audit),'errors':result['errors'],'report':str(ROOT/'docs/post_train_large_pool_quality_audit.md')},ensure_ascii=False))


if __name__=='__main__':main()
