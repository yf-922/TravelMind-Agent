"""Create small dev-only review packets without signing machine proposals."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.rag_protocol import fingerprint


def select_batch(cases, limit, city=None):
    groups={}
    for case in cases:
        if case['split']=='dev' and (city is None or case.get('city')==city):
            groups.setdefault(case['intent_group'],[]).append(case)
    selected=[]
    for group in groups.values():
        if len(selected)+len(group)<=limit:
            selected.extend(group)
    return selected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--city')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.limit<1:p.error('limit must be positive')
    all_cases=json.loads(a.cases.read_text(encoding='utf-8'))
    rows=json.loads(a.corpus.read_text(encoding='utf-8'))
    cases=select_batch(all_cases,a.limit,a.city)
    if not cases:p.error('no matching complete dev groups')
    lookup={r['chunk_id']:r for r in rows}
    selected_ids={label['chunk_id'] for c in cases for label in c.get('relevant_chunks',[])}
    evidence=[lookup[i] for i in sorted(selected_ids)]
    packet={'scope':'dev_only','annotation_status':'pending_review',
            'corpus_fingerprint':fingerprint(rows),'all_annotations_fingerprint':fingerprint(all_cases),
            'cases':cases,'evidence':evidence,
            'instructions':'Review proposals against sources; this packet does not authorize or perform signing. Transfer actual human corrections to the full review packet before importing.'}
    lines=['# 第一批 RAG 人工审核（仅开发集）','',
           f'共 {len(cases)} 条查询，{len(evidence)} 块待核验证据。状态：待审核。','',
           '这是机器提出的草稿。逐条确认问题是否真的可回答、必要事实是否齐全、来源是否仍适用。',
           '同意图的两种说法一起审核；不查看测试集调参。没有做来源实时查询的内容仍是采集快照。','']
    for case in cases:
        lines.extend([f"## {case['id']}",'',f"问题：{case['query']}",'',
                      f"候选分类：{case['type']}；意图组：{case['intent_group']}",'',
                      '候选证据：'])
        for label in case.get('relevant_chunks',[]):
            lines.append(f"- {label['chunk_id']}（草稿相关性 {label['relevance']}）")
        lines.extend(['','请填写：是否可回答 / 问题应如何改 / 必要事实 / 证据是否充分 / 来源是否有效 / 修改理由。',''])
    lines.extend(['## 证据原文',''])
    for row in evidence:
        lines.extend([f"### {row['chunk_id']}",'',f"来源：{row['url']}",'',
                      f"城市：{row['city']}；采集日期：{row['collected_at']}；类型：{row['source_type']}；有效性：{row.get('fact_validity','unknown')}",'',row['text'],''])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text('\n'.join(lines),encoding='utf-8')
    a.out.with_suffix('.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cases':len(cases),'evidence':len(evidence),'scope':'dev_only','reviewed':False}))


if __name__=='__main__':main()
