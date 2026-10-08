"""Materialize evidence-based AI proposals; never import, sign, or tune labels."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.rag_protocol import fingerprint


def main():
    corpus = json.loads((ROOT / 'knowledge/benchmark_v2_curated/chunks.json').read_text(encoding='utf-8'))
    # No test cases are inspected or emitted. This report is not a tuning run.
    all_cases = json.loads((ROOT / 'evaluation/rag_benchmark_v2_draft.json').read_text(encoding='utf-8'))
    dev = [c for c in all_cases if c['split'] == 'dev']
    first = json.loads((ROOT / 'docs/rag_review_batch_01_ai.json').read_text(encoding='utf-8'))
    decisions = json.loads((ROOT / 'docs/rag_dev_ai_remaining_decisions.json').read_text(encoding='utf-8'))
    assert first['corpus_fingerprint'] == fingerprint(corpus)
    assert first['all_annotations_fingerprint'] == fingerprint(all_cases)
    assert len(dev) == 60
    lookup = {r['chunk_id']: r for r in corpus}
    first_lookup = {i: g for g in first['groups'] for i in g['case_ids']}
    remaining = {g['intent_group']: g for g in decisions['decisions']}
    rows = []
    for case in dev:
        if case['id'] in first_lookup:
            proposal = first_lookup[case['id']]
        elif case['type'] != 'answerable':
            is_missing = case['type'] == 'in_corpus_missing'
            proposal = {
                'decision': 'retain_negative_clarify_query' if is_missing else 'retain_out_of_scope',
                'proposed_type': case['type'], 'proposed_labels': [],
                'reason': '现有语料无实时名额、天气、排队、停车或临时运营数据。城市已覆盖不代表动态事实已知。'
                          if is_missing else '请求城市或业务主题不在本旅行语料范围内；同城签证、居住政策和房贷不能借旅游证据回答。',
                'expected_behavior': '询问具体场馆或区域及绝对日期；可转实时工具，但失败时不能编造数字。'
                                     if is_missing else '说明当前语料不覆盖此请求，可引导到适用来源，不借用相邻城市政策。',
                'human_checks': ['“明天实时降雨概率”应表述为预报概率；“今日临时闭馆通知是多少”需明确场馆并改为有无适用公告。原题须保留，修改留痕。']
                               if is_missing else ['确认超出的是本语料范围，不代表整个旅行应用永远不能查询此城市。']
            }
        else:
            g = remaining[case['intent_group']]
            proposal = {'decision': g['decision'], 'proposed_type': g['proposed_type'],
                        'reason': g['reason'], 'proposed_labels': [{'chunk_id': i, 'relevance': grade} for i, grade in g['labels']],
                        'proposed_reference_facts': g['facts'], 'missing_facts': g['missing'],
                        'human_checks': [g['human_check']]}
        for label in proposal.get('proposed_labels', []):
            assert label['chunk_id'] in lookup and label['relevance'] in (0, 1, 2)
        rows.append({'id': case['id'], 'query': case['query'], 'intent_group': case['intent_group'],
                     'original_type': case['type'], 'original_labels': case.get('relevant_chunks', []),
                     'proposal': proposal, 'annotation_status': 'pending_review', 'annotator': None,
                     'reviewed_at': None})
    assert len({r['id'] for r in rows}) == 60
    for flag in decisions['corpus_flags']:
        assert all(i in lookup for i in flag['chunk_ids'])
    counts = dict(Counter(r['proposal']['proposed_type'] or 'needs_clarification' for r in rows))
    report = {'schema_version': 1, 'status': 'ai_review_proposal_pending_human', 'scope': 'dev_only',
              'reviewer': first['reviewer'], 'review_date': first['review_date'],
              'corpus_fingerprint': fingerprint(corpus), 'dev_annotations_fingerprint': fingerprint(dev),
              'original_assets_modified': False, 'importable_as_gold': False,
              'retrieval_rankings_consulted': False, 'annotation_counts_proposed': counts,
              'source_checks': first['source_checks'],
              'source_check_limit': 'Only Nanjing/Shanghai listed URLs checked live; all other cities audited against saved text, not live-certified.',
              'cases': rows, 'corpus_flags': decisions['corpus_flags']}
    out = ROOT / 'docs/rag_dev_ai_review.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 60 条开发查询的 AI 证据审核', '',
             '状态：AI 提案，待人工确认。原问题、语料、标签、生产配置与历史结果没有修改。',
             '不参考检索方案、得分或排名；不审核测试集问题。其他城市为保存文本审核，不能称实时来源全部核验。', '',
             f'建议分类（不是正式标签）：{counts}。缺事实项可提供有证据的部分答案，不等于拒绝整条请求。', '',
             '以下保留同义问法，统计时应按意图组区分，不能把它们算独立场景。', '',
             '需要澄清的请求不能强行塞进现有三分类；先由人工决定保留澄清子集或有记录地新建明确问题。', '',
             '## 逐请求判断', '']
    for row in rows:
        p = row['proposal']
        lines += [f"### {row['id']}", '', row['query'], '',
                  f"判断：{p['proposed_type'] or '需澄清'}；{p['reason']}", '',
                  '建议证据：' + (', '.join(f"{l['chunk_id']}（{l['relevance']}）" for l in p.get('proposed_labels', [])) or '没有充分证据'), '',
                  '人工复核：' + '；'.join(p.get('human_checks', [])), '']
    lines += ['## 语料问题（建议另建版本，不改旧数据）', '']
    for flag in decisions['corpus_flags']:
        lines += ['- ' + flag['reason'] + ' IDs: ' + ', '.join(flag['chunk_ids']), '']
    lines += ['## 下一步', '',
              '优先复核需澄清、缺事实、项目建议与官方规定冲突项；再随机抽查可回答项。',
              '确认来源与标签后另建版本，修订必要事实组、替代证据和多证据要求，再跑开发集。',
              '人工签名、配置冻结及正式测试仍由原门禁控制；本报告不能导入为人工金标准。',
              '逐条事实、证据及输入指纹见同名 JSON；第一批详细说明见 rag_review_batch_01_ai.md。', '']
    out.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'cases': len(rows), 'proposed_types': counts, 'human_reviewed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
