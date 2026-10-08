"""Render honest development smoke results and generation-call preflight."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    corpus = json.loads((ROOT/'knowledge/benchmark_v1/chunks.json').read_text(encoding='utf-8'))
    cases = json.loads((ROOT/'evaluation/rag_benchmark_v1_draft.json').read_text(encoding='utf-8'))
    report = json.loads((ROOT/'evaluation/rag_v2_dev_sweep_draft.json').read_text(encoding='utf-8'))
    lines = ['# RAG Retrieval Development Smoke', '',
             '> Pending machine annotations: these are pipeline smoke scores, NOT human gold, test/generalization scores or resume-ready quality improvements.', '',
             f"Corpus: {len(corpus)} extracted chunks; city counts {dict(Counter(x['city'] for x in corpus))}.",
             f"Draft queries: {len(cases)}; split counts {dict(Counter(x['split'] for x in cases))}.",
             'Target not achieved: 120-180 reviewed chunks across six cities and 120 reviewed questions with a 60/60 split.',
             'Current chunks include historical travel/transport snapshots. They must not be used as live weather, ticket prices or availability.',
             'Development only. Frozen test NOT evaluated. Production corpus, index, retriever and Top-3 default unchanged.', '',
             '| Retriever | k | Chunk Recall | Hit | MRR | nDCG (linear 0/1/2 gain) | Precision | Fact coverage | Context chars | Errors |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['reports']:
        if row['rrf_c'] != 60: continue
        m=row['metrics']
        lines.append(f"| {row['mode']} | {row['k']} | {m['recall']:.3f} | {m['hit']:.3f} | {m['rr']:.3f} | {m['ndcg']:.3f} | {m['precision']:.3f} | {m['fact_coverage']:.3f} | {row['mean_context_chars']:.0f} | {row['errors']} |")
    lines += ['', '## Interpretation', '',
              'RRF = sum 1/(c + rank), no rounded score sorting. Both fusion channels retrieve up to 20 chunks; final k=1/2/3/5/8/10. Legacy rounded RRF is a separate control over the SAME corpus and embedding.',
              'Recall is relevant chunk hits / all labeled relevant chunks. Hit is any relevant chunk. Fact coverage uses unique evidence groups. Precision uses actually returned passages; current runs return k passages.',
              'No-answer retrieval is reported separately by type in JSON. Without a calibrated abstention threshold, top-k methods often return unrelated neighbors; that is not proof of generation refusal.',
              'Machine questions were drafted from source topics. They are too generic and need manual rewrite/0-1-2 relevance review, cross-chunk and cross-city cases before conclusions.',
              'All draft k candidates reach 10. Do NOT change the production default: inspect corpus/topic quality, query labels and Chinese embedding suitability first.',
              'Independent annotation/review is still missing. Query-group bootstrap intervals are diagnostic, not confidence in real users or multi-annotator agreement.',
              'Fresh-process cold-start latency has NOT been measured. First query per strategy includes setup in a reused process; warm P50/P95 are separate, not online SLA. Context lengths are characters, not billed tokens.', '',
              '## Generation Stage Dry Run', '',
              'After review and retriever selection: 120 questions x 3 neighboring k values x 3 repeats = 1080 generation calls. One independent judge per answer adds 1080 calls (2160 logical calls before provider retries).',
              'No generation/Judge calls executed. Monetary estimate cannot be supplied until model prices, prompt/context budget and actual usage are specified. Human spot checks remain required.', '',
              '## Commands', '', '```powershell',
              'python -m pip install -r requirements-rag-eval.txt',
              'python scripts/collect_rag_corpus.py --max-pages-per-city 10 --allow-external-calls',
              'python scripts/draft_rag_questions.py',
              'python scripts/sweep_rag_v2.py',
              'python scripts/report_rag_v2.py',
              'python -m pytest -q tests/test_rag_benchmark_v2.py', '```', '',
              'Collector reruns change the fingerprint and require fresh question annotation. Do not overwrite a frozen corpus/test set to improve a reported score.']
    path=ROOT/'docs/rag_retrieval_v2_smoke.md'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
