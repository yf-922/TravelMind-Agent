"""Development-only retrieval curves. Pending labels never choose production defaults."""
import json
import sys
import time
import importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_rag_v2 import evaluate, select_k, paired_bootstrap, score_case, KS
from app.evaluation.rag_retrieval import corpus_fingerprint, load_chunks
import statistics


def main():
    cases=json.loads((ROOT/'evaluation/rag_benchmark_v1_draft.json').read_text(encoding='utf-8'))
    cases=[c for c in cases if c['split']=='dev']
    reports=[]
    lookup={x['id']:x for x in cases}
    chunks={x['chunk_id']:x for x in load_chunks()}
    for mode in ('keyword','bm25','vector','hybrid','keyword_rrf','legacy_hybrid'):
        for c in ((20,60,100) if mode in ('hybrid','keyword_rrf') else (60,)):
            # Retrieve once at k=10, then score frozen prefixes. This keeps
            # the sweep rankings identical rather than recomputing each k.
            full=evaluate(cases,mode,10,c)
            for k in KS:
                print(mode,c,k,flush=True)
                report=json.loads(json.dumps(full))
                report['k']=k
                report['latency_boundary']='Measured full Top20 channel retrieval; k prefix scoring adds no model calls. First request includes initialization, remaining are warm.'
                for row in report['cases']:
                    row['retrieved']=row['retrieved'][:k]
                    retrieved=[chunks[i] for i in row['retrieved']]
                    row['grade']=score_case(lookup[row['id']],retrieved) if not row['error'] else None
                    row['context_chars']=sum(len(x['text']) for x in retrieved)
                valid=[x['grade'] for x in report['cases'] if x['answerable'] and x['grade']]
                report['metrics']={name:statistics.mean(x[name] for x in valid) if valid else None for name in ('hit','recall','precision','rr','ndcg','fact_coverage')}
                report['mean_context_chars']=statistics.mean(x['context_chars'] for x in report['cases'])
                reports.append(report)
    out={'schema_version':2,'annotation_status':'pending_review','scope':'dev only, smoke metrics, not gold conclusions',
         'corpus_fingerprint':corpus_fingerprint(),'production_default_changed':False,
         'embedding':{'implementation':'Chroma ONNXMiniLM_L6_V2 default','model':'all-MiniLM-L6-v2','chroma_version':importlib.metadata.version('chromadb'),'distance':'squared L2','language_limitation':'English-oriented model; Chinese retrieval quality must be evaluated'},
         'tokenizer':{'name':'jieba','version':importlib.metadata.version('jieba'),'cut_all':False},
         'reports':reports,'draft_k_candidates':{m:select_k([r for r in reports if r['mode']==m and r['rrf_c']==60]) for m in ('keyword','bm25','vector','hybrid','keyword_rrf')},
         'bootstrap_hybrid_vs_bm25_k3':paired_bootstrap(next(r for r in reports if r['mode']=='bm25' and r['k']==3),next(r for r in reports if r['mode']=='hybrid' and r['k']==3 and r['rrf_c']==60))}
    (ROOT/'evaluation/rag_v2_dev_sweep_draft.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out['draft_k_candidates'])


if __name__=='__main__':main()
