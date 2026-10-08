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
import argparse
from scripts.evaluate_rag_v2 import summarize_rows
from app.evaluation.rag_retrieval import embedding_config,lexical_config
from app.evaluation.rag_protocol import fingerprint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,default=ROOT/'evaluation/rag_benchmark_v1_draft.json')
    p.add_argument('--corpus',type=Path)
    p.add_argument('--modes',nargs='+',default=['keyword','bm25','vector','hybrid','keyword_rrf','legacy_hybrid'])
    p.add_argument('--out',type=Path,default=ROOT/'evaluation/rag_v2_dev_sweep_draft.json')
    a=p.parse_args()
    cases=json.loads(a.cases.read_text(encoding='utf-8'))
    cases=[c for c in cases if c['split']=='dev']
    reports=[]
    lookup={x['id']:x for x in cases}
    corpus=json.loads(a.corpus.read_text(encoding='utf-8')) if a.corpus else load_chunks()
    chunks={x['chunk_id']:x for x in corpus}
    for mode in a.modes:
        for c in ((20,60,100) if mode in ('hybrid','keyword_rrf') else (60,)):
            # Retrieve once at k=10, then score frozen prefixes. This keeps
            # the sweep rankings identical rather than recomputing each k.
            full=evaluate(cases,mode,10,c,corpus)
            for k in KS:
                print(mode,c,k,flush=True)
                report=json.loads(json.dumps(full))
                report['k']=k
                report['latency_boundary']='Measured full Top20 channel retrieval; k prefix scoring adds no model calls. First request includes initialization, remaining are warm.'
                for row in report['cases']:
                    row['retrieved']=row['retrieved'][:k]
                    retrieved=[chunks[i] for i in row['retrieved']]
                    row['grade']=score_case(lookup[row['id']],retrieved,k) if not row['error'] else None
                    row['context_chars']=sum(len(x['text']) for x in retrieved)
                valid=[x['grade'] for x in report['cases'] if x['answerable'] and x['grade']]
                report.update(summarize_rows(report['cases']))
                reports.append(report)
    out={'schema_version':2,'annotation_status':'pending_review','scope':'dev only, smoke metrics, not gold conclusions',
         'corpus_fingerprint':corpus_fingerprint(corpus),'annotation_fingerprint':fingerprint(cases),'production_default_changed':False,
         'embedding':embedding_config(),
         'tokenizer':lexical_config(),
         'reports':reports,'draft_k_candidates':{m:select_k([r for r in reports if r['mode']==m and r['rrf_c']==60]) for m in a.modes}}
    if 'hybrid' in a.modes and 'bm25' in a.modes:
        out['bootstrap_hybrid_vs_bm25_k3']=paired_bootstrap(next(r for r in reports if r['mode']=='bm25' and r['k']==3),next(r for r in reports if r['mode']=='hybrid' and r['k']==3 and r['rrf_c']==60))
    out['draft_configuration_candidates']={}
    for mode in a.modes:
        options=[r for r in reports if r['mode']==mode and not r['errors']]
        if not options:
            out['draft_configuration_candidates'][mode]=None
            continue
        # Select c and k jointly using the same quality tolerance, dev only.
        best_recall=max(r['metrics']['recall'] for r in options)
        best_ndcg=max(r['metrics']['ndcg'] for r in options)
        eligible=[r for r in options if best_recall-r['metrics']['recall']<=.02+1e-9 and best_ndcg-r['metrics']['ndcg']<=.02+1e-9]
        selected=min(eligible,key=lambda r:(r['k'],-r['metrics']['ndcg'],abs(r['rrf_c']-60))) if eligible else max(options,key=lambda r:(r['metrics']['recall'],r['metrics']['ndcg'],-r['k']))
        out['draft_configuration_candidates'][mode]={key:selected[key] for key in ('mode','k','rrf_c')}
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out['draft_k_candidates'])


if __name__=='__main__':main()
