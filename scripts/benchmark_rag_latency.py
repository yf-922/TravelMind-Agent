"""Separate subprocess cold-start measurements from reused-process warm latency."""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.rag_retrieval import search,embedding_config,corpus_fingerprint


def percentile(values,p):
    import math
    return sorted(values)[max(0,math.ceil(len(values)*p)-1)] if values else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--query',required=True)
    p.add_argument('--mode',required=True)
    p.add_argument('--k',type=int,default=3)
    p.add_argument('--repeat',type=int,default=10)
    p.add_argument('--cold-repeat',type=int,default=3)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if not 1<=a.repeat<=100 or not 1<=a.cold_repeat<=10:p.error('invalid repeat count')
    rows=json.loads(a.corpus.read_text(encoding='utf-8'))
    if a.worker:
        search(a.query,a.mode,a.k,rows=rows)
        return
    cold=[];warm=[];errors=[]
    for i in range(a.cold_repeat):
        started=time.perf_counter()
        result=subprocess.run([sys.executable,__file__,'--corpus',str(a.corpus),'--query',a.query,'--mode',a.mode,'--k',str(a.k),'--worker'],capture_output=True,timeout=300)
        cold.append((time.perf_counter()-started)*1000)
        if result.returncode:errors.append({'phase':'cold','trial':i,'error':result.stderr.decode('utf-8',errors='replace')[-2000:]})
    try:search(a.query,a.mode,a.k,rows=rows)
    except Exception as exc:errors.append({'phase':'prewarm','error':str(exc)})
    for i in range(a.repeat):
        started=time.perf_counter()
        try:search(a.query,a.mode,a.k,rows=rows)
        except Exception as exc:errors.append({'phase':'warm','trial':i,'error':str(exc)})
        warm.append((time.perf_counter()-started)*1000)
    report={'cold_includes_interpreter_import_model_and_query':True,'cold_ms':cold,'warm_ms':warm,
        'cold_p50':statistics.median(cold),'cold_p95':percentile(cold,.95),'warm_p50':statistics.median(warm),
        'warm_p95':percentile(warm,.95),'errors':errors,'embedding':embedding_config(),
        'corpus_fingerprint':corpus_fingerprint(rows),'not_online_sla':True}
    if a.out:a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':main()
