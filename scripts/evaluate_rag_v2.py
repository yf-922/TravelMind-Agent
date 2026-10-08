"""Strict chunk-level RAG benchmark with frozen dev/test splits."""
from __future__ import annotations
import argparse, json, math, statistics, sys, time, random, importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from app.evaluation.rag_retrieval import load_chunks, search, corpus_fingerprint

MODES=("keyword","bm25","vector","hybrid","keyword_rrf","legacy_hybrid")
KS=(1,2,3,5,8,10)

def dcg(values): return sum(value / math.log2(index + 2) for index, value in enumerate(values))
def score_case(case, retrieved):
    relevant={str(x["chunk_id"]): int(x.get("relevance",2)) for x in case.get("relevant_chunks",[])}
    ranked=[str(x.get("chunk_id")) for x in retrieved]
    rel=[relevant.get(x,0) for x in ranked]
    first=next((i+1 for i,x in enumerate(rel) if x>0),None)
    ideal=sorted(relevant.values(), reverse=True)[:len(rel)]
    groups = {str(x.get("evidence_group",x["chunk_id"])) for x in case.get("relevant_chunks",[]) if x.get("relevance",2)>0}
    hit_groups = {str(x.get("evidence_group",x["chunk_id"])) for x in retrieved if relevant.get(str(x["chunk_id"]),0)>0}
    return {"hit": bool(first), "rr": 1/first if first else 0.0,
            "fact_coverage": len(hit_groups & groups)/len(groups) if groups else 0.0,
            "recall": sum(x>0 for x in rel)/max(1,sum(v>0 for v in relevant.values())),
            "precision": sum(x>0 for x in rel)/max(1,len(rel)),
            "ndcg": dcg(rel)/dcg(ideal) if ideal and dcg(ideal) else 0.0,
            "no_answer_retrieved": bool(ranked) if not relevant else None}

def evaluate(cases, mode, k, c=60):
    rows=[]; latency=[]
    for case in cases:
        started=time.perf_counter()
        error=None; result=[]
        try: result=search(case["query"],mode,k,c)
        except Exception as exc: error=type(exc).__name__
        latency.append((time.perf_counter()-started)*1000)
        grade=score_case(case,result) if not error else None
        rows.append({"id":case["id"],"intent_group":case.get("intent_group",case['id']),"type":case.get("type"),"answerable":bool(case.get("relevant_chunks")),"error":error,"retrieved":[x.get("chunk_id") for x in result],"grade":grade,
                     "context_chars":sum(len(x.get("text","")) for x in result),"latency_ms":latency[-1]})
    valid=[x["grade"] for x in rows if x["grade"] and x["answerable"]]
    negatives=[x for x in rows if not x["answerable"]]
    return {"mode":mode,"k":k,"rrf_c":c,"case_count":len(cases),"errors":sum(bool(x["error"]) for x in rows),
            "metrics": {name: round(statistics.mean(g[name] for g in valid),4) if valid else None for name in ("hit","recall","precision","rr","ndcg","fact_coverage")},
            "no_answer_by_type":{kind:{"count":len(sub),"empty_rate":sum(not x['retrieved'] and not x['error'] for x in sub)/len(sub),"retrieved_rate":sum(bool(x['retrieved']) for x in sub)/len(sub)} for kind in sorted({str(x['type']) for x in negatives}) for sub in [[x for x in negatives if str(x['type'])==kind]]},
            "mean_context_chars":statistics.mean(x['context_chars'] for x in rows),
            "initialization_first_query_ms":latency[0], "cold_process_start_measured":False,
            "warm_latency_ms":{"p50":statistics.median(latency[1:]) if len(latency)>1 else None,"p95":sorted(latency[1:])[max(0,math.ceil(.95*(len(latency)-1))-1)] if len(latency)>1 else None},
            "latency_ms":{"p50":round(statistics.median(latency),2),"p95":round(sorted(latency)[max(0,math.ceil(.95*len(latency))-1)],2)},"cases":rows}


def select_k(reports):
    usable=[x for x in reports if x['metrics']['recall'] is not None and not x['errors']]
    if not usable: return None
    best_recall=max(x['metrics']['recall'] for x in usable)
    best_ndcg=max(x['metrics']['ndcg'] for x in usable)
    eligible=[x for x in usable if best_recall-x['metrics']['recall']<=.02+1e-9 and best_ndcg-x['metrics']['ndcg']<=.02+1e-9]
    chosen=min(eligible,key=lambda x:x['k']) if eligible else max(usable,key=lambda x:(x['metrics']['recall'],x['metrics']['ndcg'],-x['k']))
    return chosen['k']


def paired_bootstrap(a,b,metric='recall',seed=20261008,iterations=1000):
    amap={x['id']:x for x in a['cases'] if x['answerable'] and x['grade']}
    bmap={x['id']:x for x in b['cases'] if x['answerable'] and x['grade']}
    ids=sorted(amap.keys() & bmap.keys())
    if not ids:return None
    delta=[bmap[i]['grade'][metric]-amap[i]['grade'][metric] for i in ids]
    groups={}
    for i,d in zip(ids,delta):groups.setdefault(amap[i].get('intent_group',i),[]).append(d)
    rng=random.Random(seed)
    samples=[]
    for _ in range(iterations):
        selected=rng.choices(list(groups),k=len(groups))
        samples.append(statistics.mean(d for group in selected for d in groups[group]))
    samples.sort()
    return {'paired_n':len(ids),'intent_groups':len(groups),'resampling_unit':'intent_group','seed':seed,'iterations':iterations,'mean_delta':statistics.mean(delta),'ci95':[samples[int(.025*iterations)],samples[int(.975*iterations)-1]],'improved':[i for i,d in zip(ids,delta) if d>0],'degraded':[i for i,d in zip(ids,delta) if d<0]}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--cases",type=Path,default=ROOT/"evaluation/rag_benchmark_v1.json"); p.add_argument("--split",choices=("dev","test"),default="dev"); p.add_argument("--mode",choices=MODES,default="keyword"); p.add_argument("--k",type=int,choices=KS,default=3); p.add_argument("--rrf-c",type=int,choices=(20,60,100),default=60); p.add_argument("--out",type=Path,default=ROOT/"evaluation/rag_v2_report.json"); p.add_argument("--allow-vector",action="store_true")
    a=p.parse_args(); cases=json.loads(a.cases.read_text(encoding="utf-8")); cases=[x for x in cases if x["split"]==a.split]
    if a.mode in {"vector","hybrid","keyword_rrf","legacy_hybrid"} and not a.allow_vector: p.error("vector/hybrid require --allow-vector and an explicit real embedding run")
    if not cases: p.error("selected split is empty")
    if a.split == 'test' and any(x.get('annotation_status') != 'human_reviewed' for x in cases): p.error('test evaluation requires human-reviewed frozen annotations')
    report=evaluate(cases,a.mode,a.k,a.rrf_c); report.update({"corpus_fingerprint":corpus_fingerprint(),"annotation_status":"pending_review","split":a.split,"formula_version":"chunk-relevance-v1"})
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report["metrics"],ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
