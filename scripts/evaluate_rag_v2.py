"""Strict chunk-level RAG benchmark with frozen dev/test splits."""
from __future__ import annotations
import argparse, json, math, statistics, sys, time, random, importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from app.evaluation.rag_retrieval import load_chunks, search, corpus_fingerprint
from app.evaluation.rag_protocol import verify_freeze, fingerprint

MODES=("keyword","bm25","vector","hybrid","keyword_rrf","legacy_hybrid")
KS=(1,2,3,5,8,10)

def summarize_rows(rows):
    names=('hit','recall','precision','rr','ndcg','fact_coverage')
    positives=[r for r in rows if r['answerable']]
    valid=[r['grade'] for r in positives if not r['error'] and r['grade']]
    negatives=[r for r in rows if not r['answerable']]
    unconditional={n:statistics.mean(r['grade'][n] if not r['error'] and r['grade'] else 0 for r in positives) if positives else None for n in names}
    return {'metrics':unconditional,
            'completed_only_metrics':{n:statistics.mean(g[n] for g in valid) if valid else None for n in names},
            'availability':sum(not r['error'] for r in rows)/len(rows) if rows else None,
            'no_answer_by_type':{kind:{'count':len(sub),'empty_rate':sum(not r['retrieved'] and not r['error'] for r in sub)/len(sub),
                'retrieved_rate':sum(bool(r['retrieved']) for r in sub)/len(sub),'error_rate':sum(bool(r['error']) for r in sub)/len(sub)}
                for kind in sorted({str(r['type']) for r in negatives}) for sub in [[r for r in negatives if str(r['type'])==kind]]},
            'mean_context_chars':statistics.mean(r['context_chars'] for r in rows) if rows else 0}

def dcg(values): return sum(value / math.log2(index + 2) for index, value in enumerate(values))
def score_case(case, retrieved, k=None):
    relevant={str(x["chunk_id"]): int(x.get("relevance",2)) for x in case.get("relevant_chunks",[])}
    ranked=[]
    for x in retrieved:
        chunk_id=str(x.get("chunk_id"))
        if chunk_id not in ranked:
            ranked.append(chunk_id)
    if k is not None:
        ranked=ranked[:k]
    rel=[relevant.get(x,0) for x in ranked]
    first=next((i+1 for i,x in enumerate(rel) if x>0),None)
    ideal=sorted(relevant.values(), reverse=True)[:k if k is not None else len(rel)]
    groups = {str(x.get("evidence_group",x["chunk_id"])) for x in case.get("relevant_chunks",[]) if x.get("relevance",2)>0}
    label_groups={str(x['chunk_id']):str(x.get('evidence_group',x['chunk_id'])) for x in case.get('relevant_chunks',[]) if x.get('relevance',2)>0}
    hit_groups = {label_groups[x] for x in ranked if x in label_groups}
    return {"hit": bool(first), "rr": 1/first if first else 0.0,
            "fact_coverage": len(hit_groups & groups)/len(groups) if groups else 0.0,
            "recall": sum(x>0 for x in rel)/max(1,sum(v>0 for v in relevant.values())),
            "precision": sum(x>0 for x in rel)/max(1,k if k is not None else len(rel)),
            "ndcg": dcg(rel)/dcg(ideal) if ideal and dcg(ideal) else 0.0,
            "no_answer_retrieved": bool(ranked) if not relevant else None}

def evaluate(cases, mode, k, c=60, corpus=None):
    if not cases:
        raise ValueError('empty evaluation split')
    rows=[]; latency=[]
    for case in cases:
        started=time.perf_counter()
        error=None; result=[]
        try: result=search(case["query"],mode,k,c,rows=corpus)
        except Exception as exc: error=type(exc).__name__+': '+str(exc)
        latency.append((time.perf_counter()-started)*1000)
        grade=score_case(case,result,k) if not error else None
        rows.append({"id":case["id"],"intent_group":case.get("intent_group",case['id']),"type":case.get("type"),"answerable":case.get('type')=='answerable' if case.get('type') else any(x.get('relevance',2)>0 for x in case.get('relevant_chunks',[])),"error":error,"retrieved":[x.get("chunk_id") for x in result],"grade":grade,
                     "context_chars":sum(len(x.get("text","")) for x in result),"latency_ms":latency[-1]})
    valid=[x["grade"] for x in rows if x["grade"] and x["answerable"]]
    answerable=[x for x in rows if x["answerable"]]
    completed=[x for x in answerable if x["grade"]]
    negatives=[x for x in rows if not x["answerable"]]
    report={"mode":mode,"k":k,"rrf_c":c,"case_count":len(cases),"errors":sum(bool(x["error"]) for x in rows),
            "metrics": {name: round(statistics.mean(g[name] for g in valid),4) if valid else None for name in ("hit","recall","precision","rr","ndcg","fact_coverage")},
            "quality_accounting": {"answerable": len(answerable), "completed": len(completed),
                "errors": len(answerable)-len(completed), "availability": len(completed)/len(answerable) if answerable else None,
                "unconditional": {name: round(sum(g[name] for g in valid)/len(answerable),4) if answerable else None for name in ("hit","recall","precision","rr","ndcg","fact_coverage")}},
            "no_answer_by_type":{kind:{"count":len(sub),"empty_rate":sum(not x['retrieved'] and not x['error'] for x in sub)/len(sub),"retrieved_rate":sum(bool(x['retrieved']) for x in sub)/len(sub)} for kind in sorted({str(x['type']) for x in negatives}) for sub in [[x for x in negatives if str(x['type'])==kind]]},
            "mean_context_chars":statistics.mean(x['context_chars'] for x in rows),
            "initialization_first_query_ms":latency[0], "cold_process_start_measured":False,
            "warm_latency_ms":{"p50":statistics.median(latency[1:]) if len(latency)>1 else None,"p95":sorted(latency[1:])[max(0,math.ceil(.95*(len(latency)-1))-1)] if len(latency)>1 else None},
            "latency_ms":{"p50":round(statistics.median(latency),2),"p95":round(sorted(latency)[max(0,math.ceil(.95*len(latency))-1)],2)},"cases":rows}
    report.update(summarize_rows(rows))
    return report


def select_k(reports):
    usable=[x for x in reports if x['metrics']['recall'] is not None and not x['errors']]
    if not usable: return None
    best_recall=max(x['metrics']['recall'] for x in usable)
    best_ndcg=max(x['metrics']['ndcg'] for x in usable)
    eligible=[x for x in usable if best_recall-x['metrics']['recall']<=.02+1e-9 and best_ndcg-x['metrics']['ndcg']<=.02+1e-9]
    chosen=min(eligible,key=lambda x:x['k']) if eligible else max(usable,key=lambda x:(x['metrics']['recall'],x['metrics']['ndcg'],-x['k']))
    return chosen['k']


def paired_bootstrap(a,b,metric='recall',seed=20261008,iterations=1000):
    amap={x['id']:x for x in a['cases'] if x['answerable']}
    bmap={x['id']:x for x in b['cases'] if x['answerable']}
    ids=sorted(amap.keys() & bmap.keys())
    if not ids:return None
    def value(row):
        return row['grade'][metric] if row['grade'] and not row.get('error') else 0
    delta=[value(bmap[i])-value(amap[i]) for i in ids]
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
    p.add_argument('--corpus',type=Path)
    p.add_argument('--freeze',type=Path)
    a=p.parse_args(); all_cases=json.loads(a.cases.read_text(encoding="utf-8")); cases=[x for x in all_cases if x["split"]==a.split]
    corpus=json.loads(a.corpus.read_text(encoding='utf-8')) if a.corpus else load_chunks()
    if a.mode in {"vector","hybrid","keyword_rrf","legacy_hybrid"} and not a.allow_vector: p.error("vector/hybrid require --allow-vector and an explicit real embedding run")
    if not cases: p.error("selected split is empty")
    if a.split == 'test' and any(x.get('annotation_status') != 'human_reviewed' for x in cases): p.error('test evaluation requires human-reviewed frozen annotations')
    config={'mode':a.mode,'k':a.k,'rrf_c':a.rrf_c}
    if a.split == 'test':
        if not a.freeze: p.error('--freeze required for test evaluation')
        try: verify_freeze(a.freeze,all_cases,corpus,config)
        except (ValueError,OSError) as exc: p.error(str(exc))
    annotation_status={str(x.get('annotation_status')) for x in cases}
    frozen = a.split == 'test' and annotation_status == {'human_reviewed'}
    report=evaluate(cases,a.mode,a.k,a.rrf_c,corpus); report.update({"corpus_fingerprint":corpus_fingerprint(corpus),"annotation_fingerprint":fingerprint(all_cases),"configuration_fingerprint":fingerprint(config),"annotation_status":next(iter(annotation_status)) if len(annotation_status)==1 else 'mixed',"split":a.split,"formula_version":"chunk-relevance-v2-linear-dcg","frozen":frozen})
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report["metrics"],ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
