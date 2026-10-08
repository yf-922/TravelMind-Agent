"""Refresh timing/bootstrap metadata from saved observations, no new model calls."""
import json
import statistics
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_rag_v2 import paired_bootstrap
from scripts.report_rag_v2 import main as render

path=ROOT/'evaluation/rag_v2_dev_sweep_draft.json'
report=json.loads(path.read_text(encoding='utf-8'))
cases={x['id']:x for x in json.loads((ROOT/'evaluation/rag_benchmark_v1_draft.json').read_text(encoding='utf-8'))}
for result in report['reports']:
    latency=[x['latency_ms'] for x in result['cases']]
    for row in result['cases']:row['intent_group']=cases[row['id']]['intent_group']
    result.pop('cold_first_query_ms',None)
    result['initialization_first_query_ms']=latency[0]
    result['cold_process_start_measured']=False
    warm=sorted(latency[1:])
    result['warm_latency_ms']={'p50':statistics.median(warm),'p95':warm[max(0,math.ceil(.95*len(warm))-1)]}
report['bootstrap_hybrid_vs_bm25_k3']=paired_bootstrap(next(r for r in report['reports'] if r['mode']=='bm25' and r['k']==3),next(r for r in report['reports'] if r['mode']=='hybrid' and r['k']==3 and r['rrf_c']==60))
path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
render()
