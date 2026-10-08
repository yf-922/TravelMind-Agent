"""Offline audit of production RAG wiring; no models or providers called."""
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core import travel_knowledge
from app.planning import graph, nodes
from app.evaluation.rag_protocol import fingerprint


def main():
    docs = travel_knowledge.load_documents()
    chunks = travel_knowledge._all_chunks()
    corpus = json.loads((ROOT / 'knowledge/benchmark_v2_curated/chunks.json').read_text(encoding='utf-8'))
    factory = inspect.getsource(graph._planner_for_graph)
    joint = inspect.getsource(nodes.make_joint_planner_node)
    legacy = inspect.getsource(nodes.make_planner_node)
    probes = [
        '重庆 历史建筑 景点内容', '南京博物院 主要展览 游览区域',
        '上海博物馆 东馆 专项预约', '丽江 玉龙雪山 索道 门票价格',
        '三亚 蜈支洲岛 游览区域', '景德镇 陶瓷博物馆 展览内容',
    ]
    results = [{'query': q, 'rows': travel_knowledge.search_travel_knowledge(q, mode='keyword')}
               for q in probes]
    report = {
        'status': 'offline_diagnostic_not_quality_benchmark',
        'external_calls': 0, 'embedding_calls': 0,
        'method': 'Read factory/node source; inventory runtime Markdown and planning facts; keyword-only probes without human relevance labels.',
        'default_planner': 'make_joint_planner_node' if 'return make_joint_planner_node(model_name)' in factory else 'unknown',
        'joint_planner_calls_rag': 'search_planning_knowledge(' in joint,
        'legacy_planner_calls_rag': 'search_travel_knowledge(' in legacy,
        'runtime_source_directory': 'knowledge/travel',
        'runtime_documents': [d['source'] for d in docs],
        'runtime_document_count': len(docs), 'runtime_chunk_count': len(chunks),
        'runtime_fingerprint': fingerprint(docs),
        'evaluation_corpus': 'knowledge/benchmark_v2_curated/chunks.json',
        'evaluation_chunk_count': len(corpus),
        'evaluation_cities': dict(Counter(r['city'] for r in corpus)),
        'evaluation_fingerprint': fingerprint(corpus),
        'runtime_uses_evaluation_corpus': 'benchmark_v2_curated' in inspect.getsource(travel_knowledge),
        'probes': results,
        'limits': ['Source inspection does not measure route quality or demonstrate RAG benefit.',
                   'Nonempty keyword results are not evidence of relevance; no success rate is calculated.',
                   'No local vector index or runtime environment secrets are inspected.'],
    }
    out = ROOT / 'docs/travel_rag_usage_audit.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('probes', 'limits')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
