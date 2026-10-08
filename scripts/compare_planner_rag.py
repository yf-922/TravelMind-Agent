"""Paired production Planner pilot. Default: offline preflight, never paid calls."""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.travel_knowledge import load_documents
from app.evaluation.rag_protocol import fingerprint
from app.planning.nodes import make_joint_planner_node
from app.planning.schemas import TravelPlanState, TravelRoute
from scripts.evaluate_rag_generation import CallBudget
from tests.eval.graders.code_graders import g1_closed_pool, g4_structure


def pilot_cases():
    facts = [d for d in load_documents() if d.get('knowledge_type') == 'stable_planning_fact']
    cases = []
    for fact in facts:
        city, name = fact['city'], fact['entities'][0]
        for kind, query in [
            ('interest', f'{city}一日游，我对{fact["topic"]}感兴趣，请说明选点依据。'),
            ('unknown_live_fact', f'{city}一日游，想去{name}，请给出明天门票价格和可预约名额，不知道就明确说。'),
            ('insufficient_pool', f'{city}一日游，想安排三个不同景点，不能重复，请说明候选不足。'),
            ('no_evidence', f'{city}一日游，但候选为空，不能编造地点。'),
        ]:
            cases.append({'id': f'{fact["source"]}-{kind}', 'kind': kind,
                          'state': {'query': query, 'destination': city, 'days': 1,
                                    'max_per_day': 3, 'pois': [] if kind == 'no_evidence' else [{'name': name}]},
                          'candidate_provenance': {'kind': 'official_snapshot_entity_not_live_poi',
                                                   'url': fact['url'], 'evidence_id': fact['evidence_id']},
                          'human_checks': ['事实是否获证据支持', '当前票价/名额未知时是否明确待核验',
                                           '是否满足当前兴趣', '候选不足时是否说明且不编造']})
    return cases


def prepare(cases):
    """Freeze lexical evidence and actual production messages without a model."""
    from app.core.travel_knowledge import search_travel_knowledge
    facts = {d['source']: d for d in load_documents() if d.get('knowledge_type') == 'stable_planning_fact'}
    prepared = []
    for case in cases:
        state = TravelPlanState(**case['state'])
        # Explicit keyword pilot avoids uncontrolled embedding downloads/model changes.
        def lookup(city, pois, query):
            eligible = [d['source'] for d in facts.values() if d['city'] == city and
                        set(d['entities']).intersection(p['name'] for p in pois)]
            rows = search_travel_knowledge(f'{city} {" ".join(p["name"] for p in pois)} {query}',
                                           mode='keyword', source_filter=eligible)
            return [{**row, **{key: facts[row['source']][key] for key in
                              ('url', 'collected_at', 'evidence_id', 'validity')}} for row in rows]
        evidence = lookup(state.destination, state.pois, state.query)
        for arm in ('without_rag', 'with_rag'):
            captured = {}
            def capture(llm, messages):
                captured['messages'] = messages
                return TravelRoute(reasoning='preflight only', days=[], notes='not generated')
            make_joint_planner_node(None, llm=object(), invoke_fn=capture,
                                   knowledge_lookup=lambda *args: evidence if arm == 'with_rag' else [])(state)
            messages = captured['messages']
            messages = messages + [('human', 'Return only JSON matching this schema: ' +
                                    json.dumps(TravelRoute.model_json_schema(), ensure_ascii=False))]
            prepared.append({'case_id': case['id'], 'arm': arm, 'messages': messages,
                             'evidence': evidence if arm == 'with_rag' else [], 'state': case['state']})
    return prepared


def summary(results, expected):
    output = {}
    for arm in ('without_rag', 'with_rag'):
        rows = [r for r in results if r['arm'] == arm]
        good = [r for r in rows if not r.get('error')]
        latencies = sorted(r['latency_ms'] for r in rows)
        measured = [r['usage'] for r in rows if r.get('usage') and all(
            type(r['usage'].get(k)) is int for k in ('input_tokens', 'output_tokens'))]
        output[arm] = {'expected': expected, 'recorded': len(rows), 'successful': len(good),
                       'availability': len(good) / expected if rows else None,
                       'basic_rule_pass_including_failures': sum(r['basic_rules']['passed'] for r in good) / expected if rows else None,
                       'p50_ms': statistics.median(latencies) if latencies else None,
                       'p95_ms': latencies[math.ceil(.95 * len(latencies))-1] if latencies else None,
                       'usage_measured_calls': len(measured),
                       'actual_input_tokens': sum(u['input_tokens'] for u in measured) if measured else None,
                       'actual_output_tokens': sum(u['output_tokens'] for u in measured) if measured else None}
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'data/evaluation/planner_rag_pilot.json')
    p.add_argument('--model', default='not_selected')
    p.add_argument('--url')
    p.add_argument('--key-env', default='RAG_GENERATOR_API_KEY')
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--input-price', type=float)
    p.add_argument('--output-price', type=float)
    p.add_argument('--max-output-tokens', type=int, default=2048)
    p.add_argument('--max-calls', type=int, default=0)
    p.add_argument('--max-cost', type=float, default=0)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.repetitions < 1 or a.max_output_tokens < 1: p.error('positive limits required')
    for price in (a.input_price, a.output_price):
        if price is not None and (not math.isfinite(price) or price < 0): p.error('invalid price')
    cases = pilot_cases(); prepared = prepare(cases)
    calls = len(prepared) * a.repetitions
    input_ceiling = sum(sum(len(t.encode('utf-8')) for _, t in r['messages']) + 256 for r in prepared) * a.repetitions
    cost = None if a.input_price is None or a.output_price is None else (
        input_ceiling*a.input_price + calls*a.max_output_tokens*a.output_price)/1_000_000
    report = {'status': 'not_run', 'cases': cases, 'prepared': prepared,
              'retrieval_diagnostic': {'rag_requests': len(cases),
                                       'with_evidence': sum(bool(r['evidence']) for r in prepared if r['arm'] == 'with_rag'),
                                       'interpretation': 'Evidence presence only, not relevance or quality score.'},
              'fingerprint': fingerprint({'cases': cases, 'prepared': prepared, 'model': a.model,
                                          'url': a.url, 'repetitions': a.repetitions, 'max_output': a.max_output_tokens}),
              'preflight': {'requests': len(cases), 'arms': 2, 'repetitions': a.repetitions,
                            'calls_ceiling': calls, 'input_token_ceiling_estimate': input_ceiling,
                            'cost_ceiling': cost, 'price_unit': 'specified currency / million tokens',
                            'estimate_method': 'UTF-8 bytes + 256 overhead per prompt; not measured usage'},
              'scope': 'Planner-only controlled pilot; keyword evidence; sequential, no retries/Judge/maps; not end-to-end quality or latency.',
              'model': a.model, 'results': []}
    def save():
        report['summary'] = summary(report['results'], len(cases)*a.repetitions)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if a.out.exists(): p.error('output already exists; choose a fresh file to preserve prior results')
    save(); print(json.dumps(report['preflight'], ensure_ascii=False))
    if not a.execute: return
    if not a.url or a.model == 'not_selected' or cost is None: p.error('model, URL and both prices required')
    if a.max_calls < calls or not math.isfinite(a.max_cost) or a.max_cost < cost: p.error('approved budget below ceiling')
    import os
    from langchain_openai import ChatOpenAI
    if not os.getenv(a.key_env): p.error('API key environment variable missing')
    client = ChatOpenAI(model=a.model, base_url=a.url, api_key=os.environ[a.key_env],
                        temperature=.3, max_tokens=a.max_output_tokens, max_retries=0, timeout=90,
                        model_kwargs={'response_format': {'type': 'json_object'}})
    budget = CallBudget(a.max_calls, a.max_cost, a.input_price, a.output_price, a.max_output_tokens)
    report['status'] = 'running'; save()
    for repetition in range(a.repetitions):
        # Alternate pair ordering to reduce systematic warm-up/order bias.
        order = prepared if repetition % 2 == 0 else [r for pair in zip(prepared[::2], prepared[1::2]) for r in reversed(pair)]
        for row in order:
            result = {'case_id': row['case_id'], 'arm': row['arm'], 'repetition': repetition,
                      'human_review_status': 'pending', 'human_scores': None}
            start = time.perf_counter()
            try:
                reservation = budget.reserve(row['messages'])
            except RuntimeError as exc:
                report['status'] = 'budget_stopped'; report['stop_reason'] = str(exc); save(); return
            try:
                response = client.invoke(row['messages'])
                result['usage'] = response.usage_metadata
                result['raw_response'] = response.content
                budget.reconcile(reservation, response.usage_metadata)
                route = TravelRoute.model_validate_json(response.content)
                result['generated'] = route.model_dump()
                days = [d.model_dump() for d in route.days]
                closed = g1_closed_pool(days, row['state']['pois'])
                structure = g4_structure(days, row['state']['pois'], row['state']['max_per_day'])
                nonempty = bool(days) and all(d['spots'] for d in days)
                result['basic_rules'] = {'passed': closed[0] and structure[0] and nonempty,
                                         'closed_pool': closed, 'structure': structure, 'nonempty': nonempty}
                # Empty-pool refusal needs human review, not an empty-route success claim.
            except Exception as exc:
                if 'usage' not in result:
                    budget.reconcile(reservation, None)
                result['error'] = type(exc).__name__ + ': ' + str(exc)
            result['latency_ms'] = (time.perf_counter()-start)*1000
            report['results'].append(result); report['budget'] = budget.snapshot(); save()
    report['status'] = 'completed_pending_human_review'; save()


if __name__ == '__main__': main()
