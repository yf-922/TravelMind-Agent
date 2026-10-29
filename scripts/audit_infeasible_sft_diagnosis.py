"""Read-only infeasibility diagnosis. No API, model generation, or training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.planning.schemas import TravelRoute
from scripts.regen_v1_v2_current_protocol import audit, make_tokenizer
from scripts.build_representative_eval50 import score_case
from scripts.prepare_sft_v2_release import explanation_faults

DATA = ROOT/'data/post_train'
OUT = DATA/'infeasible_diagnosis_20261025'


def read(path):
    text = path.read_text(encoding='utf-8')
    return json.loads(text) if text.lstrip().startswith('[') else [json.loads(x) for x in text.splitlines() if x.strip()]


def main():
    suite = DATA/'representative_eval50_v1'
    cases = json.loads((suite/'cases.json').read_text(encoding='utf-8'))
    infeasible = [c for c in cases if c['expected'] == 'explained_infeasible']
    maps = json.loads((suite/'maps.json').read_text(encoding='utf-8'))['routes']
    tokenizer = make_tokenizer()
    proofs = []
    for case in infeasible:
        state = case['state']
        kind = case['infeasibility_proof']['kind']
        names = {p['name'] for p in state['pois']}
        known_indoor = sum(p.get('indoor') is True for p in state['pois'])
        proof_holds = len(names) == 2 if kind == 'pigeonhole' else (
            known_indoor == 0 and all(p.get('indoor') is None for p in state['pois'])
            and float(state['weather_forecast'][0]['day_temp']) >= 38)
        witness = case['feasibility_witness']
        TravelRoute.model_validate(witness)
        proofs.append(dict(id=case['id'], kind=kind, proof_holds=proof_holds,
            query=state['query'], distinct_pool=len(names), known_indoor=known_indoor,
            eval_witness_faults=score_case(witness, case, maps),
            training_gate_witness_faults=audit(case, witness, maps, tokenizer)[0]))
    training = {}
    sources = {
        'v1-current': DATA/'release_current_protocol_v1_v2/v1/train.ledger.jsonl',
        'v2-current': DATA/'release_current_protocol_v1_v2/v2/train.ledger.jsonl',
        'validation': DATA/'release_current_protocol_v1_v2/v2/validation.ledger.jsonl',
        'v2-large': DATA/'v2_large_pool_teacher_contract_v3/sft_candidate/synthesis.ledger.jsonl',
    }
    for tag, path in sources.items():
        rows = read(path)
        unknown_claims = []
        minimum_count_failures = []
        for row in rows:
            state, plan = row['state'], row['chosen']
            by_name = {p['name']:p for p in state['pois']}
            spots = [s for d in plan['days'] for s in d['spots']]
            for reason in plan.get('selection_reasons', []):
                if by_name.get(reason['name'], {}).get('indoor') is None and re.search(
                        '明确室内|纯室内|全室内|室内展|室内参观|室内场', reason['reason']):
                    unknown_claims.append(dict(id=row['id'], reason=reason))
            minimum = re.search(r'全程至少([0-9]+)个不同景点', state['query'])
            if minimum and len({s['name'] for s in spots}) < int(minimum[1]):
                minimum_count_failures.append(row['id'])
        training[tag] = dict(n=len(rows), source=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            empty_day_answers=sum(any(not d['spots'] for d in r['chosen']['days']) for r in rows),
            all_empty_answers=sum(all(not d['spots'] for d in r['chosen']['days']) for r in rows),
            all_unknown_pools=sum(all(p.get('indoor') is None for p in r['state']['pois']) for r in rows),
            unknown_pool_rows=sum(any(p.get('indoor') is None for p in r['state']['pois']) for r in rows),
            explicit_infeasible_query_rows=sum(any(w in r['state']['query'] for w in
                ['若无解', '若无法同时满足', '无法满足', '不可行']) for r in rows),
            narrow_unknown_indoor_claim_flags=unknown_claims,
            explicit_total_minimum_count_failures=minimum_count_failures)
    results = {}
    ids = {c['id'] for c in infeasible}
    by_id = {c['id']:c for c in infeasible}
    for tag in ['base', 'v1-current', 'v2-current', 'v2-large']:
        for mode in ['planner-only', 'rules-rework']:
            path = suite/'current_retrained_20261025'/f'{tag}_{mode}/results.jsonl'
            rows = [r for r in read(path) if r['id'] in ids]
            rounds = [rr for r in rows for rr in r['rounds']]
            sample = []
            for row in rows:
                if row['seed'] != 42:
                    continue
                sample.append(dict(id=row['id'], faults=row['faults'], plan=row['plan'],
                    explanation_checker_faults=explanation_faults(row['plan']),
                    rework_feedback=[r['request_messages'][-1]['content'].split('\n\n确定性规则返工反馈')[-1]
                        for r in row['rounds'][1:]]))
            results[tag+'/'+mode] = dict(n=len(rows), passed=sum(r['passed'] for r in rows),
                empty_final=sum(not any(d['spots'] for d in r['plan']['days']) for r in rows),
                generation_rounds=len(rounds),
                safe_rounds=sum(not score_case(rr['plan'], by_id[r['id']], maps)
                    for r in rows for rr in r['rounds']),
                final_faults=dict(Counter(f for r in rows for f in r['faults'])),
                seed42_examples=sample)
    output = dict(scope='read-only local audit; no paid calls; no model inference; frozen artifacts unchanged',
        evaluation_proofs=proofs, training_coverage=training, evaluation_outputs=results,
        caveat='Narrow regex flags are not a full semantic audit. Causal contribution of each mechanism requires an ablation.')
    OUT.mkdir(exist_ok=True)
    (OUT/'evidence.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(proofs=proofs, training=training,
        results={k:{a:b for a,b in v.items() if a!='seed42_examples'} for k,v in results.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
