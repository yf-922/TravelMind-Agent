"""Produce an exploratory selection comparison, never a formal test claim."""
import argparse
import json
from pathlib import Path
import statistics


def analyze(report):
    pairs = {}
    for row in report['results']:
        pairs.setdefault((row['case_id'], row['repetition']), {})[row['arm']] = row
    comparisons = []
    for (case_id, repetition), pair in pairs.items():
        complete = len(pair) == 2 and all(not r.get('error') and r.get('metrics') for r in pair.values())
        item = {'case_id': case_id, 'repetition': repetition, 'complete_pair': complete}
        for arm, row in pair.items():
            metrics = row.get('metrics', {})
            item[arm] = {'status': 'failed' if row.get('error') else 'completed',
                         'selected_pois': metrics.get('selected_pois'),
                         'interest_precision': metrics.get('interest_precision'),
                         'interest_recall': metrics.get('interest_recall'),
                         'hard_constraint_pass': metrics.get('hard_constraint_pass'),
                         'verification_complete': metrics.get('hard_constraint_verification_complete'),
                         'forbidden_selected': metrics.get('forbidden_selected'),
                         'rework': metrics.get('rework_count'), 'usage': row.get('actual_usage'),
                         'usage_complete': row.get('actual_usage_complete'),
                         'elapsed_ms': row.get('elapsed_ms'), 'error': row.get('error'),
                         'selection_reasons': metrics.get('selection_reasons'),
                         'reused': bool(row.get('reused_paid_trial'))}
        if complete:
            a, b = item['without_knowledge'], item['with_knowledge']
            item['same_selection'] = set(a['selected_pois']) == set(b['selected_pois'])
            item['precision_delta'] = b['interest_precision']-a['interest_precision'] if a['interest_precision'] is not None and b['interest_precision'] is not None else None
        comparisons.append(item)
    completed = [p for p in comparisons if p['complete_pair']]
    deltas = [p['precision_delta'] for p in completed if p.get('precision_delta') is not None]
    no_benefit = bool(completed) and all(p['same_selection'] for p in completed)
    return {'status': 'exploratory_draft_labels_pending_human_review',
            'execution_status': report['status'], 'summary': report['summary'],
            'complete_pairs': len(completed), 'same_selection_pairs': sum(p['same_selection'] for p in completed),
            'mean_precision_delta_complete_pairs': statistics.mean(deltas) if deltas else None,
            'comparisons': comparisons, 'interruption_audits': report.get('interruption_audits', []),
            'cost_status': 'unpriced',
            'decision': 'keep_production_disabled_stop_selection_expansion' if no_benefit else 'keep_production_disabled_insufficient_evidence',
            'interpretation': 'No stable selection benefit demonstrated; descriptions and citations are not selection improvements. Failed pairs are not credited as RAG gains.',
            'human_review_required': ['interest labels for selected POIs', 'whether cited facts actually support selection'],
            'fingerprints': {k: report[k] for k in ('cases', 'facts', 'maps', 'implementation')},
            'configuration': report['configuration']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = analyze(json.loads(a.report.read_text(encoding='utf-8')))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
