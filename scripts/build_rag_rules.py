"""Materialize explicitly project-authored planning rules, not official facts."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.rag_protocol import fingerprint


def main():
    source=ROOT/'knowledge/benchmark_v2/planning_rules.json'
    rules=json.loads(source.read_text(encoding='utf-8'))
    rows=[]
    import hashlib
    for rule in rules:
        h=hashlib.sha256(rule['text'].encode()).hexdigest()
        rows.append({**rule,'chunk_id':'rule-'+h[:24],'content_hash':h,
            'url':'project://knowledge/benchmark_v2/planning_rules.json','source':'project-planning-rules-v2',
            'collected_at':'2026-10-08','source_type':'project_rule','evidence_group':h[:24],
            'indoor':'unknown','review_status':'pending','fact_validity':'planning_advice_not_official_fact'})
    out=ROOT/'knowledge/benchmark_v2/rule_chunks.json'
    out.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'{len(rows)} project-authored rules; not official policies')


if __name__=='__main__':main()
