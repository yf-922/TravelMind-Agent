"""Post-hoc format diagnostic; never overwrite frozen raw scores or outputs."""
import copy
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_representative_eval50 import score_case
from scripts.evaluate_representative50 import SUITE, verify
from scripts.diagnose_sft_transfer import rows
from scripts.prepare_sft_v2_release import explanation_faults

TAGS=['base','v1','v2','v3','mc','smoke']


def normalize_format(plan,state):
    """Repair only representation, not POI choices, times, order or constraints.

    Full local datetimes are reduced only if date matches the requested day,
    seconds are zero and timezone is absent or +08:00. Invalid period strings
    are mapped from an otherwise valid HH:MM start. Valid labels stay unchanged.
    """
    plan=copy.deepcopy(plan);changes=[]
    for day in plan['days']:
        if not isinstance(day.get('day'),int) or day['day']<1:
            continue
        target=(date.fromisoformat(state['travel_start_date'])+timedelta(days=day['day']-1)).isoformat()
        for spot in day['spots']:
            for field in ['start_time','end_time']:
                text=spot[field]
                m=re.fullmatch(r'(\d{4}-\d{2}-\d{2})[ T]([0-2]\d:[0-5]\d)(?::00)?(\+08:00)?',text)
                if m and m.group(1)==target and int(m.group(2)[:2])<24:
                    spot[field]=m.group(2)
                    changes.append(field+'_datetime_to_hhmm')
            t=spot['start_time']
            if spot.get('period') not in ['morning','afternoon','evening'] and re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d',t):
                hour=int(t[:2]);spot['period']='morning' if hour<12 else 'afternoon' if hour<18 else 'evening'
                changes.append('invalid_period_to_enum')
    return plan,changes


def main():
    verify()
    cases={c['id']:c for c in json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))}
    maps=json.loads((SUITE/'maps.json').read_text(encoding='utf-8'))['routes']
    out=SUITE/'all_versions_v1'
    all_results=[];summary={}
    for tag in TAGS:
        records=rows(SUITE/'runs'/('all_versions_v1_'+tag)/'results.jsonl')
        assert len(records)==150
        normalized=[]
        for r in records:
            case=cases[r['id']]
            result=dict(tag=tag,id=r['id'],seed=r['seed'],raw_passed=r['passed'],
                raw_faults=r['faults'],raw_error=r.get('error_type'),passed=False,faults=[],changes=[])
            if r.get('parse_success') and r.get('plan'):
                plan,changes=normalize_format(r['plan'],case['state'])
                result.update(plan=plan,changes=changes)
                try:
                    result['faults']=score_case(plan,case,maps)
                    result['passed']=not result['faults']
                    result['explanation_faults']=explanation_faults(plan)
                except Exception as exc:
                    result['error_type']=type(exc).__name__
            else:
                result['error_type']=r.get('error_type','parse_failure')
            normalized.append(result)
        summary[tag]=dict(raw_pass=sum(r['passed'] for r in records),
            format_repaired_pass=sum(r['passed'] for r in normalized),
            format_repaired_joint=sum(r['passed'] and not r.get('explanation_faults') for r in normalized),
            records_changed=sum(bool(r['changes']) for r in normalized),
            remaining_faults=dict(Counter(f for r in normalized for f in r['faults'])))
        all_results+=normalized
    (out/'format_diagnostic_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_results),encoding='utf-8')
    (out/'format_diagnostic_summary.json').write_text(json.dumps(dict(
        label='POST-HOC FORMAT DIAGNOSTIC; NOT FROZEN PRIMARY SCORES',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary=summary),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
