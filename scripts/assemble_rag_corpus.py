"""Assemble snapshots without padding, remove boilerplate and flag near duplicates."""
import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.rag_protocol import fingerprint,validate_corpus


def normalized(text):
    return re.sub(r'[\W\d_]+','',text)


def assemble(sources):
    result=[];excluded=[]
    for rows in sources:
        for row in rows:
            text=row['text']
            if any(x in text for x in ('经调查','行政处罚','全体干部','召开','高质量发展','活动从8月6日持续至8月31日','优惠执行价','先到先得','客流有序','国庆假期','领导','统筹','2026年春节','招标')):
                excluded.append({'chunk_id':row['chunk_id'],'reason':'expired_campaign_or_nonvisitor_content'})
                continue
            norm=normalized(text)
            existing=next((r for r in result if r['city']==row['city'] and (norm==normalized(r['text']) or SequenceMatcher(None,norm,normalized(r['text']),autojunk=False).ratio()>=.93)),None)
            if existing:
                existing.setdefault('alternate_sources',[]).append({'url':row['url'],'chunk_id':row['chunk_id']})
                excluded.append({'chunk_id':row['chunk_id'],'reason':'near_duplicate','canonical':existing['chunk_id']})
                continue
            result.append({**row,'chunking_version':'v2-semantic-char420','fact_validity':row.get('fact_validity','snapshot_not_live'),
                'review_status':'pending','valid_until':None,'indoor':'unknown'})
    return sorted(result,key=lambda r:r['chunk_id']),excluded


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows,excluded=assemble([json.loads(path.read_text(encoding='utf-8')) for path in a.inputs])
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'chunks.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest={'corpus_fingerprint':fingerprint(rows),'cities':dict(Counter(r['city'] for r in rows)),
        'inputs':[{ 'path':str(p),'fingerprint':fingerprint(json.loads(p.read_text(encoding='utf-8')))} for p in a.inputs],
        'excluded':excluded,'acceptance_errors':validate_corpus(rows,True),'status':'pending_source_and_annotation_review'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'chunks':len(rows),'cities':manifest['cities'],'acceptance_errors':manifest['acceptance_errors']},ensure_ascii=True))


if __name__=='__main__':main()
