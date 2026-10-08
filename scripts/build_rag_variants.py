"""Create immutable semantic/token variants from captured source paragraphs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.rag_chunking import split_text
from app.evaluation.rag_retrieval import embedding_tokenizer,embedding_config
from app.evaluation.rag_protocol import fingerprint,validate_corpus


def source_paragraphs(rows, source_rows):
    """Restore capture order, never join unrelated pages to inflate chunks."""
    order={r['chunk_id']:i for i,r in enumerate(source_rows)}
    if any(r['chunk_id'] not in order for r in rows):
        raise ValueError('source corpora must contain every curated parent chunk')
    grouped={}
    for row in sorted(rows,key=lambda r:order[r['chunk_id']]):
        key=(row['url'],row.get('evidence_group',row['chunk_id']))
        grouped.setdefault(key,[]).append(row)
    return [{**parts[0], 'text':''.join(r['text'] for r in parts),
             'parent_chunk_ids':[r['chunk_id'] for r in parts]} for parts in grouped.values()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',type=Path)
    p.add_argument('--source-corpora',type=Path,nargs='+',required=True,
                   help='Captured pre-curation chunks in original paragraph order, including rule chunks')
    a=p.parse_args()
    rows=json.loads(a.corpus.read_text(encoding='utf-8'))
    sources=[r for path in a.source_corpora for r in json.loads(path.read_text(encoding='utf-8'))]
    paragraphs=source_paragraphs(rows,sources)
    tokenizer=embedding_tokenizer()
    a.out.mkdir(parents=True,exist_ok=True)
    for strategy in ('semantic','token'):
        result=[]
        for row in paragraphs:
            for text in split_text(row['text'],tokenizer,strategy):
                h=hashlib.sha256(text.encode()).hexdigest()
                result.append({**row,'text':text,'content_hash':h,
                    'chunk_id':strategy+'-'+fingerprint([row['url'],text])[:24],
                    'chunking_version':'v2-'+strategy,'token_count':len(tokenizer.encode(text,add_special_tokens=True))})
        errors=validate_corpus(result)
        if errors: p.error('; '.join(errors))
        (a.out/(strategy+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (a.out/(strategy+'.manifest.json')).write_text(json.dumps({'fingerprint':fingerprint(result),'embedding':embedding_config(),
            'strategy':strategy,'soft_chars':420,'max_tokens':512,'overlap':0,'chunks':len(result),
            'input_paragraphs':len(paragraphs),'source_fingerprint':fingerprint(paragraphs)},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if a.cases:
            cases=json.loads(a.cases.read_text(encoding='utf-8'))
            children={}
            for row in result:
                for parent in row['parent_chunk_ids']:children.setdefault(parent,[]).append(row)
            for case in cases:
                remapped={}
                for label in case.get('relevant_chunks',[]):
                    for child in children[label['chunk_id']]:
                        new={'chunk_id':child['chunk_id'],'relevance':label['relevance'],
                             'evidence_group':label.get('evidence_group',child['evidence_group'])}
                        if child['chunk_id'] not in remapped or remapped[child['chunk_id']]['relevance']<new['relevance']:
                            remapped[child['chunk_id']]=new
                case['relevant_chunks']=list(remapped.values())
                case['annotation_status']='pending_review'
                case['annotator']=None
                case['reviewed_at']=None
                case['change_log'].append('Chunk variant remapped; relevance requires human reconfirmation.')
            (a.out/(strategy+'.cases.json')).write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
