"""Create immutable semantic/token variants from captured source paragraphs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.rag_chunking import split_text
from app.evaluation.rag_retrieval import embedding_model,embedding_config
from app.evaluation.rag_protocol import fingerprint,validate_corpus


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',type=Path)
    a=p.parse_args()
    rows=json.loads(a.corpus.read_text(encoding='utf-8'))
    tokenizer=embedding_model().tokenizer
    a.out.mkdir(parents=True,exist_ok=True)
    for strategy in ('semantic','token'):
        result=[]
        for row in rows:
            for text in split_text(row['text'],tokenizer,strategy):
                h=hashlib.sha256(text.encode()).hexdigest()
                result.append({**row,'text':text,'content_hash':h,'parent_chunk_id':row['chunk_id'],
                    'chunk_id':strategy+'-'+fingerprint([row['url'],text])[:24],
                    'chunking_version':'v2-'+strategy,'token_count':len(tokenizer.encode(text,add_special_tokens=True))})
        errors=validate_corpus(result)
        if errors: p.error('; '.join(errors))
        (a.out/(strategy+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (a.out/(strategy+'.manifest.json')).write_text(json.dumps({'fingerprint':fingerprint(result),'embedding':embedding_config(),
            'strategy':strategy,'soft_chars':420,'max_tokens':512,'overlap':0,'chunks':len(result)},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if a.cases:
            cases=json.loads(a.cases.read_text(encoding='utf-8'))
            children={}
            for row in result:children.setdefault(row['parent_chunk_id'],[]).append(row)
            for case in cases:
                case['relevant_chunks']=[{'chunk_id':child['chunk_id'],'relevance':label['relevance'],
                    'evidence_group':label.get('evidence_group',child['evidence_group'])}
                    for label in case.get('relevant_chunks',[]) for child in children[label['chunk_id']]]
                case['annotation_status']='pending_review'
                case['annotator']=None
                case['reviewed_at']=None
                case['change_log'].append('Chunk variant remapped; relevance requires human reconfirmation.')
            (a.out/(strategy+'.cases.json')).write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
