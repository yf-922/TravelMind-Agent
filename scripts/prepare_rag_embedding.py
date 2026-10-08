"""Explicitly download the pinned model before benchmarking; never inside a query."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation import rag_retrieval as retrieval
from app.evaluation.rag_retrieval import embedding_config,embedding_tokenizer,embedding_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--download',action='store_true')
    p.add_argument('--tokenizer-only',action='store_true')
    p.add_argument('--embedding-config',type=Path)
    a=p.parse_args()
    if a.embedding_config:retrieval.configure_embedding(json.loads(a.embedding_config.read_text(encoding='utf-8')))
    if a.download:
        from huggingface_hub import snapshot_download
        patterns=['config.json','tokenizer*.json','special_tokens_map.json','vocab.txt'] if a.tokenizer_only else ['*.json','*.txt','1_Pooling/*','model.safetensors']
        snapshot_download(retrieval.MODEL,revision=retrieval.REVISION,allow_patterns=patterns,etag_timeout=10,max_workers=2)
    tokenizer=embedding_tokenizer()
    output={'embedding':embedding_config(),'tokenizer_smoke_tokens':len(tokenizer.encode('南京博物院预约',add_special_tokens=True)),
            'model_loaded':False}
    if not a.tokenizer_only:
        model=embedding_model()
        output['embedding_shape']=list(model.encode(['南京博物院预约']).shape)
        output['model_loaded']=True
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
