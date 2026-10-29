"""Check Qwen chat-template lengths locally; never silently truncate examples."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--cutoff', type=int, default=3584)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.tokenizer), local_files_only=True)
    lengths = []
    for line in a.input.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        for side in ['chosen', 'rejected']:
            if not row.get(side):
                continue
            messages = row['messages'] + [{'role': 'assistant', 'content': json.dumps(row[side], ensure_ascii=False)}]
            length = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
            lengths.append({'id': row['id'], 'side': side, 'tokens': length, 'over_cutoff': length > a.cutoff})
    ordered = sorted(r['tokens'] for r in lengths)
    summary = {'cutoff': a.cutoff, 'records': len(lengths), 'over_cutoff': sum(r['over_cutoff'] for r in lengths),
        'p50': ordered[len(ordered)//2] if ordered else None,
        'p95': ordered[min(len(ordered)-1, int(len(ordered)*.95))] if ordered else None,
        'max': max(ordered, default=None), 'tokenizer': str(a.tokenizer),
        'policy': 'overlength samples must be filtered or resynthesized; no silent truncation', 'lengths': lengths}
    a.input.with_suffix('.tokens.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'lengths'}))


if __name__ == '__main__':
    main()
