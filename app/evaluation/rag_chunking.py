"""Semantic soft boundaries with an actual tokenizer hard ceiling."""
import re


def split_text(text, tokenizer, strategy='semantic', soft_chars=420, max_tokens=512):
    if strategy not in ('semantic','token') or soft_chars < 1 or max_tokens < 3:
        raise ValueError('invalid chunking configuration')
    def count(value):
        return len(tokenizer.encode(value, add_special_tokens=True))
    if not text.strip():
        return []
    sentences = re.findall(r'[^。！？!?\n]*[。！？!?\n]|[^。！？!?\n]+$', text)
    output=[]
    current=''
    for sentence in sentences:
        while count(sentence) > max_tokens:
            # Use character boundaries so decode normalization cannot lose text.
            lo,hi=1,len(sentence)
            while lo < hi:
                mid=(lo+hi+1)//2
                if count(sentence[:mid]) <= max_tokens: lo=mid
                else: hi=mid-1
            if count(sentence[:lo]) > max_tokens:
                raise ValueError('single character exceeds token budget')
            if current:
                output.append(current)
                current=''
            output.append(sentence[:lo])
            sentence=sentence[lo:]
        combined=current+sentence
        if current and (count(combined)>max_tokens or (strategy=='semantic' and len(combined)>soft_chars)):
            output.append(current)
            current=''
        current+=sentence
    if current: output.append(current)
    if ''.join(output) != text:
        raise ValueError('chunking changed text')
    return output
