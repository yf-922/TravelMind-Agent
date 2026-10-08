"""Capture public read-only visitor APIs discovered in official frontend code."""
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.collect_rag_corpus import paragraphs,split_paragraph,digest


def request(url,data=None):
    req=urllib.request.Request(url,data=urllib.parse.urlencode(data).encode() if data is not None else None,
        headers={'User-Agent':'TravelMind research snapshot/2.0','X-Requested-With':'XMLHttpRequest'})
    with urllib.request.urlopen(req,timeout=15) as response:return response.read(3_000_000).decode('utf-8',errors='replace')


def strings(value):
    if isinstance(value,str):
        if len(value)>=50:yield value
    elif isinstance(value,dict):
        for child in value.values():yield from strings(child)
    elif isinstance(value,list):
        for child in value:yield from strings(child)


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--allow-external-calls',action='store_true')
    a=p.parse_args()
    if not a.allow_external_calls:p.error('public website calls require opt-in')
    out=ROOT/'knowledge/benchmark_v2_dynamic'
    out.mkdir(parents=True,exist_ok=True)
    date=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    captures=[];attempts=[]
    for endpoint in ['/visit/notes','/home/daolan','/home/yylg','/service/shop','/news/logisticService']:
        url='https://www.njmuseum.com/api'+endpoint
        try:
            raw=request(url,{'language':'zh'})
            content=json.loads(raw)
            captures.append(('南京',url,endpoint,raw,list(strings(content))))
        except Exception as exc:attempts.append({'url':url,'error':str(exc)})
    url='https://www.wuzhizhou.com/strategy/page/'
    try:
        response=json.loads(request(url,{'currPage':1,'pageSize':8}))
        html=response[0]['html']
        links=[urllib.parse.urljoin(url,a['href']) for a in BeautifulSoup(html,'html.parser').find_all('a',href=True)]
        for child in list(dict.fromkeys(links))[:8]:
            if urllib.parse.urlparse(child).netloc!='www.wuzhizhou.com':continue
            try:
                raw=request(child)
                captures.append(('三亚',child,'蜈支洲岛游玩攻略',raw,paragraphs(raw)))
            except Exception as exc:attempts.append({'url':child,'error':str(exc)})
    except Exception as exc:attempts.append({'url':url,'error':str(exc)})
    rows=[];seen=set();pages=[]
    for city,url,topic,raw,texts in captures:
        sid=digest(url)[:16]
        (out/(sid+'.txt')).write_text(raw,encoding='utf-8')
        pages.append({'city':city,'url':url,'snapshot':str((out/(sid+'.txt')).relative_to(ROOT)),
            'raw_hash':digest(raw),'collected_at':date})
        for text in texts:
            text=BeautifulSoup(text,'html.parser').get_text(' ',strip=True)
            if not any(t in text for t in ('游客','参观','预约','入馆','门票','游览','公交','乘车','开放','博物院')):continue
            for part in split_paragraph(text):
                h=digest(part)
                if h in seen:continue
                seen.add(h)
                rows.append({'chunk_id':'dynamic-'+digest(url+'\n'+part)[:24],'text':part,'content_hash':h,
                    'city':city,'topic':topic,'url':url,'source':sid,'collected_at':date,
                    'source_type':'official_museum' if city=='南京' else 'official_attraction',
                    'evidence_group':digest(text)[:24],'review_status':'pending','indoor':'unknown','fact_validity':'snapshot_not_live'})
    (out/'chunks.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'pages':pages,'attempts':attempts,'chunks':len(rows)},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'chunks':len(rows),'errors':attempts},ensure_ascii=True))


if __name__=='__main__':main()
