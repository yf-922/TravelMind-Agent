"""Cache AMap evidence, then classify locally. Human review is never fabricated."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,random,re,sys,time,unicodedata,urllib.parse,urllib.request
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_no_pool_base import DEFAULT_OUT, read,dump

def norm(s):
    return re.sub(r"[\s·・]", "",unicodedata.normalize("NFKC",s)).casefold()

def entity_key(city,name):
    return hashlib.sha256((city+"\0"+name).encode()).hexdigest()

def spot_names(plan):
    return [s["name"] for day in (plan or {}).get("days",[]) for s in day.get("spots",[]) if isinstance(s.get("name"),str)]

def collect(out):
    cases={c["id"]:c for c in read(out/"cases_metadata.json")};runs=[]
    for line in (out/"generations.jsonl").read_text(encoding="utf8").splitlines():
        r=json.loads(line);a=r["attempts"][0]
        runs.append(dict(arm="no_pool_base",id=r["id"],seed=r["seed"],city=r["city"],expected=r["expected"],
                         parse_success=a["parse_success"],plan=a.get("plan"),retry_recovered=len(r["attempts"])==2 and r["attempts"][1]["parse_success"]))
    assert len(runs)==150 and len({(r['id'],r['seed']) for r in runs})==150,"Wait for all 150 first attempts"
    for r in read(out/"historical_first_round.json"):
        c=cases[r["id"]]
        runs.append(dict(r,city=c["city"],expected=c["expected"]))
    entities={}
    for r in runs:
        if not r["parse_success"]:continue
        for name in spot_names(r["plan"]):
            k=entity_key(r["city"],name);entities[k]=dict(key=k,city=r["city"],name=name)
    dump(out/"scoring_runs.json",runs);dump(out/"entities.json",list(entities.values()))
    return runs,list(entities.values())

def api(path,params,key,cache):
    query={**params,"key":key}
    started=datetime.now(timezone.utc).isoformat()
    try:
        req=urllib.request.Request("https://restapi.amap.com/"+path+"?"+urllib.parse.urlencode(query))
        with urllib.request.urlopen(req,timeout=25) as res:data=json.loads(res.read())
        entry=dict(timestamp=started,endpoint=path,query=params,response=data,ok=data.get("status")=="1")
        if not entry["ok"]:entry["error"]=data.get("info","AMap error")
    except Exception as exc:entry=dict(timestamp=started,endpoint=path,query=params,ok=False,error=type(exc).__name__)
    dump(cache,entry);return entry

def polygons(data):
    ds=(data or {}).get("districts",[])
    if not ds:return []
    return [[tuple(map(float,p.split(','))) for p in poly.split(';') if p] for poly in ds[0].get("polyline","").split('|') if poly]

def contains(point,poly):
    x,y=point;inside=False
    for (x1,y1),(x2,y2) in zip(poly,poly[1:]+poly[:1]):
        if (y1>y)!=(y2>y) and x<(x2-x1)*(y-y1)/(y2-y1)+x1:inside=not inside
    return inside

def classify(city,name,pois,boundary,lookup_ok=True):
    exact=[p for p in pois if norm(p.get("name",""))==norm(name) or norm(name) in [norm(x) for x in str(p.get("alias","")).split(';') if x]]
    if not exact:return dict(bucket="d" if lookup_ok else "pending",reason="no exact identity; near names need human confirmation" if lookup_ok else "lookup failed",matches=pois)
    if not boundary:return dict(bucket="pending",reason="city boundary unavailable",matches=exact)
    located=[]
    for p in exact:
        loc=p.get("location")
        if isinstance(loc,dict): xy=(loc.get("lng"),loc.get("lat"))
        else:
            try:xy=tuple(map(float,loc.split(',')))
            except (ValueError,AttributeError):continue
        if len(xy)!=2 or None in xy:continue
        inside=any(contains(xy,poly) for poly in boundary)
        # Conservative tourism whitelist: scenic POIs + museums/art galleries/memorials.
        code=str(p.get("typecode", ""))
        kind=p.get("type",p.get("category",""))
        attraction=code.startswith("11") or code.startswith(("140100","140200","140300")) or any(t in kind for t in ["风景名胜","博物馆","美术馆","纪念馆"])
        located.append(dict(poi=p,inside=inside,attraction=attraction))
    legal=[x for x in located if x["inside"] and x["attraction"]]
    if legal:return dict(bucket="a",reason="exact identity, coordinate in city polygon, attraction category",matches=legal)
    incity=[x for x in located if x["inside"]]
    if incity:return dict(bucket="c",reason="exact local identity but category outside preregistered tourism whitelist",matches=incity)
    if located:return dict(bucket="b",reason="exact identity outside target city polygon",matches=located)
    return dict(bucket="pending",reason="identity found without usable coordinates",matches=exact)

def prefetch(out,max_calls):
    from app.core.env import load_local_env
    load_local_env();key=os.getenv("AMAP_API_KEY","")
    if not key:raise RuntimeError("AMAP_API_KEY missing")
    _,entities=collect(out);cache=out/"amap_cache";cache.mkdir(exist_ok=True);calls=0
    def fetch(path,params,name):
        nonlocal calls
        dest=cache/(name+".json")
        if dest.exists():return read(dest)
        if calls>=max_calls:raise RuntimeError("AMap hard call budget reached; resume with a reviewed budget")
        calls+=1;entry=api(path,params,key,dest);time.sleep(.3);return entry
    cities={e["city"] for e in entities}
    boundaries={}
    for city in sorted(cities):
        e=fetch("v3/config/district",dict(keywords=city,subdistrict=0,extensions="all"),"city-"+entity_key(city,"boundary"))
        boundaries[city]=polygons(e.get("response",{})) if e["ok"] else []
    evidence=read(out/"pool_evidence.json");lookups=[]
    for ent in entities:
        local=[x["poi"] for x in evidence if x["city"]==ent["city"] and norm(x["poi"]["name"])==norm(ent["name"])]
        if local:
            # Pool origin is evidence, not an automatic pass: coordinates/category still checked.
            lookups.append(dict(**ent,source="frozen_pool",pois=local,lookup_ok=True));continue
        pois=[];ok=True
        for scope in ["city","national"]:
            for page in [1,2]:
                params=dict(keywords=ent["name"],offset=25,page=page,extensions="all")
                if scope=="city":params.update(city=ent["city"],citylimit="true")
                e=fetch("v3/place/text",params,ent["key"]+f"-{scope}-{page}")
                if not e["ok"]:ok=False;break
                found=e["response"].get("pois",[]);pois.extend(found)
                if len(found)<25:break
            if any(norm(p.get('name',''))==norm(ent['name']) for p in pois):break
        # Persist details for exact hits where an AMap ID is available.
        for p in list(pois):
            if norm(p.get('name',''))!=norm(ent['name']) or not p.get('id'):continue
            e=fetch("v3/place/detail",dict(id=p['id'],extensions="all"),"detail-"+p['id'])
            if e["ok"]:pois.extend(e["response"].get("pois",[]))
            else:ok=False
        lookups.append(dict(**ent,source="cached_amap_search_detail",pois=pois,lookup_ok=ok))
    dump(out/"lookups.json",lookups);dump(out/"boundaries.json",boundaries)
    dump(out/"prefetch_status.json",dict(complete=True,new_api_calls=calls,entities=len(entities)))
    print(json.dumps(dict(prefetch_complete=True,new_calls=calls,entities=len(entities))))

def score(out):
    runs,entities=collect(out);lookups={r['key']:r for r in read(out/'lookups.json')};boundaries=read(out/'boundaries.json')
    labels=[]
    for e in entities:
        x=lookups[e['key']];labels.append(dict(**e,**classify(e['city'],e['name'],x['pois'],boundaries.get(e['city'],[]),x['lookup_ok'])))
    dump(out/'entity_labels.json',labels);bykey={e['key']:e for e in labels}
    cases={c['id']:c for c in read(out/'cases_metadata.json')};summaries={};details=[]
    for arm in ['no_pool_base','base','C_dpo']:
        rs=[r for r in runs if r['arm']==arm and r['expected']=='feasible'];parsed=[r for r in rs if r['parse_success']]
        buckets=Counter();polluted=0;incomplete=0;short=0;excluded=0;judgeable=0
        for r in parsed:
            labs=[bykey[entity_key(r['city'],n)] for n in spot_names(r['plan'])];bs=Counter(e['bucket'] for e in labs);buckets.update(bs)
            polluted+=bool(sum(bs[b] for b in ['b','c','d']));incomplete+=bool(bs['pending'])
            rubric=cases[r['id']]['rubric'];counts=rubric.get('counts',[]);days=r['plan'].get('days',[])
            shortage=any(len(next((d.get('spots',[]) for d in days if d.get('day')==i+1),[]))<pair[0] for i,pair in enumerate(counts));short+=shortage
            denied=rubric.get('excluded',[])
            if denied:
                judgeable+=1;excluded+=bool({norm(n) for n in spot_names(r['plan'])}&{norm(n) for n in denied})
            details.append(dict(arm=arm,id=r['id'],seed=r['seed'],buckets=dict(bs),quantity_shortage=shortage))
        n=sum(buckets.values());bad=sum(buckets[b] for b in ['b','c','d']);unknown=buckets['pending']
        summaries[arm]=dict(runs=len(rs),parsed=len(parsed),first_parse_failures=len(rs)-len(parsed),parse_failure_rate_pct=100*(len(rs)-len(parsed))/len(rs),
            retry_recovered=sum(r.get('retry_recovered',False) for r in rs),entity_occurrences=n,buckets=dict(buckets),
            unverifiable_rate_pct=100*bad/n if n and not unknown else None,
            verification_pending_entities=unknown,unverifiable_bounds_pct=[100*bad/n,100*(bad+unknown)/n] if n else None,
            itinerary_pollution_rate_pct=100*polluted/len(parsed) if parsed and not unknown else None,
            polluted_itineraries_observed=polluted,itineraries_with_pending=incomplete,
            quantity_shortage_count=short,quantity_shortage_rate_pct=100*short/len(parsed) if parsed else None,
            empty_itineraries=sum(not spot_names(r['plan']) for r in parsed),
            preference_scope='explicit excluded POI identity only; semantic preference ungraded',preference_judgeable=judgeable,preference_violations=excluded)
    dump(out/'verification_summary.json',dict(status='provisional_pending_human_review',arms=summaries,
        infeasible_observations=[dict(arm=r['arm'],id=r['id'],seed=r['seed'],parse_success=r['parse_success'],entity_count=len(spot_names(r.get('plan')))) for r in runs if r['expected']!='feasible']))
    dump(out/'itinerary_scores.json',details)
    rng=random.Random(20261030);sample=[]
    for b in ['a','b','c','d','pending']:
        group=[e for e in labels if e['bucket']==b];rng.shuffle(group)
        sample.extend(group if b in ['d','pending'] else group[:max(12, math.ceil(len(group)*.1))])
    review=out/'human_review.csv'
    if not review.exists():
        with review.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['key','city','name','automatic_bucket','human_bucket','evidence','reviewer'])
            writer.writeheader()
            for e in sample:writer.writerow(dict(key=e['key'],city=e['city'],name=e['name'],automatic_bucket=e['bucket'],evidence=e['reason']))
    md=['# 无候选池不可核验率：预注册统计（待人工复核）','',
        '生成分母为45可行场景×3种子；实体按出现次数加权。历史池外率不是不可核验率，以下三组由同一核验器重新判定。',
        '接口错误/边界缺失不进入d桶，保留pending并给出区间；人工复核前不得称最终幻觉率。','',
        '|模型协议|首次解析/135|实体数|a/b/c/d/pending|不可核验率|数量不足|','|---|---:|---:|---|---:|---:|']
    for arm,s in summaries.items():md.append(f"|{arm}|{s['parsed']}/135|{s['entity_occurrences']}|{s['buckets']}|{s['unverifiable_rate_pct']}|{s['quantity_shortage_count']}|")
    md+=['','人工复核请填写 human_review.csv，逐桶估计误判率。未经人工标签不能估计核验器误差；重复实体及同场景种子不独立，不能使用普通实体二项置信区间。']
    (out/'report.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    print(json.dumps(summaries,ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prefetch','score']);ap.add_argument('--output',type=Path,default=DEFAULT_OUT);ap.add_argument('--max-api-calls',type=int,default=1000)
    args=ap.parse_args()
    if args.stage=='prefetch':prefetch(args.output,args.max_api_calls)
    else:score(args.output)
if __name__=='__main__':main()
