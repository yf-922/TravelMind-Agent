"""Bounded live Amap capture for an explicitly authorized development experiment."""
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.evaluation.task_robustness import load_assets
from app.planning.helpers import amap_key
from app.providers.amap.poi import search_city_pois, poi_to_spot
from scripts.post_train_runtime import append_json

OUT = ROOT/'data/post_train/candidate_pool_live_v1'


def identity(name):
    return re.sub(r'[\s()（）·\-]', '', name).removesuffix('景区')


def main():
    load_local_env()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'pools.json').exists() and '--rebuild' not in sys.argv:
        pools = json.loads((OUT/'pools.json').read_text(encoding='utf-8'))['pools']
        if all(len(p) >= 15 for p in pools.values()):
            print(json.dumps({c:len(p) for c,p in pools.items()},ensure_ascii=False))
            return
    source, _ = load_assets()
    pools = {}
    for case in source:
        pools.setdefault(case['destination'], copy.deepcopy(case['pois']))
    captured_at = datetime.now(timezone.utc).isoformat()
    key = amap_key()
    if not key:
        raise RuntimeError('Amap key missing')
    budget = {'remaining':24,'used':0}
    queries = [('', '110000'), ('博物馆', '140100'), ('公园', '110000'),
               ('美术馆', '140100'), ('风景名胜', '110000'), ('纪念馆', '140100')]
    for city, pool in pools.items():
        for query, types in queries:
            saved = OUT/(city+'-'+(query or '风景检索')+'.json')
            if saved.exists():
                raw = json.loads(saved.read_text(encoding='utf-8'))['rows']
            else:
                append_json(OUT/'search_calls.jsonl',dict(event='started',city=city,query=query,types=types,captured_at=captured_at))
                raw = search_city_pois(city,key,keywords=query,types=types,offset=25,request_budget=budget)
                saved.write_text(json.dumps(dict(city=city,query=query,types=types,rows=raw,captured_at=captured_at),ensure_ascii=False,indent=2),encoding='utf-8')
                append_json(OUT/'search_calls.jsonl',dict(event='completed',city=city,query=query,count=len(raw)))
            for r in raw:
                p = poi_to_spot(r)
                reason = None
                if not p or not p.get('name'):
                    reason = 'invalid_poi'
                elif str(r.get('cityname','')).removesuffix('市') != city:
                    reason = 'city_mismatch'
                elif p.get('rating') is not None and p['rating'] < 4.5:
                    reason = 'production_rating_filter'
                elif any(w in p['name'] for w in ['停车场','售票','游客中心','游客服务','出入口','检票','洗手间','码头','购物','商店','塑像']):
                    reason = 'facility_not_standalone_attraction'
                elif p['name']=='牛王岭' and '西岛' in p.get('address',''):
                    reason = 'address_confirms_subsite_of_existing_west_island'
                elif any(identity(p['name']) == identity(x['name']) or (
                        min(len(identity(p['name'])),len(identity(x['name']))) >= 3 and
                        (identity(p['name']) in identity(x['name']) or identity(x['name']) in identity(p['name']))) for x in pool):
                    reason = 'same_name_or_parent_child'
                elif any(math.hypot(p['location']['lat']-x['location']['lat'],(p['location']['lng']-x['location']['lng'])*.85) < .0015 for x in pool):
                    reason = 'near_coincident_poi_review_filter'
                if reason:
                    append_json(OUT/'filter_ledger.jsonl',dict(city=city,name=r.get('name'),id=r.get('id'),reason=reason))
                    continue
                p.update(source='amap',category=str(r.get('type') or ''),
                    provenance=dict(amap_id=r.get('id'),captured_at=captured_at,query=query,raw_file=saved.name),
                    rating_status='unknown' if p.get('rating') is None else 'known')
                pool.append(p)
                if len(pool) >= 20:
                    break
            print(json.dumps(dict(city=city,query=query,pool_count=len(pool),search_http_used=budget['used']),ensure_ascii=False),flush=True)
            if len(pool) >= 20:
                break
        if len(pool) < 15:
            raise RuntimeError(f'{city}: only {len(pool)} verified candidates; will not fabricate extras')
    (OUT/'pools.json').write_text(json.dumps(dict(pools=pools,captured_at=captured_at,
        search_http_used_this_run=budget['used'],scope='original six retained; added candidates from real Amap; indoor unknown unless legacy label',
        deduplication='exact/parent-child name and 150m coordinate heuristic; not exhaustive entity resolution'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({c:len(p) for c,p in pools.items()},ensure_ascii=False))


if __name__ == '__main__':
    main()
