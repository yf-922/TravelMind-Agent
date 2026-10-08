"""Capture bounded real POI/road responses, never fabricate missing distances."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.env import load_local_env
from app.planning.nodes import amap_key
from app.providers.amap.poi import search_city_pois, poi_to_spot
from app.providers.amap.direction import plan_route_distance
from app.evaluation.route_interest import route_key
from app.evaluation.rag_protocol import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=ROOT/'knowledge/route_interest_v1')
    p.add_argument('--max-requests', type=int, default=100)
    a = p.parse_args()
    if not 1 <= a.max_requests <= 100: p.error('request budget must be 1..100')
    load_local_env(); key = amap_key()
    cases = json.loads((a.assets/'cases.json').read_text(encoding='utf-8'))
    previous = json.loads((a.assets/'maps.json').read_text(encoding='utf-8')) if (a.assets/'maps.json').exists() else {}
    report = {'captured_at': datetime.now(timezone.utc).isoformat(), 'routes': previous.get('routes', {}), 'failures': [],
              'requests': 0, 'mode': 'drive', 'live_weather': False}
    report['previous_capture'] = {k: previous.get(k) for k in ('captured_at', 'requests')}
    pools = {c['destination']: c['pois'] for c in cases}
    if not key:
        report['failures'].append({'reason': 'amap_key_missing'})
    else:
        for city, pool in pools.items():
            for poi in pool:
                if poi.get('location'): continue
                if report['requests'] >= a.max_requests: break
                report['requests'] += 1
                try:
                    rows = search_city_pois(city, key, keywords=poi['name'], types='', offset=8)
                    aliases = {'上海博物馆东馆': {'上海博物馆东馆', '上海博物馆(东馆)'}}
                    exact = [r for r in rows if r.get('name') in aliases.get(poi['name'], {poi['name']}) and str(r.get('cityname', '')).removesuffix('市') == city]
                    parsed = poi_to_spot(exact[0]) if len(exact) == 1 else None
                    if not parsed: raise ValueError('exact POI identity unresolved')
                    label_name = poi['name']; poi.update(parsed); poi['name'] = label_name
                    poi['provenance'] = {'kind': 'real_amap_exact_city_name', 'captured_at': report['captured_at']}
                except Exception as exc:
                    report['failures'].append({'city': city, 'poi': poi['name'], 'reason': type(exc).__name__})
            for origin in pool:
                for destination in pool:
                    if origin['name'] == destination['name'] or not all(x.get('location') for x in (origin, destination)): continue
                    if report['requests'] >= a.max_requests: break
                    k = route_key(origin['location'], destination['location'], 'drive')
                    if k in report['routes']: continue
                    report['requests'] += 1
                    try:
                        value = plan_route_distance(origin['location'], destination['location'], key, mode='drive')
                        if not value: raise ValueError('road response unavailable')
                        report['routes'][k] = value
                    except Exception as exc:
                        report['failures'].append({'route_key': k, 'reason': type(exc).__name__})
    for c in cases:
        c['pois'] = pools[c['destination']]
        c['route_distance_mode'] = 'drive'
    (a.assets/'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report['case_fingerprint'] = fingerprint(cases)
    (a.assets/'maps.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    manifest = json.loads((a.assets/'manifest.json').read_text(encoding='utf-8'))
    manifest['case_fingerprint'] = fingerprint(cases)
    manifest['map_fingerprint'] = fingerprint(report)
    (a.assets/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'requests': report['requests'], 'routes': len(report['routes']), 'failures': len(report['failures'])}))


if __name__ == '__main__': main()
