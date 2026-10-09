"""Predeclared controlled audit probes; no claim of natural failure prevalence."""
import copy
import json
from pathlib import Path
from app.evaluation.route_interest import route_key

ROOT = Path(__file__).resolve().parents[1]


def build_cases():
    source = json.loads((ROOT/'knowledge/route_interest_v1/cases.json').read_text(encoding='utf-8'))
    cases = []
    for category, count in [('preference', 8), ('pace', 8), ('weather', 7), ('commute', 7)]:
        for index in range(count):
            city = ('南京', '上海')[index % 2]
            pool = copy.deepcopy(next(c['pois'] for c in source if c['destination'] == city))
            # Controlled all-day opening removes opening-hour confounding.
            for poi in pool: poi['open_time'] = '09:00-18:00'
            inside = [p for p in pool if p.get('indoor') is True]
            outside = [p for p in pool if p.get('indoor') is False]
            clean = index == 0 or (index == 1 and category in ('preference', 'pace'))
            quality = {'category': category, 'expected_clean': clean}
            state = {'destination': city, 'days': 1, 'max_per_day': 3, 'pois': pool,
                     'travel_start_date': '2026-10-15', 'travel_end_date': '2026-10-15',
                     'weather_forecast': [{'date': '2026-10-15', 'is_bad': False,
                         'day_weather': '晴', 'night_weather': '晴', 'day_temp': '20', 'night_temp': '18'}]}
            names, hours = [inside[0]['name']], [('10:00', '12:00')]
            if category == 'preference':
                excluded = outside[index % len(outside)]['name']
                state.update(query=f'{city}一日游，想参观历史或科普展览，不要安排“{excluded}”。',
                             attraction_preference=f'历史或科普展览；明确排除{excluded}')
                quality['forbidden'] = [excluded]
                if not clean: names = [excluded]
            elif category == 'pace':
                state.update(query=f'{city}一日游，带老人慢节奏，最多两个景点，每个至少游玩60分钟。', habit_preference='慢节奏')
                quality.update(max_spots=2, min_visit_minutes=60)
                if not clean:
                    names = [p['name'] for p in pool[:3]]
                    hours = [('10:00', '10:20'), ('11:00', '11:20'), ('14:00', '14:20')]
            elif category == 'weather':
                state.update(query=f'{city}一日游，雨天避免户外，选室内展览。', rain_indoor_priority=True)
                state['weather_forecast'][0].update(is_bad=True, day_weather='雨', night_weather='雨')
                if not clean: names = [outside[index % len(outside)]['name']]
            else:
                names = [p['name'] for p in pool[:2]]
                hours = [('10:00', '12:00'), ('14:00', '16:00')] if clean else [('10:00', '12:00'), ('12:05', '14:00')]
                state.update(query=f'{city}一日游，景点之间必须预留足够的驾车通勤时间，不要瞬移。')
                quality['require_road_gap'] = True
            state['route'] = [{'day': 1, 'spots': [{'name': n, 'period': 'morning' if i == 0 else 'afternoon',
                                  'start_time': hours[i][0], 'end_time': hours[i][1]} for i, n in enumerate(names)]}]
            cases.append({'id': f'quality-{category}-{index:02d}', 'expected': 'feasible', 'state': state,
                          'quality': quality, 'label_status': 'predeclared_author_draft',
                          'opening_hours': 'controlled_09_18_not_live_fact'})
    return cases


def minutes(value):
    h, m = map(int, value.split(':'))
    return h*60+m


def quality_grade(state, case):
    q = case['quality']
    spots = [s for day in state.route for s in day.get('spots', [])]
    names = [s['name'] for s in spots]
    violations = []
    if not spots: violations.append('empty_route')
    if set(names) & set(q.get('forbidden', [])): violations.append('excluded_preference')
    if len(spots) > q.get('max_spots', 100): violations.append('pace_count')
    if any(minutes(s['end_time'])-minutes(s['start_time']) < q.get('min_visit_minutes', 0) for s in spots):
        violations.append('pace_short_visit')
    if q['category'] == 'weather':
        indoor = {p['name']: p.get('indoor') for p in state.pois}
        if any(indoor.get(n) is not True for n in names): violations.append('rain_outdoor')
    if q.get('require_road_gap'):
        maps = json.loads((ROOT/'knowledge/route_interest_v1/maps.json').read_text(encoding='utf-8'))['routes']
        by_name = {p['name']: p for p in state.pois}
        for day in state.route:
            for a, b in zip(day['spots'], day['spots'][1:]):
                if a['name'] not in by_name or b['name'] not in by_name:
                    violations.append('unknown_poi'); continue
                leg = maps.get(route_key(by_name[a['name']]['location'], by_name[b['name']]['location'], 'drive'))
                if not leg or leg.get('duration_min') is None: violations.append('unverified_road_gap')
                elif minutes(b['start_time'])-minutes(a['end_time']) < float(leg['duration_min']): violations.append('insufficient_road_gap')
    return {'passed': not violations, 'violations': violations, 'category': q['category'],
            'expected_initial_clean': q['expected_clean']}
