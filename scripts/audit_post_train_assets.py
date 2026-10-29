"""Inventory missing evidence before supplementing training or held-out tests."""
import json
from collections import Counter
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_robustness import load_assets
from app.evaluation.route_interest import route_key


def main():
    _, maps = load_assets()
    coverage = []
    for path in sorted((ROOT/'tests/eval/fixtures').glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        pois = data.get('pois', [])
        possible = found = 0
        for a in pois:
            for b in pois:
                if a is b or not a.get('location') or not b.get('location'):
                    continue
                possible += 1
                leg = maps.get(route_key(a['location'], b['location'], 'drive'))
                found += bool(leg and leg.get('duration_min') is not None)
        coverage.append({'fixture': path.name, 'city': data.get('destination'),
            'pois': len(pois), 'directed_drive_pairs': possible, 'saved_drive_pairs': found,
            'main_meal_candidates': len(data.get('main_meal_candidates', [])),
            'eligible_for_multi_stop_replay': possible > 0 and found == possible})
    result = {'fixtures': coverage, 'cities': dict(Counter(r['city'] for r in coverage)),
        'policy': 'No fabricated restaurants, road durations, or claims of held-out status. Asset existence alone does not establish non-exposure.'}
    out = ROOT/'data/post_train/supplement_asset_audit.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
