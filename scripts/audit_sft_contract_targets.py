"""Narrow verifiable target claims, separate from the frozen evaluation rubric."""
import re
from app.planning.nodes import _clock_minutes


def aggregate_duration_faults(plan):
    total = sum(_clock_minutes(s['end_time'])-_clock_minutes(s['start_time'])
                for d in plan['days'] for s in d['spots'])
    text = plan.get('reasoning', '')+' '+plan.get('notes', '')
    patterns = [r'(?:合计游玩|总游玩时长)\s*(\d+)\s*分钟',
                r'共\s*(\d+)\s*分钟游玩时长']
    claims = [int(n) for pattern in patterns for n in re.findall(pattern, text)]
    return ['explanation_total_duration_mismatch'] if any(n != total for n in claims) else []
