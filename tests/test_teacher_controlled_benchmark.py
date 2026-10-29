import json

import pytest

from scripts import benchmark_teacher_controlled as bench


class Stream:
    def __init__(self, items):
        self.items = items

    def __enter__(self):
        return iter(self.items)

    def __exit__(self, *args):
        return False


def test_stream_uses_real_usage_and_never_logs_key(tmp_path, monkeypatch):
    chunks = [
        b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":null}]}\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":3}}\n',
        b'data: [DONE]\n',
    ]
    requests = []

    def send(request, timeout):
        requests.append(json.loads(request.data))
        return Stream(chunks)

    monkeypatch.setattr(bench.urllib.request, 'urlopen', send)
    teacher = bench.Teacher(tmp_path, 'teacher', 'https://example.invalid/v1/chat/completions', 'private-test-key', 1)
    text, timing = teacher.call([{'role': 'user', 'content': 'unchanged'}], 42, {'warmup': False})
    assert text == '{}'
    assert timing['prompt_tokens'] == 12
    assert timing['completion_tokens'] == 3
    assert requests[0]['temperature'] == 0
    assert requests[0]['max_tokens'] == 2048
    assert requests[0]['messages'][0]['content'] == 'unchanged'
    assert 'private-test-key' not in teacher.ledger.read_text(encoding='utf8')
    with pytest.raises(RuntimeError, match='budget exhausted'):
        teacher.call([], 42, {})
    assert len(requests) == 1


def test_missing_usage_is_not_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.urllib.request, 'urlopen', lambda *a, **k: Stream([
        b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n',
        b'data: [DONE]\n',
    ]))
    teacher = bench.Teacher(tmp_path, 'teacher', 'https://example.invalid', 'key', 1)
    _, timing = teacher.call([], 42, {})
    assert timing['completion_tokens'] is None
    assert timing['tpot_ms'] is None


def test_delivery_failure_cannot_pass_subrules():
    assert not any(bench.rule_flags(['delivery_failure'], False).values())
    assert not bench.rule_flags(['global_duplicate'], True)['non_duplicate_pass']


def test_broken_stream_stops_and_is_metered(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.urllib.request, 'urlopen', lambda *a, **k: Stream([
        b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n',
    ]))
    teacher = bench.Teacher(tmp_path, 'teacher', 'https://example.invalid', 'key', 2)
    with pytest.raises(RuntimeError, match='transport failed'):
        teacher.call([], 42, {})
    rows = [json.loads(s) for s in teacher.ledger.read_text(encoding='utf8').splitlines()]
    assert [r['event'] for r in rows] == ['started', 'failed']


def test_summary_missing_usage_and_unmeasurable_tpot_are_null():
    row = dict(delivery=False, all_rule_pass=False, round0_pass=False, rounds_used=1,
        **bench.rule_flags(['delivery_failure'], False), usage_complete=False,
        prompt_tokens=None, completion_tokens=None, output_chars=12, e2e_s=1,
        tpot_ms=None, rounds=[{'timing': {'tpot_ms': None, 'ttft_s': 1, 'finish_reason': 'length', 'content_chunks': 1}}])
    result = bench.summary([row], 'rules-rework')
    assert result['total_tokens'] is None
    assert result['tpot_p50_ms'] is None
    assert result['truncated_calls'] == 1
    assert result['single_chunk_calls'] == 1
    assert result['in_pool_faithfulness_pct'] == 0


def test_service_unavailable_retry_is_bounded_and_metered(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(bench.time, 'sleep', sleeps.append)
    monkeypatch.setattr(bench.urllib.request, 'urlopen', lambda *a, **k: Stream([
        b'data: {"error":{"message":"The model service is temporarily unavailable"}}\n',
    ]))
    teacher = bench.Teacher(tmp_path, 'teacher', 'https://example.invalid', 'key', 10)
    with pytest.raises(bench.TransientTeacherError):
        teacher.call([], 42, {})
    assert teacher.calls == 4
    assert sleeps == [60, 60, 60]
    rows = [json.loads(s) for s in teacher.ledger.read_text(encoding='utf8').splitlines()]
    assert sum(r['event'] == 'failed' for r in rows) == 4
