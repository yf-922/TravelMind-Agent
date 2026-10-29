"""Four-model GPU0 comparison after current-production v1/v2 retraining.

Uses the frozen 50-case rubric and the existing shared-message evaluation
implementation. Never resumes the stopped historical all-version experiment.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_shared_prompt_all_versions as shared
from scripts.diagnose_sft_transfer import MODELS, switch
from scripts.run_sft_validation_job import remote, ready
from scripts.evaluate_representative50 import SUITE, verify
from app.planning.prompts import JOINT_PLANNER_SYSTEM
from app.planning.schemas import TravelRoute
from scripts.post_train_runtime import capture_messages

OUT = SUITE/'current_retrained_20261025'
SEEDS = [42, 43, 44]
MODES = ['planner-only', 'rules-rework']
MODELS.clear()
MODELS.update({
    'base': ('/home/jiale/models/Qwen2.5-7B-Instruct', 'Qwen2.5-7B-Instruct'),
    'v1-current': ('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-v1-current-20261025', 'TravelMind-Planner-v1-current-20261025'),
    'v2-current': ('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-v2-current-20261025', 'TravelMind-Planner-v2-current-20261025'),
    'v2-large': ('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v2-large', 'TravelMind-Planner-SFT-v2-large'),
})


def preflight():
    manifest = verify()
    cases = json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))
    assert len(cases) == 50
    assert Counter(c['stratum'] for c in cases) == {'normal': 40, 'hard': 10}
    for case in cases:
        messages = capture_messages(case['state'])
        if messages[-1] != case['messages'][-1] or messages[0]['content'] != JOINT_PLANNER_SYSTEM:
            raise RuntimeError('Production message mismatch: '+case['id'])
    code = '''
import hashlib,json
from pathlib import Path
paths = '''+repr({tag: path for tag, (path, _) in MODELS.items()})+'''
out={}
for tag, path in paths.items():
 p=Path(path)
 index=json.loads((p/'model.safetensors.index.json').read_text())
 shards=set(index['weight_map'].values())
 if not all((p/name).is_file() and (p/name).stat().st_size>0 for name in shards):
  raise RuntimeError('Missing model shard: '+tag)
 out[tag]={'path':path,'shards':len(shards),'bytes':sum((p/name).stat().st_size for name in shards),
  'index_sha256':hashlib.sha256((p/'model.safetensors.index.json').read_bytes()).hexdigest(),
  'config_sha256':hashlib.sha256((p/'config.json').read_bytes()).hexdigest()}
print(json.dumps(out))
'''
    models = json.loads(remote(code))
    binding = dict(suite_hashes=manifest['hashes'], models=models, modes=MODES, seeds=SEEDS,
        cases=50, concurrency=4, temperature=.3, max_tokens=2048, max_model_len=8192,
        paid_api_calls=0, system_sha256=hashlib.sha256(JOINT_PLANNER_SYSTEM.encode()).hexdigest(),
        schema_sha256=hashlib.sha256(json.dumps(TravelRoute.model_json_schema(), sort_keys=True).encode()).hexdigest(),
        source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
            'scripts/evaluate_current_protocol_sft.py', 'scripts/run_shared_prompt_all_versions.py',
            'scripts/post_train_runtime.py', 'scripts/build_representative_eval50.py',
            'scripts/prepare_sft_v2_release.py', 'app/planning/prompts.py']})
    return binding, cases


def detailed_summary(rows, cases):
    summary = shared.summarize(rows, cases)
    summary['api_errors'] = dict(Counter(r['error_type'] for r in rows if r.get('error_type')))
    summary['actual_generation_calls'] = sum(len(r['rounds']) for r in rows)
    summary['mean_rework_rounds'] = sum(max(0, len(r['rounds'])-1) for r in rows)/len(rows)
    summary['round0_rules_pass'] = sum(bool(r['rounds']) and not r['rounds'][0]['faults'] for r in rows)
    summary['round0_joint_pass'] = sum(bool(r['rounds']) and not r['rounds'][0]['faults']
        and not r['rounds'][0]['explanation_faults'] for r in rows)
    summary['by_city'] = {}
    for city in sorted({c['city'] for c in cases}):
        ids = {c['id'] for c in cases if c['city'] == city}
        group = [r for r in rows if r['id'] in ids]
        summary['by_city'][city] = dict(n=len(group), rules=sum(r['passed'] for r in group),
                                      joint=sum(r['joint_passed'] for r in group))
    return summary


def paired_interval(a, b, cases, key):
    aa = {(r['id'], r['seed']): r for r in a}
    bb = {(r['id'], r['seed']): r for r in b}
    delta = {c['id']: sum(int(aa[c['id'], s][key])-int(bb[c['id'], s][key]) for s in SEEDS)/3 for c in cases}
    strata = [[c['id'] for c in cases if c['stratum'] == st] for st in ['normal', 'hard']]
    rng = random.Random(20261025)
    samples = sorted(sum(delta[rng.choice(group)] for group in strata for _ in group)/50 for _ in range(4000))
    return dict(delta_pp=100*sum(delta.values())/50, case_bootstrap_95pct=[100*samples[100], 100*samples[3899]])


def report(cases):
    summaries = {}
    comparisons = {}
    for tag in MODELS:
        for mode in MODES:
            group = shared.rows(OUT/f'{tag}_{mode}/results.jsonl')
            if len(group) != 150 or len({(r['id'], r['seed']) for r in group}) != 150:
                raise RuntimeError('Incomplete evaluation: '+tag+'/'+mode)
            summaries[tag+'/'+mode] = detailed_summary(group, cases)
    for mode in MODES:
        base = shared.rows(OUT/f'base_{mode}/results.jsonl')
        for tag in list(MODELS)[1:]:
            group = shared.rows(OUT/f'{tag}_{mode}/results.jsonl')
            comparisons[tag+'/'+mode] = {key:paired_interval(group, base, cases, key)
                                        for key in ['passed', 'joint_passed']}
        comparisons['v2-current_vs_v1-current/'+mode] = {
            key:paired_interval(shared.rows(OUT/f'v2-current_{mode}/results.jsonl'),
                                shared.rows(OUT/f'v1-current_{mode}/results.jsonl'), cases, key)
            for key in ['passed', 'joint_passed']}
    shared.atomic(OUT/'summaries.json', summaries)
    shared.atomic(OUT/'paired_comparisons.json', comparisons)
    lines = ['# 当前生产协议重训后的四模型评测', '',
        '冻结五城市50题×3 seeds（42/43/44）；40条常规、10条困难。所有模型使用同一当前生产消息、JSON Schema、temperature=0.3、max_tokens=2048、8192上下文、并发4。', '',
        '|模型|模式|规则通过/150|规则+解释/150|JSON解析/150|平均返工轮次|总Token|平均秒|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for key, s in summaries.items():
        tag, mode = key.split('/')
        a = s['all']
        lines.append(f"|{tag}|{mode}|{a['rules_pass']}|{a['joint_pass']}|{a['parsed']}|{s['mean_rework_rounds']:.2f}|{a['token_total']}|{a['mean_latency_s']:.1f}|")
    lines += ['', '## 分层结果', '', '|模型/模式|常规/120|困难/30|有解/135|无解/15|', '|---|---:|---:|---:|---:|']
    for key, s in summaries.items():
        lines.append(f"|{key}|{s['normal']['rules_pass']}|{s['hard']['rules_pass']}|{s['feasible']['rules_pass']}|{s['explained_infeasible']['rules_pass']}|")
    lines += ['', '## 配对差异', '', '|模型比较/模式|规则差异pp|95%区间|规则+解释差异pp|95%区间|', '|---|---:|---|---:|---|']
    for key, s in comparisons.items():
        lines.append(f"|{key}|{s['passed']['delta_pp']:.2f}|{s['passed']['case_bootstrap_95pct']}|{s['joint_passed']['delta_pp']:.2f}|{s['joint_passed']['case_bootstrap_95pct']}|")
    lines += ['', '## 口径', '',
        'Planner-Only只生成一次；Rules+Rework初稿后最多3次规则反馈重排。返工只由冻结rubric的规则故障触发；解释一致性独立统计，不触发返工。没有LLM Reviewer或付费API。', '',
        '两种模式分别生成初稿，使用相同case和seed配对，不宣称逐字相同的初稿。区间按案例分层bootstrap，3个seed作为同一案例内的相关重复，不能当作150道独立题。', '',
        '这套50题已多次用于诊断，属于开发基准，不构成新的独立泛化证据。v2-large不是本轮重训版本，其候选池分布、样本量和历史训练不同，差异不能归因于单一因素。API教师历史结果不自动混入异协议对照。', '',
        '每轮原始请求、输出、错误、Token及耗时保存在本轮结果目录；细分城市和故障汇总见summaries.json。']
    text = '\n'.join(lines)+'\n'
    (OUT/'report.md').write_text(text, encoding='utf-8')
    (ROOT/'docs/post_train_current_protocol_sft_evaluation.md').write_text(text, encoding='utf-8')


def main():
    binding, cases = preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    bp = OUT/'binding.json'
    if bp.exists() and json.loads(bp.read_text(encoding='utf-8')) != binding:
        raise RuntimeError('Run binding changed; do not mix protocols')
    shared.atomic(bp, binding)
    lock = OUT/'running.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    started = time.time()
    state = dict(stage='starting', started_at=started, target=1200, done=0,
                 counts={}, paid_api_calls=0, gpu=0)
    def status(stage, **extra):
        state.update(stage=stage, updated_at=time.time(), **extra)
        shared.atomic(OUT/'status.json', state)
        print(json.dumps(state), flush=True)
    shared.OUT = OUT
    shared.maps = json.loads((SUITE/'maps.json').read_text(encoding='utf-8'))['routes']
    try:
        for tag in MODELS:
            status('loading_model', current_model=tag)
            switch(tag, max_model_len=8192)
            ready(MODELS[tag][1])
            for mode in MODES:
                status('evaluating', current_mode=mode)
                records = shared.run_one(tag, mode, cases, SEEDS, 4)
                summary = detailed_summary(records, cases)
                shared.atomic(OUT/f'{tag}_{mode}/summary.json', summary)
                # Transport failures are not model-quality failures; stop before poisoning comparisons.
                transport = {k:v for k,v in summary['api_errors'].items()
                             if k in {'HTTPError', 'URLError', 'TimeoutError', 'ConnectionError'}}
                if transport:
                    raise RuntimeError('Transport failures require inspection: '+json.dumps(transport))
                state['counts'][tag+'/'+mode] = len(records)
                state['done'] = sum(state['counts'].values())
                status('mode_complete')
        report(cases)
        status('completed', elapsed_s=time.time()-started)
    except Exception as exc:
        status('failed', error=str(exc))
        raise
    finally:
        try:
            status(state['stage'], base_service_restore='starting')
            switch('base', max_model_len=8192)
            status(state['stage'], base_service_restore='ready')
        finally:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
