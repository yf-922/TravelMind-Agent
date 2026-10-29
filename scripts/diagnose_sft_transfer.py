"""Paid-API-free controlled development diagnostic; never touches heldout cities."""
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.planning.schemas import TravelRoute
from app.evaluation.task_robustness import load_assets
from scripts.evaluate_sft_validation import score
from scripts.post_train_runtime import append_json
from scripts.prepare_sft_v2_release import explanation_faults
from scripts.run_sft_validation_job import remote, ready

OUT = ROOT/'data/post_train/sft_diagnosis_v1'
MODELS = {'base': ('/home/jiale/models/Qwen2.5-7B-Instruct', 'Qwen2.5-7B-Instruct'),
    'v1': ('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT', 'TravelMind-Planner-7B-SFT'),
    'v2': ('/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v2', 'TravelMind-Planner-7B-SFT-v2')}


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]


def intervention(row, arm):
    messages = copy.deepcopy(row['messages'])
    if arm == 'original':
        return messages
    state = row['state']
    if arm == 'metadata':
        extra = '以下为输入状态已有的候选属性（非名称推断；不新增景点）：\n'+ '\n'.join(
            p['name']+'：室内属性='+('明确室内' if p.get('indoor') is True else '明确室外' if p.get('indoor') is False else '未知')
            for p in state['pois'])
    elif arm == 'rules':
        extra = ('请落实输入中已有的约束：输出恰好'+str(state['days'])+'天；同一个景点全程仅出现一次，'
            '不能为了填满每天行程而重复；用户更严格的每天数量上限优先，候选不足可少排。'
            '每个景点必须在输入开放窗口内，交通空档须充分；解释中的选点、点数、时长与最终时间表一致。'
            'period只填morning/afternoon/evening，时间用HH:MM；无餐馆候选时meal_slots留空。')
        if '38°C' in state['query']:
            extra += '本诊断把原需求的正午避热明确为11:00-15:00；此时段不安排室外或室内属性未知的景点，不根据名称猜测室内。'
        if state.get('rain_indoor_priority'):
            extra += '只选择有明确室内证据的景点；未知不等于室内。'
    else:
        raise ValueError(arm)
    messages[-1]['content'] += '\n\n【开发诊断附加信息】\n'+extra
    return messages


def switch(tag, *, max_model_len=None):
    path, name = MODELS[tag]
    code = '''
import os,signal,subprocess,time
from pathlib import Path
paths='''+repr({p.encode() for p,n in MODELS.values()})+'''
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=(proc/'cmdline').read_bytes().split(b'\\0')
  if proc.stat().st_uid==os.getuid() and b'/home/jiale/.local/bin/vllm' in cmd and b'serve' in cmd and paths.intersection(cmd):os.kill(int(proc.name),signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
for i in range(45):
 r=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 if int(r.stdout.strip())<500:break
 time.sleep(1)
else:raise RuntimeError('GPU0 occupied')
env=dict(os.environ,MODEL_PATH='''+repr(path)+''',MODEL_NAME='''+repr(name)+''')
env.update('''+repr({'MAX_MODEL_LEN': str(max_model_len)} if max_model_len else {})+''')
with open('/home/jiale/vllm_qwen.log','a') as out:
 p=subprocess.Popen(['bash','/home/jiale/TravelMind-posttrain/scripts/serve_post_train_4090.sh'],env=env,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(p.pid)
'''
    remote(code)
    ready(name)


def status(stage, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT/'status.tmp'
    temp.write_text(json.dumps(dict(stage=stage, updated_at=time.time(), paid_calls=0, **extra), indent=2), encoding='utf-8')
    temp.replace(OUT/'status.json')
    print(json.dumps(dict(stage=stage, **extra)), flush=True)


def select_cases():
    data = ROOT/'data/post_train'
    validation = rows(data/'release_v1/validation.ledger.jsonl')
    comparison = json.loads((data/'sft_validation_v2/v1_v2_comparison.json').read_text())
    selected = set(comparison['improved']+comparison['regressed'])
    a = {r['id']: r for r in rows(data/'sft_validation_v1/sft.jsonl')}
    b = {r['id']: r for r in rows(data/'sft_validation_v2/sft.jsonl')}
    for passed in [True, False]:
        controls = sorted(k for k in a if k not in selected and a[k]['passed'] == passed and b[k]['passed'] == passed)
        selected.update(controls[:3])
    return [r for r in validation if r['id'] in selected]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = select_cases()
    cases_text = json.dumps(selected, ensure_ascii=False, indent=2)
    case_path = OUT/'cases.json'
    if case_path.exists() and case_path.read_text(encoding='utf-8') != cases_text:
        raise RuntimeError('Frozen diagnostic selection changed')
    case_path.write_text(cases_text, encoding='utf-8')
    binding = hashlib.sha256(case_path.read_bytes()+Path(__file__).read_bytes()).hexdigest()
    meta = dict(binding=binding, selected=len(selected), arms=['original','rules','metadata'],
        seed=42, temperature=.3, max_tokens=2048, selection='all 9 changed dev outcomes + 3 stable successes + 3 stable failures; diagnostic selection bias',
        scope='development only, not unbiased benchmark or independent test', paid_calls=0)
    meta_path = OUT/'binding.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise RuntimeError('Diagnostic protocol changed')
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    _, maps = load_assets()
    hard = {c['id']: c for c in json.loads((ROOT/'data/post_train/hard_supplement_cases.json').read_text(encoding='utf-8'))}
    output = OUT/'results.jsonl'
    done = {(r['tag'],r['arm'],r['id']) for r in rows(output)} if output.exists() else set()
    def worker(tag, row, arm):
        request = dict(model=MODELS[tag][1], messages=intervention(row, arm), temperature=.3, seed=42,
            max_tokens=2048, response_format={'type':'json_schema','json_schema':{
                'name':'TravelRoute','strict':False,'schema':TravelRoute.model_json_schema()}})
        result = dict(tag=tag, arm=arm, id=row['id'], request=request, parse_success=False,
            passed=False, faults=[], explanation_faults=[])
        started = time.perf_counter()
        try:
            req = urllib.request.Request('http://10.14.3.82:8000/v1/chat/completions',
                data=json.dumps(request, ensure_ascii=False).encode(), headers={'Content-Type':'application/json'})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=90) as response:
                raw = json.load(response)
            result.update(response=raw,usage=raw.get('usage'))
            if raw['choices'][0]['finish_reason'] != 'stop':
                raise ValueError('incomplete_generation')
            plan = TravelRoute.model_validate_json(raw['choices'][0]['message']['content']).model_dump(mode='json')
            result.update(parse_success=True, plan=plan)
            result['faults'] = score(plan,row,maps,hard)
            result['passed'] = not result['faults']
            result['explanation_faults'] = explanation_faults(plan)
        except Exception as exc:
            result.update(error_type=type(exc).__name__)
        result['elapsed_s'] = time.perf_counter()-started
        return result
    try:
        for tag in MODELS:
            pending = [(r,a) for r in selected for a in meta['arms'] if (tag,a,r['id']) not in done]
            if not pending:
                continue
            status('loading_model', model=tag)
            switch(tag)
            status('evaluating', model=tag, pending=len(pending))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker,tag,r,a) for r,a in pending]
                for future in as_completed(futures):
                    result = future.result()
                    append_json(output,result)
                    print(json.dumps({k:result[k] for k in ['tag','arm','id','passed','faults']}),flush=True)
    finally:
        status('restoring_base')
        switch('base')
    results = rows(output)
    summary = []
    for tag in MODELS:
        for arm in meta['arms']:
            group = [r for r in results if r['tag']==tag and r['arm']==arm]
            summary.append(dict(tag=tag,arm=arm,n=len(group),passed=sum(r['passed'] for r in group),
                parsed=sum(r['parse_success'] for r in group),
                faults=dict(Counter(f for r in group for f in r['faults'])),
                explanation_fault_cases=sum(bool(r['explanation_faults']) for r in group),
                tokens=sum((r.get('usage') or {}).get('total_tokens',0) for r in group)))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    status('completed', total=len(results), summary=summary)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc))
        raise
