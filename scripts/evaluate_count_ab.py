"""GPU0 count-repair A/B generation evaluation; no merge or paid API."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_shared_prompt_all_versions as shared
from scripts.prepare_count_infeasible_ab import check, OUT as DATA
from scripts.run_sft_validation_job import remote, ready
from scripts.evaluate_representative50 import SUITE, verify

OUT = DATA/'evaluation_total96'
BASE = '/home/jiale/models/Qwen2.5-7B-Instruct'
V2 = '/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-v2-current-20261025'
MODELS = {
    'v2-current': (V2, None),
    'A': (V2, '/home/jiale/TravelMind-posttrain/saves/TravelMind-count-ab-20261025-A'),
    'B': (BASE, '/home/jiale/TravelMind-posttrain/saves/TravelMind-count-ab-total96-20261025-B'),
    'base': (BASE, None),
}

def switch(tag):
    base, adapter = MODELS[tag]
    name = 'count-ab-'+tag
    args = ['/home/jiale/.local/bin/vllm', 'serve', base, '--host', '0.0.0.0', '--port', '8000',
            '--served-model-name', name if not adapter else 'count-ab-parent', '--dtype', 'bfloat16',
            '--max-model-len', '8192', '--gpu-memory-utilization', '0.90', '--enforce-eager']
    if adapter:
        args += ['--enable-lora', '--max-lora-rank', '16', '--lora-modules', name+'='+adapter]
    remote('''
import os,signal,subprocess,time
from pathlib import Path
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().split(b'\\0')
  if p.stat().st_uid==os.getuid() and b'/home/jiale/.local/bin/vllm' in cmd and b'serve' in cmd:
   os.kill(int(p.name),signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
for _ in range(60):
 r=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 if int(r.stdout.strip())<500:break
 time.sleep(1)
else:raise RuntimeError('GPU0 occupied')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
h='/home/jiale/python-dev/extracted/usr/include'
env['CPATH']=h+'/python3.10:'+h+'/x86_64-linux-gnu/python3.10:'+h
with open('/home/jiale/vllm_count_ab_eval.log','a') as f:
 p=subprocess.Popen('''+repr(args)+''',env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(p.pid)
''')
    ready(name)
    shared.MODELS[tag] = (base, name)

def status(stage, **extra):
    shared.atomic(OUT/'status.json', dict(stage=stage, updated_at=time.time(), paid_api_calls=0, **extra))
    print(json.dumps(dict(stage=stage, **extra)), flush=True)

def count_summary(rs, cases):
    byid={c['id']:c for c in cases}
    groups={key:[r for r in rs if byid[r['id']]['expected']==key] for key in ['infeasible','feasible']}
    result={key:dict(n=len(g),passed=sum(r['passed'] for r in g)) for key,g in groups.items()}
    pairs={}
    for r in rs:pairs.setdefault((byid[r['id']]['pair_id'],r['seed']),[]).append(r)
    result['paired_both_pass']=sum(len(g)==2 and all(r['passed'] for r in g) for g in pairs.values())
    result['paired_total']=len(pairs)
    result['parsed']=sum(r['parse_success'] for r in rs)
    result['tokens']=sum(r.get('usage_total',0) for r in rs)
    return result

def report():
    text=['# 数量不可行 A/B 统一生成评测','','A完整路径96步；B从底座96步。相同步数不等于相同数据曝光。固定生产提示词、Schema、种子42/43/44、temperature=0.3；规则返工最多3轮，无付费API。', '',
          '|模型|模式|数量无解正确|配对有解通过|配对双通过|旧50题规则通过|旧50题规则+解释|',
          '|---|---|---:|---:|---:|---:|---:|']
    for tag in MODELS:
        for mode in ['planner-only','rules-rework']:
            c=OUT/'count'/f'{tag}_{mode}'/'summary.json'
            r=OUT/'regression'/f'{tag}_{mode}'/'summary.json'
            cs=json.loads(c.read_text()) if c.exists() else None
            rs=json.loads(r.read_text()) if r.exists() else None
            def ratio(x):return f"{x['passed']}/{x['n']}" if x else '待测'
            text.append(f"|{tag}|{mode}|{ratio(cs['infeasible']) if cs else '待测'}|{ratio(cs['feasible']) if cs else '待测'}|{str(cs['paired_both_pass'])+'/'+str(cs['paired_total']) if cs else '待测'}|{str(rs['all']['rules_pass'])+'/150' if rs else '待测'}|{str(rs['all']['joint_pass'])+'/150' if rs else '待测'}|")
    text+=['','数量诊断40题来自留出城市，但规则模板与训练同构，只能证明数量边界迁移；旧50题为已反复诊断的开发集。高温/未知室内无解不属于本次补数范围，须单独查看分层原始结果。所有失败保留在分母，每轮输入、输出、故障、Token与耗时均落盘。']
    (ROOT/'docs/post_train_count_ab_evaluation.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    p=DATA/'test.ledger.json'
    if hashlib.sha256(p.read_bytes()).hexdigest()!=manifest['sha256'][str(p.relative_to(ROOT)).replace('\\','/')]:
        raise RuntimeError('Count test hash mismatch')
    count=json.loads(p.read_text(encoding='utf-8'))
    for c in count:
        if check(c,c['chosen']):raise RuntimeError('Invalid frozen witness')
    verify()
    regression=json.loads((SUITE/'cases.json').read_text(encoding='utf-8'))
    shared.maps=json.loads((SUITE/'maps.json').read_text(encoding='utf-8'))['routes']
    original_score=shared.score_case
    try:
        for tag in MODELS:
            status('loading_model', model=tag)
            switch(tag)
            for suite,cases in [('count',count),('regression',regression)]:
                shared.OUT=OUT/suite
                shared.score_case=(lambda plan,case,maps:check(case,plan)) if suite=='count' else original_score
                for mode in ['planner-only','rules-rework']:
                    status('evaluating', model=tag,suite=suite,mode=mode)
                    rs=shared.run_one(tag,mode,cases,[42,43,44],4)
                    summary=count_summary(rs,cases) if suite=='count' else shared.summarize(rs,cases)
                    shared.atomic(shared.OUT/f'{tag}_{mode}'/'summary.json',summary)
                    report()
            status('model_complete',model=tag)
        status('completed')
    except Exception as exc:
        status('failed',error=str(exc));raise
    finally:
        # Restore the original serving name, not the evaluation alias.
        from scripts.diagnose_sft_transfer import switch as restore
        shared.MODELS['base']=(BASE,'Qwen2.5-7B-Instruct')
        restore('base',max_model_len=8192)

if __name__=='__main__':main()
