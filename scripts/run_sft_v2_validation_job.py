"""Wait for SFT-v2, evaluate the frozen dev inputs, then restore Base."""
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_sft_validation import run, compare
from scripts.run_sft_validation_job import remote, ready

OUT = ROOT/'data/post_train/sft_validation_v2'
MODEL = 'TravelMind-Planner-7B-SFT-v2'


def status(stage, **extra):
    row = dict(stage=stage, updated_at=time.time(), dpo_auto_start=False, **extra)
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT/'job_status.tmp'
    temporary.write_text(json.dumps(row, indent=2), encoding='utf-8')
    temporary.replace(OUT/'job_status.json')
    print(json.dumps(row), flush=True)


def switch(sft):
    path = '/home/jiale/TravelMind-posttrain/models/'+MODEL if sft else '/home/jiale/models/Qwen2.5-7B-Instruct'
    name = MODEL if sft else 'Qwen2.5-7B-Instruct'
    remote('''
import os,signal,subprocess,time
from pathlib import Path
paths={b'/home/jiale/models/Qwen2.5-7B-Instruct',b'/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v2'}
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  cmd=(process/'cmdline').read_bytes().split(b'\\0')
  if process.stat().st_uid==os.getuid() and b'/home/jiale/.local/bin/vllm' in cmd and b'serve' in cmd and paths.intersection(cmd):
   os.kill(int(process.name),signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
for i in range(45):
 r=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 if int(r.stdout.strip())<500:break
 time.sleep(1)
else:raise RuntimeError('GPU0 occupied; validation switch aborted')
env=dict(os.environ,MODEL_PATH='''+repr(path)+''',MODEL_NAME='''+repr(name)+''')
with open('/home/jiale/vllm_qwen.log','a') as stream:
 p=subprocess.Popen(['bash','/home/jiale/TravelMind-posttrain/scripts/serve_post_train_4090.sh'],env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(p.pid)
''')
    ready(name)


def main():
    status('waiting_for_sft_v2')
    for _ in range(120):
        row = json.loads(remote("from pathlib import Path\nprint(Path('/home/jiale/TravelMind-posttrain/logs/sft-v2-status.json').read_text())"))
        if row['stage'] == 'sft_failed':
            raise RuntimeError('SFT-v2 failed')
        if row['stage'] == 'sft_complete_pending_validation':
            break
        time.sleep(15)
    else:
        raise RuntimeError('SFT-v2 did not complete within 30 minutes')
    switched = False
    try:
        ready('Qwen2.5-7B-Instruct')
        status('loading_sft_v2')
        switched = True
        switch(True)
        status('evaluating_sft_v2')
        result = run('sft', MODEL, OUT)
        # Preserve original recordings: same input, temperature, seed, schema and
        # 2048 output budget. No paid teacher calls or test-set model selection.
        old = ROOT/'data/post_train/sft_validation_v1'
        for suffix in ['jsonl', 'summary.json', 'meta.json']:
            target = OUT/f'base.{suffix}'
            source = old/f'base.{suffix}'
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise RuntimeError('Recorded Base baseline changed')
            shutil.copyfile(source, target)
        base_comparison = compare(OUT)
        previous = {r['id']: r for r in [json.loads(l) for l in (old/'sft.jsonl').read_text(encoding='utf-8').splitlines()]}
        current = {r['id']: r for r in [json.loads(l) for l in (OUT/'sft.jsonl').read_text(encoding='utf-8').splitlines()]}
        if previous.keys() != current.keys() or len(current) != 30:
            raise RuntimeError('Paired SFT-v1/v2 inputs incomplete')
        for key in previous:
            a, b = previous[key]['request'], current[key]['request']
            if {k:v for k,v in a.items() if k != 'model'} != {k:v for k,v in b.items() if k != 'model'}:
                raise RuntimeError('SFT validation protocol mismatch: '+key)
        comparison = {'n': 30, 'v1_passed': sum(r['passed'] for r in previous.values()),
            'v2_passed': result['passed'],
            'improved': [k for k in previous if not previous[k]['passed'] and current[k]['passed']],
            'regressed': [k for k in previous if previous[k]['passed'] and not current[k]['passed']],
            'base_comparison': base_comparison, 'baseline_source': 'preserved v1 raw recordings, not resampled',
            'independent_test': False, 'dpo_auto_start': False}
        (OUT/'v1_v2_comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        report = ['# SFT-v2开发验证', '',
            '固定30条开发验证、零返工、相同2048输出预算；Base和SFT-v1沿用已保存原始请求响应。',
            '本表不是独立测试或四组最终结果，所有失败均计入分母。', '',
            '|模型|完整约束通过|JSON解析|总Token|', '|---|---:|---:|---:|']
        for label, folder, tag in [('Base', old, 'base'), ('SFT-v1', old, 'sft'), ('SFT-v2', OUT, 'sft')]:
            s = json.loads((folder/f'{tag}.summary.json').read_text())
            report.append(f'|{label}|{s["passed"]}/{s["n"]}|{s["parsed"]}/{s["n"]}|{s["total_tokens"]}|')
        report += ['', f'相对SFT-v1改善{len(comparison["improved"])}例，退化{len(comparison["regressed"])}例。',
            '须检查配对失败、截断和解释质量，再决定DPO。既有errors字段混合请求/解析/评分失败，不能统称接口调用错误。',
            'SFT-v2错误类别：'+json.dumps(result['faults'], ensure_ascii=False)]
        (ROOT/'docs/post_train_sft_v2_validation_report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    finally:
        if switched:
            status('restoring_base')
            switch(False)
    status('completed_pending_sft_v2_review', comparison=comparison)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc))
        raise
