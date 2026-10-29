"""Immediately follow the running server SFT job with paired local evaluation.

No DPO, paid endpoints, heldout-test inspection, or permanent service deployment.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_sft_validation import run, compare

OUTPUT = ROOT/'data/post_train/sft_validation_v1'


def remote(code):
    result = subprocess.run(['ssh', 'jiale-10.14.3.82', 'python3', '-'], input=code,
        encoding='utf-8', capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError('Server command failed: '+result.stderr[-1500:])
    return result.stdout


def write_status(stage, **extra):
    status = {'stage': stage, 'updated_at': time.time(), 'dpo_auto_start': False, **extra}
    (OUTPUT/'job_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status), flush=True)


def ready(model):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(60):
        try:
            with opener.open('http://10.14.3.82:8000/v1/models', timeout=3) as response:
                models = json.load(response)['data']
            if any(m['id'] == model for m in models):
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError('Requested vLLM model did not become ready')


def switch(sft):
    path = '/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT' if sft else '/home/jiale/models/Qwen2.5-7B-Instruct'
    name = 'TravelMind-Planner-7B-SFT' if sft else 'Qwen2.5-7B-Instruct'
    # Paths are fixed constants, never external shell text. Only owned matching
    # vLLM server processes are stopped; no other user's jobs are touched.
    remote('''
import os,signal,subprocess,time
from pathlib import Path
paths={b'/home/jiale/models/Qwen2.5-7B-Instruct',b'/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT'}
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  cmd=(process/'cmdline').read_bytes().split(b'\\0')
  if process.stat().st_uid==os.getuid() and b'/home/jiale/.local/bin/vllm' in cmd and b'serve' in cmd and paths.intersection(cmd):
   os.kill(int(process.name),signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
for i in range(45):
 result=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 if int(result.stdout.strip())<500:break
 time.sleep(1)
else:raise RuntimeError('GPU0 occupied; model switch aborted')
env=dict(os.environ,MODEL_PATH='''+repr(path)+''',MODEL_NAME='''+repr(name)+''')
log=open('/home/jiale/vllm_qwen.log','a')
p=subprocess.Popen(['bash','/home/jiale/TravelMind-posttrain/scripts/serve_post_train_4090.sh'],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(p.pid)
''')
    ready(name)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_status('waiting_for_running_sft_job')
    for _ in range(60):
        state = json.loads(remote("from pathlib import Path\nprint(Path('/home/jiale/TravelMind-posttrain/logs/sft-full-status.json').read_text())"))
        if state['stage'] == 'sft_failed':
            raise RuntimeError('SFT failed; validation not run')
        if state['stage'] == 'sft_complete_pending_validation':
            break
        time.sleep(30)
    else:
        raise RuntimeError('SFT did not finish within 30 minutes')
    ready('Qwen2.5-7B-Instruct')
    write_status('base_validation')
    run('base', 'Qwen2.5-7B-Instruct', OUTPUT)
    switched = False
    try:
        write_status('loading_sft_for_validation')
        switched = True
        switch(True)
        write_status('sft_validation')
        run('sft', 'TravelMind-Planner-7B-SFT', OUTPUT)
        result = compare(OUTPUT)
        report = ['# SFT首轮验证对照', '',
            '仅使用30条开发验证状态，零返工，不使用独立城市测试，所有失败计入分母。', '',
            '|模型|完整规则通过|JSON解析|调用错误|总Token|', '|---|---|---|---|---|']
        for tag in ['base', 'sft']:
            s = json.loads((OUTPUT/f'{tag}.summary.json').read_text())
            report.append(f'|{tag}|{s["passed"]}/{s["n"]}|{s["parsed"]}/{s["n"]}|{s["errors"]}|{s["total_tokens"]}|')
        report += ['', f'逐场景改善{len(result["improved"])}条、退化{len(result["regressed"])}条。',
            '这是开发集诊断，不能当作最终四组独立测试结论。DPO未自动启动，须检查具体退化案例和解释质量。']
        (ROOT/'docs/post_train_sft_validation_report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    finally:
        if switched:
            write_status('restoring_original_base_service')
            switch(False)
    write_status('completed_pending_sft_review', comparison=result)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_status('failed', error_type=type(exc).__name__, error=str(exc))
        raise
