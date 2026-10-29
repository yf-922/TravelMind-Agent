"""Server-side SFT job: stop owned GPU0 service, train/merge, restore Base.

Never runs DPO implicitly. Logs and an atomic status file survive SSH disconnects.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

ROOT = Path('/home/jiale/TravelMind-posttrain')


def status(**values):
    values['updated_at'] = time.time()
    target = ROOT/'logs/sft-full-status.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, indent=2))
    temporary.replace(target)
    print(json.dumps(values), flush=True)


def stop_owned_service():
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            cmd = (process/'cmdline').read_bytes().split(b'\0')
            if (process.stat().st_uid == os.getuid() and b'serve' in cmd
                    and b'/home/jiale/.local/bin/vllm' in cmd
                    and b'/home/jiale/models/Qwen2.5-7B-Instruct' in cmd):
                os.kill(int(process.name), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    for _ in range(45):
        r = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        if int(r.stdout.strip()) < 500:
            return
        time.sleep(1)
    raise RuntimeError('GPU0 is still occupied; training was not started')


def record_gpu(stop):
    with (ROOT/'logs/sft-full-gpu.jsonl').open('a') as out:
        while not stop.is_set():
            r = subprocess.run(['nvidia-smi', '-i', '0',
                '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            out.write(json.dumps({'time': time.time(), 'gpu0': r.stdout.strip()})+'\n')
            out.flush()
            stop.wait(5)


def main():
    os.chdir(ROOT)
    if (ROOT/'saves/TravelMind-Planner-SFT/trainer_state.json').exists():
        raise RuntimeError('Existing full SFT run found; choose a new output or explicit resume')
    started = time.time()
    status(stage='stopping_base_service', started_at=started, dpo_auto_start=False)
    stop_owned_service()
    monitor_stop = threading.Event()
    monitor = threading.Thread(target=record_gpu, args=(monitor_stop,), daemon=True)
    monitor.start()
    code = -1
    try:
        status(stage='sft_training_and_merge', started_at=started, dpo_auto_start=False)
        with (ROOT/'logs/sft-full-pipeline.log').open('w') as out:
            code = subprocess.run(['bash', 'scripts/run_post_train_4090.sh', '--sft-only'],
                stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    finally:
        monitor_stop.set()
        monitor.join(timeout=6)
        # Free training GPU before restoring the original inference model. A
        # successful run does not silently deploy its newly trained weights.
        log = open('/home/jiale/vllm_qwen.log', 'a')
        service = subprocess.Popen(['bash', str(ROOT/'scripts/serve_post_train_4090.sh')],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        status(stage='sft_complete_pending_validation' if code == 0 else 'sft_failed',
            exit_code=code, started_at=started, elapsed_s=time.time()-started,
            base_service_pid=service.pid, base_service_readiness='starting_not_yet_verified',
            dpo_auto_start=False)


if __name__ == '__main__':
    main()
