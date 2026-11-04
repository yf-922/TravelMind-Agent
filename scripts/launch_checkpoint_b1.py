"""Run the frozen B-1 C-DPO prompt replay and persist raw responses."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data/eval/checkpoint_modifications_v2_prompts.json"
OUT = ROOT / "data/eval/checkpoint_prompt_diff_b0/b1_c_dpo_20261030"
HOST = "jiale-10.14.3.82"
REMOTE = "/home/jiale/TravelMind-posttrain"
PY = "/home/jiale/.venvs/travelmind-train-smoke/bin/python"
MODEL = "b1-c-dpo"
PORT = 8000

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def openai_messages(messages):
    """Convert LangChain's ``human`` role to the OpenAI/vLLM ``user`` role."""
    return [{**m, "role": "user" if m.get("role") == "human" else m.get("role")}
            for m in messages]

def remote(code: str) -> str:
    p = subprocess.run(["ssh", HOST, PY, "-"], input=code, text=True,
                       capture_output=True, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:])
    return p.stdout.strip()

def call(messages, seed: int) -> dict:
    payload = {"model": MODEL, "messages": messages, "temperature": 0.3,
               "top_p": 0.95, "seed": seed, "max_tokens": 2048,
               "stream": True, "stream_options": {"include_usage": True},
               "response_format": {"type": "json_object"}}
    req = urllib.request.Request(f"http://10.14.3.82:{PORT}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"})
    started = time.perf_counter(); first = None; chunks = []; usage = {}; finish = None
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            event = json.loads(data)
            if event.get("usage"): usage = event["usage"]
            for choice in event.get("choices", []):
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    if first is None: first = time.perf_counter()
                    chunks.append(delta)
                if choice.get("finish_reason"): finish = choice["finish_reason"]
    text = "".join(chunks); ended = time.perf_counter()
    return {"request": payload, "output": text, "output_sha256": digest(text.encode()),
            "usage": usage, "finish_reason": finish, "e2e_s": ended-started,
            "ttft_s": ended-started if first is None else first-started}

def start_server(model_path: str) -> None:
    remote(f"""
import os,socket,subprocess
from pathlib import Path
s=socket.socket()
if s.connect_ex(('127.0.0.1',{PORT}))==0: raise RuntimeError('port 8000 occupied')
s.close()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
log=Path({REMOTE!r})/'logs/b1-c-dpo-20261030-vllm.log'
args=['/home/jiale/.local/bin/vllm','serve',{model_path!r},'--host','0.0.0.0','--port',str({PORT}),'--served-model-name',{MODEL!r},'--dtype','bfloat16','--max-model-len','8192','--gpu-memory-utilization','0.90','--enforce-eager','--no-enable-prefix-caching']
with log.open('a') as f:
 p=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(Path({REMOTE!r})/'logs/b1-c-dpo-20261030-vllm.pid').write_text(str(p.pid))
print(p.pid)
""")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(180):
        try:
            with opener.open(f"http://10.14.3.82:{PORT}/v1/models", timeout=3) as r:
                ids = {x["id"] for x in json.load(r).get("data", [])}
            if MODEL in ids: return
        except Exception: pass
        time.sleep(2)
    raise RuntimeError("vLLM readiness timeout")

def stop_server() -> None:
    remote(f"""
from pathlib import Path
import os,signal
p=Path({REMOTE!r})/'logs/b1-c-dpo-20261030-vllm.pid'
if p.exists():
 pid=int(p.read_text()); proc=Path('/proc')/str(pid)
 if proc.exists(): os.killpg(pid,signal.SIGTERM)
 p.unlink()
""")

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--model-path',required=True); args=ap.parse_args()
    prompts=json.loads(PROMPTS.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=False)
    manifest={"schema_version":2,"run_id":"checkpoint-b1-c-dpo-20261030",
      "measurement_declaration":"new measurement; not historical recovery",
      "launch_allowed":True,"deployment":"C-DPO merged BF16",
      "model_path":args.model_path,"temperature":0.3,"top_p":0.95,
      "max_tokens":2048,"arms":["F@seed1001","I@seed1001","F@seed1002"],
      "requests_per_arm":120,"concurrency":1,"enforce_eager":True,
      "prefix_cache":False,"prompt_source_sha256":digest(PROMPTS.read_bytes())}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    start_server(args.model_path); rows=[]
    try:
        for p in prompts:
            for arm,seed,key in [("F@seed1001",1001,"full"),("I@seed1001",1001,"local"),("F@seed1002",1002,"full")]:
                msgs=openai_messages(p[key]["planner"])
                rec={"probe_id":p["id"],"arm":arm,"seed":seed,"manifest":"manifest.json",
                     "prompt_sha256":digest(json.dumps(msgs,ensure_ascii=False,separators=(',',':')).encode())}
                try: rec.update(call(msgs,seed)); rec["status"]="ok"
                except Exception as e: rec.update(status="error",error=repr(e))
                with (OUT/'requests_outputs.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                rows.append(rec)
    finally: stop_server()
    summary={"records":len(rows),"ok":sum(r.get('status')=='ok' for r in rows),"errors":sum(r.get('status')!='ok' for r in rows),
      "rule_audit":"not_evaluable from saved prompts alone; no route-state/road ledger was stored"}
    (OUT/'b1_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__ == '__main__': main()
