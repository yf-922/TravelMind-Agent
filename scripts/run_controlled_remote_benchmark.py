"""Run the controlled benchmark on remote GPU0 with merged BF16 weights only."""
import json, subprocess, time, urllib.request, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; HOST="jiale-10.14.3.82"; REMOTE="/home/jiale/TravelMind-posttrain"
PY="/home/jiale/.venvs/travelmind-train-smoke/bin/python"; BASE="/home/jiale/models/Qwen2.5-7B-Instruct"; V2=REMOTE+"/models/TravelMind-Planner-v2-current-20261025"
OUT=ROOT/"data/post_train/controlled_benchmark_20261027"; OUT.mkdir(parents=True,exist_ok=True)
AD={"v2_sft":(V2,REMOTE+"/saves/TravelMind-count-ab-20261025-A",None), # replaced below
    "v2_dpo":(V2,REMOTE+"/saves/shared-dpo-full-20261026-v2",None),
    "A_sft":(V2,REMOTE+"/saves/TravelMind-count-ab-20261025-A",None),
    "A_dpo":(V2,REMOTE+"/saves/shared-dpo-full-20261026-A",None),
    "B_sft":(BASE,REMOTE+"/saves/TravelMind-count-ab-total96-20261025-B",None),
    "B_dpo":(BASE,REMOTE+"/saves/shared-dpo-full-20261026-B",None)}
def ssh(c,check=True): return subprocess.run(["ssh",HOST,c],check=check,text=True,capture_output=True)
def stop(): ssh("pkill -TERM -f '/home/jiale/.local/bin/vllm serve' 2>/dev/null || true",False); time.sleep(5)
def merge(tag,base,adapter):
    path=f"{REMOTE}/models/controlled-20261027-{tag}"
    ssh(f"test -f {path}/model.safetensors.index.json || {PY} {REMOTE}/scripts/post_train_train_entry.py merge --base {base} --adapter {adapter} --output {path}")
    return path
def serve(path,name):
    ssh(f"nohup /home/jiale/.local/bin/vllm serve {path} --host 0.0.0.0 --port 8000 --served-model-name {name} --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 --enforce-eager --no-enable-prefix-caching > {REMOTE}/logs/controlled-{name}.log 2>&1 < /dev/null &")
    for _ in range(180):
        try:
            with urllib.request.urlopen("http://10.14.3.82:8000/v1/models",timeout=2) as r:
                if any(x.get("id")==name for x in json.load(r).get("data",[])): return
        except Exception: pass
        time.sleep(2)
    raise RuntimeError("vLLM not ready: "+name)
def main():
    stop(); status={"status":"running","variants":[]}; (OUT/"status.json").write_text(json.dumps(status,indent=2))
    # Merge every tested weight into the same BF16 representation.
    merged={"base":BASE,"v2_sft":V2}
    merged["v2_dpo"]=merge("v2-dpo",V2,REMOTE+"/saves/shared-dpo-full-20261026-v2")
    merged["A_sft"]=merge("A-sft",V2,REMOTE+"/saves/TravelMind-count-ab-20261025-A")
    merged["A_dpo"]=merge("A-dpo",merged["A_sft"],REMOTE+"/saves/shared-dpo-full-20261026-A")
    merged["B_sft"]=merge("B-sft",BASE,REMOTE+"/saves/TravelMind-count-ab-total96-20261025-B")
    merged["B_dpo"]=merge("B-dpo",merged["B_sft"],REMOTE+"/saves/shared-dpo-full-20261026-B")
    for tag,path in merged.items():
        name="controlled-"+tag; serve(path,name)
        for mode in ["planner-only","rules-rework"]:
            folder=OUT/f"{tag}_{mode}"
            subprocess.run(["python","-u","scripts/benchmark_controlled.py","--url","http://10.14.3.82:8000/v1/chat/completions","--model",name,"--mode",mode,"--output",str(folder),"--concurrency","1"],check=True)
        stop(); status["variants"].append(tag); (OUT/"status.json").write_text(json.dumps(status,indent=2))
    stop(); status.update(status="complete",elapsed_s=time.time()); (OUT/"status.json").write_text(json.dumps(status,indent=2))
if __name__=="__main__": main()
