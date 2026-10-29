"""Evaluate C-SFT and C-DPO against the frozen seven-arm local-rules run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_online_holdout50 import OUT as SUITE, verify as verify_suite
from scripts.run_controlled_online_holdout_matrix import (
    HOST, PY, remote, resource, start, stop_owned, wait_gpu_zero,
)
from scripts.run_online_holdout_matrix import REMOTE, BASE

RUN_ID = "controlled-7arm-localrules-20261028"
RUN_DIR = SUITE / RUN_ID
ROLLING = REMOTE + "/models/controlled-c-rolling-20261028"
SFT_ADAPTER = REMOTE + "/saves/TravelMind-C-SFT-20261028"
DPO_ADAPTER = REMOTE + "/saves/TravelMind-C-DPO-20261028"
DPO_PARENT = REMOTE + "/models/TravelMind-C-SFT-DPO-parent-20261028"
MERGER = REMOTE + "/scripts/post_train_train_entry.py"
TAGS = {"C_sft": "c_sft", "C_dpo": "c_dpo"}
PROTOCOL_FILES = [
    "app/planning/graph.py", "app/planning/nodes.py", "app/planning/prompts.py",
    "app/planning/schemas.py", "app/planning/constraints.py",
    "app/evaluation/online_replay.py", "scripts/evaluate_controlled_online_holdout.py",
    "scripts/run_controlled_online_holdout_matrix.py", "scripts/prepare_controlled_holdout_weights.py",
    "scripts/report_controlled_online_holdout.py", "scripts/build_representative_eval50.py",
    "scripts/post_train_runtime.py",
]


def sha256_remote(path: str) -> dict:
    code = f"""
import hashlib,json
from pathlib import Path
p=Path({path!r})
files=sorted(p.glob('*.safetensors')) if p.is_dir() else [p]
if not files or any(not x.is_file() for x in files): raise RuntimeError('missing model/adapter files: '+str(p))
print(json.dumps({{'path':str(p),'files':[{{'name':x.name,'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()}} for x in files]}}))
"""
    return json.loads(remote(code))


def merge(tag: str, base: str, adapter: str, output: str) -> dict:
    if json.loads(remote(f"import os,json;print(json.dumps(os.path.exists({output!r})))")):
        raise RuntimeError(f"Refuse to overwrite existing merge output: {output}")
    info = json.loads(remote(f"import json;from pathlib import Path;p=Path({(adapter + '/adapter_config.json')!r});print(p.read_text())"))
    if info.get("base_model_name_or_path") != base:
        raise RuntimeError(f"{tag} adapter parent mismatch: {info.get('base_model_name_or_path')} != {base}")
    command = " ".join([
        "cd", REMOTE, "&&", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
        PY, MERGER, "merge", "--base", base, "--adapter", adapter, "--output", output,
    ])
    result = subprocess.run(["ssh", HOST, command], capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{tag} merge failed: {result.stderr[-3000:]}")
    return {"tag": tag, "base": base, "adapter": sha256_remote(adapter), "merged": sha256_remote(output)}


def cleanup_exact(path: str) -> None:
    remote(f"""
from pathlib import Path
import shutil
root=Path({ROLLING!r}).resolve(); p=Path({path!r}).resolve()
if p.parent != root or p.name not in {{'C_sft','C_dpo'}}: raise RuntimeError('refuse cleanup outside exact C rolling path')
if p.exists(): shutil.rmtree(p)
""")


def append_binding(binding: dict) -> None:
    path = RUN_DIR / "c_incremental_binding.json"
    path.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    frozen = verify_suite()
    baseline = json.loads((RUN_DIR / "binding.json").read_text(encoding="utf-8"))
    if baseline["run_id"] != RUN_ID or baseline["suite_sha256"] != frozen["hashes"]["cases.json"]:
        raise RuntimeError("Frozen suite / seven-arm binding mismatch")
    for tag in ["base", "v2_sft", "A_sft", "B_sft", "v2_dpo", "A_dpo", "B_dpo"]:
        if not (RUN_DIR / tag / "summary.json").is_file():
            raise RuntimeError("Seven-arm baseline is incomplete: " + tag)
    protocol_now = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in PROTOCOL_FILES}
    mismatches = {f: {"baseline": baseline["protocol_hashes"].get(f), "current": value}
                  for f, value in protocol_now.items() if baseline["protocol_hashes"].get(f) != value}
    if set(mismatches) - {"scripts/evaluate_controlled_online_holdout.py"}:
        raise RuntimeError("Non-evaluator protocol source drift: " + json.dumps(mismatches, ensure_ascii=False))
    binding = {
        "run_id": "controlled-c-localrules-20261028",
        "extends_run_id": RUN_ID,
        "tags": list(TAGS),
        "seeds": [1001, 1002, 1003],
        "modes": ["planner-only", "online-rework"],
        "cases_sha256": frozen["hashes"]["cases.json"],
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 2048,
        "prefix_cache": False,
        "weight_form": "full BF16",
        "warmup": 5,
        "concurrency": 1,
        "rework": "frozen deterministic local rubric; up to three extra Planner calls",
        "reviewer_api_calls": 0,
        "time_check_api_calls": 0,
        "protocol_hashes": protocol_now,
        "baseline_protocol_hash_mismatches": mismatches,
        "notes": "The evaluator file changed after the seven-arm run; its metric/result serialization hash differs. Current execution keeps the frozen cases, production graph/prompt, local-rules feedback, decoder settings, seeds, warmup, and weight form. See this binding before interpreting as byte-identical evaluator source.",
    }
    binding_path = RUN_DIR / "c_incremental_binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text(encoding="utf-8")) != binding:
        raise RuntimeError("Existing C binding differs; refusing to overwrite")
    binding_path.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    remote(f"from pathlib import Path;Path({ROLLING!r}).mkdir(parents=True,exist_ok=True)")
    started = time.time()
    weights = {}
    try:
        for tag in ["C_sft", "C_dpo"]:
            if (RUN_DIR / tag / "summary.json").exists():
                raise RuntimeError("C results already exist; refusing to overwrite: " + tag)
            state = resource()
            if not state["ready"]:
                raise RuntimeError("GPU0 or port 8000 is not idle: " + json.dumps(state))
            output = ROLLING + "/" + tag
            if tag == "C_sft":
                evidence = merge(tag, BASE, SFT_ADAPTER, output)
            else:
                evidence = merge(tag, DPO_PARENT, DPO_ADAPTER, output)
            weights[tag] = evidence
            try:
                inventory = start(tag, output)
                (RUN_DIR / (tag + "-inventory.json")).write_text(json.dumps(inventory, indent=2), encoding="utf-8")
                command = [sys.executable, "-X", "utf8", "-u", str(ROOT / "scripts/evaluate_controlled_online_holdout.py"),
                           "--suite", str(SUITE), "--tag", tag, "--model-name", "controlled-" + tag,
                           "--base-url", "http://10.14.3.82:8000/v1", "--output", str(RUN_DIR),
                           "--temperature", "0.3", "--max-tokens", "2048"]
                with (RUN_DIR / (tag + ".log")).open("x", encoding="utf-8") as stream:
                    result = subprocess.run(command, cwd=ROOT, env=dict(os.environ), stdout=stream,
                                            stderr=subprocess.STDOUT, text=True)
                if result.returncode:
                    raise RuntimeError(f"{tag} evaluator failed; inspect {tag}.log")
            finally:
                stop_owned()
                idle = wait_gpu_zero()
                cleanup_exact(output)
                (RUN_DIR / (tag + "-cleanup.json")).write_text(json.dumps({"gpu_idle": idle, "merged_deleted": output}, indent=2), encoding="utf-8")
            weights[tag]["cleanup"] = "merged evaluation directory removed after GPU release"
            (RUN_DIR / (tag + "-weights.json")).write_text(json.dumps(weights[tag], indent=2), encoding="utf-8")
        remote(f"""
from pathlib import Path
import shutil
p=Path({DPO_PARENT!r}).resolve(); allowed=Path({(REMOTE + '/models')!r}).resolve()
if p.parent != allowed or p.name != 'TravelMind-C-SFT-DPO-parent-20261028': raise RuntimeError('unexpected DPO parent path')
if p.exists(): shutil.rmtree(p)
""")
        binding["completed"] = ["C_sft", "C_dpo"]
        binding["elapsed_s"] = time.time() - started
        binding["training_dpo_parent_deleted"] = True
        binding_path.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        binding["stage"] = "failed"
        binding["error"] = str(exc)
        binding["completed_so_far"] = [tag for tag in TAGS if (RUN_DIR / tag / "summary.json").exists()]
        binding_path.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        stop_owned()


if __name__ == "__main__":
    main()
