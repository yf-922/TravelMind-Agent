"""Wait for remote C training, run only C arms, and refresh the nine-arm report."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE_STATUS = "/home/jiale/TravelMind-posttrain/logs/c-posttrain-20261028/status.json"
RUN_DIR = ROOT / "data/post_train/online_holdout50_20261027_v2/controlled-7arm-localrules-20261028"
OUT = ROOT / ".tmp/c_posttrain_orchestrator"
OUT.mkdir(parents=True, exist_ok=True)


def write_status(stage: str, **extra) -> None:
    row = {"stage": stage, "updated_at": time.time(), **extra}
    path = OUT / "status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def remote_status() -> dict:
    code = f"import json; print(open({REMOTE_STATUS!r}).read())"
    result = subprocess.run(["ssh", "jiale-10.14.3.82", "/home/jiale/.venvs/travelmind-train-smoke/bin/python -"],
                            input=code, text=True, capture_output=True, encoding="utf-8", timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    return json.loads(result.stdout)


def run_command(name: str, args: list[str]) -> None:
    log = (OUT / f"{name}.log").open("x", encoding="utf-8")
    process = subprocess.Popen(args, cwd=ROOT, env=dict(os.environ), stdout=log,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    code = process.wait()
    log.close()
    if code:
        raise RuntimeError(f"{name} exited {code}; inspect {OUT / (name + '.log')}")


def main() -> None:
    if (OUT / "status.json").exists():
        raise RuntimeError("C post-training orchestration already exists; refusing restart")
    started = time.time()
    try:
        while True:
            state = remote_status()
            write_status("waiting_for_training", remote_stage=state.get("stage"), remote_updated_at=state.get("updated_at"))
            if state.get("stage") == "training_complete_ready_for_evaluation":
                break
            if state.get("stage") == "failed":
                raise RuntimeError("Remote C training failed: " + str(state.get("error")))
            time.sleep(60)
        write_status("running_c_incremental_evaluation", training=state)
        run_command("c_incremental_evaluation", [sys.executable, "-X", "utf8", "-u",
                    str(ROOT / "scripts/run_controlled_c_incremental.py")])
        write_status("rendering_nine_model_report")
        run_command("nine_model_report", [sys.executable, "-X", "utf8",
                    str(ROOT / "scripts/report_controlled_c_ablation.py"), "--run-dir", str(RUN_DIR)])
        write_status("complete", elapsed_s=time.time() - started,
                     report=str(RUN_DIR / "controlled_9_model_ablation_report.md"))
    except Exception as exc:
        write_status("failed", error=repr(exc), elapsed_s=time.time() - started)
        raise


if __name__ == "__main__":
    main()
