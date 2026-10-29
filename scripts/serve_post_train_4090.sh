#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Locally downloaded Ubuntu development headers, unpacked without root access.
# Triton structured-output kernels compile on first request even in eager mode.
HEADER_ROOT=/home/jiale/python-dev/extracted/usr/include
test -f "$HEADER_ROOT/python3.10/Python.h"
export CPATH="$HEADER_ROOT/python3.10:$HEADER_ROOT/x86_64-linux-gnu/python3.10:$HEADER_ROOT${CPATH:+:$CPATH}"
exec /home/jiale/.local/bin/vllm serve "${MODEL_PATH:-/home/jiale/models/Qwen2.5-7B-Instruct}" \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --served-model-name "${MODEL_NAME:-Qwen2.5-7B-Instruct}" \
  --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-4096}" --gpu-memory-utilization 0.90 --enforce-eager
