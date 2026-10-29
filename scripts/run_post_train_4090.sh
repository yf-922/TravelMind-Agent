#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON="${TRAIN_PYTHON:-/home/jiale/.venvs/travelmind-train-smoke/bin/python}"
SMOKE=0
SFT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --sft-only) SFT_ONLY=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [[ "$SMOKE" == 0 && "$SFT_ONLY" == 0 ]]; then
  echo 'Run --sft-only first and evaluate it; DPO is a separate decision after the SFT comparison.' >&2
  exit 1
fi
EXTRA=()
SUFFIX=''
if [[ "$SMOKE" == 1 ]]; then
  EXTRA=(max_steps=5 max_samples=16 gradient_accumulation_steps=1 'eval_strategy="no"' 'save_strategy="no"' load_best_model_at_end=false)
  SUFFIX=-smoke
fi
mkdir -p logs
"$PYTHON" scripts/post_train_train_entry.py train configs/post_train/qwen25_7b_planner_sft.yaml \
  output_dir="saves/TravelMind-Planner-SFT${SUFFIX}" "${EXTRA[@]}" 2>&1 | tee "logs/sft${SUFFIX}.log"
# Merge on CPU, without a quantized base. A separate config uses real export keys.
"$PYTHON" scripts/post_train_train_entry.py merge \
  --base /home/jiale/models/Qwen2.5-7B-Instruct \
  --adapter "saves/TravelMind-Planner-SFT${SUFFIX}" \
  --output "models/TravelMind-Planner-7B-SFT${SUFFIX}"
if [[ "$SFT_ONLY" == 1 ]]; then
  echo 'SFT training and best-checkpoint CPU merge completed. DPO awaits SFT validation comparison.'
  exit 0
fi
"$PYTHON" scripts/post_train_train_entry.py train configs/post_train/qwen25_7b_planner_dpo.yaml \
  model_name_or_path="models/TravelMind-Planner-7B-SFT${SUFFIX}" \
  output_dir="saves/TravelMind-Planner-DPO${SUFFIX}" "${EXTRA[@]}" 2>&1 | tee "logs/dpo${SUFFIX}.log"
if [[ "$SMOKE" == 1 ]]; then
  echo 'SFT and QLoRA DPO five-step smoke completed; smoke adapters are not production models.'
  exit 0
fi
"$PYTHON" scripts/post_train_train_entry.py merge \
  --base models/TravelMind-Planner-7B-SFT \
  --adapter saves/TravelMind-Planner-DPO --output models/TravelMind-Planner-7B-DPO
echo 'Training/export complete. Start the model service separately after evaluation.'
