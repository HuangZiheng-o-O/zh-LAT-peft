#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PEFT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PEFT_DIR"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/user/mzs_h/data/hf_cache"
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export GLUE_METRIC_DIR="/home/user/mzs_h/data/hf_cache/eval_metrics/glue"
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
rm -rf ~/.config/wandb ~/.triton ~/.cache/torch_extensions || true

# CoLA: ablate targets
EXPS=(
  "cfg/exps/benchmark/glue/cola_gla/007_lora_r8_gla_q.yaml"
  "cfg/exps/benchmark/glue/cola_gla/008_lora_r8_gla_v.yaml"
  "cfg/exps/benchmark/glue/cola_gla/009_lora_r8_gla_qv.yaml"
  "cfg/exps/benchmark/glue/cola_gla/010_lora_r8_gla_g.yaml"
  "cfg/exps/benchmark/glue/cola_gla/011_lora_r8_gla_gk.yaml"
)

GPUS=(0 1 2 3 4)

idx=0
for CFG in "${EXPS[@]}"; do
  gpu=${GPUS[$idx]}
  echo "[GPU $gpu] $CFG"
  CUDA_VISIBLE_DEVICES=$gpu python train.py --cfg "$CFG" --device $gpu --prec bf16 --lock &
  idx=$((idx+1))
done
wait
echo "Target ablations finished."


