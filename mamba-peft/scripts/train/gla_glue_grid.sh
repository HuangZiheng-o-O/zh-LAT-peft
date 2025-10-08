#!/bin/bash
set -euo pipefail

# Relative to this script's directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PEFT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PEFT_DIR"

# Environment mirrors run_gla_finetune.sh but uses relative paths
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

# Experiments list (CFG paths relative to mamba-peft/)
EXPS=(
  "cfg/exps/benchmark/glue/cola_gla/001_lora_r8_gla_qkvo.yaml"
  "cfg/exps/benchmark/glue/cola_gla/002_lora_r8_gla_qkvo_mlp.yaml"
  "cfg/exps/benchmark/glue/cola_gla/003_lora_r8_gla_out.yaml"
  "cfg/exps/benchmark/glue/cola_gla/004_lora_r8_gla_full.yaml"
  "cfg/exps/benchmark/glue/rte_gla/001_lora_r8_gla_qkvo.yaml"
  "cfg/exps/benchmark/glue/rte_gla/002_lora_r8_gla_qkvo_mlp.yaml"
  "cfg/exps/benchmark/glue/rte_gla/003_lora_r8_gla_out.yaml"
  "cfg/exps/benchmark/glue/rte_gla/004_lora_r8_gla_full.yaml"
)

# GPU list (one experiment per GPU)
GPUS=(0 1 2 3 4 5 6)

idx=0
while [[ $idx -lt ${#EXPS[@]} ]]; do
  for gpu in "${GPUS[@]}"; do
    [[ $idx -ge ${#EXPS[@]} ]] && break
    CFG=${EXPS[$idx]}
    echo "[GPU $gpu] $CFG"
    CUDA_VISIBLE_DEVICES=$gpu python train.py --cfg "$CFG" --device $gpu --prec bf16 --lock &
    idx=$((idx+1))
  done
  wait
done

echo "All experiments finished."


