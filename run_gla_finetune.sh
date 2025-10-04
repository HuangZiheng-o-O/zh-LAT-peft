#!/bin/bash
set -e
source /data1/xy/anaconda3/etc/profile.d/conda.sh
conda activate mzsz
# =====================================================================
# GLA LoRA Fine-tuning (严格复现原命令；支持可选输出目录覆盖)
# =====================================================================

# 【可选】如果你想临时覆盖输出目录，就把下面的变量改成非空：
# 例：OUTPUT_DIR="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/benchmark/glue/cola_gla/001_lora_r8_gla_qkvo_debug"
# OUTPUT_DIR=""
# ---------------- 原始固定路径（保持不变） ----------------
DEFAULT_OUTPUT_DIR="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/benchmark/glue/cola_gla/001_lora_r8_gla_qkvo"
PEFT_DIR="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft"
CFG_FILE="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/exps/benchmark/glue/cola_gla/001_lora_r8_gla_qkvo.yaml"

# ---------------- 环境准备（与原命令保持一致） ----------------
# conda activate mzsz

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

# 清理（保持原逻辑）
rm -rf ~/.config/wandb
rm -rf ~/.triton ~/.cache/torch_extensions

# 与原命令一致：先删默认输出目录
# rm -rf "$DEFAULT_OUTPUT_DIR"

# 如果你手动指定了 OUTPUT_DIR，且与默认不同，也一并清掉（可选增强，不影响默认行为）
if [[ -n "$OUTPUT_DIR" && "$OUTPUT_DIR" != "$DEFAULT_OUTPUT_DIR" ]]; then
  rm -rf "$OUTPUT_DIR"
fi

# 进入训练目录（保持一致）
cd "$PEFT_DIR"

CUDA_VISIBLE_DEVICES=0 python "$PEFT_DIR/train.py" \
--cfg "$CFG_FILE"
