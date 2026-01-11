#!/bin/bash
# 测试单个实验的评估 - 用于调试

set -euo pipefail

EXP_NAME="${1:-E1_QKVO_r8_alpha16}"
GPU_ID="${2:-0}"

BASE_MODEL="${BASE_MODEL:-/home/user/mzs_h/model/retnet-1.3B-100B}"
BASE_OUTPUT_DIR="/home/user/mzs_h/code/zh-LAT-peft/output/benchmark/retnet/commonsense_170k_seed87"
EVAL_OUTPUT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/output/lm_eval"
EVAL_TASKS="${EVAL_TASKS:-boolq,piqa}"  # 只测试2个任务，更快

echo "========================================================================"
echo "测试单个实验评估"
echo "========================================================================"
echo "实验名: ${EXP_NAME}"
echo "GPU: ${GPU_ID}"
echo "Base模型: ${BASE_MODEL}"
echo "任务: ${EVAL_TASKS}"
echo "========================================================================"
echo ""

exp_dir="${BASE_OUTPUT_DIR}/${EXP_NAME}"

# 检查目录
echo "检查实验目录: ${exp_dir}"
if [[ ! -d "${exp_dir}" ]]; then
    echo "❌ 目录不存在"
    exit 1
fi
echo "✅ 目录存在"
echo ""

# 检查adapter文件
echo "检查adapter文件..."
ls -lh "${exp_dir}/adapter_config.json" "${exp_dir}/adapter_model.safetensors"
echo ""

# 设置GPU
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo ""

# 运行eval
echo "开始评估..."
echo "========================================================================"

python eval_lat.py \
    --model-type retnet \
    --model "${BASE_MODEL}" \
    --peft-weights "${exp_dir}" \
    --tasks "${EVAL_TASKS}" \
    --eval-batch-size 64 \
    --output-root "${EVAL_OUTPUT_ROOT}"

echo ""
echo "========================================================================"
echo "✅ 评估完成"
echo "========================================================================"
