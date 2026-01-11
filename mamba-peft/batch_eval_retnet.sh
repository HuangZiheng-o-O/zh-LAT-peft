#!/bin/bash
# 批量评估retnet的所有训练实验
# 使用修复后的dataset代码（字母labels）

set -euo pipefail

# 配置
MODEL_TYPE="retnet"
BASE_OUTPUT_DIR="/home/user/mzs_h/code/zh-LAT-peft/output/benchmark/retnet/commonsense_170k_seed87"
EVAL_OUTPUT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/output/lm_eval"
EVAL_TASKS="${EVAL_TASKS:-boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"

# 所有实验配置（按照你的tree输出）
EXPERIMENTS=(
    "E1_QKVO_plus_G_plus_MLP_r8_alpha16"
    "E1_QKVO_plus_G_r8_alpha16"
    "E1_QKVO_plus_MLP_r8_alpha16"
    "E1_QKVO_r8_alpha16"
    "E2_OMLP_r8_alpha16"
    "E4_KONLY_r8_alpha16"
    "E4_MLPONLY_r8_alpha16"
    "E4_QONLY_r8_alpha16"
    "E4_VONLY_r8_alpha16"
    "E6_QVONLY_r8_alpha16"
    "E6_VOONLY_r8_alpha16"
    "E7_KVONLY_r8_alpha16"
    "E11_OONLY_r8_alpha16"
)

echo "========================================================================"
echo "批量评估 RetNet 实验"
echo "========================================================================"
echo "模型类型: ${MODEL_TYPE}"
echo "实验数量: ${#EXPERIMENTS[@]}"
echo "评估任务: ${EVAL_TASKS}"
echo "Batch Size: ${EVAL_BATCH_SIZE}"
echo "输出目录: ${EVAL_OUTPUT_ROOT}"
echo "========================================================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_EXPS=()

for exp_name in "${EXPERIMENTS[@]}"; do
    exp_dir="${BASE_OUTPUT_DIR}/${exp_name}"

    echo ""
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评估: ${exp_name}"
    echo "========================================================================"

    # 检查实验目录是否存在
    if [[ ! -d "${exp_dir}" ]]; then
        echo "⚠️  警告: 实验目录不存在，跳过: ${exp_dir}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_EXPS+=("${exp_name} (目录不存在)")
        continue
    fi

    # 检查adapter文件是否存在
    if [[ ! -f "${exp_dir}/adapter_config.json" ]] || [[ ! -f "${exp_dir}/adapter_model.safetensors" ]]; then
        echo "⚠️  警告: adapter文件不存在，跳过: ${exp_name}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_EXPS+=("${exp_name} (adapter文件缺失)")
        continue
    fi

    # 构造输出目录名称
    eval_output_dir="${EVAL_OUTPUT_ROOT}/retnet_commonsense_170k_seed87_${exp_name}"

    # 运行eval
    echo "PEFT weights: ${exp_dir}"
    echo "输出目录: ${eval_output_dir}"
    echo ""

    if python eval_lat.py \
        --model-type "${MODEL_TYPE}" \
        --peft-weights "${exp_dir}" \
        --tasks "${EVAL_TASKS}" \
        --eval-batch-size "${EVAL_BATCH_SIZE}" \
        --output-root "${EVAL_OUTPUT_ROOT}"; then

        echo "✅ 成功: ${exp_name}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ 失败: ${exp_name}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_EXPS+=("${exp_name}")
    fi
done

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========================================================================"
echo "批量评估完成"
echo "========================================================================"
echo "总实验数: ${#EXPERIMENTS[@]}"
echo "成功: ${SUCCESS_COUNT}"
echo "失败: ${FAILED_COUNT}"
echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"

if [[ ${FAILED_COUNT} -gt 0 ]]; then
    echo ""
    echo "失败的实验:"
    for failed_exp in "${FAILED_EXPS[@]}"; do
        echo "  - ${failed_exp}"
    done
fi

echo ""
echo "评估结果保存在: ${EVAL_OUTPUT_ROOT}"
echo "========================================================================"

# 返回失败计数作为退出码（0表示全部成功）
exit ${FAILED_COUNT}
