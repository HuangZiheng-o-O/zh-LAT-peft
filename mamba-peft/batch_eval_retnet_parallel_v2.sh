#!/bin/bash
# 批量评估retnet的所有训练实验 - 并行版本v2（增强日志）
# 使用修复后的dataset代码（字母labels）

set -euo pipefail

# 配置
MODEL_TYPE="retnet"
BASE_MODEL="${BASE_MODEL:-/home/user/mzs_h/model/retnet-1.3B-100B}"  # base模型路径
BASE_OUTPUT_DIR="/home/user/mzs_h/code/zh-LAT-peft/output/benchmark/retnet/commonsense_170k_seed87"
EVAL_OUTPUT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/output/lm_eval"
EVAL_TASKS="${EVAL_TASKS:-boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"

# 并行配置
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

# 所有实验配置
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

# 将GPU_IDS转换为数组
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"

echo "========================================================================"
echo "批量评估 RetNet 实验（并行版本v2 - 增强日志）"
echo "========================================================================"
echo "模型类型: ${MODEL_TYPE}"
echo "Base模型: ${BASE_MODEL}"
echo "实验数量: ${#EXPERIMENTS[@]}"
echo "评估任务: ${EVAL_TASKS}"
echo "Batch Size: ${EVAL_BATCH_SIZE}"
echo "并行GPU数: ${NUM_GPUS} (${GPU_IDS})"
echo "输出目录: ${EVAL_OUTPUT_ROOT}"
echo "========================================================================"
echo ""

# 创建临时目录存放日志和状态
TEMP_DIR="/tmp/batch_eval_retnet_$$"
mkdir -p "${TEMP_DIR}"
echo "📁 临时目录: ${TEMP_DIR}"
echo ""

# 单个实验的评估函数
run_eval() {
    local exp_name=$1
    local gpu_id=$2
    local log_file="${TEMP_DIR}/${exp_name}.log"
    local status_file="${TEMP_DIR}/${exp_name}.status"
    local pid_file="${TEMP_DIR}/${exp_name}.pid"

    # 记录PID
    echo $$ > "${pid_file}"

    local exp_dir="${BASE_OUTPUT_DIR}/${exp_name}"

    {
        echo "=========================================="
        echo "实验: ${exp_name}"
        echo "GPU: ${gpu_id}"
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="

        # 检查实验目录
        echo "[检查] 实验目录: ${exp_dir}"
        if [[ ! -d "${exp_dir}" ]]; then
            echo "❌ ERROR: 实验目录不存在"
            echo "FAILED: 目录不存在" > "${status_file}"
            return 1
        fi
        echo "✅ 实验目录存在"

        # 检查adapter文件
        echo "[检查] adapter文件..."
        if [[ ! -f "${exp_dir}/adapter_config.json" ]]; then
            echo "❌ ERROR: adapter_config.json 不存在"
            echo "FAILED: adapter文件缺失" > "${status_file}"
            return 1
        fi
        if [[ ! -f "${exp_dir}/adapter_model.safetensors" ]]; then
            echo "❌ ERROR: adapter_model.safetensors 不存在"
            echo "FAILED: adapter文件缺失" > "${status_file}"
            return 1
        fi
        echo "✅ adapter文件完整"

        # 设置CUDA设备
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        echo "[设置] CUDA_VISIBLE_DEVICES=${gpu_id}"
        echo ""

        # 运行eval
        echo "=========================================="
        echo "开始评估..."
        echo "=========================================="

        if python eval_lat.py \
            --model-type "${MODEL_TYPE}" \
            --model "${BASE_MODEL}" \
            --peft-weights "${exp_dir}" \
            --tasks "${EVAL_TASKS}" \
            --eval-batch-size "${EVAL_BATCH_SIZE}" \
            --output-root "${EVAL_OUTPUT_ROOT}"; then

            echo ""
            echo "=========================================="
            echo "✅ 评估成功"
            echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="
            echo "SUCCESS" > "${status_file}"
            return 0
        else
            echo ""
            echo "=========================================="
            echo "❌ 评估失败"
            echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="
            echo "FAILED: 评估失败" > "${status_file}"
            return 1
        fi
    } &> "${log_file}"

    # 同时输出到主控台（带前缀）
    local exit_code=$?
    if [[ ${exit_code} -eq 0 ]]; then
        echo "[GPU ${gpu_id}] ✅ ${exp_name} - 成功"
    else
        echo "[GPU ${gpu_id}] ❌ ${exp_name} - 失败 (查看日志: ${log_file})"
    fi
    return ${exit_code}
}

# 记录开始时间
START_TIME=$(date +%s)

# 并行调度
echo "🚀 开始并行评估..."
echo ""

declare -A running_jobs  # 记录正在运行的任务
job_count=0

for exp_name in "${EXPERIMENTS[@]}"; do
    # 选择GPU
    gpu_idx=$((job_count % NUM_GPUS))
    gpu_id="${GPU_ARRAY[$gpu_idx]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 [$(date '+%H:%M:%S')] 调度任务 $((job_count + 1))/${#EXPERIMENTS[@]}"
    echo "   实验: ${exp_name}"
    echo "   GPU: ${gpu_id}"

    # 检查目录是否存在（快速预检）
    exp_dir="${BASE_OUTPUT_DIR}/${exp_name}"
    if [[ ! -d "${exp_dir}" ]]; then
        echo "   ⚠️  警告: 目录不存在，跳过"
        echo "FAILED: 目录不存在" > "${TEMP_DIR}/${exp_name}.status"
        job_count=$((job_count + 1))
        continue
    fi

    echo "   日志: ${TEMP_DIR}/${exp_name}.log"
    echo "   启动中..."

    # 后台启动
    run_eval "${exp_name}" "${gpu_id}" &
    local pid=$!
    running_jobs[$pid]="${exp_name} (GPU ${gpu_id})"

    echo "   ✅ 已启动 (PID: ${pid})"
    echo ""

    job_count=$((job_count + 1))

    # 如果已启动NUM_GPUS个任务，等待一个完成
    if (( job_count % NUM_GPUS == 0 )) && (( job_count < ${#EXPERIMENTS[@]} )); then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "⏸️  [$(date '+%H:%M:%S')] 第一批GPU已满，等待完成..."
        echo "   正在运行的任务: ${#running_jobs[@]}"
        for pid in "${!running_jobs[@]}"; do
            echo "   - PID ${pid}: ${running_jobs[$pid]}"
        done
        echo ""

        # 等待任意一个任务完成
        wait -n
        completed_pid=$!

        echo "   ✅ 任务完成 (PID: ${completed_pid})"
        unset running_jobs[$completed_pid]
        echo "   继续调度剩余任务..."
        echo ""
    fi
done

# 等待所有任务完成
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ [$(date '+%H:%M:%S')] 等待所有任务完成..."
echo "   剩余任务: ${#running_jobs[@]}"
for pid in "${!running_jobs[@]}"; do
    echo "   - PID ${pid}: ${running_jobs[$pid]}"
done
echo ""

wait

echo "✅ 所有任务已完成"
echo ""

# 统计结果
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 收集评估结果..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0
declare -a FAILED_EXPS

for exp_name in "${EXPERIMENTS[@]}"; do
    status_file="${TEMP_DIR}/${exp_name}.status"
    log_file="${TEMP_DIR}/${exp_name}.log"

    if [[ -f "${status_file}" ]]; then
        status=$(cat "${status_file}")
        if [[ "${status}" == "SUCCESS" ]]; then
            echo "✅ ${exp_name}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ ${exp_name}: ${status}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_EXPS+=("${exp_name}")
        fi
    else
        echo "⚠️  ${exp_name}: 状态未知 (可能启动失败)"
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
echo "📈 批量评估完成"
echo "========================================================================"
echo "总实验数: ${#EXPERIMENTS[@]}"
echo "✅ 成功: ${SUCCESS_COUNT}"
echo "❌ 失败: ${FAILED_COUNT}"
echo "⏱️  总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"

if [[ ${FAILED_COUNT} -gt 0 ]]; then
    echo ""
    echo "失败的实验:"
    for failed_exp in "${FAILED_EXPS[@]}"; do
        echo ""
        echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  ❌ ${failed_exp}"
        log_file="${TEMP_DIR}/${failed_exp}.log"
        if [[ -f "${log_file}" ]]; then
            echo "  最后10行日志:"
            tail -n 10 "${log_file}" | sed 's/^/     /'
        else
            echo "  （日志文件不存在）"
        fi
    done
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

echo ""
echo "📂 评估结果保存在: ${EVAL_OUTPUT_ROOT}"
echo "📝 详细日志目录: ${TEMP_DIR}"
echo ""
echo "查看单个日志: cat ${TEMP_DIR}/<实验名>.log"
echo "清理临时文件: rm -rf ${TEMP_DIR}"
echo "========================================================================"

exit ${FAILED_COUNT}
