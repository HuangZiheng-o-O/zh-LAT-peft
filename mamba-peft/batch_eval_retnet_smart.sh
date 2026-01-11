#!/bin/bash
# 智能批量评估retnet - 自动找best和last checkpoint并分别评估

set -euo pipefail

# 配置
MODEL_TYPE="retnet"
BASE_MODEL="${BASE_MODEL:-/home/user/mzs_h/model/retnet-1.3B-100B}"
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

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"

echo "========================================================================"
echo "智能批量评估 RetNet 实验 (Best + Last Checkpoints)"
echo "========================================================================"
echo "模型类型: ${MODEL_TYPE}"
echo "Base模型: ${BASE_MODEL}"
echo "实验数量: ${#EXPERIMENTS[@]}"
echo "评估任务: ${EVAL_TASKS}"
echo "Batch Size: ${EVAL_BATCH_SIZE}"
echo "并行GPU数: ${NUM_GPUS} (${GPU_IDS})"
echo "输出目录: ${EVAL_OUTPUT_ROOT}/retnet/{best,last}"
echo "========================================================================"
echo ""

TEMP_DIR="/tmp/batch_eval_retnet_smart_$$"
mkdir -p "${TEMP_DIR}"
echo "📁 临时目录: ${TEMP_DIR}"
echo ""

# 查找最佳checkpoint的函数
find_best_checkpoint() {
    local exp_dir=$1
    local best_ckpt=""
    local best_step=-1
    local best_metric=999999  # 用于eval_loss（越小越好）

    # 遍历所有checkpoint目录
    for ckpt_dir in "${exp_dir}"/checkpoint-*; do
        if [[ ! -d "${ckpt_dir}" ]]; then
            continue
        fi

        local trainer_state="${ckpt_dir}/trainer_state.json"
        if [[ ! -f "${trainer_state}" ]]; then
            continue
        fi

        # 提取step number
        local step=$(basename "${ckpt_dir}" | sed 's/checkpoint-//')

        # 提取best_metric (优先用eval_token_accuracy，fallback到eval_loss)
        local metric=$(python3 -c "
import json, sys
try:
    with open('${trainer_state}') as f:
        data = json.load(f)
    # 优先查找eval_token_accuracy（越大越好），否则用eval_loss（越小越好）
    if 'best_metric' in data:
        print(data['best_metric'])
    else:
        # 如果没有best_metric，从log_history找最后一次eval
        for entry in reversed(data.get('log_history', [])):
            if 'eval_token_accuracy' in entry:
                # 用负值，这样仍然可以用min比较
                print(-entry['eval_token_accuracy'])
                break
            elif 'eval_loss' in entry:
                print(entry['eval_loss'])
                break
except:
    print('999999')
" 2>/dev/null)

        # 比较（假设metric越小越好，如果是accuracy会取负值）
        if (( $(echo "${metric} < ${best_metric}" | bc -l 2>/dev/null || echo 0) )); then
            best_metric=${metric}
            best_step=${step}
            best_ckpt="${ckpt_dir}"
        fi
    done

    echo "${best_ckpt}"
}

# 查找最后checkpoint的函数
find_last_checkpoint() {
    local exp_dir=$1
    local last_ckpt=""
    local max_step=-1

    for ckpt_dir in "${exp_dir}"/checkpoint-*; do
        if [[ ! -d "${ckpt_dir}" ]]; then
            continue
        fi

        local step=$(basename "${ckpt_dir}" | sed 's/checkpoint-//')
        if (( step > max_step )); then
            max_step=${step}
            last_ckpt="${ckpt_dir}"
        fi
    done

    echo "${last_ckpt}"
}

# 单个评估任务的函数
run_eval() {
    local exp_name=$1
    local ckpt_path=$2
    local ckpt_type=$3  # "best" or "last"
    local gpu_id=$4
    local log_file="${TEMP_DIR}/${exp_name}_${ckpt_type}.log"
    local status_file="${TEMP_DIR}/${exp_name}_${ckpt_type}.status"

    {
        echo "=========================================="
        echo "实验: ${exp_name}"
        echo "Checkpoint: ${ckpt_type} ($(basename ${ckpt_path}))"
        echo "GPU: ${gpu_id}"
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="

        if [[ ! -d "${ckpt_path}" ]]; then
            echo "❌ ERROR: Checkpoint目录不存在: ${ckpt_path}"
            echo "FAILED: checkpoint不存在" > "${status_file}"
            return 1
        fi

        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export TOKENIZERS_PARALLELISM=false

        local eval_output="${EVAL_OUTPUT_ROOT}/retnet/${ckpt_type}/${exp_name}"

        if python eval_lat.py \
            --model-type "${MODEL_TYPE}" \
            --model "${BASE_MODEL}" \
            --peft-weights "${ckpt_path}" \
            --tasks "${EVAL_TASKS}" \
            --eval-batch-size "${EVAL_BATCH_SIZE}" \
            --output-root "${EVAL_OUTPUT_ROOT}/retnet/${ckpt_type}"; then

            echo ""
            echo "=========================================="
            echo "✅ 评估成功"
            echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "结果: ${eval_output}"
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

    local exit_code=$?
    if [[ ${exit_code} -eq 0 ]]; then
        echo "[GPU ${gpu_id}] ✅ ${exp_name} (${ckpt_type}) - 成功"
    else
        echo "[GPU ${gpu_id}] ❌ ${exp_name} (${ckpt_type}) - 失败"
    fi
    return ${exit_code}
}

# 主流程
START_TIME=$(date +%s)

echo "🔍 第一步: 分析所有实验的checkpoints..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

declare -A BEST_CKPTS
declare -A LAST_CKPTS
declare -A NEED_BOTH

for exp_name in "${EXPERIMENTS[@]}"; do
    exp_dir="${BASE_OUTPUT_DIR}/${exp_name}"

    echo ""
    echo "📊 分析: ${exp_name}"

    if [[ ! -d "${exp_dir}" ]]; then
        echo "   ⚠️  实验目录不存在，跳过"
        continue
    fi

    best_ckpt=$(find_best_checkpoint "${exp_dir}")
    last_ckpt=$(find_last_checkpoint "${exp_dir}")

    if [[ -z "${best_ckpt}" ]] || [[ -z "${last_ckpt}" ]]; then
        echo "   ⚠️  未找到有效checkpoint，跳过"
        continue
    fi

    BEST_CKPTS[$exp_name]="${best_ckpt}"
    LAST_CKPTS[$exp_name]="${last_ckpt}"

    echo "   Best: $(basename ${best_ckpt})"
    echo "   Last: $(basename ${last_ckpt})"

    if [[ "${best_ckpt}" == "${last_ckpt}" ]]; then
        echo "   ℹ️  Best == Last，只需评估一次"
        NEED_BOTH[$exp_name]="no"
    else
        echo "   ℹ️  Best != Last，需分别评估"
        NEED_BOTH[$exp_name]="yes"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 第二步: 开始并行评估..."
echo ""

declare -A running_jobs
job_count=0
total_jobs=0

# 计算总任务数
for exp_name in "${EXPERIMENTS[@]}"; do
    if [[ -n "${BEST_CKPTS[$exp_name]:-}" ]]; then
        if [[ "${NEED_BOTH[$exp_name]:-no}" == "yes" ]]; then
            total_jobs=$((total_jobs + 2))  # best + last
        else
            total_jobs=$((total_jobs + 1))  # 只需一次，但写入两个目录
        fi
    fi
done

echo "📋 总任务数: ${total_jobs}"
echo ""

# 调度评估任务
for exp_name in "${EXPERIMENTS[@]}"; do
    if [[ -z "${BEST_CKPTS[$exp_name]:-}" ]]; then
        continue
    fi

    best_ckpt="${BEST_CKPTS[$exp_name]}"
    last_ckpt="${LAST_CKPTS[$exp_name]}"

    # 评估best checkpoint
    gpu_idx=$((job_count % NUM_GPUS))
    gpu_id="${GPU_ARRAY[$gpu_idx]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 [$(date '+%H:%M:%S')] 调度任务 $((job_count + 1))/${total_jobs}"
    echo "   实验: ${exp_name}"
    echo "   类型: BEST ($(basename ${best_ckpt}))"
    echo "   GPU: ${gpu_id}"

    run_eval "${exp_name}" "${best_ckpt}" "best" "${gpu_id}" &
    local pid=$!
    running_jobs[$pid]="${exp_name} (best)"

    echo "   ✅ 已启动 (PID: ${pid})"
    job_count=$((job_count + 1))

    # 如果best != last，还需要评估last
    if [[ "${NEED_BOTH[$exp_name]:-no}" == "yes" ]]; then
        # 等待有GPU空闲
        if (( ${#running_jobs[@]} >= NUM_GPUS )); then
            wait -n
            for p in "${!running_jobs[@]}"; do
                if ! kill -0 $p 2>/dev/null; then
                    unset running_jobs[$p]
                fi
            done
        fi

        gpu_idx=$((job_count % NUM_GPUS))
        gpu_id="${GPU_ARRAY[$gpu_idx]}"

        echo ""
        echo "📋 [$(date '+%H:%M:%S')] 调度任务 $((job_count + 1))/${total_jobs}"
        echo "   实验: ${exp_name}"
        echo "   类型: LAST ($(basename ${last_ckpt}))"
        echo "   GPU: ${gpu_id}"

        run_eval "${exp_name}" "${last_ckpt}" "last" "${gpu_id}" &
        local pid=$!
        running_jobs[$pid]="${exp_name} (last)"

        echo "   ✅ 已启动 (PID: ${pid})"
        job_count=$((job_count + 1))
    else
        # best == last，创建符号链接或复制结果
        echo "   ℹ️  创建last目录的软链接（指向best）"
        # 这个在评估完成后处理
    fi

    echo ""

    # GPU已满，等待一个完成
    if (( ${#running_jobs[@]} >= NUM_GPUS )); then
        echo "⏸️  GPU已满，等待任务完成..."
        wait -n
        for p in "${!running_jobs[@]}"; do
            if ! kill -0 $p 2>/dev/null; then
                unset running_jobs[$p]
            fi
        done
        echo "   ✅ 继续调度..."
        echo ""
    fi
done

echo "⏳ 等待所有任务完成..."
wait

echo "✅ 所有评估任务已完成"
echo ""

# 处理best == last的情况（创建软链接）
echo "🔗 处理best == last的实验..."
for exp_name in "${EXPERIMENTS[@]}"; do
    if [[ "${NEED_BOTH[$exp_name]:-no}" == "no" ]] && [[ -n "${BEST_CKPTS[$exp_name]:-}" ]]; then
        best_dir="${EVAL_OUTPUT_ROOT}/retnet/best/${exp_name}"
        last_dir="${EVAL_OUTPUT_ROOT}/retnet/last/${exp_name}"

        if [[ -d "${best_dir}" ]]; then
            mkdir -p "$(dirname ${last_dir})"
            if [[ ! -e "${last_dir}" ]]; then
                ln -s "${best_dir}" "${last_dir}"
                echo "   ✅ ${exp_name}: last -> best (软链接)"
            fi
        fi
    fi
done

echo ""

# 统计结果
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 收集评估结果..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0
declare -a FAILED_TASKS

for exp_name in "${EXPERIMENTS[@]}"; do
    if [[ -z "${BEST_CKPTS[$exp_name]:-}" ]]; then
        continue
    fi

    # 检查best
    status_file="${TEMP_DIR}/${exp_name}_best.status"
    if [[ -f "${status_file}" ]]; then
        status=$(cat "${status_file}")
        if [[ "${status}" == "SUCCESS" ]]; then
            echo "✅ ${exp_name} (best)"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ ${exp_name} (best): ${status}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_TASKS+=("${exp_name} (best)")
        fi
    fi

    # 检查last（如果需要的话）
    if [[ "${NEED_BOTH[$exp_name]:-no}" == "yes" ]]; then
        status_file="${TEMP_DIR}/${exp_name}_last.status"
        if [[ -f "${status_file}" ]]; then
            status=$(cat "${status_file}")
            if [[ "${status}" == "SUCCESS" ]]; then
                echo "✅ ${exp_name} (last)"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
                echo "❌ ${exp_name} (last): ${status}"
                FAILED_COUNT=$((FAILED_COUNT + 1))
                FAILED_TASKS+=("${exp_name} (last)")
            fi
        fi
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========================================================================"
echo "📈 批量评估完成"
echo "========================================================================"
echo "总任务数: ${total_jobs}"
echo "✅ 成功: ${SUCCESS_COUNT}"
echo "❌ 失败: ${FAILED_COUNT}"
echo "⏱️  总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"

if [[ ${FAILED_COUNT} -gt 0 ]]; then
    echo ""
    echo "失败的任务:"
    for failed in "${FAILED_TASKS[@]}"; do
        echo "  ❌ ${failed}"
    done
fi

echo ""
echo "📂 评估结果:"
echo "   Best checkpoints: ${EVAL_OUTPUT_ROOT}/retnet/best/"
echo "   Last checkpoints: ${EVAL_OUTPUT_ROOT}/retnet/last/"
echo "📝 详细日志: ${TEMP_DIR}"
echo ""
echo "清理临时文件: rm -rf ${TEMP_DIR}"
echo "========================================================================"

exit ${FAILED_COUNT}
