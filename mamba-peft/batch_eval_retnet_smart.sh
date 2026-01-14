#!/bin/bash
# 批量评估retnet - 仅评估最后保存的checkpoint

set -euo pipefail

# 配置
MODEL_TYPE="retnet"
BASE_MODEL="${BASE_MODEL:-/home/user/mzs_h/model/retnet-1.3B-100B}"
BASE_OUTPUT_DIR="/home/user/mzs_h/code/zh-LAT-peft/output/benchmark/retnet/commonsense_170k_seed87"
EVAL_OUTPUT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/output/lm_eval"
EVAL_TASKS="${EVAL_TASKS:-boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

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
echo "批量评估 RetNet 实验（仅最新checkpoint）"
echo "========================================================================"
echo "模型类型: ${MODEL_TYPE}"
echo "Base模型: ${BASE_MODEL}"
echo "实验数量: ${#EXPERIMENTS[@]}"
echo "评估任务: ${EVAL_TASKS}"
echo "Batch Size: ${EVAL_BATCH_SIZE}"
echo "并行GPU数: ${NUM_GPUS} (${GPU_IDS})"
echo "输出目录: ${EVAL_OUTPUT_ROOT}/retnet/last"
echo "========================================================================"
echo ""

TEMP_DIR="/tmp/batch_eval_retnet_smart_$$"
mkdir -p "${TEMP_DIR}"
echo "📁 临时目录: ${TEMP_DIR}"
echo ""

mkdir -p "${EVAL_OUTPUT_ROOT}/retnet/last"

# 查找最佳checkpoint的函数（目前已停用，保留逻辑备查）
: <<'BEST_LOGIC'
find_best_checkpoint() {
    local exp_dir=$1
    local -a checkpoints=()

    shopt -s nullglob
    for ckpt_dir in "${exp_dir}"/checkpoint-*; do
        [[ -d "${ckpt_dir}" ]] || continue
        checkpoints+=("${ckpt_dir}")
    done
    shopt -u nullglob

    if (( ${#checkpoints[@]} == 0 )); then
        echo ""
        return
    fi

    local sorted_ckpts
    IFS=$'\n' read -r -d '' -a sorted_ckpts < <(printf "%s\n" "${checkpoints[@]}" | sort -t- -k2,2n && printf '\0')

    local trainer_state="${exp_dir}/trainer_state.json"
    local candidate=""

    if [[ -f "${trainer_state}" ]]; then
        candidate=$(python3 - "$trainer_state" "$exp_dir" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
exp_dir = Path(sys.argv[2])
try:
    data = json.loads(state_path.read_text())
except Exception:
    sys.exit(0)

best_model = data.get("best_model_checkpoint")
if not best_model:
    sys.exit(0)

path = Path(best_model)
if not path.is_absolute():
    path = exp_dir / path
print(path)
PY
)
        if [[ -n "${candidate}" && -d "${candidate}" ]]; then
            echo "${candidate}"
            return
        fi
    fi

    local count=${#sorted_ckpts[@]}
    if (( count >= 2 )); then
        echo "${sorted_ckpts[$((count - 2))]}"
    else
        echo "${sorted_ckpts[0]}"
    fi
}
BEST_LOGIC

find_best_checkpoint() {
    echo ""
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

wait_with_progress() {
    local total=$1
    local interval=$2
    local completed

    if (( total <= 0 )); then
        return
    fi

    echo "⏳ 等待所有任务完成... (每${interval}s刷新进度)"
    while true; do
        completed=$(find "${TEMP_DIR}" -maxdepth 1 -name '*.status' -print | wc -l | tr -d ' ')
        echo "[进度] $(date '+%H:%M:%S') 已完成 ${completed}/${total}"
        if (( completed >= total )); then
            break
        fi
        sleep "${interval}"
    done
}

cleanup_finished_jobs() {
    for pid in "${!running_jobs[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local gpu_id="${JOB_GPU_MAP[$pid]:-}"
            if [[ -n "${gpu_id}" ]]; then
                local cnt=${GPU_JOB_COUNTS[$gpu_id]:-1}
                if (( cnt < 0 )); then
                    cnt=0
                fi
                GPU_JOB_COUNTS[$gpu_id]=$cnt
            fi
            unset running_jobs[$pid]
            unset JOB_GPU_MAP[$pid]
        fi
    done
}

acquire_gpu_slot() {
    local selected=""
    while true; do
        cleanup_finished_jobs
        for gpu_id in "${GPU_ARRAY[@]}"; do
            local cnt=${GPU_JOB_COUNTS[$gpu_id]:-0}
            if (( cnt < MAX_JOBS_PER_GPU )); then
                selected="${gpu_id}"
                break
            fi
        done

        if [[ -n "${selected}" ]]; then
            SELECTED_GPU="${selected}"
            return 0
        fi

        if (( ${#running_jobs[@]} == 0 )); then
            sleep 1
        else
            wait -n || true
        fi
    done
}

# 单个评估任务的函数
run_eval() {
    local exp_name=$1
    local ckpt_path=$2
    local ckpt_type=$3  # 当前仅传 "last"
    local gpu_id=$4
    local log_file="${TEMP_DIR}/${exp_name}_${ckpt_type}.log"
    local status_file="${TEMP_DIR}/${exp_name}_${ckpt_type}.status"
    local eval_output="${EVAL_OUTPUT_ROOT}/retnet/${ckpt_type}/${exp_name}"

    local attempt=1
    local eval_bs=${EVAL_BATCH_SIZE}
    local min_bs=${MIN_EVAL_BATCH_SIZE}
    local reduce_factor=${OOM_BATCH_REDUCTION_FACTOR}
    local max_retries=${MAX_OOM_RETRIES}

    : > "${log_file}"

    if [[ ! -d "${ckpt_path}" ]]; then
        {
            echo "=========================================="
            echo "实验: ${exp_name}"
            echo "Checkpoint: ${ckpt_type} ($(basename ${ckpt_path}))"
            echo "GPU: ${gpu_id}"
            echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="
            echo "❌ ERROR: Checkpoint目录不存在: ${ckpt_path}"
        } >> "${log_file}"
        echo "FAILED: checkpoint不存在" > "${status_file}"
        return 1
    fi

    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export TOKENIZERS_PARALLELISM=false

    while true; do
        {
            echo "=========================================="
            echo "实验: ${exp_name}"
            echo "Checkpoint: ${ckpt_type} ($(basename ${ckpt_path}))"
            echo "GPU: ${gpu_id}"
            echo "尝试: ${attempt}"
            echo "Batch Size: ${eval_bs}"
            echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="
        } >> "${log_file}"

        echo "[GPU ${gpu_id}] ▶️ ${exp_name} (${ckpt_type}) attempt=${attempt} bs=${eval_bs}"

        if python eval_lat.py \
            --model-type "${MODEL_TYPE}" \
            --model "${BASE_MODEL}" \
            --peft-weights "${ckpt_path}" \
            --tasks "${EVAL_TASKS}" \
            --eval-batch-size "${eval_bs}" \
            --output-root "${eval_output}" >> "${log_file}" 2>&1; then

            {
                echo ""
                echo "=========================================="
                echo "✅ 评估成功"
                echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
                echo "结果: ${eval_output}"
                echo "=========================================="
            } >> "${log_file}"
            echo "SUCCESS" > "${status_file}"
            echo "[GPU ${gpu_id}] ✅ ${exp_name} (${ckpt_type}) - 成功"
            return 0
        fi

        if tail -n 200 "${log_file}" | grep -qi "out of memory"; then
            if (( eval_bs <= min_bs )) || (( attempt >= max_retries )); then
                {
                    echo ""
                    echo "=========================================="
                    echo "❌ 评估失败 (OOM，已达到最小batch或最大重试次数)"
                    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
                echo "=========================================="
            } >> "${log_file}"
            echo "FAILED: OOM (batch_size=${eval_bs})" > "${status_file}"
            echo "[GPU ${gpu_id}] ❌ ${exp_name} (${ckpt_type}) - OOM (bs=${eval_bs})"
            return 1
        fi

        local new_bs=$(( eval_bs / reduce_factor ))
            if (( new_bs < min_bs )); then
                new_bs=${min_bs}
            fi
            if (( new_bs == eval_bs )); then
                new_bs=$(( eval_bs - 1 ))
                if (( new_bs < 1 )); then
                    new_bs=1
                fi
            fi

            {
                echo ""
                echo "⚠️  检测到OOM，batch size从${eval_bs}调整为${new_bs}，准备重试"
            } >> "${log_file}"
            echo "[GPU ${gpu_id}] ⚠️ ${exp_name} OOM, 调整batch ${eval_bs} -> ${new_bs}"
            eval_bs=${new_bs}
            attempt=$((attempt + 1))
            sleep 2
            continue
        fi

        {
            echo ""
            echo "=========================================="
            echo "❌ 评估失败"
            echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="
        } >> "${log_file}"
        echo "FAILED: 评估失败" > "${status_file}"
        echo "[GPU ${gpu_id}] ❌ ${exp_name} (${ckpt_type}) - 失败"
        return 1
    done
}

# 主流程
START_TIME=$(date +%s)

echo "🔍 第一步: 分析所有实验的checkpoints..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

declare -A LAST_CKPTS

for exp_name in "${EXPERIMENTS[@]}"; do
    exp_dir="${BASE_OUTPUT_DIR}/${exp_name}"

    echo ""
    echo "📊 分析: ${exp_name}"

    if [[ ! -d "${exp_dir}" ]]; then
        echo "   ⚠️  实验目录不存在，跳过"
        continue
    fi

    last_ckpt=$(find_last_checkpoint "${exp_dir}")

    if [[ -z "${last_ckpt}" ]]; then
        echo "   ⚠️  未找到最新checkpoint，跳过"
        continue
    fi

    LAST_CKPTS[$exp_name]="${last_ckpt}"

    echo "   Last: $(basename ${last_ckpt})"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 第二步: 开始并行评估..."
echo ""

declare -A running_jobs
declare -A GPU_JOB_COUNTS
declare -A JOB_GPU_MAP
job_count=0
total_jobs=0

# 计算总任务数（仅最新checkpoint）
for exp_name in "${EXPERIMENTS[@]}"; do
    if [[ -n "${LAST_CKPTS[$exp_name]:-}" ]]; then
        total_jobs=$((total_jobs + 1))
    fi
done

echo "📋 总任务数: ${total_jobs}"
echo ""

# 调度评估任务
for exp_name in "${EXPERIMENTS[@]}"; do
    if [[ -z "${LAST_CKPTS[$exp_name]:-}" ]]; then
        continue
    fi

    last_ckpt="${LAST_CKPTS[$exp_name]}"

    acquire_gpu_slot
    gpu_id="${SELECTED_GPU}"
    GPU_JOB_COUNTS[$gpu_id]=$(( ${GPU_JOB_COUNTS[$gpu_id]:-0} + 1 ))

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 [$(date '+%H:%M:%S')] 调度任务 $((job_count + 1))/${total_jobs}"
    echo "   实验: ${exp_name}"
    echo "   类型: LAST ($(basename ${last_ckpt}))"
    echo "   GPU: ${gpu_id}"

    run_eval "${exp_name}" "${last_ckpt}" "last" "${gpu_id}" &
    pid=$!
    running_jobs[$pid]="${exp_name} (last)"
    JOB_GPU_MAP[$pid]="${gpu_id}"

    echo "   ✅ 已启动 (PID: ${pid})"
    job_count=$((job_count + 1))
done

wait_with_progress "${total_jobs}" "${PROGRESS_INTERVAL}"
wait
cleanup_finished_jobs

echo "✅ 所有评估任务已完成"
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
    if [[ -z "${LAST_CKPTS[$exp_name]:-}" ]]; then
        continue
    fi

    status_file="${TEMP_DIR}/${exp_name}_last.status"
    if [[ -f "${status_file}" ]]; then
        status=$(cat "${status_file}")
        if [[ "${status}" == "SUCCESS" ]]; then
            echo "✅ ${exp_name}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ ${exp_name}: ${status}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_TASKS+=("${exp_name}")
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
echo "   Last checkpoints: ${EVAL_OUTPUT_ROOT}/retnet/last/"
echo "📝 详细日志: ${TEMP_DIR}"
echo ""
echo "清理临时文件: rm -rf ${TEMP_DIR}"
echo "========================================================================"

exit ${FAILED_COUNT}
