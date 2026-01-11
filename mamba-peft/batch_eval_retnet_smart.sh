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
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"

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

mkdir -p "${EVAL_OUTPUT_ROOT}/retnet/best" "${EVAL_OUTPUT_ROOT}/retnet/last"

# 查找最佳checkpoint的函数
find_best_checkpoint() {
    local exp_dir=$1
    local trainer_state="${exp_dir}/trainer_state.json"

    if [[ ! -f "${trainer_state}" ]]; then
        for ckpt_state in "${exp_dir}"/checkpoint-*/trainer_state.json; do
            if [[ -f "${ckpt_state}" ]]; then
                trainer_state="${ckpt_state}"
                break
            fi
        done
    fi

    if [[ ! -f "${trainer_state}" ]]; then
        echo ""
        return
    fi

    python3 - "$exp_dir" "$trainer_state" <<'PY'
import json
import sys
from pathlib import Path

exp_dir = Path(sys.argv[1])
state_path = Path(sys.argv[2])
try:
    data = json.loads(state_path.read_text())
except Exception:
    print("")
    sys.exit(0)

log_history = data.get("log_history") or []
best_step = None
best_metric = None

for entry in log_history:
    step = entry.get("step")
    if step is None:
        continue
    if "eval_token_accuracy" in entry:
        value = entry["eval_token_accuracy"]
        if best_step is None or value > best_metric:
            best_metric = value
            best_step = step

if best_step is None:
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "eval_loss" in entry:
            value = entry["eval_loss"]
            if best_step is None or value < best_metric:
                best_metric = value
                best_step = step

if best_step is not None:
    ckpt_dir = exp_dir / f"checkpoint-{int(best_step)}"
    if ckpt_dir.is_dir():
        print(str(ckpt_dir))
        sys.exit(0)

best_model = data.get("best_model_checkpoint")
if best_model:
    best_path = Path(best_model)
    if not best_path.is_absolute():
        best_path = exp_dir / best_path
    print(str(best_path))
else:
    print("")
PY
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

sync_dir_contents() {
    local src_dir=$1
    local dst_dir=$2

    if command -v rsync >/dev/null 2>&1; then
        mkdir -p "${dst_dir}"
        rsync -a --delete "${src_dir}/" "${dst_dir}/"
    else
        python3 - "$src_dir" "$dst_dir" <<'PY'
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
if not src.exists():
    sys.exit(0)
if dst.exists():
    shutil.rmtree(dst)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(src, dst)
PY
    fi
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

# 单个评估任务的函数
run_eval() {
    local exp_name=$1
    local ckpt_path=$2
    local ckpt_type=$3  # "best" or "last"
    local gpu_id=$4
    local log_file="${TEMP_DIR}/${exp_name}_${ckpt_type}.log"
    local status_file="${TEMP_DIR}/${exp_name}_${ckpt_type}.status"
    local eval_output="${EVAL_OUTPUT_ROOT}/retnet/${ckpt_type}/${exp_name}"

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

        if python eval_lat.py \
            --model-type "${MODEL_TYPE}" \
            --model "${BASE_MODEL}" \
            --peft-weights "${ckpt_path}" \
            --tasks "${EVAL_TASKS}" \
            --eval-batch-size "${EVAL_BATCH_SIZE}" \
            --output-root "${eval_output}"; then

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

    if [[ -z "${last_ckpt}" ]]; then
        echo "   ⚠️  未找到 last checkpoint，跳过"
        continue
    fi

    if [[ -z "${best_ckpt}" ]] || [[ ! -d "${best_ckpt}" ]]; then
        echo "   ⚠️  best checkpoint 无效，退回到 last (${last_ckpt})"
        best_ckpt="${last_ckpt}"
    fi

    if [[ ! -d "${best_ckpt}" ]]; then
        echo "   ⚠️  best checkpoint 仍不存在，跳过"
        continue
    fi

    BEST_CKPTS[$exp_name]="${best_ckpt}"
    LAST_CKPTS[$exp_name]="${last_ckpt}"

    echo "   Best: $(basename ${best_ckpt})"
    echo "   Last: $(basename ${last_ckpt})"

    if [[ "${best_ckpt}" == "${last_ckpt}" ]]; then
        echo "   ℹ️  Best == Last，只需评估一次，结果会复制到last目录"
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
    pid=$!
    running_jobs[$pid]="${exp_name} (best)"

    echo "   ✅ 已启动 (PID: ${pid})"
    job_count=$((job_count + 1))

    # 如果best != last，还需要评估last
    if [[ "${NEED_BOTH[$exp_name]:-no}" == "yes" ]]; then
        gpu_idx=$((job_count % NUM_GPUS))
        gpu_id="${GPU_ARRAY[$gpu_idx]}"

        echo ""
        echo "📋 [$(date '+%H:%M:%S')] 调度任务 $((job_count + 1))/${total_jobs}"
        echo "   实验: ${exp_name}"
        echo "   类型: LAST ($(basename ${last_ckpt}))"
        echo "   GPU: ${gpu_id}"

        run_eval "${exp_name}" "${last_ckpt}" "last" "${gpu_id}" &
        pid=$!
        running_jobs[$pid]="${exp_name} (last)"

        echo "   ✅ 已启动 (PID: ${pid})"
        job_count=$((job_count + 1))
    else
        # best == last，评估一次后同步结果
        echo "   ℹ️  best与last一致，后续会把结果同步到last目录"
        # 这个在评估完成后处理
    fi

    echo ""

done

wait_with_progress "${total_jobs}" "${PROGRESS_INTERVAL}"
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
            sync_dir_contents "${best_dir}" "${last_dir}"
            echo "   ✅ ${exp_name}: last目录同步自best"
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
