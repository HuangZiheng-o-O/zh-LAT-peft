#!/bin/bash
# Unified Linear Attention Training Round Script
# Supports GLA, RetNet, Mamba2 and other FLA models through MODEL_TYPE env var.
#
# This is the canonical round training script for the LAT framework.
# MODEL_TYPE defaults to "auto" (auto-detect from model config).
#
set -euo pipefail

# MODEL_TYPE: gla, retnet, mamba2, or auto (default: auto)
MODEL_TYPE="${MODEL_TYPE:-auto}"
LAT_MODEL="${LAT_MODEL:-${GLA_MODEL:-}}"
LAT_PREC="${LAT_PREC:-${HP_PREC:-}}"

# Launcher Python script - uses unified entry point
LAUNCHER_PY="train_lat.py"
EVAL_PY="eval_lat.py"

# Eval controls (optional)
# - EVAL_AFTER_TRAIN=1: run eval_lat.py after each training job (same GPU)
# - EVAL_ONLY=1: skip training and only run eval_lat.py for each config
# - EVAL_TASKS=...: comma-separated tasks (default handled in eval_lat.py)
# - EVAL_BATCH_SIZE / HP_EVAL_BATCH_SIZE: eval batch size
# - HP_VAL_SPLIT: val|test (eval_lat.py will respect)
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

#########
#                               USER CONFIG HERE                              #
#########
: "${ROUND_E_MASTER[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E_MASTER=()
# Configuration arrays (same as gla_round_clean.sh)
ROUND_E15=(# 26 个 分组更清晰：FULL → 主轴 → Attn细粒度 → Head → Gating → MLP → 增量 → O-MLP

  # 0) FULL 多模块（参考上限）
  "E1_QKVO_plus_G_plus_GK_plus_MLP_r8_alpha16.yaml"

  # 1) 主轴基线（Attn / Gating / MLP）
  "E1_QKVO_r8_alpha16.yaml"
  "E3_GATINGONLY_r8_alpha16.yaml"
  "E4_MLPONLY_r8_alpha16.yaml"

  # 2) Attention 细粒度
  # 2.1 单打点（q/k/v/o）
  "E4_QONLY_r8_alpha16.yaml"
  "E4_KONLY_r8_alpha16.yaml"
  "E4_VONLY_r8_alpha16.yaml"
  "E11_OONLY_r8_alpha16.yaml"
  # 2.2 两两组合
  "E6_QKONLY_r8_alpha16.yaml"
  "E7_KVONLY_r8_alpha16.yaml"
  "E6_QVONLY_r8_alpha16.yaml"
  "E6_QOONLY_r8_alpha16.yaml"
  "E6_KOONLY_r8_alpha16.yaml"
  "E6_VOONLY_r8_alpha16.yaml"

  # 3) Head 相关
#  "E10_HEADONLY_r8_alpha16.yaml"
#  "E9_OplusHEAD_r8_alpha16.yaml"

  # 4) Gating 细粒度
  "E3_GPROJONLY_r8_alpha16.yaml"
#  "E3_GK0ONLY_r8_alpha16.yaml"
#  "E3_GK1ONLY_r8_alpha16.yaml"
  "E3_GKONLY_r8_alpha16.yaml"

  # 5) MLP 细粒度（SwiGLU）
  "E4_MLPGATEONLY_r8_alpha16.yaml"
#  "E4_MLPUPONLY_r8_alpha16.yaml"
#  "E4_MLPDOWNONLY_r8_alpha16.yaml"
  "E4_MLPUPDOWN_r8_alpha16.yaml"

  # 6) 结构增量：在 QKVO 上增量 Gates/MLP
  "E1_QKVO_plus_G_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_r8_alpha16.yaml"
  "E1_QKVO_plus_MLP_r8_alpha16.yaml"

  # 7) O-MLP 骨架
  "E2_OMLP_r8_alpha16.yaml"
  "E2_OMLP_plus_G_r8_alpha16.yaml"
  "E2_OMLP_plus_GK_r8_alpha16.yaml"
  "E2_OMLP_plus_G_plus_GK_r8_alpha16.yaml"
)

ROUND_E11=(
  "E1_QKVO_r8_alpha16.yaml"
)
ROUND_E12=(

  "E1_QKVO_plus_G_plus_MLP_r8_alpha16.yaml"
  "E1_QKVO_plus_MLP_r8_alpha16.yaml"

  "E1_QKVO_r8_alpha16.yaml"
  "E1_QKVO_plus_G_r8_alpha16.yaml"
  "E4_MLPONLY_r8_alpha16.yaml"
  "E2_OMLP_r8_alpha16.yaml"

  "E4_QONLY_r8_alpha16.yaml"
  "E4_KONLY_r8_alpha16.yaml"
  "E4_VONLY_r8_alpha16.yaml"
  "E11_OONLY_r8_alpha16.yaml"

  "E7_KVONLY_r8_alpha16.yaml"
  "E6_QVONLY_r8_alpha16.yaml"
  "E6_VOONLY_r8_alpha16.yaml"

)

# =============================================================================
# ROUND_E13_RETNET: RetNet 专用配置（无 gk_proj，RetNet 架构不包含此模块）
# =============================================================================
# 调用指南：
#   - RetNet: 运行脚本时使用 --suite E13
#   - 其余模型: 保持原有 E12/E15 等套件
# 说明：RetNet (Retentive Network) 使用 MultiScaleRetention 层，其结构与 GLA 不同：
#   - RetNet 有: q_proj, k_proj, v_proj, o_proj, g_proj (输出门控)
#   - RetNet 没有: gk_proj (GLA 特有的低秩门控投影)
#   - MLP 相同: gate_proj, up_proj, down_proj (SwiGLU)
# 因此所有包含 gk_proj 的配置在 RetNet 上会报错，需要使用此专用 ROUND。
# =============================================================================
ROUND_E13=(#ROUND_E12_RETNET


  "E1_QKVO_plus_MLP_r8_alpha16.yaml"

  "E1_QKVO_r8_alpha16.yaml"
  "E4_MLPONLY_r8_alpha16.yaml"
  "E2_OMLP_r8_alpha16.yaml"

  "E4_QONLY_r8_alpha16.yaml"
  "E4_KONLY_r8_alpha16.yaml"
  "E4_VONLY_r8_alpha16.yaml"
  "E11_OONLY_r8_alpha16.yaml"

  "E7_KVONLY_r8_alpha16.yaml"
  "E6_QVONLY_r8_alpha16.yaml"
  "E6_VOONLY_r8_alpha16.yaml"

  "E1_QKVO_plus_G_plus_MLP_r8_alpha16.yaml"
  "E1_QKVO_plus_G_r8_alpha16.yaml"


)
# =============================================================================
# ROUND_E12_DELTANET: DeltaNet 专用配置（无 gk_proj, 默认无 g_proj）
# =============================================================================
# 调用指南：
#   - DeltaNet: 运行脚本时使用 --suite E14
#   - 若模型开启 use_gate，可自行复制该 suite 并加入相应配置
# 说明：DeltaNet (Delta Rule Linear Transformer) 架构与 GLA 显著不同：
#   - DeltaNet 有: q_proj, k_proj, v_proj, o_proj, b_proj (beta 写入强度)
#   - DeltaNet 默认无: g_proj (use_gate=False), gk_proj (GLA 特有)
#   - b_proj 输出仅 num_heads 个标量 (如16)，不适合 LoRA，故不包含
#   - MLP 相同: gate_proj, up_proj, down_proj (SwiGLU)
# 因此 DeltaNet 使用 QKVO + MLP 作为上限配置，删除所有 gk/g 相关配置。
# 参考论文: https://arxiv.org/abs/2406.06484
# =============================================================================
ROUND_E14=(# ROUND_E12_DELTANET

  "E1_QKVO_plus_MLP_r8_alpha16.yaml"

  "E1_QKVO_r8_alpha16.yaml"
  "E4_MLPONLY_r8_alpha16.yaml"
  "E2_OMLP_r8_alpha16.yaml"

  "E4_QONLY_r8_alpha16.yaml"
  "E4_KONLY_r8_alpha16.yaml"
  "E4_VONLY_r8_alpha16.yaml"
  "E11_OONLY_r8_alpha16.yaml"

  "E7_KVONLY_r8_alpha16.yaml"
  "E6_QVONLY_r8_alpha16.yaml"
  "E6_VOONLY_r8_alpha16.yaml"

)

# =============================================================================
  # ROUND_E15_BASED: Based 专用配置（无 g_proj, 无 gk_proj）
  # =============================================================================
  # 调用指南：
  #   - Based: 运行脚本时使用 --suite E15 或 MODEL_TYPE=based
  # 说明：Based (Simple Linear Attention with Taylor Feature Map) 架构最为简洁：
  #   - Based 有: q_proj, k_proj, v_proj, o_proj
  #   - Based 没有: g_proj (无输出门控), gk_proj (GLA 特有)
  #   - 使用 TaylorFeatureMap 替代门控机制: φ(q)^T φ(k) = 1 + qk + (qk)²/2
  #   - MLP 相同: gate_proj, up_proj, down_proj (SwiGLU)
  # 参考论文: https://arxiv.org/abs/2402.18668
  # =============================================================================
  ROUND_E15=(# ROUND_E15_BASED

    "E1_QKVO_plus_MLP_r8_alpha16.yaml"

    "E1_QKVO_r8_alpha16.yaml"
    "E4_MLPONLY_r8_alpha16.yaml"
    "E2_OMLP_r8_alpha16.yaml"

    "E4_QONLY_r8_alpha16.yaml"
    "E4_KONLY_r8_alpha16.yaml"
    "E4_VONLY_r8_alpha16.yaml"
    "E11_OONLY_r8_alpha16.yaml"

    "E7_KVONLY_r8_alpha16.yaml"
    "E6_QVONLY_r8_alpha16.yaml"
    "E6_VOONLY_r8_alpha16.yaml"

  )


#####################################################################
#                           Core Logic                               #
#####################################################################

declare -a PIDS=()
declare -a COMPLETED_ROUNDS=()
declare -a RUN_QUEUE=()
declare -a DETECTED_GPUS=()
declare -a Round_all=()

CURRENT_ROUND=""
FAILED_ROUND=""

# Log tag based on model type
LOG_TAG="${MODEL_TYPE^^}"
[[ "$LOG_TAG" == "AUTO" ]] && LOG_TAG="LAT"

print_interruption_summary() {
  echo ""
  echo "SUMMARY:"
  if (( ${#COMPLETED_ROUNDS[@]} > 0 )); then
    echo "  Experiments completed: ${COMPLETED_ROUNDS[*]}."
  else
    echo "  Experiments completed: none."
  fi
  if [[ -n "${CURRENT_ROUND:-}" ]]; then
    echo "  Experiment ${CURRENT_ROUND} exited abnormally (interrupted)."
  fi
}

print_failure_summary() {
  echo ""
  echo "SUMMARY:"
  if (( ${#COMPLETED_ROUNDS[@]} > 0 )); then
    echo "  Experiments completed: ${COMPLETED_ROUNDS[*]}."
  else
    echo "  Experiments completed: none."
  fi
  if [[ -n "${FAILED_ROUND:-}" ]]; then
    echo "  Experiment ${FAILED_ROUND} failed. Stopping."
  fi
}

cleanup() {
  for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done

  if [[ -n "${EXP_ROOT:-}" ]]; then
    pkill -f -- "${LAUNCHER_PY} --cfg ${EXP_ROOT}/" 2>/dev/null || true
  fi

  # Best-effort email notification for interruption
  _email_interrupt="${SWANLAB_EMAIL_ON_INTERRUPT:-1}"
  if [[ "${_email_interrupt}" == "1" ]] && command -v python >/dev/null 2>&1; then
    (
      cd "$(dirname "$0")/.." || true
      python -m scripts.utils.email_notify \
        --event INTERRUPTED \
        --group "suite=${SELECT_SUITE} round=${CURRENT_ROUND} data=${DATA} model=${MODEL_TYPE}" \
        --yaml "${SWANLAB_EMAIL_YAML:-}" >/dev/null 2>&1 || true
    )
  fi

  print_interruption_summary
  exit 130
}
trap cleanup INT TERM

ROUND="${1:-1}"
FORCE_SEED=87
DATA="${DATA:-glue-tvt_cola}"

# Remote workspace
PEFT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft"
cd "$PEFT_ROOT"

# Environment setup
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

# Echo invocation & key env overrides
echo "CMD: $0 $*"
echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "ENV_OVERRIDES:"
for _k in \
  MODEL_TYPE \
  GPU_IDS \
  GPU_PLAN \
  CUDA_VISIBLE_DEVICES \
  DATA \
  HP_VAL_SPLIT \
  SPIDER_LOCAL_DIR \
  NLTK_DATA \
  LAT_FORCE_LEFT_PAD \
  LAT_USE_MAX_NEW_TOKENS \
  LAT_VERBOSE \
  LAT_LOG_PADDING_STATS \
  LAT_LAUNCH_STAGGER_MINUTES \
  EVAL_GEN EVAL_GEN_MAX_LENGTH EVAL_GEN_MIN_LENGTH EVAL_GEN_NUM_BEAMS \
  PYTORCH_CUDA_ALLOC_CONF TOKENIZERS_PARALLELISM OMP_NUM_THREADS MKL_NUM_THREADS \
  GRADIENT_CHECKPOINTING \
  LOGITS_TO_KEEP \
  NUM_DATA_WORKERS \
  FORCE_SEED \
  HP_DATA HP_BATCH_SIZE HP_LR HP_EPOCHS HP_EVAL_BATCH_SIZE HP_PREC HP_SEED \
  HP_PEFT_R HP_PEFT_ALPHA HP_PEFT_DROPOUT HP_INIT HP_PISSA_FAST \
  HP_MAX_STEPS HP_EVAL_STEPS HP_SAVE_STEPS HP_LOGGING_STEPS \
  LR_SCHEDULER_TYPE LR_WARMUP_STEPS LR_WARMUP_RATIO
do
  v="${!_k-}"
  if [[ -n "${v:-}" ]]; then
    echo "  ${_k}=${v}"
  fi
done
if command -v env >/dev/null 2>&1; then
  echo "HP_* (all):"; env | grep -E '^HP_' | sort || true
fi

# Paths
EXP_ROOT="${EXP_ROOT:-cfg/my_lora_exp}"
CFG_DIR="${CFG_DIR:-${EXP_ROOT}/yaml}"
PEFT_DIR="${PEFT_DIR:-${EXP_ROOT}/peft}"

if [[ ! -d "$CFG_DIR" ]]; then
  echo "Config directory not found: $CFG_DIR" >&2
  exit 1
fi

# GPU detection
parse_gpu_list() {
  local s="${1:-}"
  s="${s//,/ }"
  DETECTED_GPUS=()
  for tok in $s; do
    [[ -n "$tok" ]] && DETECTED_GPUS+=("$tok")
  done
}

detect_gpus() {
  if [[ -n "${GPU_IDS:-}" ]]; then
    parse_gpu_list "$GPU_IDS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    parse_gpu_list "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    local cnt
    cnt="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    DETECTED_GPUS=()
    for ((i=0;i<cnt;i++)); do DETECTED_GPUS+=("$i"); done
  elif command -v rocm-smi >/dev/null 2>&1; then
    local cnt
    cnt="$(rocm-smi --showid 2>/dev/null | grep -E 'GPU\[|GPU' | wc -l | tr -d ' ')"
    DETECTED_GPUS=()
    for ((i=0;i<cnt;i++)); do DETECTED_GPUS+=("$i"); done
  else
    echo "ERROR: Could not detect GPUs." >&2
    exit 1
  fi
}

detect_gpus
NUM_GPUS="${#DETECTED_GPUS[@]}"

if (( NUM_GPUS < 1 )); then
  echo "ERROR: No GPUs detected." >&2
  exit 1
fi

# Per-GPU concurrency plan
GPU_PLAN_STR="${GPU_PLAN:-}"
declare -a GPU_PLAN_ARR=()
if [[ -z "$GPU_PLAN_STR" ]]; then
  for _ in "${DETECTED_GPUS[@]}"; do GPU_PLAN_ARR+=(1); done
else
  GPU_PLAN_STR="${GPU_PLAN_STR//,/ }"
  read -r -a GPU_PLAN_ARR <<<"$GPU_PLAN_STR"
  if (( ${#GPU_PLAN_ARR[@]} == 1 )); then
    val="${GPU_PLAN_ARR[0]}"; GPU_PLAN_ARR=()
    for _ in "${DETECTED_GPUS[@]}"; do GPU_PLAN_ARR+=("$val"); done
  elif (( ${#GPU_PLAN_ARR[@]} != NUM_GPUS )); then
    echo "ERROR: GPU_PLAN length mismatch." >&2
    exit 1
  fi
fi

# Build GPU_SLOTS
declare -a GPU_SLOTS=()
for i in "${!DETECTED_GPUS[@]}"; do
  gpu="${DETECTED_GPUS[$i]}"
  cnt="${GPU_PLAN_ARR[$i]}"
  if [[ -z "$cnt" || "$cnt" -le 0 ]]; then cnt=0; fi
  for ((j=0;j<cnt;j++)); do GPU_SLOTS+=("$gpu"); done
done
N_SLOTS="${#GPU_SLOTS[@]}"
if (( N_SLOTS < 1 )); then
  echo "ERROR: Effective parallel slots is zero." >&2
  exit 1
fi

# Suite selector
SELECT_SUITE="ALL"

append_suite_into_master() {
  local var="$1"
  if eval "[[ -v ${var} && \${#${var}[@]} -gt 0 ]]"; then
    local tmp=( $(eval "printf '%q ' \"\${${var}[@]}\"") )
    if ((${#tmp[@]} > 0)); then
      read -r -a tmp <<<"$(eval "printf '%s ' \"\${${var}[@]}\"")"
      Round_all+=("${tmp[@]}")
    fi
  fi
}

if [[ "${1:-}" =~ ^([Ee][0-9]+)$ ]]; then
  suite="${BASH_REMATCH[1]}"
  suite="${suite^^}"
  SELECT_SUITE="$suite"
  shift

  varname="ROUND_${suite}"

  if ! eval "[[ -v ${varname} && \${#${varname}[@]} -gt 0 ]]"; then
    echo "ERROR: Suite '${suite}' is not defined." >&2
    exit 1
  fi

  Round_all=()
  append_suite_into_master "${varname}"
  ROUND="${1:-all}"
else
  if (( ${#Round_all[@]} == 0 )); then
    Round_all=()
    for i in {1..20}; do
      append_suite_into_master "ROUND_E${i}"
    done
    if (( ${#Round_all[@]} == 0 )); then
      echo "ERROR: No configs found." >&2
      exit 1
    fi
  fi
fi

# Dynamic round slicing
TOTAL_CFGS="${#Round_all[@]}"
N_ROUNDS=$(( (TOTAL_CFGS + N_SLOTS - 1) / N_SLOTS ))

defined_rounds_str() {
  local out=""
  for ((r=1;r<=N_ROUNDS;r++)); do out+="${r} "; done
  printf "%s" "$out"
}

canonical_cfg_path() {
  local entry="$1"
  local path="${CFG_DIR}/${entry}"
  if [[ -f "$path" ]]; then
    printf '%s\n' "$path"; return 0
  else
    printf '%s\n' "$path"; return 1
  fi
}

declare -a SELECT_SET=()
get_round_configs() {
  local r="$1"
  if (( r < 1 || r > N_ROUNDS )); then
    return 1
  fi
  local start=$(( (r-1)*N_SLOTS ))
  local end=$(( r*N_SLOTS ))
  if (( end > TOTAL_CFGS )); then end="$TOTAL_CFGS"; fi
  SELECT_SET=()
  local i
  for ((i=start;i<end;i++)); do
    SELECT_SET+=("${Round_all[i]}")
  done
  (( ${#SELECT_SET[@]} > 0 ))
}

make_tmp_cfg_with_data() {
  local src="$1"; local outdir="$2"
  local base
  base="$(basename "$src")"
  local name ext
  name="${base%.*}"; ext="${base##*.}"
  local out
  out="$outdir/${name}.${ext}"
  if [[ -e "$out" ]]; then
    local k=1
    while :; do
      local cand="$outdir/${name}__rep${k}.${ext}"
      if [[ ! -e "$cand" ]]; then out="$cand"; break; fi
      k=$((k+1))
    done
  fi
  cp "$src" "$out"
  printf '\n# injected by lat_round.sh\ndata: %s\n' "$DATA" >>"$out"
  local ndw
  ndw="${NUM_DATA_WORKERS:-8}"
  printf 'num_data_workers: %s\n' "$ndw" >>"$out"
  if [[ -n "${GRADIENT_CHECKPOINTING:-}" ]]; then
    case "${GRADIENT_CHECKPOINTING,,}" in
      1|true|yes|on)
        printf 'gradient_checkpointing: true\n' >>"$out"
        ;;
    esac
  fi
  if [[ -n "${LOGITS_TO_KEEP:-}" ]]; then
    printf 'logits_to_keep: %s\n' "$LOGITS_TO_KEEP" >>"$out"
  fi
  printf '%s\n' "$out"
}

run_round () {
  local r="$1"

  if ! get_round_configs "$r"; then
    echo "Round ${r} is empty or out of range. Valid rounds: $(defined_rounds_str)" >&2
    return 1
  fi

  local missing=()
  local -a RESOLVED_CFGS=()
  local resolved=""
  for f in "${SELECT_SET[@]}"; do
    if resolved="$(canonical_cfg_path "$f")"; then
      RESOLVED_CFGS+=("$resolved")
    else
      missing+=("$resolved")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    echo "Missing configs (expected under $CFG_DIR):" >&2
    printf '  %s\n' "${missing[@]}" >&2
    return 1
  fi

  local num_jobs="${#RESOLVED_CFGS[@]}"
  echo "=== Starting Round ${r} (${num_jobs} jobs; MODEL_TYPE=${MODEL_TYPE}; FORCE_SEED=${FORCE_SEED}; N_SLOTS=${N_SLOTS}) ==="
  echo "SUITE   = ${SELECT_SUITE}"
  echo "CFG_DIR = $CFG_DIR"
  echo "PEFT_DIR= $PEFT_DIR"
  echo "GPUs    = ${DETECTED_GPUS[*]}"
  echo "PLAN    = ${GPU_PLAN_ARR[*]}"
  echo "SLOTS   = ${GPU_SLOTS[*]}"
  echo "DATA    = ${DATA}"

  local __round_start_epoch
  __round_start_epoch="$(date +%s)"
  local __round_start_iso
  __round_start_iso="$(date +%F_%T)"
  echo "[${__round_start_iso}] ROUND=${r} START"

  PIDS=()
  local i
  local TMP_CFG_DIR
  TMP_CFG_DIR="$(mktemp -d /tmp/lat_data_XXXXXX)"

  # Stagger minutes (LAT_* already merged with GLA_* in batch script)
  local _stagger_min="${LAT_LAUNCH_STAGGER_MINUTES:-0}"
  if ! [[ "${_stagger_min}" =~ ^[0-9]+$ ]]; then
    _stagger_min=0
  fi

  for i in "${!RESOLVED_CFGS[@]}"; do
    local CFG="${RESOLVED_CFGS[$i]}"
    local slot_index=$(( i % N_SLOTS ))
    local GPU="${GPU_SLOTS[$slot_index]}"
    local CFG_INJ
    CFG_INJ="$(make_tmp_cfg_with_data "$CFG" "$TMP_CFG_DIR")"
    echo "[GPU ${GPU}] ${CFG_INJ}  (MODEL_TYPE=${MODEL_TYPE}; HP_SEED=${FORCE_SEED}; data=${DATA})"
    # Stagger: fixed delay between consecutive launches (not cumulative)
    if (( _stagger_min > 0 && i > 0 )); then
      local _delay_sec=$(( _stagger_min * 60 ))
      echo "[GPU ${GPU}] delaying launch by ${_delay_sec}s (stagger=${_stagger_min}min)"
      sleep "${_delay_sec}"
    fi
    # Pass MODEL_TYPE and optional model/precision overrides to the unified launcher.
    # Training mode:
    # - default: overwrite (fresh run) to match historical behavior
    # - resume: set LAT_TRAIN_RESUME=1 (or use lat_batch_tmux.sh --resume)
    local -a _cmd=(python "$LAUNCHER_PY" --cfg "$CFG_INJ" --model-type "${MODEL_TYPE}")
    if [[ "${LAT_TRAIN_RESUME:-0}" == "1" ]]; then
      _cmd+=("--resume")
    elif [[ "${LAT_TRAIN_OVERWRITE:-1}" == "1" ]]; then
      _cmd+=("--overwrite")
    fi
    if [[ -n "${LAT_MODEL:-}" ]]; then
      _cmd+=("--model" "${LAT_MODEL}")
    fi
    if [[ -n "${LAT_PREC:-}" ]]; then
      _cmd+=("--prec" "${LAT_PREC}")
    fi

    # Eval command (reuses same cfg, uses env/overrides to locate output + adapter)
    local -a _eval_cmd=(python "$EVAL_PY" --cfg "$CFG_INJ" --model-type "${MODEL_TYPE}")
    if [[ -n "${EVAL_BACKEND:-}" ]]; then
      _eval_cmd+=("--backend" "${EVAL_BACKEND}")
    fi
    if [[ -n "${EVAL_TASKS:-}" ]]; then
      _eval_cmd+=("--tasks" "${EVAL_TASKS}")
    fi
    if [[ -n "${EVAL_OUTPUT_ROOT:-}" ]]; then
      _eval_cmd+=("--output-root" "${EVAL_OUTPUT_ROOT}")
    fi
    if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
      _eval_cmd+=("--eval-batch-size" "${EVAL_BATCH_SIZE}")
    fi

    if [[ "${EVAL_ONLY}" == "1" ]]; then
      MODEL_TYPE="${MODEL_TYPE}" HP_SEED=${FORCE_SEED} CUDA_VISIBLE_DEVICES="$GPU" \
        "${_eval_cmd[@]}" &
    elif [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
      MODEL_TYPE="${MODEL_TYPE}" HP_SEED=${FORCE_SEED} CUDA_VISIBLE_DEVICES="$GPU" \
        bash -lc "$(printf '%q ' "${_cmd[@]}") && $(printf '%q ' "${_eval_cmd[@]}")" &
    else
      MODEL_TYPE="${MODEL_TYPE}" HP_SEED=${FORCE_SEED} CUDA_VISIBLE_DEVICES="$GPU" \
        "${_cmd[@]}" &
    fi
    PIDS+=("$!")
  done

  local any_failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      any_failed=1
    fi
  done

  rm -rf "$TMP_CFG_DIR" || true

  local __round_end_epoch
  __round_end_epoch="$(date +%s)"
  local __round_end_iso
  __round_end_iso="$(date +%F_%T)"
  local __round_elapsed
  __round_elapsed=$(( __round_end_epoch - __round_start_epoch ))
  local __round_h=$(( __round_elapsed / 3600 ))
  local __round_m=$(( (__round_elapsed % 3600) / 60 ))
  local __round_s=$(( __round_elapsed % 60 ))
  printf '[%s] ROUND=%s END elapsed=%02d:%02d:%02d (%ds)\n' "${__round_end_iso}" "${r}" "${__round_h}" "${__round_m}" "${__round_s}" "${__round_elapsed}"

  if (( any_failed )); then
    return 1
  fi

  echo "ROUND=${r} finished (MODEL_TYPE=${MODEL_TYPE}; HP_SEED=${FORCE_SEED})."
  return 0
}

# Build the run queue
if (( $# == 0 )); then
  if [[ "${ROUND:-}" == "all" ]]; then
    for ((r=1;r<=N_ROUNDS;r++)); do RUN_QUEUE+=("$r"); done
  else
    RUN_QUEUE+=("${ROUND:-1}")
  fi
else
  for arg in "$@"; do
    if [[ "$arg" == "all" ]]; then
      for ((r=1;r<=N_ROUNDS;r++)); do RUN_QUEUE+=("$r"); done
    else
      RUN_QUEUE+=("$arg")
    fi
  done
fi

# Validate run queue
for r in "${RUN_QUEUE[@]}"; do
  if ! [[ "$r" =~ ^[0-9]+$ ]] || (( r < 1 || r > N_ROUNDS )); then
    echo "Invalid round '$r'. Valid values: $(defined_rounds_str) or 'all'." >&2
    exit 1
  fi
done

# Execute
for r in "${RUN_QUEUE[@]}"; do
  CURRENT_ROUND="$r"
  if run_round "$r"; then
    COMPLETED_ROUNDS+=("$r")
    CURRENT_ROUND=""
  else
    FAILED_ROUND="$r"
    for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
    if [[ -n "${EXP_ROOT:-}" ]]; then
      pkill -f -- "${LAUNCHER_PY} --cfg ${EXP_ROOT}/" 2>/dev/null || true
    fi
    print_failure_summary
    exit 1
  fi
done

exit 0
