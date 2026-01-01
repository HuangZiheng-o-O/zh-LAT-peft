#!/bin/bash
# SD-LoRA Sparse Training Round Script
# Batch testing of SD-LoRA configurations for GLA models
#
# Key differences from lat_round.sh:
# - Uses SD-LoRA JSON configs from cfg/my_lora_exp/sparse_peft/
# - Passes JSON via --peft argument (not embedded in YAML)
# - Uses a single base YAML for training parameters
#
set -euo pipefail

# MODEL_TYPE: gla, retnet, mamba2, or auto (default: gla for SD-LoRA)
MODEL_TYPE="${MODEL_TYPE:-gla}"
LAT_MODEL="${LAT_MODEL:-${GLA_MODEL:-}}"
LAT_PREC="${LAT_PREC:-${HP_PREC:-}}"

# Resolve repo root relative to this script unless explicitly provided.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PEFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PEFT_ROOT="${PEFT_ROOT:-${LAT_PEFT_ROOT:-$DEFAULT_PEFT_ROOT}}"

# Launcher Python script
LAUNCHER_PY="train_lat.py"

#########
#                           SD-LoRA CONFIGS                                    #
#########
# Base YAML for training parameters (all params from env vars)
BASE_YAML="cfg/my_lora_exp/yaml_sparse/base_sdlora.yaml"

# SD-LoRA JSON configs directory
SDLORA_CFG_DIR="${SDLORA_CFG_DIR:-cfg/my_lora_exp/sparse_peft}"

# SD-LoRA configurations to test (15 configs total)
# KV: 5 train ratios (01, 05, 10, 20, 30)
# QKVO: 5 train ratios (01, 05, 10, 20, 30)
# Others: only train05 (v, vo, qkvog, qkvo_plus_mlp, omlp)
ROUND_SPARSE=(
  # KV configurations (5)
  "gla_sdlora_kv_train01.json"
  "gla_sdlora_kv_train05.json"
  "gla_sdlora_kv_train10.json"
  "gla_sdlora_kv_train20.json"
  "gla_sdlora_kv_train30.json"

  # QKVO configurations (5)
  "gla_sdlora_qkvo_train01.json"
  "gla_sdlora_qkvo_train05.json"
  "gla_sdlora_qkvo_train10.json"
  "gla_sdlora_qkvo_train20.json"
  "gla_sdlora_qkvo_train30.json"

  # Other configurations - only train05 (5)
  "gla_sdlora_v_train05.json"
  "gla_sdlora_vo_train05.json"
  "gla_sdlora_qkvog_train05.json"
  "gla_sdlora_qkvo_plus_mlp_train05.json"
  "gla_sdlora_omlp_train05.json"
)

#####################################################################
#                           Core Logic                               #
#####################################################################

declare -a PIDS=()
declare -a COMPLETED_ROUNDS=()
declare -a RUN_QUEUE=()
declare -a DETECTED_GPUS=()

CURRENT_ROUND=""
FAILED_ROUND=""

LOG_TAG="SDLORA"

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

  print_interruption_summary
  exit 130
}
trap cleanup INT TERM

ROUND="${1:-1}"
FORCE_SEED=87
DATA="${DATA:-glue-tvt_cola}"

cd "$PEFT_ROOT"

# Environment setup - use env vars if set, otherwise use defaults
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/user/mzs_h/data/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export GLUE_METRIC_DIR="${GLUE_METRIC_DIR:-$HF_HOME/eval_metrics/glue}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# Respect HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE if set by user
[[ -n "${HF_HUB_OFFLINE:-}" ]] && export HF_HUB_OFFLINE
[[ -n "${TRANSFORMERS_OFFLINE:-}" ]] && export TRANSFORMERS_OFFLINE
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
rm -rf ~/.config/wandb ~/.triton ~/.cache/torch_extensions || true

# Echo invocation & key env overrides
echo "CMD: $0 $*"
echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "SDLORA_CFG_DIR: ${SDLORA_CFG_DIR}"
echo "BASE_YAML: ${BASE_YAML}"
echo "ENV_OVERRIDES:"
for _k in \
  MODEL_TYPE \
  GPU_IDS \
  GPU_PLAN \
  CUDA_VISIBLE_DEVICES \
  DATA \
  HP_VAL_SPLIT \
  LAT_FORCE_LEFT_PAD GLA_FORCE_LEFT_PAD \
  LAT_USE_MAX_NEW_TOKENS GLA_USE_MAX_NEW_TOKENS \
  EVAL_GEN EVAL_GEN_MAX_LENGTH EVAL_GEN_MIN_LENGTH EVAL_GEN_NUM_BEAMS \
  PYTORCH_CUDA_ALLOC_CONF TOKENIZERS_PARALLELISM OMP_NUM_THREADS MKL_NUM_THREADS \
  GRADIENT_CHECKPOINTING \
  NUM_DATA_WORKERS \
  FORCE_SEED \
  HP_DATA HP_BATCH_SIZE HP_LR HP_EPOCHS HP_EVAL_BATCH_SIZE HP_PREC HP_SEED \
  HP_MAX_STEPS HP_EVAL_STEPS HP_SAVE_STEPS HP_LOGGING_STEPS \
  LR_SCHEDULER_TYPE LR_WARMUP_STEPS LR_WARMUP_RATIO \
  LAT_LAUNCH_STAGGER_MINUTES GLA_LAUNCH_STAGGER_MINUTES
do
  v="${!_k-}"
  if [[ -n "${v:-}" ]]; then
    echo "  ${_k}=${v}"
  fi
done

# Verify base YAML exists
if [[ ! -f "$BASE_YAML" ]]; then
  echo "ERROR: Base YAML not found: $BASE_YAML" >&2
  exit 1
fi

# Verify SD-LoRA config directory exists
if [[ ! -d "$SDLORA_CFG_DIR" ]]; then
  echo "ERROR: SD-LoRA config directory not found: $SDLORA_CFG_DIR" >&2
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

# Use ROUND_SPARSE as the config array
Round_all=("${ROUND_SPARSE[@]}")
SELECT_SUITE="SPARSE"

# Dynamic round slicing
TOTAL_CFGS="${#Round_all[@]}"
N_ROUNDS=$(( (TOTAL_CFGS + N_SLOTS - 1) / N_SLOTS ))

defined_rounds_str() {
  local out=""
  for ((r=1;r<=N_ROUNDS;r++)); do out+="${r} "; done
  printf "%s" "$out"
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
  out="$(mktemp "$outdir/${name}.XXXXXX.${ext}")"
  cp "$src" "$out"
  printf '\n# injected by lat_round_sparse.sh\ndata: %s\n' "$DATA" >>"$out"
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
  printf '%s\n' "$out"
}

run_round () {
  local r="$1"

  if ! get_round_configs "$r"; then
    echo "Round ${r} is empty or out of range. Valid rounds: $(defined_rounds_str)" >&2
    return 1
  fi

  # Verify all SD-LoRA JSON configs exist
  local missing=()
  for f in "${SELECT_SET[@]}"; do
    local json_path="${SDLORA_CFG_DIR}/${f}"
    if [[ ! -f "$json_path" ]]; then
      missing+=("$json_path")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    echo "Missing SD-LoRA configs:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    return 1
  fi

  local num_jobs="${#SELECT_SET[@]}"
  echo "=== Starting Round ${r} (${num_jobs} SD-LoRA configs; MODEL_TYPE=${MODEL_TYPE}; FORCE_SEED=${FORCE_SEED}; N_SLOTS=${N_SLOTS}) ==="
  echo "SUITE      = ${SELECT_SUITE}"
  echo "SDLORA_DIR = $SDLORA_CFG_DIR"
  echo "BASE_YAML  = $BASE_YAML"
  echo "GPUs       = ${DETECTED_GPUS[*]}"
  echo "PLAN       = ${GPU_PLAN_ARR[*]}"
  echo "SLOTS      = ${GPU_SLOTS[*]}"
  echo "DATA       = ${DATA}"
  echo ""
  echo "Configs in this round:"
  for f in "${SELECT_SET[@]}"; do
    echo "  - $f"
  done
  echo ""

  local __round_start_epoch
  __round_start_epoch="$(date +%s)"
  local __round_start_iso
  __round_start_iso="$(date +%F_%T)"
  echo "[${__round_start_iso}] ROUND=${r} START"

  PIDS=()
  local i
  local TMP_CFG_DIR
  TMP_CFG_DIR="$(mktemp -d /tmp/sdlora_data_XXXXXX)"

  # Stagger support
  local _stagger_min="${LAT_LAUNCH_STAGGER_MINUTES:-${GLA_LAUNCH_STAGGER_MINUTES:-0}}"
  if ! [[ "${_stagger_min}" =~ ^[0-9]+$ ]]; then
    _stagger_min=0
  fi

  for i in "${!SELECT_SET[@]}"; do
    local SDLORA_JSON="${SELECT_SET[$i]}"
    local SDLORA_PATH="${SDLORA_CFG_DIR}/${SDLORA_JSON}"
    local slot_index=$(( i % N_SLOTS ))
    local GPU="${GPU_SLOTS[$slot_index]}"

    # Create temp YAML with data injection
    local CFG_INJ
    CFG_INJ="$(make_tmp_cfg_with_data "$BASE_YAML" "$TMP_CFG_DIR")"

    echo "[GPU ${GPU}] ${SDLORA_JSON}  (MODEL_TYPE=${MODEL_TYPE}; HP_SEED=${FORCE_SEED}; data=${DATA})"

    if (( _stagger_min > 0 )); then
      local _delay_sec=$(( _stagger_min * 60 * i ))
      if (( _delay_sec > 0 )); then
        echo "[GPU ${GPU}] delaying launch by ${_delay_sec}s"
        sleep "${_delay_sec}"
      fi
    fi

    # Build command with --peft pointing to SD-LoRA JSON
    local -a _cmd=(python "$LAUNCHER_PY" --cfg "$CFG_INJ" --peft "$SDLORA_PATH" --model-type "${MODEL_TYPE}" --overwrite)
    if [[ -n "${LAT_MODEL:-}" ]]; then
      _cmd+=("--model" "${LAT_MODEL}")
    fi
    if [[ -n "${LAT_PREC:-}" ]]; then
      _cmd+=("--prec" "${LAT_PREC}")
    fi

    MODEL_TYPE="${MODEL_TYPE}" HP_SEED=${FORCE_SEED} CUDA_VISIBLE_DEVICES="$GPU" \
      "${_cmd[@]}" &
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
    print_failure_summary
    exit 1
  fi
done

exit 0
