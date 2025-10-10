#!/bin/bash
set -euo pipefail

# Unified launcher for GLA LoRA experiments (dynamic rounds from ROUND_FILES).
# - Uses pre-authored configs under cfg/my_lora_exp/ (no generation here).
# - Launches one round at a time; each round may contain N jobs (not forced to 7).
# - GLOBAL seed for training is FORCE_SEED=127 (overrides any filename/config seed).
#
# Usage examples:
#   bash scripts/train/new/gla_round_new.sh 1
#   TASK=rte SEED=127 bash scripts/train/new/gla_round_new.sh 2
#   bash scripts/train/new/gla_round_new.sh all            # Run all defined rounds (dynamic)
#   bash scripts/train/new/gla_round_new.sh 3 1            # Run rounds 3 then 1 in strict order
#
# Optional:
#   export GPU_IDS="0 1 2 3 4 5 6"                         # Explicit GPU mapping; length must >= jobs per round
#   # If not set, GPU IDs default to 0..(num_jobs-1)

declare -a PIDS=()
declare -a COMPLETED_ROUNDS=()
declare -a RUN_QUEUE=()

CURRENT_ROUND=""
FAILED_ROUND=""

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
  # Best-effort terminate any in-flight jobs of the current round by PIDs
  for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done

  # Fallback: kill any orphaned train.py that still match our config base path
  # NOTE: this is intentionally broad to avoid GPU leaks; adjust if you need narrower scope.
  if [[ -n "${BASE_DIR:-}" ]]; then
    pkill -f -- "train.py --cfg ${BASE_DIR}/" 2>/dev/null || true
  fi

  print_interruption_summary
  exit 130
}
trap cleanup INT TERM

ROUND="${1:-1}"        # first arg kept for backward compat/docs; may be number or 'all'
TASK="${TASK:-cola}"   # informational only
SEED="${SEED:-42}"     # informational only (NOT used for training)
FORCE_SEED=127         # actual seed used in training (HP_SEED)

# Remote workspace expected by train.py
PEFT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft"
cd "$PEFT_ROOT"

# Env mirrors/caches (same as original)
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

# ---------------------------------------------------------------------------
# Pre-authored configs only
# ---------------------------------------------------------------------------
BASE_DIR="cfg/my_lora_exp"

declare -A ROUND_FILES
ROUND_FILES[1]="round1_E0_ZS_r0_alpha0_seed42.yaml round1_E1_QKVO_r8_alpha8_seed42.yaml round1_E2_OMLP_r8_alpha8_seed42.yaml round1_E3_QV_r8_alpha8_seed42.yaml round1_E4_OONLY_r4_alpha8_seed42.yaml round1_E5_MLPONLY_r8_alpha8_seed42.yaml round1_E6_QKV_r8_alpha8_seed42.yaml"
ROUND_FILES[2]="round2_E7_GONLY_r8_alpha8_seed127.yaml round2_E8_QKVO_G_r8_alpha8_seed127.yaml round2_E1_QKVO_R16_r16_alpha16_seed127.yaml round2_E1_QKVO_seed_r8_alpha8_seed127.yaml round2_E2_OMLP_seed_r8_alpha8_seed127.yaml round2_E3_QV_seed_r8_alpha8_seed127.yaml round2_E4_OONLY_seed_r4_alpha8_seed127.yaml"
ROUND_FILES[3]="round3_E1_QKVO_last6_r8_alpha8_seed42.yaml round3_E2_OMLP_last6_r8_alpha8_seed42.yaml round3_E1_QKVO_first6_r8_alpha8_seed42.yaml round3_E2_OMLP_middle6_r8_alpha8_seed42.yaml round3_E1_QKVO_alpha2r_r8_alpha16_seed42.yaml round3_E1_QKVO_dropout0_r8_alpha8_seed42.yaml round3_E1_QKVO_lr1e-4_r8_alpha8_seed42.yaml"
ROUND_FILES[4]="round4_E1_QKVO_r4_equal_r4_alpha8_seed42.yaml round4_E4_OONLY_r16_equal_r16_alpha16_seed42.yaml round4_E2_OMLP_r6_equal_r6_alpha6_seed42.yaml round4_E1_QKVO_DoRA_r8_alpha8_seed42.yaml round4_E2_OMLP_DoRA_r8_alpha8_seed42.yaml round4_E1_QKVO_RSLoRA_r8_alpha8_seed42.yaml round4_E1_QKVO_plus_GK_last6_r8_alpha8_seed42.yaml"
ROUND_FILES[5]="round5_QKVO_r16_a16_seed127.yaml round5_QKVO_plus_G_r16_a16_seed127.yaml round5_OMLP_r8_a8_seed127.yaml round5_OMLP_plus_G_r8_a8_seed127.yaml round5_QKVO_RSLoRA_r8_a8_seed127.yaml round5_QKVO_plus_G_RSLoRA_r8_a8_seed127.yaml round2_E1_QKVO_seed_r8_alpha8_seed127.yaml"

# -------- Dynamic helpers (no hardcoding of 1..4) --------

# Return 0 if round 'r' is defined in ROUND_FILES (even if empty string), else 1.
validate_round() {
  local r="$1"
  [[ -v "ROUND_FILES[$r]" ]]
}

# Fill SELECT_SET with the config list for round r; return non-zero if none.
get_round_configs() {
  local r="$1"
  local raw="${ROUND_FILES[$r]-}"
  if [[ -z "$raw" ]]; then
    return 1
  fi
  IFS=' ' read -r -a SELECT_SET <<< "$raw"
  # Export as a nameref/global (Bash scope): caller will read SELECT_SET[@]
  # Here we just rely on global SELECT_SET used right away by run_round.
  return 0
}

# Get sorted list of defined rounds (keys of ROUND_FILES), stored into global DEFINED_ROUNDS array.
declare -a DEFINED_ROUNDS=()
mapfile -t DEFINED_ROUNDS < <(printf "%s\n" "${!ROUND_FILES[@]}" | LC_ALL=C sort -n)

# Pretty string for error messages.
defined_rounds_str() {
  printf "%s " "${DEFINED_ROUNDS[@]}"
}

run_round () {
  local r="$1"

  # Load configs for this round (dynamic count, not forced to 7)
  if ! get_round_configs "$r"; then
    echo "Round ${r} is defined but empty. Please populate ROUND_FILES[${r}]."
    return 1
  fi

  # Verify existence of all configs to avoid half-start
  local missing=()
  for f in "${SELECT_SET[@]}"; do
    [[ -f "$BASE_DIR/$f" ]] || missing+=("$BASE_DIR/$f")
  done
  if (( ${#missing[@]} > 0 )); then
    echo "Missing configs:"; printf '%s\n' "${missing[@]}"; return 1
  fi

  local num_jobs="${#SELECT_SET[@]}"
  echo "=== Starting Round ${r} (${num_jobs} jobs) ==="

  # Build GPU list:
  #  - If GPU_IDS env provided, use it (must have at least num_jobs IDs).
  #  - Else default to 0..(num_jobs-1)
  local -a GPUS=()
  if [[ -n "${GPU_IDS:-}" ]]; then
    # shellcheck disable=SC2206
    GPUS=(${GPU_IDS})
    if (( ${#GPUS[@]} < num_jobs )); then
      echo "GPU_IDS provides ${#GPUS[@]} GPUs but ${num_jobs} jobs are requested."
      return 1
    fi
  else
    for ((i=0;i<num_jobs;i++)); do GPUS+=("$i"); done
  fi

  PIDS=()  # reset PIDS for this round

  # Launch jobs
  for i in "${!SELECT_SET[@]}"; do
    local CFG="$BASE_DIR/${SELECT_SET[$i]}"
    local GPU="${GPUS[$i]}"

    # Parse seed from filename (debug only); actual training uses FORCE_SEED
    local SEED_FROM_NAME
    SEED_FROM_NAME="$(echo "${SELECT_SET[$i]}" | sed -n 's/.*_seed\([0-9][0-9]*\)\.yaml/\1/p')"

    echo "[GPU ${GPU}] ${CFG} (seed ${FORCE_SEED})"
    HP_SEED=${FORCE_SEED} CUDA_VISIBLE_DEVICES=$GPU python train.py --cfg "$CFG" --overwrite &
    PIDS+=("$!")
  done

  # Wait each PID and track failures explicitly
  local any_failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      any_failed=1
    fi
  done

  if (( any_failed )); then
    return 1
  fi

  echo "ROUND=${r} TASK=${TASK} SEED=${SEED} finished."
  return 0
}

# -------------------------
# Build the run queue
# -------------------------
if (( $# == 0 )); then
  # backward compat: no args -> use default ROUND (arg1 defaulted to 1)
  RUN_QUEUE+=("$ROUND")
else
  for arg in "$@"; do
    if [[ "$arg" == "all" ]]; then
      # Expand to all defined rounds (sorted)
      RUN_QUEUE+=("${DEFINED_ROUNDS[@]}")
    else
      RUN_QUEUE+=("$arg")
    fi
  done
fi

# Validate run queue dynamically against ROUND_FILES keys
for r in "${RUN_QUEUE[@]}"; do
  if ! validate_round "$r"; then
    echo "Invalid round '$r'. Valid values: $(defined_rounds_str)or 'all'."
    exit 1
  fi
done

# -------------------------
# Execute strictly in order
# -------------------------
for r in "${RUN_QUEUE[@]}"; do
  CURRENT_ROUND="$r"
  if run_round "$r"; then
    COMPLETED_ROUNDS+=("$r")
    CURRENT_ROUND=""
  else
    FAILED_ROUND="$r"

    # Ensure no stray jobs remain for this round (PID-based)
    for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
    # Fallback kill by command line pattern (base-dir scoped, as requested)
    pkill -f -- "train.py --cfg ${BASE_DIR}/" 2>/dev/null || true

    print_failure_summary
    exit 1
  fi
done

# All requested rounds completed successfully (no extra summary to keep original behavior)
exit 0