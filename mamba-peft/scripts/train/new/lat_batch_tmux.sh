#!/usr/bin/env bash
set -euo pipefail
#
# lat_batch_tmux.sh
# Unified Linear Attention batch training script.
# Supports GLA, RetNet, Mamba2 and other FLA models through MODEL_TYPE env var.
#
# Based on gla_batch_tmux_clean.sh with MODEL_TYPE generalization.
# When MODEL_TYPE is unset or "gla", behavior is identical to gla_batch_tmux_clean.sh.
#
# Usage examples:
#   # GLA (default, backward compatible)
#   ./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_qnli"
#
#   # RetNet
#   MODEL_TYPE=retnet ./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_qnli"
#
#   # Mamba2
#   ./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_qnli" --model-type mamba2
#
#   # SD-LoRA training (two-phase sparse dimension tuning)
#   HP_PEFT_TYPE=sdlora ./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_cola"
#
# Requirements: tmux, awk, nohup. Place this script next to lat_round.sh.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/lat_round.sh"

if [[ ! -x "$LAUNCHER" ]]; then
  echo "ERROR: lat_round.sh not found or not executable at: $LAUNCHER" >&2
  exit 1
fi

# Defaults
SUITE="E2"
ROUND="all"
PAIRS=""
SESSION_NAME=""
LOG_DIR="/home/user/mzs_h/log"
MODEL_TYPE="${MODEL_TYPE:-auto}"  # NEW: Model type (gla, retnet, mamba2, auto)
MODEL_PATH="${LAT_MODEL:-${GLA_MODEL:-}}"
MODEL_PREC="${LAT_PREC:-${HP_PREC:-}}"

print_help() {
  cat <<'EOF'
Usage:
  lat_batch_tmux.sh --suite <E1|E2|...> --round <N|all> --pairs "SEED:DATA[,SEED:DATA ...]" [options]

Options:
  --suite       Suite passed to launcher (default: E2)
  --round       Round index or 'all' (default: all)
  --pairs       Comma- or space-separated list of seed:data, e.g. "127:AAA,87:BBB"
  --model-type  Model type: gla, retnet, mamba2, auto (default: auto)
  --model       Base model path or HF id (overrides YAML model field)
  --prec        Precision override passed to train_lat.py (bf16|fp16|fp32)
  --name        Optional tmux session name (auto-generated if omitted)
  --logdir      Where to store logs (default: /home/user/mzs_h/log)
  --gpus        Space- or comma-separated GPU IDs (overrides auto-detect)
  --gpu-plan    Comma/space ints per GPU concurrency (e.g. "1,1,1" or single int)
  --pissa-fast  Enable fast PiSSA init
  -h, --help    Show this help

Environment:
  MODEL_TYPE    Can also be set via environment variable

Example:
  # GLA training (default)
  ./lat_batch_tmux.sh --suite E15 --round all --pairs "87:spider-tvt"

  # RetNet training
  ./lat_batch_tmux.sh --suite E15 --round all --pairs "87:spider-tvt" --model-type retnet

  # Mamba2 training
  MODEL_TYPE=mamba2 ./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_cola"
EOF
}

# Parse args
PISSA_FAST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)      SUITE="$2"; shift 2;;
    --round)      ROUND="$2"; shift 2;;
    --pairs)      PAIRS="$2"; shift 2;;
    --model-type) MODEL_TYPE="$2"; shift 2;;
    --model)      MODEL_PATH="$2"; shift 2;;
    --prec)       MODEL_PREC="$2"; HP_PREC="$2"; shift 2;;
    --name)       SESSION_NAME="$2"; shift 2;;
    --logdir)     LOG_DIR="$2"; shift 2;;
    --gpus)       export GPU_IDS="$2"; shift 2;;
    --gpu-plan)   export GPU_PLAN="$2"; shift 2;;
    --pissa-fast) PISSA_FAST=1; shift 1;;
    -h|--help)    print_help; exit 0;;
    *)            echo "Unknown arg: $1" >&2; print_help; exit 2;;
  esac
done

if [[ -z "$PAIRS" ]]; then
  echo "ERROR: --pairs is required (e.g., --pairs \"127:AAA,87:BBB\")" >&2
  exit 2
fi

# Export MODEL_TYPE for child processes
export MODEL_TYPE
LAT_MODEL="$MODEL_PATH"
LAT_PREC="$MODEL_PREC"
if [[ -n "${LAT_PREC:-}" ]]; then
  HP_PREC="$LAT_PREC"
fi
if [[ -n "${LAT_MODEL:-}" ]]; then
  export LAT_MODEL
fi
if [[ -n "${LAT_PREC:-}" ]]; then
  export LAT_PREC
fi

# Normalize separators -> space list
PAIRS_NORM="$(echo "$PAIRS" | tr ',' ' ')"
mkdir -p "$LOG_DIR"

# Auto session name
if [[ -z "$SESSION_NAME" ]]; then
  ts="$(date +%m%d_%H%M%S)"
  SESSION_NAME="batch_lat_${MODEL_TYPE}_${SUITE}_${ROUND}_${ts}"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION_NAME" >&2
  exit 3
fi

# Build a small runner script that will execute sequentially inside tmux.
RUNNER="$(mktemp /tmp/lat_batch_runner_XXXXXX.sh)"
chmod +x "$RUNNER"

{
  cat <<'HDR'
#!/usr/bin/env bash
set -euo pipefail
cleanup_tmpfiles=()
on_exit(){ for f in "${cleanup_tmpfiles[@]:-}"; do rm -f "$f" 2>/dev/null || true; done; }
trap on_exit EXIT
HDR

  printf 'export SCRIPT_DIR=%q\n' "$SCRIPT_DIR"
  printf 'export LAUNCHER=%q\n' "$LAUNCHER"
  printf 'export SUITE=%q\n' "$SUITE"
  printf 'export ROUND=%q\n' "$ROUND"
  printf 'export LOG_DIR=%q\n' "$LOG_DIR"
  printf 'export MODEL_TYPE=%q\n' "$MODEL_TYPE"
  printf 'export GPU_IDS=%q\n' "${GPU_IDS:-}"
  printf 'export GPU_PLAN=%q\n' "${GPU_PLAN:-}"
  printf 'export PISSA_FAST=%q\n' "${PISSA_FAST:-0}"
  printf 'export LAT_MODEL=%q\n' "${LAT_MODEL:-}"
  printf 'export LAT_PREC=%q\n' "${LAT_PREC:-}"

  # SwanLab env
  printf 'export SWANLAB_ENABLE=%q\n' "${SWANLAB_ENABLE:-}"
  printf 'export SWANLAB_MODE=%q\n' "${SWANLAB_MODE:-}"
  printf 'export SWANLAB_PROJECT=%q\n' "${SWANLAB_PROJECT:-}"
  printf 'export SWANLAB_EXPERIMENT_PREFIX=%q\n' "${SWANLAB_EXPERIMENT_PREFIX:-}"
  printf 'export SWANLAB_LOGDIR=%q\n' "${SWANLAB_LOGDIR:-/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/my_swanlog/local_eval_logs}"
  printf 'export SWANLAB_EMAIL_YAML=%q\n' "${SWANLAB_EMAIL_YAML:-}"
  printf 'export SWANLAB_EMAIL_ON_START=%q\n' "${SWANLAB_EMAIL_ON_START:-}"
  printf 'export SWANLAB_EMAIL_ON_FINISH=%q\n' "${SWANLAB_EMAIL_ON_FINISH:-}"

  # LAT/GLA toggles (support both prefixes)
  printf 'export LAT_FORCE_LEFT_PAD=%q\n' "${LAT_FORCE_LEFT_PAD:-${GLA_FORCE_LEFT_PAD:-}}"
  printf 'export LAT_USE_MAX_NEW_TOKENS=%q\n' "${LAT_USE_MAX_NEW_TOKENS:-${GLA_USE_MAX_NEW_TOKENS:-}}"
  printf 'export LAT_VERBOSE=%q\n' "${LAT_VERBOSE:-${GLA_VERBOSE:-}}"
  printf 'export LAT_USE_FUSED_SWIGLU=%q\n' "${LAT_USE_FUSED_SWIGLU:-${GLA_USE_FUSED_SWIGLU:-}}"
  # Also export GLA_* for backward compatibility
  printf 'export GLA_FORCE_LEFT_PAD=%q\n' "${GLA_FORCE_LEFT_PAD:-}"
  printf 'export GLA_USE_MAX_NEW_TOKENS=%q\n' "${GLA_USE_MAX_NEW_TOKENS:-}"
  printf 'export GLA_VERBOSE=%q\n' "${GLA_VERBOSE:-}"
  printf 'export GLA_USE_FUSED_SWIGLU=%q\n' "${GLA_USE_FUSED_SWIGLU:-}"

  # HP_* hyperparameters
  printf 'export HP_EVAL_STEPS=%q\n' "${HP_EVAL_STEPS:-}"
  printf 'export HP_SAVE_STEPS=%q\n' "${HP_SAVE_STEPS:-}"
  printf 'export HP_LOGGING_STEPS=%q\n' "${HP_LOGGING_STEPS:-}"
  printf 'export HP_VAL_SPLIT=%q\n' "${HP_VAL_SPLIT:-}"
  printf 'export HP_DATA=%q\n' "${HP_DATA:-}"
  printf 'export HP_LR=%q\n' "${HP_LR:-}"
  printf 'export HP_BATCH_SIZE=%q\n' "${HP_BATCH_SIZE:-}"
  printf 'export HP_EPOCHS=%q\n' "${HP_EPOCHS:-}"
  printf 'export HP_EVAL_BATCH_SIZE=%q\n' "${HP_EVAL_BATCH_SIZE:-}"
  printf 'export HP_NO_SAVE=%q\n' "${HP_NO_SAVE:-}"

  # PEFT configuration (LoRA / SD-LoRA)
  printf 'export HP_PEFT_TYPE=%q\n' "${HP_PEFT_TYPE:-}"
  printf 'export HP_PEFT_R=%q\n' "${HP_PEFT_R:-}"
  printf 'export HP_PEFT_ALPHA=%q\n' "${HP_PEFT_ALPHA:-}"
  printf 'export HP_PEFT_DROPOUT=%q\n' "${HP_PEFT_DROPOUT:-}"
  printf 'export HP_INIT=%q\n' "${HP_INIT:-}"
  printf 'export HP_USE_DORA=%q\n' "${HP_USE_DORA:-}"
  printf 'export HP_USE_RSLoRA=%q\n' "${HP_USE_RSLoRA:-}"

  # SD-LoRA specific parameters
  printf 'export HP_WARMUP_IT=%q\n' "${HP_WARMUP_IT:-}"
  printf 'export HP_ZERO_RATIO=%q\n' "${HP_ZERO_RATIO:-}"
  printf 'export HP_FREEZE_RATIO=%q\n' "${HP_FREEZE_RATIO:-}"

  # LR scheduler
  printf 'export LR_SCHEDULER_TYPE=%q\n' "${LR_SCHEDULER_TYPE:-cosine}"
  printf 'export LR_WARMUP_STEPS=%q\n' "${LR_WARMUP_STEPS:-}"
  printf 'export LR_WARMUP_RATIO=%q\n' "${LR_WARMUP_RATIO:-0.1}"

  # EVAL_GEN parameters
  printf 'export EVAL_GEN=%q\n' "${EVAL_GEN:-}"
  printf 'export EVAL_GEN_MAX_LENGTH=%q\n' "${EVAL_GEN_MAX_LENGTH:-}"
  printf 'export EVAL_GEN_MIN_LENGTH=%q\n' "${EVAL_GEN_MIN_LENGTH:-}"
  printf 'export EVAL_GEN_NUM_BEAMS=%q\n' "${EVAL_GEN_NUM_BEAMS:-}"

  # Launch staggering (support both LAT_* and GLA_*)
  printf 'export LAT_LAUNCH_STAGGER_MINUTES=%q\n' "${LAT_LAUNCH_STAGGER_MINUTES:-${GLA_LAUNCH_STAGGER_MINUTES:-}}"
  printf 'export GLA_LAUNCH_STAGGER_MINUTES=%q\n' "${GLA_LAUNCH_STAGGER_MINUTES:-}"

  # Other common env vars
  printf 'export GRADIENT_CHECKPOINTING=%q\n' "${GRADIENT_CHECKPOINTING:-}"
  printf 'export LOGITS_TO_KEEP=%q\n' "${LOGITS_TO_KEEP:-}"
  printf 'export NUM_DATA_WORKERS=%q\n' "${NUM_DATA_WORKERS:-}"
  printf 'export PYTORCH_CUDA_ALLOC_CONF=%q\n' "${PYTORCH_CUDA_ALLOC_CONF:-}"
  printf 'export TOKENIZERS_PARALLELISM=%q\n' "${TOKENIZERS_PARALLELISM:-}"
  printf 'export OMP_NUM_THREADS=%q\n' "${OMP_NUM_THREADS:-}"
  printf 'export MKL_NUM_THREADS=%q\n' "${MKL_NUM_THREADS:-}"

  # Data roots
  printf 'export SPIDER_LOCAL_DIR=%q\n' "${SPIDER_LOCAL_DIR:-}"
  printf 'export NLTK_DATA=%q\n' "${NLTK_DATA:-}"
  printf 'export SAMSUM_LOCAL_DIR=%q\n' "${SAMSUM_LOCAL_DIR:-}"

  echo 'mkdir -p "$LOG_DIR"'

  # Emit the job list as an array of "SEED:DATA"
  echo 'declare -a JOBS=('
  for pair in $PAIRS_NORM; do
    printf '  %q\n' "$pair"
  done
  echo ')'

  cat <<'BODY'

echo "== Batch plan (MODEL_TYPE=${MODEL_TYPE}) =="
for j in "${JOBS[@]}"; do echo "  - $j"; done
echo ""

idx=0
for item in "${JOBS[@]}"; do
  idx=$((idx+1))
  seed="${item%%:*}"
  data="${item#*:}"
  ts="$(date +%m%d_%H%M%S)"
  sess_step="step${idx}_${MODEL_TYPE}_s${seed}_${data}_${ts}"
  log_file="${LOG_DIR}/${sess_step}.log"

  # Prepare temp launcher with FORCE_SEED replaced
  tmp_launcher="$(mktemp /tmp/lat_round_XXXXXX.sh)"
  awk -v s="$seed" '{
    if ($0 ~ /^FORCE_SEED=/) { print "FORCE_SEED=" s }
    else                     { print $0 }
  }' "$LAUNCHER" > "$tmp_launcher"
  chmod +x "$tmp_launcher"
  cleanup_tmpfiles+=("$tmp_launcher")

  echo "[$(date +%F_%T)] START idx=${idx} model=${MODEL_TYPE} seed=${seed} data=${data}  -> ${log_file}"
  job_start_epoch="$(date +%s)"
  (
    cd "$SCRIPT_DIR"
    GPU_IDS="$GPU_IDS" GPU_PLAN="$GPU_PLAN" DATA="$data" MODEL_TYPE="$MODEL_TYPE" \
      HP_PISSA_FAST="$PISSA_FAST" \
      SWANLAB_ENABLE="$SWANLAB_ENABLE" SWANLAB_MODE="$SWANLAB_MODE" \
      SWANLAB_PROJECT="$SWANLAB_PROJECT" SWANLAB_EXPERIMENT_PREFIX="$SWANLAB_EXPERIMENT_PREFIX" \
      SWANLAB_LOGDIR="$SWANLAB_LOGDIR" SWANLAB_EMAIL_YAML="$SWANLAB_EMAIL_YAML" \
      SPIDER_LOCAL_DIR="$SPIDER_LOCAL_DIR" NLTK_DATA="$NLTK_DATA" \
      bash "$tmp_launcher" "$SUITE" "$ROUND" 2>&1 | tee "$log_file"
  )
  status=$?
  job_end_epoch="$(date +%s)"
  job_elapsed=$(( job_end_epoch - job_start_epoch ))
  job_h=$(( job_elapsed / 3600 ))
  job_m=$(( (job_elapsed % 3600) / 60 ))
  job_s=$(( job_elapsed % 60 ))
  printf '[%s] END   idx=%s model=%s seed=%s data=%s  status=%s  elapsed=%02d:%02d:%02d (%ds)\n' \
    "$(date +%F_%T)" "$idx" "$MODEL_TYPE" "$seed" "$data" "$status" "$job_h" "$job_m" "$job_s" "$job_elapsed" | tee -a "$log_file"
  if [[ $status -ne 0 ]]; then
    echo "Job failed (idx=${idx}). Stopping the batch." | tee -a "$log_file"
    exit $status
  fi
done

echo "All jobs finished successfully (MODEL_TYPE=${MODEL_TYPE})."
BODY
} > "$RUNNER"

MASTER_LOG="${LOG_DIR}/${SESSION_NAME}.log"

echo "==> tmux session  : $SESSION_NAME"
echo "==> master log    : $MASTER_LOG"
echo "==> model type    : $MODEL_TYPE"
echo "==> suite/round   : $SUITE / $ROUND"
echo "==> jobs (--pairs): $PAIRS_NORM"
echo "==> command       : $0 $*"
echo "==> env (GPU_IDS/GPU_PLAN): GPU_IDS='${GPU_IDS:-}' GPU_PLAN='${GPU_PLAN:-}'"
echo ""

CMD="start_iso=\$(date +%F_%T); echo \"[\$start_iso] BATCH_CMD: $0 $*\" | tee -a \"$MASTER_LOG\"; bash \"$RUNNER\" | tee -a \"$MASTER_LOG\""

echo "Starting batch in new tmux session. Your terminal will be attached."
echo "To detach (and leave it running), press: Ctrl-b d"
echo "To re-attach later, use: tmux attach -t \"$SESSION_NAME\""
sleep 2

tmux new-session -s "$SESSION_NAME" "cd \"$SCRIPT_DIR\"; $CMD"

echo ""
echo "tmux session '$SESSION_NAME' has finished or been detached."
echo "To re-attach: tmux attach -t \"$SESSION_NAME\""
echo "Master log is at: $MASTER_LOG"
echo "tail -n 200 $MASTER_LOG"

tmux send-keys -t "$SESSION_NAME" "trap 'rm -f \"$RUNNER\"' EXIT" C-m >/dev/null 2>&1 || true
