#!/usr/bin/env bash
set -euo pipefail
#
# gla_batch_tmux_clean.sh
# Run multiple gla_round_clean.sh jobs sequentially in ONE tmux session,
# each job wrapped with nohup and its own log file. Clean GLA-only path.
#
# Usage examples:
#   ./gla_batch_tmux_clean.sh --suite E5 --round all --pairs "127:spider-tvt,87:spider-tvt"
#   ./gla_batch_tmux_clean.sh --suite E5 --round all --pairs "127:spider-tvt 87:spider-tvt" --name batch_glaclean
#   ./gla_batch_tmux_clean.sh --suite E5 --round all --pairs "127:glue-tvt_mrpc 127:glue-tvt_cola" --gpus "0 1" --gpu-plan "1,1"
#
# Requirements: tmux, awk, nohup. Place this script next to gla_round_clean.sh.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/gla_round_clean.sh"

if [[ ! -x "$LAUNCHER" ]]; then
  echo "ERROR: gla_round_clean.sh not found or not executable at: $LAUNCHER" >&2
  exit 1
fi

# Defaults
SUITE="E2"
ROUND="all"
PAIRS=""           # e.g., "127:AAA,87:BBB" or "127:AAA 87:BBB"
SESSION_NAME=""
LOG_DIR="${SCRIPT_DIR}/logs"

print_help() {
  cat <<'EOF'
Usage:
  gla_batch_tmux_clean.sh --suite <E1|E2|...> --round <N|all> --pairs "SEED:DATA[,SEED:DATA ...]" [--name <session>] [--logdir <dir>] [--pissa-fast]

Flags:
  --suite    Suite passed to launcher (default: E2)
  --round    Round index or 'all' (default: all)
  --pairs    Comma- or space-separated list of seed:data, e.g. "127:AAA,87:BBB"
  --name     Optional tmux session name (auto-generated if omitted)
  --logdir   Where to store logs (default: ./logs next to this script)
  --gpus     Space- or comma-separated GPU IDs (overrides auto-detect)
  --gpu-plan Comma/space ints per GPU concurrency (e.g. "1,1,1" or single int)
  --pissa-fast Enable fast PiSSA init (maps init_lora_weights=pissa -> pissa_niter_4 when present)
  -h, --help   Show this help

Example:
  ./gla_batch_tmux_clean.sh --suite E5 --round all --pairs "127:spider-tvt,87:spider-tvt"
EOF
}

# Parse args
PISSA_FAST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)  SUITE="$2"; shift 2;;
    --round)  ROUND="$2"; shift 2;;
    --pairs)  PAIRS="$2"; shift 2;;
    --name)   SESSION_NAME="$2"; shift 2;;
    --logdir) LOG_DIR="$2"; shift 2;;
    --gpus)   export GPU_IDS="$2"; shift 2;;
    --gpu-plan) export GPU_PLAN="$2"; shift 2;;
    --pissa-fast) PISSA_FAST=1; shift 1;;
    -h|--help) print_help; exit 0;;
    *) echo "Unknown arg: $1" >&2; print_help; exit 2;;
  esac
done

if [[ -z "$PAIRS" ]]; then
  echo "ERROR: --pairs is required (e.g., --pairs \"127:AAA,87:BBB\")" >&2
  exit 2
fi

# Normalize separators -> space list
PAIRS_NORM="$(echo "$PAIRS" | tr ',' ' ')"
mkdir -p "$LOG_DIR"

# Auto session name
if [[ -z "$SESSION_NAME" ]]; then
  ts="$(date +%m%d_%H%M%S)"
  SESSION_NAME="batch_clean_${SUITE}_${ROUND}_${ts}"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION_NAME" >&2
  exit 3
fi

# Build a small runner script that will execute sequentially inside tmux.
RUNNER="$(mktemp /tmp/gla_batch_clean_runner_XXXXXX.sh)"
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
  # Capture GPU env into the runner so tmux sessions get the right values even if tmux server env differs
  printf 'export GPU_IDS=%q\n' "${GPU_IDS:-}"
  printf 'export GPU_PLAN=%q\n' "${GPU_PLAN:-}"
  printf 'export PISSA_FAST=%q\n' "${PISSA_FAST:-0}"
  # SwanLab env (optional)
  printf 'export SWANLAB_ENABLE=%q\n' "${SWANLAB_ENABLE:-}"
  printf 'export SWANLAB_MODE=%q\n' "${SWANLAB_MODE:-}"
  printf 'export SWANLAB_PROJECT=%q\n' "${SWANLAB_PROJECT:-}"
  printf 'export SWANLAB_EXPERIMENT_PREFIX=%q\n' "${SWANLAB_EXPERIMENT_PREFIX:-}"
  printf 'export SWANLAB_LOGDIR=%q\n' "${SWANLAB_LOGDIR:-}"
  # GLA-specific toggles
  printf 'export GLA_FORCE_LEFT_PAD=%q\n' "${GLA_FORCE_LEFT_PAD:-}"
  printf 'export GLA_USE_MAX_NEW_TOKENS=%q\n' "${GLA_USE_MAX_NEW_TOKENS:-}"
  printf 'export GLA_VERBOSE=%q\n' "${GLA_VERBOSE:-}"
  # HP_* hyperparameters (eval_steps, logging_steps, etc.)
  printf 'export HP_EVAL_STEPS=%q\n' "${HP_EVAL_STEPS:-}"
  printf 'export HP_SAVE_STEPS=%q\n' "${HP_SAVE_STEPS:-}"
  printf 'export HP_LOGGING_STEPS=%q\n' "${HP_LOGGING_STEPS:-}"
  printf 'export HP_VAL_SPLIT=%q\n' "${HP_VAL_SPLIT:-}"
  printf 'export HP_DATA=%q\n' "${HP_DATA:-}"
  # EVAL_GEN parameters for generation tasks
  printf 'export EVAL_GEN=%q\n' "${EVAL_GEN:-}"
  printf 'export EVAL_GEN_MAX_LENGTH=%q\n' "${EVAL_GEN_MAX_LENGTH:-}"
  printf 'export EVAL_GEN_MIN_LENGTH=%q\n' "${EVAL_GEN_MIN_LENGTH:-}"
  printf 'export EVAL_GEN_NUM_BEAMS=%q\n' "${EVAL_GEN_NUM_BEAMS:-}"
  # Other common env vars
  printf 'export GRADIENT_CHECKPOINTING=%q\n' "${GRADIENT_CHECKPOINTING:-}"
  printf 'export LOGITS_TO_KEEP=%q\n' "${LOGITS_TO_KEEP:-}"
  printf 'export NUM_DATA_WORKERS=%q\n' "${NUM_DATA_WORKERS:-}"
  printf 'export PYTORCH_CUDA_ALLOC_CONF=%q\n' "${PYTORCH_CUDA_ALLOC_CONF:-}"
  printf 'export TOKENIZERS_PARALLELISM=%q\n' "${TOKENIZERS_PARALLELISM:-}"
  printf 'export OMP_NUM_THREADS=%q\n' "${OMP_NUM_THREADS:-}"
  printf 'export MKL_NUM_THREADS=%q\n' "${MKL_NUM_THREADS:-}"
  # Data roots and NLTK resources (ensure they propagate into tmux jobs)
  printf 'export SPIDER_LOCAL_DIR=%q\n' "${SPIDER_LOCAL_DIR:-}"
  printf 'export NLTK_DATA=%q\n' "${NLTK_DATA:-}"

  echo 'mkdir -p "$LOG_DIR"'

  # Emit the job list as an array of "SEED:DATA"
  echo 'declare -a JOBS=('
  for pair in $PAIRS_NORM; do
    printf '  %q\n' "$pair"
  done
  echo ')'

  cat <<'BODY'

echo "== Batch plan =="
for j in "${JOBS[@]}"; do echo "  - $j"; done
echo ""

idx=0
for item in "${JOBS[@]}"; do
  idx=$((idx+1))
  seed="${item%%:*}"
  data="${item#*:}"
  ts="$(date +%m%d_%H%M%S)"
  sess_step="step${idx}_s${seed}_${data}_${ts}"
  log_file="${LOG_DIR}/${sess_step}.log"

  # Prepare temp launcher with FORCE_SEED replaced
  tmp_launcher="$(mktemp /tmp/gla_round_clean_XXXXXX.sh)"
  awk -v s="$seed" '{
    if ($0 ~ /^FORCE_SEED=/) { print "FORCE_SEED=" s }
    else                     { print $0 }
  }' "$LAUNCHER" > "$tmp_launcher"
  chmod +x "$tmp_launcher"
  cleanup_tmpfiles+=("$tmp_launcher")

  echo "[$(date +%F_%T)] START idx=${idx} seed=${seed} data=${data}  -> ${log_file}"
  job_start_epoch="$(date +%s)"
  (
    cd "$SCRIPT_DIR"
    GPU_IDS="$GPU_IDS" GPU_PLAN="$GPU_PLAN" DATA="$data" \
      HP_PISSA_FAST="$PISSA_FAST" \
      SWANLAB_ENABLE="$SWANLAB_ENABLE" SWANLAB_MODE="$SWANLAB_MODE" \
      SWANLAB_PROJECT="$SWANLAB_PROJECT" SWANLAB_EXPERIMENT_PREFIX="$SWANLAB_EXPERIMENT_PREFIX" \
      SWANLAB_LOGDIR="$SWANLAB_LOGDIR" \
      SPIDER_LOCAL_DIR="$SPIDER_LOCAL_DIR" NLTK_DATA="$NLTK_DATA" \
      bash "$tmp_launcher" "$SUITE" "$ROUND" 2>&1 | tee "$log_file"
  )
  status=$?
  job_end_epoch="$(date +%s)"
  job_elapsed=$(( job_end_epoch - job_start_epoch ))
  job_h=$(( job_elapsed / 3600 ))
  job_m=$(( (job_elapsed % 3600) / 60 ))
  job_s=$(( job_elapsed % 60 ))
  printf '[%s] END   idx=%s seed=%s data=%s  status=%s  elapsed=%02d:%02d:%02d (%ds)\n' \
    "$(date +%F_%T)" "$idx" "$seed" "$data" "$status" "$job_h" "$job_m" "$job_s" "$job_elapsed" | tee -a "$log_file"
  if [[ $status -ne 0 ]]; then
    echo "Job failed (idx=${idx}). Stopping the batch." | tee -a "$log_file"
    exit $status
  fi
done

echo "All jobs finished successfully."
BODY
} > "$RUNNER"

MASTER_LOG="${LOG_DIR}/${SESSION_NAME}.log"

echo "==> tmux session  : $SESSION_NAME"
echo "==> master log    : $MASTER_LOG"
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
echo "tail -n 50 $MASTER_LOG"

# Runner will clean up its temp launchers at exit; we can also clean the runner script when session ends.
tmux send-keys -t "$SESSION_NAME" "trap 'rm -f \"$RUNNER\"' EXIT" C-m >/dev/null 2>&1 || true
