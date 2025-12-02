#!/usr/bin/env bash
set -euo pipefail
#
# gla_batch_tmux_clean_multidata.sh
# Run multiple data tasks in parallel, each pinned to a dedicated GPU,
# using the same suite (YAML set) and round. Suitable for:
#   GPU 0 -> DATA_A
#   GPU 1 -> DATA_B
#   ...
# Jobs exceeding the number of GPUs are scheduled in waves.
#
# Example:
#   ./gla_batch_tmux_clean_multidata.sh \
#     --suite E158 \
#     --round 1 \
#     --pairs "87:glue-tvt_cola 87:glue-tvt_rte 87:glue-tvt_mnli 87:glue-tvt_qnli" \
#     --gpus "0 1 2 3" \
#     --name glue_multidata
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
PAIRS=""           # space/comma separated "SEED:DATA"
SESSION_NAME=""
#LOG_DIR="${SCRIPT_DIR}/logs"
LOG_DIR="/home/user/mzs_h/log"

print_help() {
  cat <<'EOF'
Usage:
  gla_batch_tmux_clean_multidata.sh --suite <E1|E2|...> --round <N|all> \
    --pairs "SEED:DATA [SEED:DATA ...]" --gpus "ID [ID ...]" [--name <session>] [--logdir <dir>] [--pissa-fast]

Behavior:
  - Launches one job per GPU concurrently (GPU_PLAN=1) so that each GPU runs a distinct DATA task.
  - If pairs > number of GPUs, jobs are scheduled in waves until all complete.
  - Uses the same suite (YAML set) for all pairs; DATA is injected per-job.

Flags:
  --suite      Suite passed to launcher (e.g., E158)
  --round      Round index or 'all'
  --pairs      Space/comma separated list of seed:data, e.g. "87:glue-tvt_cola 87:glue-tvt_rte"
  --gpus       Space/comma separated GPU IDs, e.g. "0 1 2 3"
  --name       Optional tmux session name
  --logdir     Directory for logs (default: /home/user/mzs_h/log)
  --pissa-fast Enable fast PiSSA init (map init_lora_weights=pissa -> pissa_niter_4 when present)
  -h, --help   Show help
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
    --gpus)   GPU_IDS_RAW="$2"; shift 2;;
    --pissa-fast) PISSA_FAST=1; shift 1;;
    -h|--help) print_help; exit 0;;
    *) echo "Unknown arg: $1" >&2; print_help; exit 2;;
  esac
done

if [[ -z "$PAIRS" ]]; then
  echo "ERROR: --pairs is required (e.g., --pairs \"87:glue-tvt_cola 87:glue-tvt_rte\")" >&2
  exit 2
fi
if [[ -z "${GPU_IDS_RAW:-}" ]]; then
  echo "ERROR: --gpus is required (e.g., --gpus \"0 1 2 3\")" >&2
  exit 2
fi

# Normalize whitespace/commas
PAIRS_NORM="$(echo "$PAIRS" | tr ',' ' ')"
GPU_IDS="$(echo "$GPU_IDS_RAW" | tr ',' ' ')"
mkdir -p "$LOG_DIR"

# Auto session name
if [[ -z "$SESSION_NAME" ]]; then
  ts="$(date +%m%d_%H%M%S)"
  SESSION_NAME="multidata_${SUITE}_${ROUND}_${ts}"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION_NAME" >&2
  exit 3
fi

# Build a runner script that launches per-GPU jobs concurrently in waves
RUNNER="$(mktemp /tmp/gla_batch_multidata_runner_XXXXXX.sh)"
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

  # Capture user env that should propagate
  printf 'export GPU_IDS_STR=%q\n' "$GPU_IDS"
  printf 'export PISSA_FAST=%q\n' "${PISSA_FAST:-0}"
  # SwanLab and common env
  printf 'export SWANLAB_ENABLE=%q\n' "${SWANLAB_ENABLE:-}"
  printf 'export SWANLAB_MODE=%q\n' "${SWANLAB_MODE:-}"
  printf 'export SWANLAB_PROJECT=%q\n' "${SWANLAB_PROJECT:-}"
  printf 'export SWANLAB_EXPERIMENT_PREFIX=%q\n' "${SWANLAB_EXPERIMENT_PREFIX:-}"
  printf 'export SWANLAB_LOGDIR=%q\n' "${SWANLAB_LOGDIR:-/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/my_swanlog/local_eval_logs}"
  printf 'export SWANLAB_EMAIL_YAML=%q\n' "${SWANLAB_EMAIL_YAML:-}"
  printf 'export SWANLAB_EMAIL_ON_START=%q\n' "${SWANLAB_EMAIL_ON_START:-}"
  printf 'export SWANLAB_EMAIL_ON_FINISH=%q\n' "${SWANLAB_EMAIL_ON_FINISH:-}"
  printf 'export SWANLAB_EMAIL_ON_INTERRUPT=%q\n' "${SWANLAB_EMAIL_ON_INTERRUPT:-}"
  # GLA toggles
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
  printf 'export HP_NO_SAVE=%q\n' "${HP_NO_SAVE:-}"
  printf 'export HP_LR=%q\n' "${HP_LR:-}"
  printf 'export HP_BATCH_SIZE=%q\n' "${HP_BATCH_SIZE:-}"
  printf 'export HP_EPOCHS=%q\n' "${HP_EPOCHS:-}"
  printf 'export HP_EVAL_BATCH_SIZE=%q\n' "${HP_EVAL_BATCH_SIZE:-}"
  # LR scheduler
  printf 'export LR_SCHEDULER_TYPE=%q\n' "${LR_SCHEDULER_TYPE:-}"
  printf 'export LR_WARMUP_STEPS=%q\n' "${LR_WARMUP_STEPS:-}"
  printf 'export LR_WARMUP_RATIO=%q\n' "${LR_WARMUP_RATIO:-}"
  # EVAL_GEN
  printf 'export EVAL_GEN=%q\n' "${EVAL_GEN:-}"
  printf 'export EVAL_GEN_MAX_LENGTH=%q\n' "${EVAL_GEN_MAX_LENGTH:-}"
  printf 'export EVAL_GEN_MIN_LENGTH=%q\n' "${EVAL_GEN_MIN_LENGTH:-}"
  printf 'export EVAL_GEN_NUM_BEAMS=%q\n' "${EVAL_GEN_NUM_BEAMS:-}"
  # Runtime
  printf 'export GRADIENT_CHECKPOINTING=%q\n' "${GRADIENT_CHECKPOINTING:-}"
  printf 'export LOGITS_TO_KEEP=%q\n' "${LOGITS_TO_KEEP:-}"
  printf 'export NUM_DATA_WORKERS=%q\n' "${NUM_DATA_WORKERS:-}"
  printf 'export PYTORCH_CUDA_ALLOC_CONF=%q\n' "${PYTORCH_CUDA_ALLOC_CONF:-}"
  printf 'export TOKENIZERS_PARALLELISM=%q\n' "${TOKENIZERS_PARALLELISM:-}"
  printf 'export OMP_NUM_THREADS=%q\n' "${OMP_NUM_THREADS:-}"
  printf 'export MKL_NUM_THREADS=%q\n' "${MKL_NUM_THREADS:-}"
  # Data roots and NLTK
  printf 'export SPIDER_LOCAL_DIR=%q\n' "${SPIDER_LOCAL_DIR:-}"
  printf 'export NLTK_DATA=%q\n' "${NLTK_DATA:-}"
  printf 'export SAMSUM_LOCAL_DIR=%q\n' "${SAMSUM_LOCAL_DIR:-}"
  printf 'export DART_LOCAL_DIR=%q\n' "${DART_LOCAL_DIR:-}"

  echo 'mkdir -p "$LOG_DIR"'

  # Build arrays in runner
  echo 'declare -a JOBS=()'
  for pair in $PAIRS_NORM; do
    printf 'JOBS+=(%q)\n' "$pair"
  done
  echo 'declare -a GPUS=()'
  for g in $GPU_IDS; do
    printf 'GPUS+=(%q)\n' "$g"
  done

  cat <<'BODY'

echo "== Multi-data plan =="
echo "  GPUs: ${GPUS[*]}"
echo "  Jobs: ${JOBS[*]}"
echo ""

total_jobs="${#JOBS[@]}"
total_gpus="${#GPUS[@]}"
if (( total_gpus < 1 )); then
  echo "ERROR: No GPUs specified." >&2
  exit 1
fi

wave_start=0
job_idx=0
wave=0

while (( job_idx < total_jobs )); do
  wave=$((wave+1))
  echo "[wave ${wave}] launching up to ${total_gpus} jobs"

  declare -a PIDS=()
  for ((i=0; i<total_gpus && job_idx<total_jobs; i++)); do
    gpu="${GPUS[$i]}"
    item="${JOBS[$job_idx]}"
    seed="${item%%:*}"
    data="${item#*:}"
    ts="$(date +%m%d_%H%M%S)"
    tag="gpu${gpu}_s${seed}_${data// /_}_${ts}"
    log_file="${LOG_DIR}/multidata_${tag}.log"

    # Temp launcher with FORCE_SEED replaced
    tmp_launcher="$(mktemp /tmp/gla_round_clean_XXXXXX.sh)"
    awk -v s="$seed" '{
      if ($0 ~ /^FORCE_SEED=/) { print "FORCE_SEED=" s }
      else                     { print $0 }
    }' "$LAUNCHER" > "$tmp_launcher"
    chmod +x "$tmp_launcher"
    cleanup_tmpfiles+=("$tmp_launcher")

    echo "  -> [GPU ${gpu}] ${data}  (seed=${seed})  log=${log_file}"
    (
      cd "$SCRIPT_DIR"
      GPU_IDS="$gpu" GPU_PLAN="1" DATA="$data" \
        HP_PISSA_FAST="$PISSA_FAST" \
        SWANLAB_ENABLE="$SWANLAB_ENABLE" SWANLAB_MODE="$SWANLAB_MODE" \
        SWANLAB_PROJECT="$SWANLAB_PROJECT" SWANLAB_EXPERIMENT_PREFIX="$SWANLAB_EXPERIMENT_PREFIX" \
        SWANLAB_LOGDIR="$SWANLAB_LOGDIR" SWANLAB_EMAIL_YAML="$SWANLAB_EMAIL_YAML" \
        NLTK_DATA="$NLTK_DATA" SPIDER_LOCAL_DIR="$SPIDER_LOCAL_DIR" SAMSUM_LOCAL_DIR="$SAMSUM_LOCAL_DIR" DART_LOCAL_DIR="$DART_LOCAL_DIR" \
        bash "$tmp_launcher" "$SUITE" "$ROUND" 2>&1 | tee "$log_file"
    ) &
    PIDS+=("$!")
    job_idx=$((job_idx+1))
  done

  # Wait for this wave to finish
  fail=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if (( fail != 0 )); then
    echo "At least one job failed in wave ${wave}. Stopping." >&2
    exit 1
  fi
done

echo "All multi-data jobs completed successfully."
BODY
} > "$RUNNER"

MASTER_LOG="${LOG_DIR}/${SESSION_NAME}.log"

echo "==> tmux session  : $SESSION_NAME"
echo "==> master log    : $MASTER_LOG"
echo "==> suite/round   : $SUITE / $ROUND"
echo "==> pairs         : $PAIRS_NORM"
echo "==> gpus          : $GPU_IDS"
echo ""

CMD="start_iso=\$(date +%F_%T); echo \"[\$start_iso] MULTIDATA_CMD: $0 $*\" | tee -a \"$MASTER_LOG\"; bash \"$RUNNER\" | tee -a \"$MASTER_LOG\""

echo "Starting multi-data batch in new tmux session. Your terminal will be attached."
echo "To detach: Ctrl-b d"
echo "To re-attach later: tmux attach -t \"$SESSION_NAME\""
sleep 2

tmux new-session -s "$SESSION_NAME" "cd \"$SCRIPT_DIR\"; $CMD"

echo ""
echo "tmux session '$SESSION_NAME' has finished or been detached."
echo "Master log is at: $MASTER_LOG"
echo "tail -n 200 $MASTER_LOG"


