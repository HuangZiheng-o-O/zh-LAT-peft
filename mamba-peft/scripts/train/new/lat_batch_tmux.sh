#!/usr/bin/env bash
set -euo pipefail
#
# lat_batch_tmux.sh
# Unified Linear Attention batch training script.
# Supports GLA, RetNet, Mamba2 and other FLA models through MODEL_TYPE env var.
#
# This is the canonical batch training script for the LAT framework.
# MODEL_TYPE defaults to "auto" (auto-detect from model config).
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
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
EVAL_TASKS="${EVAL_TASKS:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-}"
EVAL_BACKEND="${EVAL_BACKEND:-}"   # lat | lm_eval (handled in eval_lat.py)

# Train controls (propagated to train_lat.py via lat_round.sh)
# - default behavior historically was overwrite (always start fresh)
TRAIN_OVERWRITE="${LAT_TRAIN_OVERWRITE:-1}"
TRAIN_RESUME="${LAT_TRAIN_RESUME:-0}"

# NEW: Parallelize across --pairs (datasets) for quick smoke-testing.
# - LAT_BATCH_PAIR_CONCURRENCY=1 (default): sequential
# - LAT_BATCH_PAIR_CONCURRENCY=auto: run up to one dataset per GPU (from --gpus / GPU_IDS)
# - LAT_BATCH_PAIR_CONCURRENCY=N: run up to N datasets concurrently
# Each dataset job will be pinned to a single GPU from the GPU list (round-robin).
PAIR_CONCURRENCY="${LAT_BATCH_PAIR_CONCURRENCY:-${BATCH_PAIR_CONCURRENCY:-1}}"
PAIR_GPU_PLAN="${LAT_BATCH_PAIR_GPU_PLAN:-${BATCH_PAIR_GPU_PLAN:-}}"

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
  --eval-after-train  Run eval after each training job (calls eval_lat.py)
  --eval-only   Only run eval_lat.py (no training)
  --eval-backend Eval backend: lat|lm_eval (default: lat)
  --eval-tasks  Comma-separated tasks for eval_lat.py (default handled in eval_lat.py)
  --eval-batch-size  Eval batch size override
  --eval-output-root Where to write eval outputs (default: mamba-peft/outputs/lm_eval/)
  --resume      Resume training if output_dir exists (train_lat.py --resume). Disables overwrite.
  --overwrite   Force overwrite training output_dir (default behavior). Disables resume.
  --pairs-parallel <N|auto>  Run multiple SEED:DATA pairs concurrently (default: 1 / sequential)
  --pair-gpu-plan <N>        Per-dataset GPU_PLAN when pairs-parallel is enabled (default: infer or 1)
  -h, --help    Show this help

Environment:
  MODEL_TYPE    Can also be set via environment variable
  LAT_BATCH_PAIR_CONCURRENCY  1|auto|N (parallelize across --pairs)
  LAT_BATCH_PAIR_GPU_PLAN     Per-pair GPU_PLAN when pairs-parallel is enabled

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
    --eval-after-train) EVAL_AFTER_TRAIN=1; shift 1;;
    --eval-only)  EVAL_ONLY=1; shift 1;;
    --eval-backend) EVAL_BACKEND="$2"; shift 2;;
    --eval-tasks) EVAL_TASKS="$2"; shift 2;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2;;
    --eval-output-root) EVAL_OUTPUT_ROOT="$2"; shift 2;;
    --resume)     TRAIN_RESUME=1; TRAIN_OVERWRITE=0; shift 1;;
    --overwrite)  TRAIN_OVERWRITE=1; TRAIN_RESUME=0; shift 1;;
    --pairs-parallel) PAIR_CONCURRENCY="$2"; shift 2;;
    --pair-gpu-plan)  PAIR_GPU_PLAN="$2"; shift 2;;
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
  printf 'export EVAL_AFTER_TRAIN=%q\n' "${EVAL_AFTER_TRAIN:-0}"
  printf 'export EVAL_ONLY=%q\n' "${EVAL_ONLY:-0}"
  printf 'export EVAL_TASKS=%q\n' "${EVAL_TASKS:-}"
  printf 'export EVAL_BATCH_SIZE=%q\n' "${EVAL_BATCH_SIZE:-}"
  printf 'export EVAL_OUTPUT_ROOT=%q\n' "${EVAL_OUTPUT_ROOT:-}"
  printf 'export EVAL_BACKEND=%q\n' "${EVAL_BACKEND:-}"
  printf 'export LAT_TRAIN_OVERWRITE=%q\n' "${TRAIN_OVERWRITE:-1}"
  printf 'export LAT_TRAIN_RESUME=%q\n' "${TRAIN_RESUME:-0}"
  printf 'export GPU_IDS=%q\n' "${GPU_IDS:-}"
  printf 'export GPU_PLAN=%q\n' "${GPU_PLAN:-}"
  printf 'export LAT_BATCH_PAIR_CONCURRENCY=%q\n' "${PAIR_CONCURRENCY:-1}"
  printf 'export LAT_BATCH_PAIR_GPU_PLAN=%q\n' "${PAIR_GPU_PLAN:-}"
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

  # LAT environment toggles (Python handles LAT_* > GLA_* fallback internally)
  # We merge LAT_* and GLA_* here to support both naming conventions
  printf 'export LAT_FORCE_LEFT_PAD=%q\n' "${LAT_FORCE_LEFT_PAD:-${GLA_FORCE_LEFT_PAD:-1}}"
  printf 'export LAT_USE_MAX_NEW_TOKENS=%q\n' "${LAT_USE_MAX_NEW_TOKENS:-${GLA_USE_MAX_NEW_TOKENS:-1}}"
  printf 'export LAT_VERBOSE=%q\n' "${LAT_VERBOSE:-${GLA_VERBOSE:-0}}"
  printf 'export LAT_USE_FUSED_SWIGLU=%q\n' "${LAT_USE_FUSED_SWIGLU:-${GLA_USE_FUSED_SWIGLU:-0}}"
  printf 'export LAT_LOG_PADDING_STATS=%q\n' "${LAT_LOG_PADDING_STATS:-${GLA_LOG_PADDING_STATS:-0}}"

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

  # LR scheduler
  printf 'export LR_SCHEDULER_TYPE=%q\n' "${LR_SCHEDULER_TYPE:-cosine}"
  printf 'export LR_WARMUP_STEPS=%q\n' "${LR_WARMUP_STEPS:-}"
  printf 'export LR_WARMUP_RATIO=%q\n' "${LR_WARMUP_RATIO:-0.1}"

  # EVAL_GEN parameters
  printf 'export EVAL_GEN=%q\n' "${EVAL_GEN:-}"
  printf 'export EVAL_GEN_MAX_LENGTH=%q\n' "${EVAL_GEN_MAX_LENGTH:-}"
  printf 'export EVAL_GEN_MIN_LENGTH=%q\n' "${EVAL_GEN_MIN_LENGTH:-}"
  printf 'export EVAL_GEN_NUM_BEAMS=%q\n' "${EVAL_GEN_NUM_BEAMS:-}"

  # Launch staggering (merge LAT_* and GLA_*)
  printf 'export LAT_LAUNCH_STAGGER_MINUTES=%q\n' "${LAT_LAUNCH_STAGGER_MINUTES:-${GLA_LAUNCH_STAGGER_MINUTES:-0}}"

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
# pairs-parallel settings
pair_conc="${LAT_BATCH_PAIR_CONCURRENCY:-1}"
pair_gpu_plan="${LAT_BATCH_PAIR_GPU_PLAN:-}"

# Normalize GPU list for assignment (when pinning one dataset per GPU)
gpu_list_str="${GPU_IDS:-}"
gpu_list_str="${gpu_list_str//,/ }"
declare -a GPU_LIST=()
for tok in $gpu_list_str; do
  [[ -n "$tok" ]] && GPU_LIST+=("$tok")
done
num_gpus="${#GPU_LIST[@]}"

if [[ "$pair_conc" == "auto" ]]; then
  if (( num_gpus > 0 )); then
    pair_conc="$num_gpus"
  else
    pair_conc="1"
  fi
fi
if ! [[ "$pair_conc" =~ ^[0-9]+$ ]]; then
  pair_conc="1"
fi
if (( pair_conc < 1 )); then
  pair_conc="1"
fi

# Infer a single-int GPU_PLAN for per-pair pinning
if [[ -z "$pair_gpu_plan" ]]; then
  plan_str="${GPU_PLAN:-}"
  plan_str="${plan_str//,/ }"
  declare -a _plan_arr=()
  if [[ -n "$plan_str" ]]; then
    read -r -a _plan_arr <<<"$plan_str"
  fi
  if (( ${#_plan_arr[@]} == 0 )); then
    pair_gpu_plan="1"
  elif (( ${#_plan_arr[@]} == 1 )); then
    pair_gpu_plan="${_plan_arr[0]}"
  else
    all_eq=1
    first="${_plan_arr[0]}"
    for v in "${_plan_arr[@]}"; do
      if [[ "$v" != "$first" ]]; then all_eq=0; break; fi
    done
    if (( all_eq )); then
      pair_gpu_plan="$first"
    else
      pair_gpu_plan="1"
      echo "[lat_batch_tmux][warn] GPU_PLAN varies per GPU; set LAT_BATCH_PAIR_GPU_PLAN=<N> for pairs-parallel pinning."
    fi
  fi
fi

supports_wait_n=0
if [[ -n "${BASH_VERSINFO[0]:-}" ]]; then
  if (( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then
    supports_wait_n=1
  fi
fi

declare -a PAIR_PIDS=()

echo "[lat_batch_tmux] pairs-parallel concurrency=${pair_conc} (num_gpus=${num_gpus}, per_pair_gpu_plan=${pair_gpu_plan})"
if (( pair_conc > 1 )) && (( num_gpus == 0 )); then
  echo "[lat_batch_tmux][warn] pairs-parallel requested but no --gpus/GPU_IDS provided; falling back to sequential."
  pair_conc=1
fi

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

  # Default: reuse global GPU_IDS/GPU_PLAN
  job_gpu_ids="${GPU_IDS:-}"
  job_gpu_plan="${GPU_PLAN:-}"

  # In parallel mode, pin each pair to a single GPU (round-robin) + single-int GPU_PLAN
  if (( pair_conc > 1 )); then
    gi=$(( (idx - 1) % num_gpus ))
    job_gpu_ids="${GPU_LIST[$gi]}"
    job_gpu_plan="${pair_gpu_plan}"
  fi

  echo "[$(date +%F_%T)] START idx=${idx} model=${MODEL_TYPE} seed=${seed} data=${data} gpus='${job_gpu_ids}' plan='${job_gpu_plan}' -> ${log_file}"
  (
    job_start_epoch="$(date +%s)"
    (
      cd "$SCRIPT_DIR"
      GPU_IDS="$job_gpu_ids" GPU_PLAN="$job_gpu_plan" DATA="$data" MODEL_TYPE="$MODEL_TYPE" \
        HP_PISSA_FAST="$PISSA_FAST" \
        SWANLAB_ENABLE="$SWANLAB_ENABLE" SWANLAB_MODE="$SWANLAB_MODE" \
        SWANLAB_PROJECT="$SWANLAB_PROJECT" SWANLAB_EXPERIMENT_PREFIX="$SWANLAB_EXPERIMENT_PREFIX" \
        SWANLAB_LOGDIR="$SWANLAB_LOGDIR" SWANLAB_EMAIL_YAML="$SWANLAB_EMAIL_YAML" \
        SPIDER_LOCAL_DIR="$SPIDER_LOCAL_DIR" NLTK_DATA="$NLTK_DATA" \
        bash "$tmp_launcher" "$SUITE" "$ROUND" 2>&1 | sed -u "s/^/[${sess_step}] /" | tee "$log_file"
    )
    status=$?
    job_end_epoch="$(date +%s)"
    job_elapsed=$(( job_end_epoch - job_start_epoch ))
    job_h=$(( job_elapsed / 3600 ))
    job_m=$(( (job_elapsed % 3600) / 60 ))
    job_s=$(( job_elapsed % 60 ))
    printf '[%s] END   idx=%s model=%s seed=%s data=%s  status=%s  elapsed=%02d:%02d:%02d (%ds)\n' \
      "$(date +%F_%T)" "$idx" "$MODEL_TYPE" "$seed" "$data" "$status" "$job_h" "$job_m" "$job_s" "$job_elapsed" | tee -a "$log_file"
    exit "$status"
  ) &
  pid="$!"
  PAIR_PIDS+=("$pid")

  # Sequential mode: wait immediately to keep original fail-fast behavior
  if (( pair_conc <= 1 )); then
    if ! wait "$pid"; then
      exit 1
    fi
    PAIR_PIDS=()
    continue
  fi

  # Parallel mode: enforce concurrency with wait -n (or fallback)
  while (( ${#PAIR_PIDS[@]} >= pair_conc )); do
    if (( supports_wait_n )); then
      if ! wait -n; then
        echo "[lat_batch_tmux][error] One pair job failed. Stopping remaining jobs."
        for p in "${PAIR_PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
        exit 1
      fi
    else
      p0="${PAIR_PIDS[0]}"
      if ! wait "$p0"; then
        echo "[lat_batch_tmux][error] One pair job failed. Stopping remaining jobs."
        for p in "${PAIR_PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
        exit 1
      fi
    fi
    # Prune finished PIDs
    new_pids=()
    for p in "${PAIR_PIDS[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        new_pids+=("$p")
      fi
    done
    PAIR_PIDS=("${new_pids[@]:-}")
  done
done

# Wait remaining background jobs in parallel mode
if (( pair_conc > 1 )); then
  if (( supports_wait_n )); then
    while (( ${#PAIR_PIDS[@]} > 0 )); do
      if ! wait -n; then
        echo "[lat_batch_tmux][error] One pair job failed. Stopping remaining jobs."
        for p in "${PAIR_PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
        exit 1
      fi
      new_pids=()
      for p in "${PAIR_PIDS[@]}"; do
        if kill -0 "$p" 2>/dev/null; then
          new_pids+=("$p")
        fi
      done
      PAIR_PIDS=("${new_pids[@]:-}")
    done
  else
    for p in "${PAIR_PIDS[@]}"; do
      wait "$p"
    done
  fi
fi

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
