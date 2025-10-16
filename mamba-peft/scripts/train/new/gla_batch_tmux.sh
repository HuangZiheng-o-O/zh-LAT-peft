#!/usr/bin/env bash
set -euo pipefail

# gla_batch_tmux.sh
# Run multiple gla_round_new.sh jobs sequentially in ONE tmux session,
# each job wrapped with nohup and its own log file. Original launcher is not modified.
#
# Usage examples:
#   # Two jobs back-to-back:
#   ./gla_batch_tmux.sh --suite E2 --round all --pairs "127:AAA,87:BBB"
#
#   # Space-separated also works:
#   ./gla_batch_tmux.sh --suite E2 --round all --pairs "127:AAA 87:BBB"
#
#   # Custom session name and log dir:
#   ./gla_batch_tmux.sh --suite E2 --round all --pairs "127:AAA,87:BBB" --name batch_exp --logdir /path/to/logs
#   # Per-GPU settings:
#   ./gla_batch_tmux.sh --suite E2 --round all --pairs "127:glue-tvt_mrpc 127:glue-tvt_cola" --gpus "0 1 2 3 5 6" --gpu-plan "3,3,3,3,0,3,3"
#
# Requirements: tmux, awk, nohup. Place this script next to gla_round_new.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/gla_round_new.sh"

if [[ ! -x "$LAUNCHER" ]]; then
  echo "ERROR: gla_round_new.sh not found or not executable at: $LAUNCHER" >&2
  exit 1
fi

# Defaults
SUITE="E2"
ROUND="all"
PAIRS=""           # e.g., "127:AAA,87:BBB" or "127:AAA 87:BBB"
SESSION_NAME=""
#LOG_DIR="${SCRIPT_DIR}/logs"
LOG_DIR="/home/user/mzs_h/log"
print_help() {
  cat <<'EOF'
Usage:
  gla_batch_tmux.sh --suite <E1|E2|...> --round <N|all> --pairs "SEED:DATA[,SEED:DATA ...]" [--name <session>] [--logdir <dir>]

Flags:
  --suite    Suite passed to launcher (default: E2)
  --round    Round index or 'all' (default: all)
  --pairs    Comma- or space-separated list of seed:data pairs, e.g. "127:AAA,87:BBB"
  --name     Optional tmux session name (auto-generated if omitted)
  --logdir   Where to store logs (default: ./logs next to this script)
  --gpus     Space- or comma-separated GPU IDs (overrides auto-detect)
  --gpu-plan Comma/space ints per GPU concurrency (e.g. "3,3,3,3,0,3,3" or single int)
  -h, --help Show this help

Example:
  ./gla_batch_tmux.sh --suite E2 --round all --pairs "127:AAA,87:BBB"
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)  SUITE="$2"; shift 2;;
    --round)  ROUND="$2"; shift 2;;
    --pairs)  PAIRS="$2"; shift 2;;
    --name)   SESSION_NAME="$2"; shift 2;;
    --logdir) LOG_DIR="$2"; shift 2;;
    --gpus)   export GPU_IDS="$2"; shift 2;;
    --gpu-plan) export GPU_PLAN="$2"; shift 2;;
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
  # Compact session name (avoid special chars)
  SESSION_NAME="batch_${SUITE}_${ROUND}_${ts}"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION_NAME" >&2
  exit 3
fi

# Build a small runner script that will execute sequentially inside tmux.
RUNNER="$(mktemp /tmp/gla_batch_runner_XXXXXX.sh)"
chmod +x "$RUNNER"

{
  cat <<'HDR'
#!/usr/bin/env bash
set -euo pipefail
cleanup_tmpfiles=()
on_exit(){ for f in "${cleanup_tmpfiles[@]:-}"; do rm -f "$f" 2>/dev/null || true; done; }
trap on_exit EXIT
HDR

  printf 'SCRIPT_DIR=%q\n' "$SCRIPT_DIR"
  printf 'LAUNCHER=%q\n' "$LAUNCHER"
  printf 'SUITE=%q\n' "$SUITE"
  printf 'ROUND=%q\n' "$ROUND"
  printf 'LOG_DIR=%q\n' "$LOG_DIR"

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
  tmp_launcher="$(mktemp /tmp/gla_round_XXXXXX.sh)"
  awk -v s="$seed" '{
    if ($0 ~ /^FORCE_SEED=/) { print "FORCE_SEED=" s }
    else                     { print $0 }
  }' "$LAUNCHER" > "$tmp_launcher"
  chmod +x "$tmp_launcher"
  cleanup_tmpfiles+=("$tmp_launcher")

  echo "[$(date +%F_%T)] START idx=${idx} seed=${seed} data=${data}  -> ${log_file}"
  # Run the job, teeing output to the log file and stdout.
  (
    cd "$SCRIPT_DIR"
    DATA="$data" bash "$tmp_launcher" "$SUITE" "$ROUND" 2>&1 | tee "$log_file"
  )
  status=$?
  echo "[$(date +%F_%T)] END   idx=${idx} seed=${seed} data=${data}  status=${status}" | tee -a "$log_file"
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
echo ""

CMD="bash \"$RUNNER\" | tee -a \"$MASTER_LOG\""

echo "Starting batch in new tmux session. Your terminal will be attached."
echo "To detach (and leave it running), press: Ctrl-b d"
echo "To re-attach later, use: tmux attach -t \"$SESSION_NAME\""
sleep 3 # Give user time to read

tmux new-session -s "$SESSION_NAME" "cd \"$SCRIPT_DIR\"; $CMD"

# This will be printed after the tmux session is detached or ends.
echo ""
echo "tmux session '$SESSION_NAME' has finished or been detached."
echo "To re-attach: tmux attach -t \"$SESSION_NAME\""
echo "Master log is at: $MASTER_LOG"


# Runner will clean up its temp launchers at exit; we can also clean the runner script when session ends.
tmux send-keys -t "$SESSION_NAME" "trap 'rm -f \"$RUNNER\"' EXIT" C-m >/dev/null 2>&1 || true
