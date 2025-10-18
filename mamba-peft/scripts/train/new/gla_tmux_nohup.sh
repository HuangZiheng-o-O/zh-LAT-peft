#!/usr/bin/env bash
set -euo pipefail

# Wrapper: tmux + nohup + logfile around gla_round_new.sh
# - Rewrites FORCE_SEED=<seed> on a TEMP COPY of gla_round_new.sh (original is untouched).
# - Injects dataset via env DATA=<data> (works with the minimal-patched launcher you downloaded).
# - Starts each run in its own tmux session, logs to ./logs/<session>.log
#
# Usage examples:
#   # Example 1 :
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_tmux_nohup.sh --suite E2 --round all --seed 127 --data glue-tvt_mrpc
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_tmux_nohup.sh --suite E2 --round all --seed 87  --data glue-tvt_mrpc
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh E2 all
#   # With a custom session name and log directory:
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_tmux_nohup.sh --suite E2 --round all --seed 127 --data glue-tvt_mrpc --name exp_AAA --logdir /path/to/logs
#   # Per-GPU settings:
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_tmux_nohup.sh --suite E2 --round all --data glue-tvt_mrpc --gpus "0 1 2 3 5 6" --gpu-plan "3,3,3,3,0,3,3"
#
# Notes:
# - Requires: tmux, nohup.
# - This script must sit alongside gla_round_new.sh.
# - If you use my minimal-patched launcher, DATA will be injected into a temp YAML automatically.
#   If you use the original unmodified launcher, make sure each YAML already specifies the dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/gla_round_new.sh"

if [[ ! -x "$LAUNCHER" ]]; then
  echo "ERROR: gla_round_new.sh not found or not executable at: $LAUNCHER" >&2
  exit 1
fi

# Defaults
SUITE="E2"
ROUND="all"
SEED="127"
DATA_VAL="glue-tvt_cola"
SESSION_NAME=""
#LOG_DIR="${SCRIPT_DIR}/logs"
LOG_DIR="/home/user/mzs_h/log"
print_help() {
  cat <<'EOF'
Usage:
  gla_tmux_nohup.sh --suite <E1|E2|...> --round <N|all> --seed <int> --data <name> [--name <session>] [--logdir <dir>]

Flags:
  --suite    Suite name passed to gla_round_new.sh (default: E2)
  --round    Round index or 'all' (default: all)
  --seed     Seed to force (overrides FORCE_SEED inside a temp copy) (default: 127)
  --data     Dataset code/name injected via env DATA=... (default: glue-tvt_cola)
  --name     Optional tmux session name (auto-generated if omitted)
  --logdir   Log directory (default: ./logs next to this wrapper)
  --gpus     Space- or comma-separated GPU IDs (overrides auto-detect)
  --gpu-plan Comma/space ints per GPU concurrency (e.g. "3,3,3,3,0,3,3" or single int)
  -h, --help Show this help

Examples:
  ./gla_tmux_nohup.sh --suite E2 --round all --seed 127 --data AAA
  ./gla_tmux_nohup.sh --suite E2 --round all --seed 87  --data BBB
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)  SUITE="$2"; shift 2;;
    --round)  ROUND="$2"; shift 2;;
    --seed)   SEED="$2"; shift 2;;
    --data)   DATA_VAL="$2"; shift 2;;
    --name)   SESSION_NAME="$2"; shift 2;;
    --logdir) LOG_DIR="$2"; shift 2;;
    --gpus)   export GPU_IDS="$2"; shift 2;;
    --gpu-plan) export GPU_PLAN="$2"; shift 2;;
    -h|--help) print_help; exit 0;;
    *) echo "Unknown arg: $1" >&2; print_help; exit 2;;
  esac
done

mkdir -p "$LOG_DIR"

# Auto-generate a session name if not provided
if [[ -z "$SESSION_NAME" ]]; then
  ts="$(date +%m%d_%H%M%S)"
  SESSION_NAME="gla_${SUITE}_${ROUND}_s${SEED}_${DATA_VAL}_${ts}"
fi

LOG_FILE="${LOG_DIR}/${SESSION_NAME}.log"

# Create a temp copy of the launcher with FORCE_SEED replaced by the requested seed.
TMP_LAUNCHER="$(mktemp /tmp/gla_round_XXXXXX.sh)"
# Replace a line that starts with FORCE_SEED=...; leave the rest unchanged.
awk -v s="$SEED" '{
  if ($0 ~ /^FORCE_SEED=/) { print "FORCE_SEED=" s }
  else                     { print $0 }
}' "$LAUNCHER" > "$TMP_LAUNCHER"
chmod +x "$TMP_LAUNCHER"

echo "==> tmux session: $SESSION_NAME"
echo "==> log file    : $LOG_FILE"
echo "==> suite/round : $SUITE / $ROUND"
echo "==> seed        : $SEED"
echo "==> data        : $DATA_VAL"
echo ""

# Start in tmux with nohup; export DATA so that the minimal-patched launcher can inject dataset.
CMD="cd \"$SCRIPT_DIR\"; \nstart_epoch=\$(date +%s); start_iso=\$(date +%F_%T); echo \"[\$start_iso] SESSION=$SESSION_NAME START\" | tee -a \"$LOG_FILE\"; \nDATA=\"$DATA_VAL\" bash \"$TMP_LAUNCHER\" \"$SUITE\" \"$ROUND\" 2>&1 | tee -a \"$LOG_FILE\"; \nend_epoch=\$(date +%s); end_iso=\$(date +%F_%T); elapsed=\$(( end_epoch - start_epoch )); h=\$(( elapsed / 3600 )); m=\$(( (elapsed % 3600) / 60 )); s=\$(( elapsed % 60 )); printf '[%s] SESSION=%s END elapsed=%02d:%02d:%02d (%ds)\n' \"$end_iso\" \"$SESSION_NAME\" \"$h\" \"$m\" \"$s\" \"$elapsed\" | tee -a \"$LOG_FILE\""
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session already exists: $SESSION_NAME" >&2
  exit 3
fi

tmux new-session -s "$SESSION_NAME" "$CMD"

echo ""
echo "tmux session '$SESSION_NAME' has finished or been detached."
echo "To re-attach: tmux attach -t \"$SESSION_NAME\""
echo "Or tail the log:"
echo "  tail -f \"$LOG_FILE\""

# Cleanup temp launcher when the tmux session exits (best-effort)
tmux send-keys -t "$SESSION_NAME" "trap 'rm -f \"$TMP_LAUNCHER\"' EXIT" C-m >/dev/null 2>&1 || true
