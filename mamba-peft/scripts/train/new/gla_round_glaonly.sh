#!/usr/bin/env bash
set -euo pipefail
#
# Minimal round launcher that reuses dynamic YAML injection but calls train_gla_only.py
# Usage:
#   ./gla_round_glaonly.sh --suite E5 --round all --pairs "87:spider-tvt" --gpus "7" --gpu-plan "1"
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PEFT_ROOT"

LAUNCHER_PY="train_gla_only.py"

# Defaults
SUITE="E2"
ROUND="all"
PAIRS=""
SESSION_NAME=""
LOG_DIR="${SCRIPT_DIR}/logs"
CFG_PATH="cfg/my_lora_exp/yaml/E1_QKVO_r8_alpha16.yaml"

print_help() {
  cat <<'EOF'
Usage:
  gla_round_glaonly.sh --suite <E1|E2|...> --round <N|all> --pairs "SEED:DATA[,SEED:DATA ...]" [--gpus "ID ..."] [--gpu-plan "N,N,..."]

Flags:
  --suite     Suite name (only used for log naming)
  --round     Round label (only used for log naming)
  --pairs     Comma- or space-separated list of seed:data (e.g., "87:spider-tvt")
  --gpus      Space/comma separated GPU IDs to use (default: CUDA_VISIBLE_DEVICES as-is)
  --gpu-plan  Ignored; present for compatibility
  -h|--help   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)  SUITE="$2"; shift 2;;
    --round)  ROUND="$2"; shift 2;;
    --pairs)  PAIRS="$2"; shift 2;;
    --gpus)   export CUDA_VISIBLE_DEVICES="$2"; shift 2;;
    --cfg)    CFG_PATH="$2"; shift 2;;
    --gpu-plan) shift 2;; # compatibility no-op
    -h|--help) print_help; exit 0;;
    *) echo "Unknown arg: $1" >&2; print_help; exit 2;;
  esac
done

if [[ -z "$PAIRS" ]]; then
  echo "ERROR: --pairs is required" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
PAIRS_NORM="$(echo "$PAIRS" | tr ',' ' ')"

# Iterate over jobs sequentially; one process at a time
idx=0
for item in $PAIRS_NORM; do
  idx=$((idx+1))
  seed="${item%%:*}"
  data="${item#*:}"
  ts="$(date +%m%d_%H%M%S)"
  log_file="${LOG_DIR}/glaonly_${SUITE}_${ROUND}_step${idx}_s${seed}_${data}_${ts}.log"

  echo "[${ts}] START seed=${seed} data=${data} -> ${log_file}"
  HP_SEED="$seed" HP_DATA="$data" \
    python "$LAUNCHER_PY" --cfg "$CFG_PATH" 2>&1 | tee "$log_file"
done

echo "All gla-only jobs finished."


