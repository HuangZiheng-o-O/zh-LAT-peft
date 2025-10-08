#!/bin/bash
set -euo pipefail

# Unified launcher + config emitter for GLA LoRA experiments (Rounds 1-4).
# It deterministically writes all PEFT JSON and 28 YAML configs under:
#   cfg/my_lora_exp/
# and launches one round (7 jobs) with global TASK/SEED.
#
# Usage examples:
#   bash scripts/train/new/gla_round_new.sh 1            # Round 1, TASK=cola, SEED=42
#   TASK=rte SEED=127 bash scripts/train/new/gla_round_new.sh 2
#   TASK=cola SEED=42  bash scripts/train/new/gla_round_new.sh 3
#   TASK=cola SEED=42  bash scripts/train/new/gla_round_new.sh 4

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
  exit 130
}
trap cleanup INT TERM

ROUND="${1:-1}"        # 1|2|3|4
TASK="${TASK:-cola}"   # cola|rte|mrpc|sst2|qnli|qqp|mnli
SEED="${SEED:-42}"     # global seed

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
# Use pre-authored, fully resolved configs only (no generation here)
# ---------------------------------------------------------------------------
BASE_DIR="cfg/my_lora_exp"

declare -A ROUND_FILES
ROUND_FILES[1]="round1_E0_ZS_r0_alpha0_seed42.yaml round1_E1_QKVO_r8_alpha8_seed42.yaml round1_E2_OMLP_r8_alpha8_seed42.yaml round1_E3_QV_r8_alpha8_seed42.yaml round1_E4_OONLY_r4_alpha8_seed42.yaml round1_E5_MLPONLY_r8_alpha8_seed42.yaml round1_E6_QKV_r8_alpha8_seed42.yaml"
ROUND_FILES[2]="round2_E7_GONLY_r8_alpha8_seed127.yaml round2_E8_QKVO_G_r8_alpha8_seed127.yaml round2_E1_QKVO_R16_r16_alpha16_seed127.yaml round2_E1_QKVO_seed_r8_alpha8_seed127.yaml round2_E2_OMLP_seed_r8_alpha8_seed127.yaml round2_E3_QV_seed_r8_alpha8_seed127.yaml round2_E4_OONLY_seed_r4_alpha8_seed127.yaml"
ROUND_FILES[3]="round3_E1_QKVO_last6_r8_alpha8_seed42.yaml round3_E2_OMLP_last6_r8_alpha8_seed42.yaml round3_E1_QKVO_first6_r8_alpha8_seed42.yaml round3_E2_OMLP_middle6_r8_alpha8_seed42.yaml round3_E1_QKVO_alpha2r_r8_alpha16_seed42.yaml round3_E1_QKVO_dropout0_r8_alpha8_seed42.yaml round3_E1_QKVO_lr1e-4_r8_alpha8_seed42.yaml"
ROUND_FILES[4]="round4_E1_QKVO_r4_equal_r4_alpha8_seed42.yaml round4_E4_OONLY_r16_equal_r16_alpha16_seed42.yaml round4_E2_OMLP_r6_equal_r6_alpha6_seed42.yaml round4_E1_QKVO_DoRA_r8_alpha8_seed42.yaml round4_E2_OMLP_DoRA_r8_alpha8_seed42.yaml round4_E1_QKVO_RSLoRA_r8_alpha8_seed42.yaml round4_E1_QKVO_plus_GK_last6_r8_alpha8_seed42.yaml"

IFS=' ' read -r -a SELECT_SET <<< "${ROUND_FILES[$ROUND]:-}"
if [[ ${#SELECT_SET[@]} -ne 7 ]]; then echo "ROUND must be 1..4"; exit 1; fi

# Seed discipline: ensure filenames match requested SEED to avoid accidental mismatch
for f in "${SELECT_SET[@]}"; do
  case "$ROUND" in
    2) [[ "$f" == *"seed${SEED}.yaml"* ]] || { echo "Seed mismatch: $f expects 127. Set SEED=127."; exit 1; };;
    *) [[ "$f" == *"seed${SEED}.yaml"* ]] || { echo "Seed mismatch: $f expects 42. Set SEED=42."; exit 1; };;
  esac
done

GPUS=(0 1 2 3 4 5 6)
for i in $(seq 0 6); do
  CFG="$BASE_DIR/${SELECT_SET[$i]}"
  GPU="${GPUS[$i]}"
  [[ -f "$CFG" ]] || { echo "Missing config: $CFG"; exit 1; }
  echo "[GPU ${GPU}] ${CFG}"
  CUDA_VISIBLE_DEVICES=$GPU python train.py --cfg "$CFG" --overwrite &
  PIDS+=("$!")
done
wait
echo "ROUND=${ROUND} TASK=${TASK} SEED=${SEED} finished."


