#!/bin/bash
set -euo pipefail

# Round 1 (7 jobs) with fully-resolved configs in this script.
# Effect matches scripts/train/gla_rounds.sh ROUND=1, but all YAML and PEFT JSON
# are deterministically written from here for clarity and reproducibility.

# Usage:
#   bash scripts/train/gla_round1_resolved.sh            # TASK=cola by default
#   TASK=rte bash scripts/train/gla_round1_resolved.sh    # switch task

# ----------------------------- env & workspace -----------------------------
PIDS=()
cleanup() {
  echo "Caught signal, stopping ${#PIDS[@]} jobs..."
  for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
  exit 130
}
trap cleanup INT TERM

TASK="${TASK:-cola}"    # cola | rte | mrpc | sst2 | qnli | qqp | mnli

# Remote workspace used by train.py in this environment (same as original)
PEFT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft"
cd "$PEFT_ROOT"

# Mirrors / caches (identical to original for runtime parity)
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

# ------------------------------- task mapping ------------------------------
declare -A TASK_DIR=(
  [cola]=cola_gla
  [rte]=rte_gla
  [mrpc]=mrpc_gla
  [sst2]=sst2_gla
  [qnli]=qnli_gla
  [qqp]=qqp_gla
  [mnli]=mnli_gla
)
if [[ -z "${TASK_DIR[$TASK]+x}" ]]; then
  echo "Unsupported TASK=$TASK. Use one of: ${!TASK_DIR[@]}"; exit 1
fi

CFG_DIR="cfg/exps/benchmark/glue/${TASK_DIR[$TASK]}"
mkdir -p "$CFG_DIR"

# ----------------------------- PEFT JSON writers ---------------------------
# We inline the exact PEFT target modules used by the original script.
# Files are written deterministically to ensure clarity.

peft_qkvo() {
  # r8 QKVO
  cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj"
  ],
  "modules_to_save": []
}
JSON
}

peft_omlp() {
  # r8 O + MLP
  cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": [
    "attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj"
  ],
  "modules_to_save": []
}
JSON
}

peft_qv() {
  # r8 QV
  cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": [
    "attn.q_proj",
    "attn.v_proj"
  ],
  "modules_to_save": []
}
JSON
}

peft_mlp_only() {
  # r8 MLP-only
  cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": [
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj"
  ],
  "modules_to_save": []
}
JSON
}

peft_qkv() {
  # r8 QKV
  cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj"
  ],
  "modules_to_save": []
}
JSON
}

peft_out_r4() {
  # r4 O-only
  cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 4,
  "lora_alpha": 4,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": [
    "attn.o_proj"
  ],
  "modules_to_save": []
}
JSON
}

# ----------------------------- YAML emit function --------------------------
emit_yaml() {
  local path="$1"; shift
  local batch_size="$1"; shift
  local data_id="$1"; shift
  local learning_rate="$1"; shift
  local model_path="$1"; shift
  local no_save="$1"; shift
  local num_epochs="$1"; shift
  local peft_path="$1"; shift
  local prec="$1"; shift
  local seed_val="${1:-}"
  {
    echo "batch_size: ${batch_size}"
    echo "data: glue-tvt_${data_id}"
    echo "learning_rate: ${learning_rate}"
    echo "model: ${model_path}"
    echo "no_save: ${no_save}"
    echo "num_epochs: ${num_epochs}"
    echo "peft: ${peft_path}"
    echo "prec: ${prec}"
    if [[ -n "${seed_val}" ]]; then
      echo "seed: ${seed_val}"
    fi
  } > "${path}"
}

# ------------------------------- resolve paths -----------------------------
PEFT_DIR="cfg/peft/resolved/r1"
mkdir -p "$PEFT_DIR"

# Pre-write all PEFT JSONs this round uses (deterministically)
PEFT_QKVO_JSON="$PEFT_DIR/lora_gla_qkvo.json";      peft_qkvo      "$PEFT_QKVO_JSON"
PEFT_OMLP_JSON="$PEFT_DIR/lora_gla_omlp.json";      peft_omlp      "$PEFT_OMLP_JSON"
PEFT_QV_JSON="$PEFT_DIR/lora_gla_qv.json";          peft_qv        "$PEFT_QV_JSON"
PEFT_MLP_JSON="$PEFT_DIR/lora_gla_mlp.json";        peft_mlp_only  "$PEFT_MLP_JSON"
PEFT_QKV_JSON="$PEFT_DIR/lora_gla_qkv.json";        peft_qkv       "$PEFT_QKV_JSON"
PEFT_OONLY_R4_JSON="$PEFT_DIR/lora_gla_out_r4.json"; peft_out_r4   "$PEFT_OONLY_R4_JSON"

# Common training knobs (same as original builder)
BATCH_SIZE=4
LR=0.0003
MODEL="/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B"
NO_SAVE=false
EPOCHS=10
PREC=bf16

# ------------------------------- write YAMLs -------------------------------
# E0: zero-shot
emit_yaml "$CFG_DIR/E0_ZS.yaml" "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" null "$PREC"
# E1..E6 with resolved PEFT JSON paths
emit_yaml "$CFG_DIR/E1_QKVO.yaml"    "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_QKVO_JSON" "$PREC"
emit_yaml "$CFG_DIR/E2_OMLP.yaml"    "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_OMLP_JSON" "$PREC"
emit_yaml "$CFG_DIR/E3_QV.yaml"      "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_QV_JSON" "$PREC"
emit_yaml "$CFG_DIR/E4_OONLY.yaml"   "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_OONLY_R4_JSON" "$PREC"
emit_yaml "$CFG_DIR/E5_MLPONLY.yaml" "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_MLP_JSON" "$PREC"
emit_yaml "$CFG_DIR/E6_QKV.yaml"     "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_QKV_JSON" "$PREC"

# ------------------------------- launch round ------------------------------
GPUS=(0 1 2 3 4 5 6)
ROUND_SET=(
  "E0_ZS.yaml"
  "E1_QKVO.yaml"
  "E2_OMLP.yaml"
  "E3_QV.yaml"
  "E4_OONLY.yaml"
  "E5_MLPONLY.yaml"
  "E6_QKV.yaml"
)

for i in $(seq 0 6); do
  CFG="$CFG_DIR/${ROUND_SET[$i]}"
  GPU="${GPUS[$i]}"
  echo "[GPU ${GPU}] ${CFG}"
  CUDA_VISIBLE_DEVICES=$GPU python train.py --cfg "$CFG" --overwrite &
  PIDS+=("$!")
done
wait

echo "Round 1 resolved for TASK=${TASK} finished."


