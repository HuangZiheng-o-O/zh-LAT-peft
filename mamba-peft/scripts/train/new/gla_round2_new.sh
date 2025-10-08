#!/bin/bash
set -euo pipefail

# Round 2 (7 jobs) resolved. Mirrors ROUND=2 of gla_rounds.sh exactly, but with
# explicit YAML and PEFT JSON written deterministically here.

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

TASK="${TASK:-cola}"
PEFT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft"
cd "$PEFT_ROOT"

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

declare -A TASK_DIR=( [cola]=cola_gla [rte]=rte_gla [mrpc]=mrpc_gla [sst2]=sst2_gla [qnli]=qnli_gla [qqp]=qqp_gla [mnli]=mnli_gla )
if [[ -z "${TASK_DIR[$TASK]+x}" ]]; then
  echo "Unsupported TASK=$TASK. Use one of: ${!TASK_DIR[@]}"; exit 1
fi
CFG_DIR="cfg/exps/benchmark/glue/${TASK_DIR[$TASK]}"
mkdir -p "$CFG_DIR"

# ----------------------------- PEFT JSON writers ---------------------------
peft_qkvo() { cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"],
  "modules_to_save": []
}
JSON
}
peft_qkvo_r16() { cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 16,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"],
  "modules_to_save": []
}
JSON
}
peft_omlp() { cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"],
  "modules_to_save": []
}
JSON
}
peft_qv() { cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.q_proj","attn.v_proj"],
  "modules_to_save": []
}
JSON
}
peft_qkvo_g() { cat > "$1" <<'JSON'
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
    "attn.o_proj",
    "attn.g_proj"
  ],
  "modules_to_save": []
}
JSON
}
peft_out_r4() { cat > "$1" <<'JSON'
{
  "peft_type": "LORA",
  "r": 4,
  "lora_alpha": 4,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.o_proj"],
  "modules_to_save": []
}
JSON
}

PEFT_DIR="cfg/peft/resolved/r2"
mkdir -p "$PEFT_DIR"
PEFT_QKVO_JSON="$PEFT_DIR/lora_gla_qkvo.json";     peft_qkvo      "$PEFT_QKVO_JSON"
PEFT_QKVO_R16_JSON="$PEFT_DIR/lora_gla_qkvo_r16.json"; peft_qkvo_r16 "$PEFT_QKVO_R16_JSON"
PEFT_OMLP_JSON="$PEFT_DIR/lora_gla_omlp.json";     peft_omlp      "$PEFT_OMLP_JSON"
PEFT_QV_JSON="$PEFT_DIR/lora_gla_qv.json";         peft_qv        "$PEFT_QV_JSON"
PEFT_OONLY_R4_JSON="$PEFT_DIR/lora_gla_out_r4.json"; peft_out_r4  "$PEFT_OONLY_R4_JSON"
PEFT_QKVO_G_JSON="$PEFT_DIR/lora_gla_qkvo_g.json"; peft_qkvo_g    "$PEFT_QKVO_G_JSON"

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
    if [[ -n "${seed_val}" ]]; then echo "seed: ${seed_val}"; fi
  } > "${path}"
}

BATCH_SIZE=4; LR=0.0003; MODEL="/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B"; NO_SAVE=false; EPOCHS=10; PREC=bf16

# Round 2 YAMLs
emit_yaml "$CFG_DIR/E7_GONLY.yaml"         "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_OMLP_JSON" "$PREC"
emit_yaml "$CFG_DIR/E8_QKVO_G.yaml"       "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_QKVO_G_JSON" "$PREC"
emit_yaml "$CFG_DIR/E1_QKVO_R16.yaml"     "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_QKVO_R16_JSON" "$PREC"
emit_yaml "$CFG_DIR/E1_QKVO_seed127.yaml" "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_QKVO_JSON" "$PREC" 127
emit_yaml "$CFG_DIR/E2_OMLP_seed127.yaml" "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_OMLP_JSON" "$PREC" 127
emit_yaml "$CFG_DIR/E3_QV_seed127.yaml"   "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_QV_JSON" "$PREC" 127
emit_yaml "$CFG_DIR/E4_OONLY_seed127.yaml" "$BATCH_SIZE" "$TASK" "$LR" "$MODEL" "$NO_SAVE" "$EPOCHS" "$PEFT_OONLY_R4_JSON" "$PREC" 127

GPUS=(0 1 2 3 4 5 6)
ROUND_SET=(
  "E7_GONLY.yaml"
  "E8_QKVO_G.yaml"
  "E1_QKVO_R16.yaml"
  "E1_QKVO_seed127.yaml"
  "E2_OMLP_seed127.yaml"
  "E3_QV_seed127.yaml"
  "E4_OONLY_seed127.yaml"
)
for i in $(seq 0 6); do
  CFG="$CFG_DIR/${ROUND_SET[$i]}"; GPU="${GPUS[$i]}"
  echo "[GPU ${GPU}] ${CFG}"
  CUDA_VISIBLE_DEVICES=$GPU python train.py --cfg "$CFG" --overwrite &
  PIDS+=("$!")
done
wait
echo "Round 2 resolved for TASK=${TASK} finished."


