#!/bin/bash
set -euo pipefail

# Round 3 resolved. Matches ensure_round3 + ROUND=3 set from gla_rounds.sh.

PIDS=()
trap 'for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done; exit 130' INT TERM

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
if [[ -z "${TASK_DIR[$TASK]+x}" ]]; then echo "Unsupported TASK=$TASK"; exit 1; fi
CFG_DIR="cfg/exps/benchmark/glue/${TASK_DIR[$TASK]}"; mkdir -p "$CFG_DIR"

# -------------------- helpers to construct layered targets -----------------
make_layered_targets_qkvo() {
  local -a layers=("$@"); local IFS=','; local acc=()
  for li in "${layers[@]}"; do
    acc+=("\"layers.${li}.attn.q_proj\"")
    acc+=("\"layers.${li}.attn.k_proj\"")
    acc+=("\"layers.${li}.attn.v_proj\"")
    acc+=("\"layers.${li}.attn.o_proj\"")
  done
  echo "[${acc[*]}]"
}
make_layered_targets_omlp() {
  local -a layers=("$@"); local IFS=','; local acc=()
  for li in "${layers[@]}"; do
    acc+=("\"layers.${li}.attn.o_proj\"")
    acc+=("\"layers.${li}.mlp.gate_proj\"")
    acc+=("\"layers.${li}.mlp.up_proj\"")
    acc+=("\"layers.${li}.mlp.down_proj\"")
  done
  echo "[${acc[*]}]"
}

peft_json_from_targets() {
  local path="$1"; local targets_json="$2"; local rnk="${3:-8}"; local alpha="${4:-8}"; local drop="${5:-0.05}"
  cat > "$path" <<JSON
{
  "peft_type": "LORA",
  "r": ${rnk},
  "lora_alpha": ${alpha},
  "lora_dropout": ${drop},
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ${targets_json},
  "modules_to_save": []
}
JSON
}

peft_qkvo_alpha2r() {
  local path="$1"; cat > "$path" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"],
  "modules_to_save": []
}
JSON
}

peft_qkvo_dropout0() {
  local path="$1"; cat > "$path" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.0,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"],
  "modules_to_save": []
}
JSON
}

PEFT_DIR="cfg/peft/resolved/r3"; mkdir -p "$PEFT_DIR"
# last6/first6 indices for 24-layer GLA
LAST6=(18 19 20 21 22 23); FIRST6=(0 1 2 3 4 5); MIDDLE6=(9 10 11 12 13 14)

# QKVO last6
T_QKVO_LAST6=$(make_layered_targets_qkvo "${LAST6[@]}")
peft_json_from_targets "$PEFT_DIR/lora_gla_qkvo_last6.json" "$T_QKVO_LAST6" 8 8 0.05
# OMLP last6
T_OMLP_LAST6=$(make_layered_targets_omlp "${LAST6[@]}")
peft_json_from_targets "$PEFT_DIR/lora_gla_omlp_last6.json" "$T_OMLP_LAST6" 8 8 0.05
# QKVO first6
T_QKVO_FIRST6=$(make_layered_targets_qkvo "${FIRST6[@]}")
peft_json_from_targets "$PEFT_DIR/lora_gla_qkvo_first6.json" "$T_QKVO_FIRST6" 8 8 0.05
# OMLP middle6
T_OMLP_MIDDLE6=$(make_layered_targets_omlp "${MIDDLE6[@]}")
peft_json_from_targets "$PEFT_DIR/lora_gla_omlp_middle6.json" "$T_OMLP_MIDDLE6" 8 8 0.05
# alpha=2r
peft_qkvo_alpha2r "$PEFT_DIR/lora_gla_qkvo_alpha2r.json"
# dropout=0
peft_qkvo_dropout0 "$PEFT_DIR/lora_gla_qkvo_dropout0.json"

emit_yaml() {
  local path="$1"; local peft_path="$2"; shift 2
  local batch_size=4; local data_id="$TASK"; local lr=0.0003; local model="/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B"; local no_save=false; local epochs=10; local prec=bf16
  {
    echo "batch_size: ${batch_size}"
    echo "data: glue-tvt_${data_id}"
    echo "learning_rate: ${lr}"
    echo "model: ${model}"
    echo "no_save: ${no_save}"
    echo "num_epochs: ${epochs}"
    echo "peft: ${peft_path}"
    echo "prec: ${prec}"
  } > "$path"
}

emit_yaml "$CFG_DIR/E1_QKVO_last6.yaml"     "$PEFT_DIR/lora_gla_qkvo_last6.json"
emit_yaml "$CFG_DIR/E2_OMLP_last6.yaml"     "$PEFT_DIR/lora_gla_omlp_last6.json"
emit_yaml "$CFG_DIR/E1_QKVO_first6.yaml"    "$PEFT_DIR/lora_gla_qkvo_first6.json"
emit_yaml "$CFG_DIR/E2_OMLP_middle6.yaml"   "$PEFT_DIR/lora_gla_omlp_middle6.json"
emit_yaml "$CFG_DIR/E1_QKVO_alpha2r.yaml"   "$PEFT_DIR/lora_gla_qkvo_alpha2r.json"
emit_yaml "$CFG_DIR/E1_QKVO_dropout0.yaml"  "$PEFT_DIR/lora_gla_qkvo_dropout0.json"
# lr 1e-4 variant
{
  echo "batch_size: 4"
  echo "data: glue-tvt_${TASK}"
  echo "learning_rate: 0.0001"
  echo "model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B"
  echo "no_save: false"
  echo "num_epochs: 10"
  echo "peft: ${PEFT_DIR}/lora_gla_qkvo.json" # base QKVO (we write it now)
  echo "prec: bf16"
} > "$CFG_DIR/E1_QKVO_lr1e-4.yaml"
# base QKVO for lr1e-4
cat > "$PEFT_DIR/lora_gla_qkvo.json" <<'JSON'
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

GPUS=(0 1 2 3 4 5 6)
ROUND_SET=(
  "E1_QKVO_last6.yaml"
  "E2_OMLP_last6.yaml"
  "E1_QKVO_first6.yaml"
  "E2_OMLP_middle6.yaml"
  "E1_QKVO_alpha2r.yaml"
  "E1_QKVO_dropout0.yaml"
  "E1_QKVO_lr1e-4.yaml"
)
for i in $(seq 0 6); do
  CFG="$CFG_DIR/${ROUND_SET[$i]}"; GPU="${GPUS[$i]}"
  echo "[GPU ${GPU}] ${CFG}"
  CUDA_VISIBLE_DEVICES=$GPU python train.py --cfg "$CFG" --overwrite &
  PIDS+=("$!")
done
wait
echo "Round 3 resolved for TASK=${TASK} finished."


