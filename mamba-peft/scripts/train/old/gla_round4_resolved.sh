#!/bin/bash
set -euo pipefail

# Round 4 resolved. Mirrors ensure_round4 + ROUND=4 from gla_rounds.sh with
# DoRA/RSLoRA and rank variants explicitly emitted.

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

PEFT_DIR="cfg/peft/resolved/r4"; mkdir -p "$PEFT_DIR/r16" "$PEFT_DIR/r6" "$PEFT_DIR/r8"

# ------------------------------- PEFT writers ------------------------------
write_out_rank() {
  local path="$1"; local rnk="$2"; local alpha="$3"; cat > "$path" <<JSON
{
  "peft_type": "LORA",
  "r": ${rnk},
  "lora_alpha": ${alpha},
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.o_proj"],
  "modules_to_save": []
}
JSON
}
write_omlp_rank() {
  local path="$1"; local rnk="$2"; local alpha="$3"; cat > "$path" <<JSON
{
  "peft_type": "LORA",
  "r": ${rnk},
  "lora_alpha": ${alpha},
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
write_qkvo_dora() {
  local path="$1"; cat > "$path" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "use_dora": true,
  "target_modules": ["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"],
  "modules_to_save": []
}
JSON
}
write_omlp_dora() {
  local path="$1"; cat > "$path" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "use_dora": true,
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
write_qkvo_rs() {
  local path="$1"; cat > "$path" <<'JSON'
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "use_rslora": true,
  "target_modules": ["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"],
  "modules_to_save": []
}
JSON
}

write_qkvo_plus_gk_last6() {
  local path="$1"; local last6=(18 19 20 21 22 23); local entries=()
  for li in "${last6[@]}"; do
    entries+=("\"layers.${li}.attn.q_proj\"")
    entries+=("\"layers.${li}.attn.k_proj\"")
    entries+=("\"layers.${li}.attn.v_proj\"")
    entries+=("\"layers.${li}.attn.o_proj\"")
    entries+=("\"layers.${li}.attn.gk_proj.0\"")
    entries+=("\"layers.${li}.attn.gk_proj.1\"")
  done
  local IFS=','; local targets="[${entries[*]}]"
  cat > "$path" <<JSON
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ${targets},
  "modules_to_save": []
}
JSON
}

write_out_rank "$PEFT_DIR/r16/lora_gla_out.json" 16 16
write_omlp_rank "$PEFT_DIR/r6/lora_gla_omlp.json" 6 6
write_qkvo_dora "$PEFT_DIR/r8/lora_gla_qkvo_dora.json"
write_omlp_dora "$PEFT_DIR/r8/lora_gla_omlp_dora.json"
write_qkvo_rs   "$PEFT_DIR/r8/lora_gla_qkvo_rs.json"
write_qkvo_plus_gk_last6 "$PEFT_DIR/r8/lora_gla_qkvo_gk_last6.json"

emit_yaml() {
  local path="$1"; local peft_path="$2"; shift 2
  {
    echo "batch_size: 4"
    echo "data: glue-tvt_${TASK}"
    echo "learning_rate: 0.0003"
    echo "model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B"
    echo "no_save: false"
    echo "num_epochs: 10"
    echo "peft: ${peft_path}"
    echo "prec: bf16"
  } > "$path"
}

emit_yaml "$CFG_DIR/E1_QKVO_r4_equal.yaml"       "$PEFT_DIR/r4/lora_gla_qkvo.json"   # if needed, write r4 QKVO now
mkdir -p "$PEFT_DIR/r4"
cat > "$PEFT_DIR/r4/lora_gla_qkvo.json" <<'JSON'
{
  "peft_type": "LORA",
  "r": 4,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"],
  "modules_to_save": []
}
JSON

emit_yaml "$CFG_DIR/E4_OONLY_r16_equal.yaml"     "$PEFT_DIR/r16/lora_gla_out.json"
emit_yaml "$CFG_DIR/E2_OMLP_r6_equal.yaml"       "$PEFT_DIR/r6/lora_gla_omlp.json"
emit_yaml "$CFG_DIR/E1_QKVO_DoRA.yaml"           "$PEFT_DIR/r8/lora_gla_qkvo_dora.json"
emit_yaml "$CFG_DIR/E2_OMLP_DoRA.yaml"           "$PEFT_DIR/r8/lora_gla_omlp_dora.json"
emit_yaml "$CFG_DIR/E1_QKVO_RSLoRA.yaml"         "$PEFT_DIR/r8/lora_gla_qkvo_rs.json"
emit_yaml "$CFG_DIR/E1_QKVO_plus_GK_last6.yaml"  "$PEFT_DIR/r8/lora_gla_qkvo_gk_last6.json"

GPUS=(0 1 2 3 4 5 6)
ROUND_SET=(
  "E1_QKVO_r4_equal.yaml"
  "E4_OONLY_r16_equal.yaml"
  "E2_OMLP_r6_equal.yaml"
  "E1_QKVO_DoRA.yaml"
  "E2_OMLP_DoRA.yaml"
  "E1_QKVO_RSLoRA.yaml"
  "E1_QKVO_plus_GK_last6.yaml"
)
for i in $(seq 0 6); do
  CFG="$CFG_DIR/${ROUND_SET[$i]}"; GPU="${GPUS[$i]}"
  echo "[GPU ${GPU}] ${CFG}"
  CUDA_VISIBLE_DEVICES=$GPU python train.py --cfg "$CFG" --overwrite &
  PIDS+=("$!")
done
wait
echo "Round 4 resolved for TASK=${TASK} finished."


