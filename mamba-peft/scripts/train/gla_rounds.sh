#!/bin/bash
set -euo pipefail

# Graceful stop: kill all launched child jobs on Ctrl+C/TERM
PIDS=()
cleanup() {
  echo "Caught signal, stopping ${#PIDS[@]} jobs..."
  for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
  exit 130
}
trap cleanup INT TERM

# Usage:
#   bash gla_rounds.sh 1              # Round 1 on default TASK=cola
#   TASK=rte bash gla_rounds.sh 2     # Round 2 on RTE
#
# Path mapping:
#   local  -> /Users/huangziheng/PycharmProjects/code/zh-LAT-peft
#   remote -> /home/user/mzs_h/code/zh-LAT-peft  (this script runs on remote)

ROUND="${1:-1}"
TASK="${TASK:-cola}"  # cola | rte | mrpc | sst2 | qnli | qqp | mnli

# --- Resolve workspace (remote machine expected) ---
PEFT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft"
cd "$PEFT_ROOT"

# --- Env (mirror, caches, NCCL, wandb) ---
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

# --- Task dir mapping (placeholders are ready for all GLUE tasks) ---
declare -A TASK_DIR
TASK_DIR=( [cola]=cola_gla [rte]=rte_gla [mrpc]=mrpc_gla [sst2]=sst2_gla [qnli]=qnli_gla [qqp]=qqp_gla [mnli]=mnli_gla )

if [[ -z "${TASK_DIR[$TASK]+x}" ]]; then
  echo "Unsupported TASK=$TASK. Use one of: ${!TASK_DIR[@]}"; exit 1
fi

CFG_DIR="cfg/exps/benchmark/glue/${TASK_DIR[$TASK]}"
mkdir -p "$CFG_DIR"

# --- PEFT JSONs (designed targets) ---
# r8 main sets
PEFT_QKVO="cfg/peft/lora/r8/lora_gla_qkvo.json"
PEFT_OMLP="cfg/peft/lora/r8/lora_gla_omlp.json"
PEFT_QV="cfg/peft/lora/r8/lora_gla_qv.json"
PEFT_MLPONLY="cfg/peft/lora/r8/lora_gla_mlp.json"
PEFT_QKV="cfg/peft/lora/r8/lora_gla_qkv.json"
PEFT_GONLY="cfg/peft/lora/r8/lora_gla_g.json"
PEFT_QKVO_G="cfg/peft/lora/r8/lora_gla_qkvo_g.json"
# O-only uses r4 per design
PEFT_OONLY_R4="cfg/peft/lora/r4/lora_gla_out.json"
# capacity variant for Round 2
PEFT_QKVO_R16="cfg/peft/lora/r16/lora_gla_qkvo.json"

# --- Build a YAML if missing ---
build_yaml () {
  local yaml="$1"; local data_id="$2"; local peft_path="$3"
  # lr=3e-4, bf16, epochs=10 per design
  cat > "$yaml" <<EOF
batch_size: 4
data: glue-tvt_${data_id}
learning_rate: 0.0003
model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
no_save: false
num_epochs: 10
peft: ${peft_path}
prec: bf16
EOF
}

# --- Ensure E0–E8 YAMLs exist for the TASK ---
ensure_all_groups () {
  # E0: zero-shot (peft: null)
  [[ -f "$CFG_DIR/E0_ZS.yaml" ]] || cat > "$CFG_DIR/E0_ZS.yaml" <<EOF
batch_size: 4
data: glue-tvt_${TASK}
learning_rate: 0.0003
model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
no_save: false
num_epochs: 10
peft: null
prec: bf16
EOF
  # E1..E8
  [[ -f "$CFG_DIR/E1_QKVO.yaml"    ]] || build_yaml "$CFG_DIR/E1_QKVO.yaml"    "$TASK" "$PEFT_QKVO"
  [[ -f "$CFG_DIR/E2_OMLP.yaml"    ]] || build_yaml "$CFG_DIR/E2_OMLP.yaml"    "$TASK" "$PEFT_OMLP"
  [[ -f "$CFG_DIR/E3_QV.yaml"      ]] || build_yaml "$CFG_DIR/E3_QV.yaml"      "$TASK" "$PEFT_QV"
  [[ -f "$CFG_DIR/E4_OONLY.yaml"   ]] || build_yaml "$CFG_DIR/E4_OONLY.yaml"   "$TASK" "$PEFT_OONLY_R4"
  [[ -f "$CFG_DIR/E5_MLPONLY.yaml" ]] || build_yaml "$CFG_DIR/E5_MLPONLY.yaml" "$TASK" "$PEFT_MLPONLY"
  [[ -f "$CFG_DIR/E6_QKV.yaml"     ]] || build_yaml "$CFG_DIR/E6_QKV.yaml"     "$TASK" "$PEFT_QKV"
  [[ -f "$CFG_DIR/E7_GONLY.yaml"   ]] || build_yaml "$CFG_DIR/E7_GONLY.yaml"   "$TASK" "$PEFT_GONLY"
  [[ -f "$CFG_DIR/E8_QKVO_G.yaml"  ]] || build_yaml "$CFG_DIR/E8_QKVO_G.yaml"  "$TASK" "$PEFT_QKVO_G"
  # extra capacity variant for Round 2:
  [[ -f "$CFG_DIR/E1_QKVO_R16.yaml" ]] || build_yaml "$CFG_DIR/E1_QKVO_R16.yaml" "$TASK" "$PEFT_QKVO_R16"

  # seed variants for Round 2 (stability checks)
  if [[ ! -f "$CFG_DIR/E1_QKVO_seed127.yaml" ]]; then
cat > "$CFG_DIR/E1_QKVO_seed127.yaml" <<EOF
batch_size: 4
data: glue-tvt_${TASK}
learning_rate: 0.0003
model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
no_save: false
num_epochs: 10
peft: ${PEFT_QKVO}
prec: bf16
seed: 127
EOF
  fi
  if [[ ! -f "$CFG_DIR/E2_OMLP_seed127.yaml" ]]; then
cat > "$CFG_DIR/E2_OMLP_seed127.yaml" <<EOF
batch_size: 4
data: glue-tvt_${TASK}
learning_rate: 0.0003
model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
no_save: false
num_epochs: 10
peft: ${PEFT_OMLP}
prec: bf16
seed: 127
EOF
  fi
  if [[ ! -f "$CFG_DIR/E3_QV_seed127.yaml" ]]; then
cat > "$CFG_DIR/E3_QV_seed127.yaml" <<EOF
batch_size: 4
data: glue-tvt_${TASK}
learning_rate: 0.0003
model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
no_save: false
num_epochs: 10
peft: ${PEFT_QV}
prec: bf16
seed: 127
EOF
  fi
  if [[ ! -f "$CFG_DIR/E4_OONLY_seed127.yaml" ]]; then
cat > "$CFG_DIR/E4_OONLY_seed127.yaml" <<EOF
batch_size: 4
data: glue-tvt_${TASK}
learning_rate: 0.0003
model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
no_save: false
num_epochs: 10
peft: ${PEFT_OONLY_R4}
prec: bf16
seed: 127
EOF
  fi
}

ensure_all_groups

# --- Round 3 helpers: generate layered/variant JSONs & YAMLs (do not affect Round1/2) ---
make_layered_targets() {
  local kind="$1"; shift
  local -a layers=("$@")
  local entries=()
  case "$kind" in
    QKVO)
      local suffixes=("attn.q_proj" "attn.k_proj" "attn.v_proj" "attn.o_proj")
      ;;
    OMLP)
      local suffixes=("attn.o_proj" "mlp.gate_proj" "mlp.up_proj" "mlp.down_proj")
      ;;
    *) echo "unknown kind $kind"; return 1;;
  esac
  for li in "${layers[@]}"; do
    for s in "${suffixes[@]}"; do
      entries+=("\"layers.${li}.${s}\"")
    done
  done
  local IFS=','
  echo "[${entries[*]}]"
}

write_json_qkvo_layered() {
  local path="$1"; shift; local -a layers=("$@")
  local targets; targets=$(make_layered_targets QKVO "${layers[@]}")
  cat > "$path" <<EOF
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
EOF
}

write_json_omlp_layered() {
  local path="$1"; shift; local -a layers=("$@")
  local targets; targets=$(make_layered_targets OMLP "${layers[@]}")
  cat > "$path" <<EOF
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
EOF
}

write_json_qkvo_alpha() {
  local path="$1"; local alpha="$2"
  cat > "$path" <<EOF
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": ${alpha},
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"],
  "modules_to_save": []
}
EOF
}

write_json_qkvo_dropout() {
  local path="$1"; local drop="$2"
  cat > "$path" <<EOF
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": ${drop},
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": ["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"],
  "modules_to_save": []
}
EOF
}

ensure_round3() {
  # layer indices for 24-layer GLA
  local -a LAST6=(18 19 20 21 22 23)
  local -a FIRST6=(0 1 2 3 4 5)
  local -a MIDDLE6=(9 10 11 12 13 14)
  # JSON variants dir
  local JV="cfg/peft/lora/r8"
  # QKVO layered
  [[ -f "$JV/lora_gla_qkvo_last6.json"   ]] || write_json_qkvo_layered   "$JV/lora_gla_qkvo_last6.json"   "${LAST6[@]}"
  [[ -f "$JV/lora_gla_qkvo_first6.json"  ]] || write_json_qkvo_layered   "$JV/lora_gla_qkvo_first6.json"  "${FIRST6[@]}"
  # O+MLP layered
  [[ -f "$JV/lora_gla_omlp_last6.json"   ]] || write_json_omlp_layered   "$JV/lora_gla_omlp_last6.json"   "${LAST6[@]}"
  [[ -f "$JV/lora_gla_omlp_middle6.json" ]] || write_json_omlp_layered   "$JV/lora_gla_omlp_middle6.json" "${MIDDLE6[@]}"
  # alpha / dropout variants
  [[ -f "$JV/lora_gla_qkvo_alpha2r.json" ]] || write_json_qkvo_alpha     "$JV/lora_gla_qkvo_alpha2r.json" 16
  [[ -f "$JV/lora_gla_qkvo_dropout0.json" ]]|| write_json_qkvo_dropout   "$JV/lora_gla_qkvo_dropout0.json" 0.0

  # YAMLs for Round 3
  [[ -f "$CFG_DIR/E1_QKVO_last6.yaml"      ]] || build_yaml "$CFG_DIR/E1_QKVO_last6.yaml"      "$TASK" "$JV/lora_gla_qkvo_last6.json"
  [[ -f "$CFG_DIR/E2_OMLP_last6.yaml"      ]] || build_yaml "$CFG_DIR/E2_OMLP_last6.yaml"      "$TASK" "$JV/lora_gla_omlp_last6.json"
  [[ -f "$CFG_DIR/E1_QKVO_first6.yaml"     ]] || build_yaml "$CFG_DIR/E1_QKVO_first6.yaml"     "$TASK" "$JV/lora_gla_qkvo_first6.json"
  [[ -f "$CFG_DIR/E2_OMLP_middle6.yaml"    ]] || build_yaml "$CFG_DIR/E2_OMLP_middle6.yaml"    "$TASK" "$JV/lora_gla_omlp_middle6.json"
  [[ -f "$CFG_DIR/E1_QKVO_alpha2r.yaml"    ]] || build_yaml "$CFG_DIR/E1_QKVO_alpha2r.yaml"    "$TASK" "$JV/lora_gla_qkvo_alpha2r.json"
  [[ -f "$CFG_DIR/E1_QKVO_dropout0.yaml"   ]] || build_yaml "$CFG_DIR/E1_QKVO_dropout0.yaml"   "$TASK" "$JV/lora_gla_qkvo_dropout0.json"
  # lr 1e-4 variant (same peft as base QKVO, different LR)
  if [[ ! -f "$CFG_DIR/E1_QKVO_lr1e-4.yaml" ]]; then
cat > "$CFG_DIR/E1_QKVO_lr1e-4.yaml" <<EOF
batch_size: 4
data: glue-tvt_${TASK}
learning_rate: 0.0001
model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
no_save: false
num_epochs: 10
peft: ${PEFT_QKVO}
prec: bf16
EOF
  fi
}

# --- GPU assignment (7 GPUs per round) ---
GPUS=(0 1 2 3 4 5 6)

# Round 1 set (7 items):
ROUND1=(
  "E0_ZS.yaml"
  "E1_QKVO.yaml"
  "E2_OMLP.yaml"
  "E3_QV.yaml"
  "E4_OONLY.yaml"
  "E5_MLPONLY.yaml"
  "E6_QKV.yaml"
)

# Round 2 set (7 items):
ROUND2=(
  "E7_GONLY.yaml"
  "E8_QKVO_G.yaml"
  "E1_QKVO_R16.yaml"
  "E1_QKVO_seed127.yaml"
  "E2_OMLP_seed127.yaml"
  "E3_QV_seed127.yaml"
  "E4_OONLY_seed127.yaml"
)

select_round=()
case "$ROUND" in
  1) select_round=("${ROUND1[@]}");;
  2) select_round=("${ROUND2[@]}");;
  3) ensure_round3; select_round=(
        "E1_QKVO_last6.yaml"
        "E2_OMLP_last6.yaml"
        "E1_QKVO_first6.yaml"
        "E2_OMLP_middle6.yaml"
        "E1_QKVO_alpha2r.yaml"
        "E1_QKVO_dropout0.yaml"
        "E1_QKVO_lr1e-4.yaml"
     );;
  4)
     ensure_round4() {
       local JV
       JV="cfg/peft/lora"
       write_json_out_rank() {
         local dir="$1"; local path="$2"; local rnk="$3"; local alpha="$4"; mkdir -p "$dir"
         cat > "$path" <<EOF
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
EOF
       }
       write_json_omlp_rank() {
         local dir="$1"; local path="$2"; local rnk="$3"; local alpha="$4"; mkdir -p "$dir"
         cat > "$path" <<EOF
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
EOF
       }
       write_json_qkvo_dora() {
         local path="$1"
         cat > "$path" <<EOF
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
EOF
       }
       write_json_omlp_dora() {
         local path="$1"
         cat > "$path" <<EOF
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
EOF
       }
       write_json_qkvo_rs() {
         local path="$1"
         cat > "$path" <<EOF
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
EOF
       }
       make_layered_targets_gk() {
         local -a layers=("$@")
         local entries=()
         local suffixes=("attn.gk_proj.0" "attn.gk_proj.1")
         for li in "${layers[@]}"; do
           for s in ${suffixes[@]}; do
             entries+=("\"layers.${li}.${s}\"")
           done
         done
         local IFS=','
         echo "[${entries[*]}]"
       }
       write_json_qkvo_plus_gk_last6() {
         local path="$1"
         local -a LAST6=(18 19 20 21 22 23)
         local t_qkvo; t_qkvo=$(make_layered_targets QKVO "${LAST6[@]}")
         local t_gk;   t_gk=$(make_layered_targets_gk "${LAST6[@]}")
         local merged
         merged="${t_qkvo%]} , ${t_gk#[}"
         cat > "$path" <<EOF
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 8,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": [ ${merged} ],
  "modules_to_save": []
}
EOF
       }
       mkdir -p "${JV}/r16" "${JV}/r6" "${JV}/r8"
       [[ -f "${JV}/r16/lora_gla_out.json"      ]] || write_json_out_rank   "${JV}/r16" "${JV}/r16/lora_gla_out.json" 16 16
       [[ -f "${JV}/r6/lora_gla_omlp.json"      ]] || write_json_omlp_rank  "${JV}/r6"  "${JV}/r6/lora_gla_omlp.json" 6 6
       [[ -f "${JV}/r8/lora_gla_qkvo_dora.json" ]] || write_json_qkvo_dora  "${JV}/r8/lora_gla_qkvo_dora.json"
       [[ -f "${JV}/r8/lora_gla_omlp_dora.json" ]] || write_json_omlp_dora  "${JV}/r8/lora_gla_omlp_dora.json"
       [[ -f "${JV}/r8/lora_gla_qkvo_rs.json"   ]] || write_json_qkvo_rs    "${JV}/r8/lora_gla_qkvo_rs.json"
       [[ -f "${JV}/r8/lora_gla_qkvo_gk_last6.json" ]] || write_json_qkvo_plus_gk_last6 "${JV}/r8/lora_gla_qkvo_gk_last6.json"
       [[ -f "$CFG_DIR/E1_QKVO_r4_equal.yaml"      ]] || build_yaml "$CFG_DIR/E1_QKVO_r4_equal.yaml"      "$TASK" "${JV}/r4/lora_gla_qkvo.json"
       [[ -f "$CFG_DIR/E4_OONLY_r16_equal.yaml"    ]] || build_yaml "$CFG_DIR/E4_OONLY_r16_equal.yaml"    "$TASK" "${JV}/r16/lora_gla_out.json"
       [[ -f "$CFG_DIR/E2_OMLP_r6_equal.yaml"      ]] || build_yaml "$CFG_DIR/E2_OMLP_r6_equal.yaml"      "$TASK" "${JV}/r6/lora_gla_omlp.json"
       [[ -f "$CFG_DIR/E1_QKVO_DoRA.yaml"          ]] || build_yaml "$CFG_DIR/E1_QKVO_DoRA.yaml"          "$TASK" "${JV}/r8/lora_gla_qkvo_dora.json"
       [[ -f "$CFG_DIR/E2_OMLP_DoRA.yaml"          ]] || build_yaml "$CFG_DIR/E2_OMLP_DoRA.yaml"          "$TASK" "${JV}/r8/lora_gla_omlp_dora.json"
       [[ -f "$CFG_DIR/E1_QKVO_RSLoRA.yaml"        ]] || build_yaml "$CFG_DIR/E1_QKVO_RSLoRA.yaml"        "$TASK" "${JV}/r8/lora_gla_qkvo_rs.json"
       [[ -f "$CFG_DIR/E1_QKVO_plus_GK_last6.yaml" ]] || build_yaml "$CFG_DIR/E1_QKVO_plus_GK_last6.yaml" "$TASK" "${JV}/r8/lora_gla_qkvo_gk_last6.json"
     }
     ensure_round4
     select_round=(
       "E1_QKVO_r4_equal.yaml"
       "E4_OONLY_r16_equal.yaml"
       "E2_OMLP_r6_equal.yaml"
       "E1_QKVO_DoRA.yaml"
       "E2_OMLP_DoRA.yaml"
       "E1_QKVO_RSLoRA.yaml"
       "E1_QKVO_plus_GK_last6.yaml"
     );;
  *) echo "ROUND must be 1 or 2 or 3 or 4"; exit 1;;
esac

# --- Fire 7 jobs, one per GPU ---
for i in $(seq 0 6); do
  CFG="${CFG_DIR}/${select_round[$i]}"
  GPU="${GPUS[$i]}"
  echo "[GPU ${GPU}] ${CFG}"
  CUDA_VISIBLE_DEVICES=$GPU python train.py --cfg "$CFG" --overwrite &
  PIDS+=("$!")
done
wait

echo "Round ${ROUND} for TASK=${TASK} finished."