#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PEFT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PEFT_DIR"

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

TASKS=(cola rte mrpc sst2 qnli qqp mnli)
# Map to cfg dir names
declare -A TASK_DIR
TASK_DIR=( [cola]=cola_gla [rte]=rte_gla [mrpc]=mrpc_gla [sst2]=sst2_gla [qnli]=qnli_gla [qqp]=qqp_gla [mnli]=mnli_gla )

# 9 groups per task (E0..E8)
GROUPS=(
  E0_ZS:null
  E1_QKVO:cfg/peft/lora/r8/lora_gla_qkvo.json
  E2_OMLP:cfg/peft/lora/r8/lora_gla_omlp.json
  E3_QV:cfg/peft/lora/r8/lora_gla_qv.json
  E4_OONLY:cfg/peft/lora/r4/lora_gla_out.json
  E5_MLPONLY:cfg/peft/lora/r8/lora_gla_mlp.json
  E6_QKV:cfg/peft/lora/r8/lora_gla_qkv.json
  E7_GONLY:cfg/peft/lora/r8/lora_gla_g.json
  E8_QKVO_G:cfg/peft/lora/r8/lora_gla_qkvo_g.json
)

build_yaml(){
  local task=$1; local group_name=$2; local peft_path=$3; local out_yaml=$4
  local data_id="glue-tvt_${task}"
  cat > "$out_yaml" <<EOF
batch_size: 4
data: $data_id
learning_rate: 0.0003
model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
no_save: false
num_epochs: 10
peft: ${peft_path}
prec: bf16
EOF
}

GPUS=(0 1 2 3 4 5 6)
jobs=()

for t in "${TASKS[@]}"; do
  dir="cfg/exps/benchmark/glue/${TASK_DIR[$t]}"
  mkdir -p "$dir"
  for item in "${GROUPS[@]}"; do
    name=${item%%:*}
    peft=${item#*:}
    yaml="$dir/${name}.yaml"
    if [[ "$peft" == "null" ]]; then
      cat > "$yaml" <<EOF
batch_size: 4
data: glue-tvt_${t}
learning_rate: 0.0003
model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
no_save: false
num_epochs: 10
peft: null
prec: bf16
EOF
    else
      build_yaml "$t" "$name" "$peft" "$yaml"
    fi
  done
done

cfgs=()
for t in "${TASKS[@]}"; do
  dir="cfg/exps/benchmark/glue/${TASK_DIR[$t]}"
  for item in "${GROUPS[@]}"; do
    name=${item%%:*}
    cfgs+=("$dir/${name}.yaml")
  done
done

idx=0
while [[ $idx -lt ${#cfgs[@]} ]]; do
  for gpu in "${GPUS[@]}"; do
    [[ $idx -ge ${#cfgs[@]} ]] && break
    CFG=${cfgs[$idx]}
    echo "[GPU $gpu] $CFG"
    CUDA_VISIBLE_DEVICES=$gpu python train.py --cfg "$CFG" --device $gpu --prec bf16 --lock &
    idx=$((idx+1))
  done
  wait
done

echo "All GLUE tasks/groups finished."

