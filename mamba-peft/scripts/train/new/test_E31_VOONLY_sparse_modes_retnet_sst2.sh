#!/usr/bin/env bash
set -euo pipefail

# Test launcher: VOONLY sparse selective tuning (8 modes) via lat_batch_tmux.sh
#
# Suite:
#   - E31 = VOONLY__* (4 scopes × 2 budget types)
# REF policy:
#   - All REF YAMLs are hard-wired to match trainable count of:
#       cfg/my_lora_exp/yaml/E1_QKVO_plus_MLP_r8_alpha16.yaml
#   - "match" is enforced after sparse selection (post-warmup), i.e., the final trainable count
#     used by the optimizer during training is what matches.
#
# NOTE:
#   - ENV overrides YAML. You can override K/rho/etc per run if desired.
#   - These YAMLs also provide defaults:
#       HP_INIT=pissa, HP_SAVE_MODE=best_last, HP_SAVE_FULL_MODEL=0
#     (only if those env vars are not already set).

conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=retnet
export LAT_MODEL=/home/user/mzs_h/model/retnet-1.3B-100B/
export LAT_PREC=bf16

# Offline caches (optional)
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

# Task selection
export HP_DATA=sst2
export EVAL_GEN=0
export HP_VAL_SPLIT=test

# Train hparams (example)
export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0003

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=200
export HP_SAVE_STEPS=400
export HP_LOGGING_STEPS=100

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

# Dataloader / perf
export NUM_DATA_WORKERS=4
export DATALOADER_PREFETCH_FACTOR=2
export DATALOADER_PIN_MEMORY=1
export DATALOADER_PERSISTENT_WORKERS=0
export GRADIENT_CHECKPOINTING=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

# Logging (optional)
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="retnet-sst2-VOONLY-sparse-modes-E31"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

# Launch (8 YAMLs)
./lat_batch_tmux.sh \
  --suite E31 \
  --round all \
  --pairs "87:glue-tvt_sst2" \
  --gpus "1 2 4 5" \
  --gpu-plan "1,1,1,1" \
  --model-type retnet

