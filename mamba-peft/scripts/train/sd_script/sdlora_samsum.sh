#!/bin/bash
# =========================
# SD-LoRA SamSum 训练命令
# =========================
# 批量测试15个SD-LoRA配置在SamSum数据集上
# SamSum: 对话摘要生成任务

conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

############################
# 模型配置
############################
export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16

############################
# 数据路径
############################
export SAMSUM_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/samsum

############################
# 生成/解码配置
############################
export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=128
export EVAL_GEN_MIN_LENGTH=8

export GLA_FORCE_LEFT_PAD=1
export GLA_USE_MAX_NEW_TOKENS=1
export GLA_VERBOSE=1
export GLA_USE_FUSED_SWIGLU=0

############################
# 训练超参
############################
export HP_EPOCHS=8
export HP_BATCH_SIZE=8
export HP_LR=0.0005

############################
# 评测/保存/日志
############################
export HP_EVAL_BATCH_SIZE=16
export HP_EVAL_STEPS=1000
export HP_SAVE_STEPS=1000
export HP_LOGGING_STEPS=50

############################
# 学习率调度
############################
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

############################
# DataLoader / Runtime
############################
export NUM_DATA_WORKERS=2
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

############################
# SwanLab
############################
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-sdlora-samsum-Jan01"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

############################
# 启动 SD-LoRA 训练
############################
./lat_batch_tmux_sparse.sh \
  --round all \
  --pairs "87:samsum" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "1,1,1,1,1,1,1" \
  --model-type gla
