#!/bin/bash
# =========================
# SD-LoRA QNLI 训练命令
# =========================
# 批量测试15个SD-LoRA配置在QNLI数据集上
# QNLI: Question Natural Language Inference (二分类)
# 训练集约 104k 样本

conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

############################
# 模型配置
############################
export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16

############################
# 严格离线 + 本地 HF 缓存
############################
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

############################
# 任务配置
############################
# QNLI: 分类任务，禁用生成
export EVAL_GEN=0
export HP_VAL_SPLIT=test

############################
# 训练超参
############################
export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0004           # 4e-4

############################
# 评测/保存/日志 (按 QNLI 量级)
############################
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=1500
export HP_SAVE_STEPS=3000
export HP_LOGGING_STEPS=100

############################
# 学习率调度
############################
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

############################
# DataLoader / Runtime
############################
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

############################
# SwanLab
############################
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-sdlora-qnli-Jan01"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

############################
# 启动 SD-LoRA 训练
############################
./lat_batch_tmux_sparse.sh \
  --round all \
  --pairs "87:glue-tvt_qnli" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "1,1,1,1,1,1,1" \
  --model-type gla
