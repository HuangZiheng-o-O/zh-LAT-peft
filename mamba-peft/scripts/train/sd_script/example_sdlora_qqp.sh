#!/bin/bash
# =========================
# SD-LoRA QQP 训练命令示例
# =========================
# 批量测试15个SD-LoRA配置在QQP数据集上
# QQP: 句对复述检测 (二分类), ~363k训练样本

############################
# 1. 环境激活
############################
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

############################
# 2. 模型配置
############################
export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16

############################
# 3. 严格离线 + 本地 HF 缓存
############################
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

############################
# 4. 任务配置
############################
# QQP: 句对复述检测 (二分类)
export EVAL_GEN=0
export HP_VAL_SPLIT=test

############################
# 5. 训练超参
############################
# QQP 训练集 363k+，大规模数据
export HP_EPOCHS=5            # 大数据任务适中迭代
export HP_BATCH_SIZE=8
export HP_LR=0.0003           # 3e-4: 比小任务稍低，更稳定

############################
# 6. 评测 / 保存 / 日志
############################
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=500      # 大任务稀疏评测
export HP_SAVE_STEPS=1000
export HP_LOGGING_STEPS=100

############################
# 7. 学习率调度
############################
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

############################
# 8. DataLoader / Runtime
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
# 9. SwanLab 监控
############################
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-sdlora-qqp-Jan01"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

############################
# 10. 启动交错间隔 (可选)
############################
export LAT_LAUNCH_STAGGER_MINUTES=5

############################
# 11. 启动 SD-LoRA 批量训练
############################
./lat_batch_tmux_sparse.sh \
  --round all \
  --pairs "87:glue-tvt_qqp" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "1,1,1,1,1,1,1" \
  --model-type gla
