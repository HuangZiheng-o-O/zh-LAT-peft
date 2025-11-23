#!/bin/bash
# DART 训练 - 最终命令

set -e

echo "=========================================="
echo "DART 训练启动"
echo "=========================================="

# 1. 清理旧缓存
echo ""
echo "[1] 清理旧的训练集缓存"
echo "------------------------------------------"
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft
rm -fv data/GEM_dart/cache_GEM_dart_train.pkl
rm -fv data/GEM_dart/cache_GEM_dart_train_gen.pkl
rm -fv data/GEM_dart/parts/cache_GEM_dart_train_part_*.pkl

# 2. 切换到训练目录
echo ""
echo "[2] 启动训练"
echo "------------------------------------------"
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

# 数据路径（本地 JSON；不要管 py 脚本）
export DART_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart

# 基础解码配置（DART 文本较短，建议 64~128，取 96）
export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=96
export EVAL_GEN_MIN_LENGTH=8
unset EVAL_GEN_NUM_BEAMS      # 保持贪心，避免 reorder_cache 兼容问题

# GLA 相关（左填充 + max_new_tokens 语义；减少无用日志）
export GLA_FORCE_LEFT_PAD=1
export GLA_USE_MAX_NEW_TOKENS=1
export GLA_USE_FUSED_SWIGLU=0
export GLA_VERBOSE=0

# 学习率调度（现代：余弦 + 10% warmup）
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

# 评测/保存频率（生成评测较重，调中等频率）
export HP_EVAL_STEPS=1000
export HP_SAVE_STEPS=1000
export HP_LOGGING_STEPS=50

# DataLoader / 资源
export NUM_DATA_WORKERS=4
export DATALOADER_PREFETCH_FACTOR=2
export DATALOADER_PIN_MEMORY=1
export DATALOADER_PERSISTENT_WORKERS=0
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1

# CPU/Tokenizer/显存配置
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SwanLab（云端 + 本地诊断日志 + 邮件）
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-dart-E16-clean-r1"
export SWANLAB_EMAIL_YAML="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/dangerous/email_notify.yaml"
export SWANLAB_EMAIL_ON_START=1
export SWANLAB_EMAIL_ON_FINISH=1
export SWANLAB_EMAIL_ON_INTERRUPT=1

# 运行（DART 对应 data 前缀为 dart；seed 可按需改）
./gla_batch_tmux_clean.sh \
  --suite E16 \
  --round 1 \
  --pairs "87:dart" \
  --gpus "0 1 3 4 6" \
  --gpu-plan "2,2,2,2,2"
echo ""
echo "=========================================="
echo "训练命令已执行"
echo "=========================================="
echo ""
echo "请检查日志以确认训练正常启动："
echo "  tail -f /home/user/mzs_h/log/step1_s87_dart_*.log"

