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

# 3. 运行训练
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUM_DATA_WORKERS=4 \
GRADIENT_CHECKPOINTING=true \
LOGITS_TO_KEEP=1 \
HP_EVAL_STEPS=2000 \
HP_SAVE_STEPS=2000 \
HP_LOGGING_STEPS=200 \
EVAL_GEN=1 \
EVAL_GEN_MAX_LENGTH=128 \
EVAL_GEN_MIN_LENGTH=5 \
EVAL_GEN_NUM_BEAMS=5 \
./gla_batch_tmux.sh --suite E10 --round all \
  --pairs "87:dart" \
  --gpus "1" \
  --gpu-plan "1"

echo ""
echo "=========================================="
echo "训练命令已执行"
echo "=========================================="
echo ""
echo "请检查日志以确认训练正常启动："
echo "  tail -f /home/user/mzs_h/log/step1_s87_dart_*.log"

