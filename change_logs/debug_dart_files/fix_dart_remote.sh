#!/bin/bash
# 在远程服务器上执行此脚本以修复 DART 训练问题

set -e  # 遇到错误立即退出

echo "=========================================="
echo "DART Training Fix - Remote Execution"
echo "=========================================="
echo ""

# 切换到项目目录
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft

echo "Step 1: 清理损坏的 DART 训练集缓存"
echo "----------------------------------------"
# 删除所有训练集相关的缓存文件
rm -fv data/GEM_dart/cache_GEM_dart_train.pkl
rm -fv data/GEM_dart/cache_GEM_dart_train_gen.pkl
rm -fv data/GEM_dart/parts/cache_GEM_dart_train_part_*.pkl

echo ""
echo "Step 2: 验证 parallel_processor_fs.py 已更新"
echo "---------------------------------------------"
if grep -q "Error processing idx=" utils/parallel_processor_fs.py; then
    echo "✓ parallel_processor_fs.py 已包含错误处理代码"
else
    echo "✗ WARNING: parallel_processor_fs.py 可能未正确更新"
    echo "  请确认文件已从本地上传到远程服务器"
fi

echo ""
echo "Step 3: 快速验证数据集加载（不使用缓存）"
echo "------------------------------------------"
cat > /tmp/test_dart_quick.py << 'PYTEST'
import os
import sys
sys.path.insert(0, "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")

os.environ["DART_LOCAL_DIR"] = "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart"

print("Importing modules...")
from transformers import AutoTokenizer
from dataset.dart_data import DartDataset

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B",
    trust_remote_code=True
)

print("Loading DART train dataset (no cache, first 10 samples only)...")
try:
    # 只加载前10个样本来快速验证
    ds_train = DartDataset(tokenizer, split="train", use_cache=False, subset_size=10)
    print(f"✓ Train subset loaded: {len(ds_train)} samples")
    
    # 测试第一个样本
    print("Testing first sample...")
    sample = ds_train[0]
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  label_ids shape: {sample['label_ids'].shape}")
    print("✓ First sample processed successfully")
    print("")
    print("=== Quick validation PASSED ===")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTEST

python /tmp/test_dart_quick.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ 数据集验证失败，请检查上面的错误信息"
    exit 1
fi

echo ""
echo "Step 4: 重新运行训练（将重建完整缓存）"
echo "----------------------------------------"
echo "现在可以重新运行训练命令。训练启动时会："
echo "  1. 并行处理所有训练样本"
echo "  2. 显示任何处理错误（如果有）"
echo "  3. 生成新的缓存文件"
echo "  4. 开始训练循环"
echo ""
echo "执行以下命令："
echo ""
echo "cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new"
echo ""
echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
echo "TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\"
echo "NUM_DATA_WORKERS=4 \\"
echo "GRADIENT_CHECKPOINTING=true \\"
echo "LOGITS_TO_KEEP=1 \\"
echo "HP_EVAL_STEPS=2000 HP_SAVE_STEPS=2000 HP_LOGGING_STEPS=200 \\"
echo "EVAL_GEN=1 EVAL_GEN_MAX_LENGTH=128 EVAL_GEN_MIN_LENGTH=5 EVAL_GEN_NUM_BEAMS=5 \\"
echo "./gla_batch_tmux.sh --suite E10 --round all \\"
echo "  --pairs \"87:dart\" \\"
echo "  --gpus \"1\" \\"
echo "  --gpu-plan \"1\""
echo ""
echo "=========================================="
echo "准备完成！请执行上面的训练命令。"
echo "=========================================="

