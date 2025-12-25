#!/usr/bin/env python3
"""验证 DART 修复"""
import os
import sys
sys.path.insert(0, "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")
os.environ["DART_LOCAL_DIR"] = "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart"

print("=" * 80)
print("验证 DART 修复")
print("=" * 80)

from transformers import AutoTokenizer
from dataset.dart_data import DartDataset

print("\n[1] 加载 tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    "/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B",
    trust_remote_code=True
)
print("✓ tokenizer 加载成功")

print("\n[2] 测试 DartDataset.load_df()")
print("-" * 80)
ds = DartDataset.__new__(DartDataset)
ds.tokenizer = tokenizer
ds.path = "GEM/dart"
ds.split = "train"
ds.mode = "lm"
ds.sep_token = tokenizer.sep_token or getattr(tokenizer, "eos_token", "</s>")
ds.df = None

df = ds.load_df()
print(f"✓ load_df() 返回: {len(df)} 行")

if len(df) > 0:
    print(f"  列: {list(df.columns)}")
    print(f"\n  第一行:")
    first = df.iloc[0]
    for col in df.columns:
        val = first[col]
        val_str = str(val)[:80] + "..." if len(str(val)) > 80 else str(val)
        print(f"    {col}: {val_str}")
    print("\n✓✓✓ 修复成功！数据加载正常！")
else:
    print("✗ 仍然返回空 DataFrame")
    sys.exit(1)

print("\n[3] 测试完整初始化（使用缓存，前10个样本）")
print("-" * 80)
# 清理可能存在的 subset 缓存
from pathlib import Path
subset_cache = Path("data/GEM_dart/cache_GEM_dart_train_10.pkl")
if subset_cache.exists():
    subset_cache.unlink()
    print(f"  清理旧缓存: {subset_cache}")

ds_full = DartDataset(
    tokenizer,
    split="train",
    use_cache=True,  # 必须为 True 才能初始化 self.data
    num_parallel_workers=0,
    subset_size=10
)
print(f"✓ 初始化成功: {len(ds_full)} 个样本")

if len(ds_full) > 0:
    sample = ds_full[0]
    print(f"✓ 第一个样本:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  label_ids shape: {sample['label_ids'].shape}")
    print("\n✓✓✓ 完整初始化成功！")
else:
    print("✗ 数据集长度为 0")
    sys.exit(1)

print("\n" + "=" * 80)
print("所有测试通过！DART 数据集修复成功！")
print("=" * 80)
print("\n现在可以重新运行训练了：")
print("  1. 确保已上传修改后的 dart_data.py 到远程服务器")
print("  2. 清理旧缓存: rm -f data/GEM_dart/cache_GEM_dart_train*.pkl")
print("  3. 重新运行训练命令")

