#!/usr/bin/env python3
"""快速调试 DART 数据加载"""
import os
import sys
sys.path.insert(0, "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")
os.environ["DART_LOCAL_DIR"] = "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart"

print("=" * 80)
print("快速调试：DART 数据加载")
print("=" * 80)

# 1. 检查原始 JSON 文件
print("\n[1] 检查原始 train.json")
print("-" * 80)
from pathlib import Path
import json

train_file = Path("/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart/train.json")
if train_file.exists():
    print(f"✓ 文件存在: {train_file} ({train_file.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # 读取前3个样本
    with open(train_file) as f:
        samples = []
        for i, line in enumerate(f):
            if i >= 3:
                break
            samples.append(json.loads(line))
    
    print(f"\n前 3 个样本的字段:")
    for i, sample in enumerate(samples):
        print(f"\n样本 {i}:")
        for key in sample.keys():
            val = sample[key]
            val_type = type(val).__name__
            if isinstance(val, (list, dict)):
                val_len = len(val)
                print(f"  {key}: {val_type}(len={val_len})")
                if isinstance(val, list) and val_len > 0:
                    print(f"    首元素类型: {type(val[0]).__name__}")
                    if isinstance(val[0], dict):
                        print(f"    首元素字段: {list(val[0].keys())}")
            else:
                val_str = str(val)[:50]
                print(f"  {key}: {val_type} = {val_str}")
else:
    print(f"✗ 文件不存在: {train_file}")
    sys.exit(1)

# 2. 用 HF datasets 加载
print("\n[2] 使用 HF datasets 加载")
print("-" * 80)
from datasets import load_dataset

ds_hf = load_dataset("json", data_files={"train": str(train_file)})
print(f"✓ 加载成功: {len(ds_hf['train'])} 样本")
print(f"  字段: {ds_hf['train'].column_names}")

first = ds_hf['train'][0]
print(f"\n第一个样本详情:")
for k, v in first.items():
    v_type = type(v).__name__
    if isinstance(v, (list, dict)):
        print(f"  {k}: {v_type}(len={len(v)})")
        if isinstance(v, list) and len(v) > 0:
            print(f"    首元素: {v[0]}")
    else:
        print(f"  {k}: {v}")

# 3. 转换为 pandas DataFrame 并检查
print("\n[3] 转换为 pandas DataFrame")
print("-" * 80)
import pandas as pd

df = pd.DataFrame(ds_hf['train'])
print(f"✓ DataFrame: {len(df)} 行 × {len(df.columns)} 列")
print(f"  列: {list(df.columns)}")

# 检查 annotations 字段
if 'annotations' in df.columns:
    print(f"\n检查 'annotations' 字段:")
    ann_sample = df.iloc[0]['annotations']
    print(f"  类型: {type(ann_sample)}")
    print(f"  内容: {ann_sample}")
    
    if isinstance(ann_sample, list) and len(ann_sample) > 0:
        print(f"  首元素类型: {type(ann_sample[0])}")
        if isinstance(ann_sample[0], dict):
            print(f"  首元素字段: {list(ann_sample[0].keys())}")
            if 'text' in ann_sample[0]:
                print(f"  首元素['text']: {ann_sample[0]['text']}")

# 4. 模拟 build_lists 处理
print("\n[4] 模拟 build_lists 处理")
print("-" * 80)

def test_build_lists(row):
    """简化版的 build_lists 逻辑"""
    annotations = row.get("annotations")
    
    if annotations is None:
        return None, None
    
    # 尝试提取 text
    if isinstance(annotations, list):
        texts = []
        for ann in annotations:
            if isinstance(ann, dict) and 'text' in ann:
                texts.append(ann['text'])
        if texts:
            sources = [""] * len(texts)
            return sources, texts
    
    return None, None

# 测试前3行
print("测试前 3 行:")
for idx in range(min(3, len(df))):
    sources, texts = test_build_lists(df.iloc[idx])
    print(f"  行 {idx}: sources={sources}, texts={texts}")

# 5. 检查完整的 build_lists 处理后有多少行保留
print("\n[5] 检查经过 build_lists 后的结果")
print("-" * 80)

results = df.apply(test_build_lists, axis=1)
valid_count = sum(1 for r in results if r[0] is not None and r[1] is not None and len(r[1]) > 0)
print(f"✓ 有效行数: {valid_count} / {len(df)}")

if valid_count == 0:
    print("\n✗ 所有行都被过滤掉了！")
    print("  原因可能是:")
    print("  1. 'annotations' 字段格式不符合预期")
    print("  2. 'annotations' 中没有 'text' 字段")
    print("  3. 'text' 字段为空")
    print("\n  请检查上面的样本详情，看看数据格式是否正确。")
else:
    print(f"✓ 数据格式正确，应该可以正常加载")

print("\n" + "=" * 80)
print("调试完成")
print("=" * 80)

