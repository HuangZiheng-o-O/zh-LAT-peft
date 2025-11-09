#!/usr/bin/env python3
"""对比两种加载方式的差异"""
import os
import sys
sys.path.insert(0, "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")
os.environ["DART_LOCAL_DIR"] = "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart"

from pathlib import Path
from datasets import load_dataset
import pandas as pd

print("=" * 80)
print("对比两种加载方式")
print("=" * 80)

train_file = Path("/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart/train.json")

# 方法1：直接 load_dataset("json", ...)
print("\n[方法1] 直接 load_dataset('json', data_files=...)")
print("-" * 80)
ds1 = load_dataset("json", data_files={"train": str(train_file)})["train"]
print(f"✓ 加载成功: {len(ds1)} 样本")
print(f"  字段: {ds1.column_names}")

df1 = ds1.to_pandas()
print(f"✓ to_pandas(): {len(df1)} 行")

first1 = df1.iloc[0]
print(f"\n第一行 annotations:")
print(f"  类型: {type(first1['annotations'])}")
print(f"  值: {first1['annotations']}")
if isinstance(first1['annotations'], list) and len(first1['annotations']) > 0:
    print(f"  首元素类型: {type(first1['annotations'][0])}")
    print(f"  首元素: {first1['annotations'][0]}")

# 方法2：通过 DartDataset.load_hf_dataset_split()
print("\n[方法2] 通过 DartDataset.load_hf_dataset_split()")
print("-" * 80)
from transformers import AutoTokenizer
from dataset.dart_data import DartDataset

tokenizer = AutoTokenizer.from_pretrained(
    "/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B",
    trust_remote_code=True
)

ds_obj = DartDataset.__new__(DartDataset)
ds_obj.tokenizer = tokenizer
ds_obj.path = "GEM/dart"
ds_obj.split = "train"
ds_obj.mode = "lm"
ds_obj.sep_token = tokenizer.sep_token or getattr(tokenizer, "eos_token", "</s>")
ds_obj.df = None

ds2 = ds_obj.load_hf_dataset_split()
print(f"✓ load_hf_dataset_split(): {len(ds2)} 样本")
print(f"  字段: {ds2.column_names}")

df2 = ds2.to_pandas()
print(f"✓ to_pandas(): {len(df2)} 行")

first2 = df2.iloc[0]
print(f"\n第一行 annotations:")
print(f"  类型: {type(first2['annotations'])}")
print(f"  值: {first2['annotations']}")
if isinstance(first2['annotations'], list) and len(first2['annotations']) > 0:
    print(f"  首元素类型: {type(first2['annotations'][0])}")
    print(f"  首元素: {first2['annotations'][0]}")

# 对比
print("\n" + "=" * 80)
print("对比结果")
print("=" * 80)
print(f"方法1 annotations 类型: {type(first1['annotations'])}")
print(f"方法2 annotations 类型: {type(first2['annotations'])}")

if type(first1['annotations']) != type(first2['annotations']):
    print("\n✗ 类型不同！这就是问题所在。")
else:
    print("\n✓ 类型相同")
    
# 检查是否是 pandas 序列化问题
print("\n检查 pandas 序列化:")
print(f"方法1 annotations 是否为 pandas Series: {isinstance(first1['annotations'], pd.Series)}")
print(f"方法2 annotations 是否为 pandas Series: {isinstance(first2['annotations'], pd.Series)}")

# 尝试访问 annotations 的内容
print("\n尝试访问 annotations[0]:")
try:
    ann1_0 = first1['annotations'][0] if isinstance(first1['annotations'], list) else first1['annotations']
    print(f"方法1: {ann1_0}")
except Exception as e:
    print(f"方法1 失败: {e}")

try:
    ann2_0 = first2['annotations'][0] if isinstance(first2['annotations'], list) else first2['annotations']
    print(f"方法2: {ann2_0}")
except Exception as e:
    print(f"方法2 失败: {e}")

print("\n" + "=" * 80)
print("对比完成")
print("=" * 80)

