#!/usr/bin/env python3
"""完整测试 DartDataset 的执行路径"""
import os
import sys
sys.path.insert(0, "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")
os.environ["DART_LOCAL_DIR"] = "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart"

print("=" * 80)
print("完整测试 DartDataset")
print("=" * 80)

from transformers import AutoTokenizer
from dataset.dart_data import DartDataset

print("\n[1] 加载 tokenizer")
print("-" * 80)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B",
    trust_remote_code=True
)
print(f"✓ tokenizer 加载成功")
print(f"  sep_token: {tokenizer.sep_token}")
print(f"  eos_token: {tokenizer.eos_token}")

print("\n[2] 创建 DartDataset 实例（手动，不初始化）")
print("-" * 80)
ds = DartDataset.__new__(DartDataset)
ds.tokenizer = tokenizer
ds.path = "GEM/dart"
ds.split = "train"
ds.mode = "lm"
ds.sep_token = tokenizer.sep_token or getattr(tokenizer, "eos_token", "</s>")
ds.df = None
print(f"✓ 实例创建成功")
print(f"  path: {ds.path}")
print(f"  split: {ds.split}")
print(f"  mode: {ds.mode}")
print(f"  sep_token: {ds.sep_token}")

print("\n[3] 调用 load_hf_dataset_split()")
print("-" * 80)
try:
    hf_ds = ds.load_hf_dataset_split()
    print(f"✓ load_hf_dataset_split() 成功")
    print(f"  类型: {type(hf_ds)}")
    print(f"  长度: {len(hf_ds)}")
    print(f"  字段: {hf_ds.column_names}")
    
    if len(hf_ds) > 0:
        print(f"\n  第一个样本:")
        first = hf_ds[0]
        for k, v in first.items():
            v_str = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
            print(f"    {k}: {v_str}")
    else:
        print("\n✗ load_hf_dataset_split() 返回空 Dataset！")
except Exception as e:
    print(f"✗ load_hf_dataset_split() 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4] 转换为 pandas DataFrame")
print("-" * 80)
try:
    df_raw = hf_ds.to_pandas()
    print(f"✓ to_pandas() 成功")
    print(f"  行数: {len(df_raw)}")
    print(f"  列: {list(df_raw.columns)}")
except Exception as e:
    print(f"✗ to_pandas() 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[5] 调用完整的 load_df()")
print("-" * 80)
try:
    # 重置 df
    ds.df = None
    df_final = ds.load_df()
    print(f"✓ load_df() 成功")
    print(f"  行数: {len(df_final)}")
    print(f"  列: {list(df_final.columns)}")
    
    if len(df_final) > 0:
        print(f"\n  第一行:")
        first_row = df_final.iloc[0]
        for col in df_final.columns:
            val = first_row[col]
            val_str = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
            print(f"    {col}: {val_str}")
    else:
        print("\n✗ load_df() 返回空 DataFrame！")
        print("\n  调试信息:")
        print(f"    原始 HF Dataset 长度: {len(hf_ds)}")
        print(f"    原始 pandas DataFrame 长度: {len(df_raw)}")
        print(f"    最终 DataFrame 长度: {len(df_final)}")
        
        # 尝试手动执行 load_df 的逻辑
        print("\n  手动执行 build_lists 逻辑:")
        import pandas as pd
        
        def build_lists(row):
            if "annotations" in row and row["annotations"] is not None:
                ann = row["annotations"]
                if isinstance(ann, list):
                    texts = []
                    sources = []
                    for a in ann:
                        if isinstance(a, dict):
                            t = a.get("text") or a.get("target") or a.get("reference")
                            s = a.get("source", "")
                            if isinstance(t, str) and t.strip():
                                texts.append(t)
                                sources.append(s)
                        elif isinstance(a, str):
                            texts.append(a)
                            sources.append("")
                    return sources, texts
            return [""], [""]
        
        # 测试前3行
        for i in range(min(3, len(df_raw))):
            row = df_raw.iloc[i]
            sources, texts = build_lists(row)
            print(f"    行 {i}: sources={sources}, texts={texts}")
        
except Exception as e:
    print(f"✗ load_df() 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[6] 测试完整初始化（use_cache=False, num_parallel_workers=0）")
print("-" * 80)
try:
    print("  创建 DartDataset(split='train', use_cache=False, num_parallel_workers=0, subset_size=10)...")
    ds_full = DartDataset(
        tokenizer, 
        split="train", 
        use_cache=False, 
        num_parallel_workers=0,  # 禁用并行处理
        subset_size=10
    )
    print(f"✓ 初始化成功")
    print(f"  数据集长度: {len(ds_full)}")
    print(f"  self.data 类型: {type(ds_full.data)}")
    print(f"  self.data 长度: {len(ds_full.data) if ds_full.data else 'None'}")
    
    if len(ds_full) > 0:
        print(f"\n  测试 __getitem__(0)...")
        sample = ds_full[0]
        print(f"  ✓ 样本获取成功:")
        print(f"    input_ids shape: {sample['input_ids'].shape}")
        print(f"    label_ids shape: {sample['label_ids'].shape}")
    else:
        print(f"\n✗ 数据集长度为 0！")
except Exception as e:
    print(f"✗ 初始化失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

