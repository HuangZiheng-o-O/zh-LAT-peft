#!/usr/bin/env python3
"""
深度调试 DART 数据集加载问题
"""
import os
import sys
sys.path.insert(0, "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")

os.environ["DART_LOCAL_DIR"] = "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart"

print("=" * 80)
print("DART Dataset Loading Debug")
print("=" * 80)

# Step 1: 检查数据文件
print("\n[Step 1] 检查 DART 数据文件")
print("-" * 80)
from pathlib import Path
dart_dir = Path("/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart")
if dart_dir.exists():
    print(f"✓ DART 目录存在: {dart_dir}")
    files = list(dart_dir.rglob("*.json")) + list(dart_dir.rglob("*.parquet"))
    print(f"  找到 {len(files)} 个数据文件:")
    for f in sorted(files)[:20]:  # 只显示前20个
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"    - {f.relative_to(dart_dir)} ({size_mb:.2f} MB)")
else:
    print(f"✗ DART 目录不存在: {dart_dir}")
    sys.exit(1)

# Step 2: 尝试直接用 HF datasets 加载
print("\n[Step 2] 使用 HF datasets 直接加载 JSON 文件")
print("-" * 80)
try:
    from datasets import load_dataset
    train_file = dart_dir / "train.json"
    if train_file.exists():
        print(f"加载: {train_file}")
        ds_hf = load_dataset("json", data_files={"train": str(train_file)})
        print(f"✓ HF datasets 加载成功:")
        print(f"  train 样本数: {len(ds_hf['train'])}")
        print(f"  字段: {ds_hf['train'].column_names}")
        print(f"\n  第一个样本:")
        first = ds_hf['train'][0]
        for k, v in first.items():
            v_str = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
            print(f"    {k}: {v_str}")
    else:
        print(f"✗ train.json 不存在: {train_file}")
except Exception as e:
    print(f"✗ HF datasets 加载失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Step 3: 测试 DartDataset 的 load_df
print("\n[Step 3] 测试 DartDataset.load_df()")
print("-" * 80)
try:
    from transformers import AutoTokenizer
    from dataset.dart_data import DartDataset
    
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B",
        trust_remote_code=True
    )
    
    # 创建数据集实例但不初始化数据
    print("创建 DartDataset 实例...")
    ds = DartDataset.__new__(DartDataset)
    ds.tokenizer = tokenizer
    ds.path = "GEM/dart"
    ds.split = "train"
    ds.mode = "lm"
    ds.sep_token = tokenizer.sep_token or getattr(tokenizer, "eos_token", "</s>")
    ds.df = None
    
    print("调用 load_df()...")
    df = ds.load_df()
    
    print(f"✓ load_df() 成功:")
    print(f"  DataFrame 行数: {len(df)}")
    print(f"  DataFrame 列: {list(df.columns)}")
    
    if len(df) > 0:
        print(f"\n  第一行数据:")
        first_row = df.iloc[0]
        for col in df.columns:
            val = first_row[col]
            val_str = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
            print(f"    {col}: {val_str}")
        
        # 检查关键字段
        print(f"\n  检查关键字段:")
        print(f"    'tripleset' 类型: {type(first_row.get('tripleset', None))}")
        print(f"    'source' 类型: {type(first_row.get('source', None))}")
        print(f"    'text' 类型: {type(first_row.get('text', None))}")
        
        if 'source' in df.columns:
            print(f"    'source' 示例: {first_row['source']}")
        if 'text' in df.columns:
            print(f"    'text' 示例: {first_row['text']}")
    else:
        print("✗ DataFrame 为空！")
        
except Exception as e:
    print(f"✗ load_df() 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Step 4: 测试 get_input_label
print("\n[Step 4] 测试 DartDataset.get_input_label()")
print("-" * 80)
try:
    if len(df) > 0:
        print("调用 get_input_label(0)...")
        input_text, label_text = ds.get_input_label(0)
        print(f"✓ get_input_label() 成功:")
        print(f"  input: {input_text[:200]}..." if len(input_text) > 200 else f"  input: {input_text}")
        print(f"  label: {label_text[:200]}..." if len(label_text) > 200 else f"  label: {label_text}")
    else:
        print("跳过（DataFrame 为空）")
except Exception as e:
    print(f"✗ get_input_label() 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Step 5: 测试 preproc
print("\n[Step 5] 测试 DartDataset.preproc()")
print("-" * 80)
try:
    if len(df) > 0:
        ds.max_seqlen = None  # 确保不会因为长度过滤
        print("调用 preproc(0)...")
        result = ds.preproc(0)
        if result is None:
            print("✗ preproc() 返回 None（样本被过滤）")
        else:
            input_ids, label_ids = result
            print(f"✓ preproc() 成功:")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  label_ids shape: {label_ids.shape}")
    else:
        print("跳过（DataFrame 为空）")
except Exception as e:
    print(f"✗ preproc() 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Step 6: 测试完整初始化（不使用缓存，只加载前10个）
print("\n[Step 6] 测试完整 DartDataset 初始化（subset_size=10）")
print("-" * 80)
try:
    print("创建 DartDataset(split='train', use_cache=False, subset_size=10)...")
    ds_full = DartDataset(tokenizer, split="train", use_cache=False, subset_size=10)
    print(f"✓ 初始化成功:")
    print(f"  数据集长度: {len(ds_full)}")
    print(f"  self.data 类型: {type(ds_full.data)}")
    print(f"  self.data 长度: {len(ds_full.data) if ds_full.data else 'None'}")
    
    if len(ds_full) > 0:
        print("\n  测试 __getitem__(0)...")
        sample = ds_full[0]
        print(f"  ✓ 样本获取成功:")
        print(f"    input_ids shape: {sample['input_ids'].shape}")
        print(f"    label_ids shape: {sample['label_ids'].shape}")
    else:
        print("\n✗ 数据集长度为 0！")
        print("  检查 self.data:")
        print(f"    self.data = {ds_full.data}")
        
except Exception as e:
    print(f"✗ 完整初始化失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("调试完成")
print("=" * 80)

