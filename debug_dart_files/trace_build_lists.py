#!/usr/bin/env python3
"""追踪 build_lists 函数的执行"""
import os
import sys
sys.path.insert(0, "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")
os.environ["DART_LOCAL_DIR"] = "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart"

from pathlib import Path
from datasets import load_dataset
import pandas as pd

print("=" * 80)
print("追踪 build_lists 函数执行")
print("=" * 80)

# 1. 加载数据
train_file = Path("/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart/train.json")
print(f"\n加载: {train_file}")
ds_hf = load_dataset("json", data_files={"train": str(train_file)})
df = pd.DataFrame(ds_hf['train'])
print(f"✓ 加载 {len(df)} 行")

# 2. 定义 build_lists（从 dart_data.py 复制）
def build_lists(row):
    """从 dart_data.py 复制的 build_lists 函数"""
    # Prefer standard annotations
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
        if isinstance(ann, dict):
            # dict-of-lists or dict-of-str
            texts = ann.get("text") or ann.get("target") or ann.get("targets") or []
            sources = ann.get("source") or [""] * (len(texts) if isinstance(texts, list) else 1)
            if isinstance(texts, str):
                texts = [texts]
            if isinstance(sources, str):
                sources = [sources]
            if isinstance(texts, list) and not isinstance(sources, list):
                sources = [""] * len(texts)
            return sources, texts

    # Alternative fields when annotations missing
    texts = None
    if "references" in row and isinstance(row["references"], list):
        cand = []
        for r in row["references"]:
            if isinstance(r, dict) and "text" in r:
                cand.append(r["text"])
            elif isinstance(r, str):
                cand.append(r)
        texts = cand
    if texts is None and isinstance(row.get("targets"), list):
        texts = [t for t in row["targets"] if isinstance(t, str)]
    if texts is None and isinstance(row.get("target"), str):
        texts = [row["target"]]
    if texts is None and isinstance(row.get("text"), str):
        texts = [row["text"]]
    if texts is None:
        texts = []
    sources = [""] * len(texts)
    return sources, texts

# 3. 测试前5行
print("\n" + "=" * 80)
print("测试前 5 行")
print("=" * 80)

for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    print(f"\n[行 {idx}]")
    print(f"  annotations 类型: {type(row['annotations'])}")
    
    if isinstance(row['annotations'], list):
        print(f"  annotations 长度: {len(row['annotations'])}")
        if len(row['annotations']) > 0:
            first_ann = row['annotations'][0]
            print(f"  首个 annotation 类型: {type(first_ann)}")
            if isinstance(first_ann, dict):
                print(f"  首个 annotation 字段: {list(first_ann.keys())}")
                for k, v in first_ann.items():
                    v_str = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
                    print(f"    {k}: {v_str}")
    
    # 调用 build_lists
    try:
        sources, texts = build_lists(row)
        print(f"  build_lists 返回:")
        print(f"    sources: {sources}")
        print(f"    texts 长度: {len(texts)}")
        if texts:
            print(f"    texts[0]: {texts[0][:100]}..." if len(texts[0]) > 100 else f"    texts[0]: {texts[0]}")
    except Exception as e:
        print(f"  ✗ build_lists 失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# 4. 应用到所有行并统计
print("\n" + "=" * 80)
print("应用到所有行")
print("=" * 80)

try:
    built = df.apply(build_lists, axis=1, result_type="reduce")
    print(f"✓ apply 成功，结果类型: {type(built)}")
    print(f"  结果长度: {len(built)}")
    
    # 检查前几个结果
    print(f"\n前 3 个结果:")
    for i in range(min(3, len(built))):
        sources, texts = built.iloc[i]
        print(f"  [{i}] sources={sources}, texts 长度={len(texts)}")
    
    # 统计有效结果
    valid_count = sum(1 for x in built if x[1] and len(x[1]) > 0)
    print(f"\n✓ 有效行数: {valid_count} / {len(built)}")
    
    # 创建 DataFrame
    sources_col = built.apply(lambda x: x[0])
    texts_col = built.apply(lambda x: x[1])
    out = pd.DataFrame({
        "tripleset": df["tripleset"],
        "source": sources_col,
        "text": texts_col,
    })
    print(f"\n✓ 创建 DataFrame: {len(out)} 行")
    print(f"  列: {list(out.columns)}")
    
    # 检查第一行
    if len(out) > 0:
        print(f"\n第一行:")
        first = out.iloc[0]
        print(f"  tripleset: {first['tripleset']}")
        print(f"  source: {first['source']}")
        print(f"  text: {first['text']}")
        print(f"  text 类型: {type(first['text'])}")
        print(f"  text 长度: {len(first['text']) if isinstance(first['text'], list) else 'N/A'}")
    
    # 应用 to_str_list
    print(f"\n应用 to_str_list 过滤...")
    def to_str_list(x):
        if isinstance(x, list):
            out_list = []
            for e in x:
                if isinstance(e, (str, int, float)) or e is None:
                    s = "" if e is None else str(e)
                    if s.strip() != "":
                        out_list.append(s)
            return out_list
        if isinstance(x, (str, int, float)) or x is None:
            s = "" if x is None else str(x)
            return [s] if s.strip() != "" else []
        return []
    
    out["source"] = out["source"].apply(to_str_list)
    out["text"] = out["text"].apply(to_str_list)
    print(f"✓ 应用 to_str_list 后: {len(out)} 行")
    
    # 过滤空 text
    print(f"\n过滤空 text...")
    out_filtered = out[out["text"].apply(lambda lst: isinstance(lst, list) and len(lst) > 0)].reset_index(drop=True)
    print(f"✓ 过滤后: {len(out_filtered)} 行")
    
    if len(out_filtered) == 0:
        print("\n✗ 所有行都被过滤掉了！")
        print("  检查前 5 行的 text 字段:")
        for i in range(min(5, len(out))):
            text_val = out.iloc[i]['text']
            print(f"    [{i}] text={text_val}, 类型={type(text_val)}, 长度={len(text_val) if isinstance(text_val, list) else 'N/A'}")
    else:
        print(f"✓ 成功！最终数据集有 {len(out_filtered)} 行")
        
except Exception as e:
    print(f"✗ 处理失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("追踪完成")
print("=" * 80)

