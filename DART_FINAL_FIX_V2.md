# DART 验证集为空问题 - 最终修复

**时间**: 2025-11-08  
**问题**: 训练正常，评估时崩溃 `'NoneType' object is not subscriptable`  
**根本原因**: 验证集缓存为空（所有样本处理失败）

---

## 问题诊断

### 错误信息

```
File "trainer/mamba_trainer.py", line 150, in generation_step
    input_ids, label_ids = inputs["input_ids"], inputs["label_ids"]
TypeError: 'NoneType' object is not subscriptable
```

### 根本原因

1. **验证集缓存为空**：`self.data = []`
2. **DataLoader 返回 None**：空数据集导致 `inputs = None`
3. **所有样本处理失败**：`get_input_label()` 对所有样本抛出异常或返回无效数据

### 深层原因

**问题 1**: `to_str_list()` 函数（`dart_data.py` 第 263-275 行）无法处理嵌套列表/numpy 数组

```python
# 旧版（错误）
def to_str_list(x):
    if isinstance(x, list):
        out = []
        for e in x:
            if isinstance(e, (str, int, float)) or e is None:  # ← 嵌套列表被跳过
                s = "" if e is None else str(e)
                if s.strip() != "":
                    out.append(s)
        return out
    # ...
```

当 `build_lists()` 返回 `sources = [['WikiTableQuestions_mturk']]`（嵌套列表）时，`to_str_list()` 无法提取内层字符串，导致 `text` 列变成空列表。

**问题 2**: `get_input_label()` 中的异常处理（第 392-393 行）

```python
if len(text) == 0:
    raise ValueError(f"Sample {idx} has no valid text references after filtering")
```

当 `text` 为空时抛出异常，导致 `preproc()` 失败，样本被标记为 `None`。如果所有样本都失败，缓存变成空列表。

---

## 修复方案

### 修复 1: 增强 `to_str_list()` 处理嵌套结构

**文件**: `mamba-peft/dataset/dart_data.py`  
**位置**: 第 263-289 行

```python
# Ensure list[str] for both columns (hardened)
def to_str_list(x):
    # Handle numpy arrays first
    if isinstance(x, np.ndarray):
        x = x.tolist()
    
    if isinstance(x, list):
        out = []
        for e in x:
            # Recursively handle nested structures
            if isinstance(e, (list, np.ndarray)):
                # Flatten one level
                for sub_e in (e.tolist() if isinstance(e, np.ndarray) else e):
                    if isinstance(sub_e, (str, int, float)) or sub_e is None:
                        s = "" if sub_e is None else str(sub_e)
                        if s.strip() != "":
                            out.append(s)
            elif isinstance(e, (str, int, float)) or e is None:
                s = "" if e is None else str(e)
                if s.strip() != "":
                    out.append(s)
        return out
    if isinstance(x, (str, int, float)) or x is None:
        s = "" if x is None else str(x)
        return [s] if s.strip() != "" else []
    return []
out["source"] = out.get("source", pd.Series([[]] * len(out))).apply(to_str_list)
out["text"]   = out.get("text",   pd.Series([[]] * len(out))).apply(to_str_list)
```

**改进点**：
- 递归处理嵌套列表和 numpy 数组
- 展平一层嵌套
- 确保最终返回 `list[str]`

### 修复 2: 改进 `get_input_label()` 错误处理

**文件**: `mamba-peft/dataset/dart_data.py`  
**位置**: 第 387-429 行

```python
else:
    # need to handle multiple references (generation mode)
    # Ensure source and text are lists (not numpy arrays)
    if isinstance(source, np.ndarray):
        source = source.tolist()
    if isinstance(text, np.ndarray):
        text = text.tolist()
    
    # Ensure they are lists
    if not isinstance(source, list):
        source = [source] if source else []
    if not isinstance(text, list):
        text = [text] if text else []
    
    # Flatten nested lists (defensive)
    def flatten_once(lst):
        result = []
        for item in lst:
            if isinstance(item, (list, np.ndarray)):
                result.extend(item.tolist() if isinstance(item, np.ndarray) else item)
            else:
                result.append(item)
        return result
    
    text = flatten_once(text)
    source = flatten_once(source)
    
    # Filter out any non-string elements
    text = [str(t).strip() for t in text if t is not None and str(t).strip()]
    
    if len(text) == 0:
        # Don't raise, return None so preproc filters it out
        print(f"[DART] Warning: Sample {idx} has no valid text after filtering, skipping")
        return None, None
    
    # Check for sep_token collision
    if any(self.sep_token in t for t in text):
        print(f"[DART] Warning: Sample {idx} contains sep_token '{self.sep_token}', replacing with space")
        text = [t.replace(self.sep_token, " ") for t in text]
    
    label = self.sep_token.join(text)

return input, label
```

**改进点**：
- 添加 `flatten_once()` 函数处理嵌套列表
- 不抛出异常，而是返回 `(None, None)`
- 将 sep_token 冲突从断言改为警告+替换

### 修复 3: 在 `base.py` 中处理 None 返回值

**文件**: `mamba-peft/dataset/base.py`  
**位置**: 第 111-124 行

```python
def preproc(self, idx):
    input, label = self.get_input_label(idx)
    
    # Handle case where get_input_label returns (None, None) for invalid samples
    if input is None or label is None:
        return None
    
    input_prepoc, label_preproc = self.preproc_input_label(input, label)
    input_ids, label_ids = self.encode(input_prepoc), self.encode(label_preproc)

    if self.max_seqlen is not None and (input_ids.shape[0] + label_ids.shape[0]) > self.max_seqlen:
        return None

    return input_ids, label_ids
```

**改进点**：
- 在编码前检查 `input` 和 `label` 是否为 `None`
- 提前返回 `None`，避免后续处理失败

---

## 部署步骤

### 步骤 1: 上传修复后的文件到服务器

```bash
# 从 Mac 上传
scp /Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft/dataset/dart_data.py \
    user@your-server:/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/dataset/

scp /Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft/dataset/base.py \
    user@your-server:/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/dataset/
```

### 步骤 2: 在服务器上清理缓存

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft

# 删除所有 DART 缓存（训练集和验证集）
rm -fv data/GEM_dart/cache_GEM_dart_*.pkl
rm -rfv data/GEM_dart/parts/

# 清理 Python 缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
```

### 步骤 3: 测试验证集加载

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft

python3 - <<'PY'
import os, sys
sys.path.insert(0, ".")
os.environ["DART_LOCAL_DIR"] = "data/GEM_dart"

from transformers import AutoTokenizer
from dataset.dart_data import DartDataset

tok = AutoTokenizer.from_pretrained(
    "/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B",
    trust_remote_code=True
)

print("测试验证集加载...")
ds_val = DartDataset(tok, split="val", mode="gen", use_cache=True)
print(f"✓ 验证集: {len(ds_val)} 样本")

if len(ds_val) > 0:
    sample = ds_val[0]
    print(f"✓ 第一个样本:")
    print(f"    input_ids: {sample['input_ids'].shape}")
    print(f"    label_ids: {sample['label_ids'].shape}")
else:
    print("✗ 验证集仍然为空！")
    sys.exit(1)

print("\n✓✓✓ 验证集加载成功！")
PY
```

### 步骤 4: 重启训练

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export DART_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart/

EVAL_GEN=1 \
EVAL_GEN_MAX_LENGTH=128 \
EVAL_GEN_MIN_LENGTH=5 \
EVAL_GEN_NUM_BEAMS=4 \
HP_EVAL_STEPS=3000 \
HP_SAVE_STEPS=3000 \
HP_LOGGING_STEPS=300 \
SWANLAB_ENABLE=1 \
SWANLAB_MODE=cloud \
SWANLAB_PROJECT="gla-mamba-dart-fixed" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUM_DATA_WORKERS=8 \
PREFETCH_FACTOR=4 \
GRADIENT_CHECKPOINTING=true \
LOGITS_TO_KEEP=1 \
./gla_batch_tmux.sh --suite E5 --round all \
  --pairs "87:dart" \
  --gpus "1 2 3 4 5" \
  --gpu-plan "2,2,2,2,2"
```

---

## 验证修复

修复后，应该看到：

1. **缓存生成时**：
   ```
   Parallel processing: 100%|██████| 2768/2768 [00:05<00:00, 500.00it/s]
   Warning: 10/2768 samples returned None (will be filtered out)
   ✓ val_gen cache warmed: 2758 samples
   ```

2. **训练日志**：
   ```
   1%|▏ | 2000/156650 [12:00<10:30:00, 4.10it/s]
   Evaluate: 100%|██████| 2758/2758 [05:30<00:00, 8.35it/s]
   {'eval_meteor': 0.35, 'eval_bleu': 0.28, ...}
   ```

3. **不再出现**：
   ```
   ✗ TypeError: 'NoneType' object is not subscriptable
   ✗ ERROR: All samples were filtered out
   ```

---

## 修复总结

| 修复 | 文件 | 位置 | 作用 |
|------|------|------|------|
| 1 | `dart_data.py` | 263-289 | 处理嵌套列表/numpy 数组 |
| 2 | `dart_data.py` | 387-429 | 改进错误处理，返回 None 而不是抛异常 |
| 3 | `base.py` | 111-124 | 在 preproc 中处理 None 返回值 |

**关键改进**：
- ✅ 递归展平嵌套结构
- ✅ 优雅处理无效样本（返回 None）
- ✅ 防止空缓存导致训练崩溃
- ✅ 详细的警告信息便于调试

---

## 如果仍然失败

如果修复后仍然出现问题，运行诊断：

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft

python3 - <<'PY'
import os, sys
sys.path.insert(0, ".")
os.environ["DART_LOCAL_DIR"] = "data/GEM_dart"

from transformers import AutoTokenizer
from dataset.dart_data import DartDataset

tok = AutoTokenizer.from_pretrained(
    "/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B",
    trust_remote_code=True
)

# 测试单个样本（不使用缓存）
ds = DartDataset(tok, split="val", mode="gen", use_cache=False, subset_size=1)
df = ds.load_df()

print(f"DataFrame 行数: {len(df)}")
print(f"第一行 text 类型: {type(df.iloc[0]['text'])}")
print(f"第一行 text 值: {df.iloc[0]['text']}")

# 测试 get_input_label
try:
    input, label = ds.get_input_label(0)
    print(f"✓ get_input_label 成功")
    print(f"  input: {input[:100]}")
    print(f"  label: {label[:100]}")
except Exception as e:
    print(f"✗ get_input_label 失败: {e}")
    import traceback
    traceback.print_exc()
PY
```

将输出发送给我进行进一步诊断。

---

**修复完成！现在训练应该能够正常进行评估了。** 🚀

