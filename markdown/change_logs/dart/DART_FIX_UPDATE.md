# DART 修复更新 - 第二轮

## 🔍 新发现的问题

在实际训练时，发现了第二个 `numpy.ndarray` 相关的问题：

```
ValueError: The truth value of an array with more than one element is ambiguous. 
Use a.any() or a.all()
```

**位置**：`dart_data.py` 第 339 行的 `linearize_triples` 方法

```python
triples = triples or []  # ← 当 triples 是 numpy.ndarray 时失败
```

## ✅ 修复方案

### 修改 `linearize_triples` 方法

**第 339-340 行**，将：
```python
triples = triples or []
```

改为：
```python
# Handle numpy.ndarray (pandas may convert lists to arrays)
if triples is None or (isinstance(triples, (list, np.ndarray)) and len(triples) == 0):
    triples = []
```

### 完整修改后的方法

```python
def linearize_triples(self, triples):
    def as_str(x):
        s = "" if x is None else str(x)
        return s.replace("\n", " ").strip()

    # Handle numpy.ndarray (pandas may convert lists to arrays)
    if triples is None or (isinstance(triples, (list, np.ndarray)) and len(triples) == 0):
        triples = []
    return " | ".join([" : ".join(as_str(ti) for ti in t) for t in triples])
```

## 📋 执行步骤

### 1. 上传更新后的 `dart_data.py` 到远程服务器

确保文件已更新。

### 2. 清理所有缓存（包括部分缓存）

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft
rm -f data/GEM_dart/cache_GEM_dart_train*.pkl
rm -f data/GEM_dart/parts/cache_GEM_dart_train_part_*.pkl
```

**重要**：必须清理 `parts/` 目录下的部分缓存，因为之前的运行已经生成了损坏的缓存文件。

### 3. 重新运行训练

```bash
cd /home/user/mzs_h/code/zh-LAT-peft
bash run_dart_training.sh
```

或手动执行：

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUM_DATA_WORKERS=4 \
GRADIENT_CHECKPOINTING=true \
LOGITS_TO_KEEP=1 \
HP_EVAL_STEPS=2000 HP_SAVE_STEPS=2000 HP_LOGGING_STEPS=200 \
EVAL_GEN=1 EVAL_GEN_MAX_LENGTH=128 EVAL_GEN_MIN_LENGTH=5 EVAL_GEN_NUM_BEAMS=5 \
./gla_batch_tmux.sh --suite E10 --round all \
  --pairs "87:dart" \
  --gpus "1" \
  --gpu-plan "1"
```

## 🎯 预期结果

修复后，训练应该能够正常处理所有样本，不再出现 `ValueError`。

日志中应该看到：
```
Parallel processing: ...
Wrote data/GEM_dart/parts/cache_GEM_dart_train_part_000.pkl
...
Aggregating: 100%|██████████| 16/16 [00:00<00:00, ...]
Warning: X/62659 samples returned None (will be filtered out)  ← 如果有少量失败是正常的
trainable params: 2,752,512 || all params: 1,368,266,752
Loaded model
[训练开始]
```

## 📊 修复总结

### 两个 numpy.ndarray 问题

1. **`build_lists` 中的 `annotations` 字段**
   - 问题：`isinstance(ann, list)` 失败
   - 修复：改为 `isinstance(ann, (list, np.ndarray))`
   - 影响：导致 `load_df()` 返回空 DataFrame

2. **`linearize_triples` 中的 `tripleset` 字段**
   - 问题：`triples or []` 触发 numpy 歧义错误
   - 修复：显式检查 `None` 和空数组
   - 影响：导致并行处理时大量样本失败

### 根本原因

pandas 的 `to_pandas()` 方法会将 HF Dataset 中的嵌套 `list` 字段转换为 `numpy.ndarray` 以提高性能。但 numpy 数组在布尔上下文中的行为与 Python 列表不同：

- `list or []` ✓ 正常工作
- `np.array([]) or []` ✗ 抛出 `ValueError`
- `isinstance(arr, list)` ✗ 返回 `False`

### 经验教训

1. **处理 pandas DataFrame 时要考虑 numpy 类型**
2. **避免使用 `or` 运算符处理可能是数组的变量**
3. **使用 `isinstance(x, (list, np.ndarray))` 同时支持两种类型**
4. **改进的错误处理非常有价值**（让我们快速定位了这个问题）

## 🚀 下一步

上传修改后的文件，清理缓存，重新训练。这次应该能成功了！

如果还有其他错误，改进的 `parallel_processor_fs.py` 会显示详细信息。

