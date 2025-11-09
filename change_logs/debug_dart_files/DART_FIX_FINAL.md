# DART 训练修复 - 最终解决方案

## 🎯 问题根源

经过深入调试，找到了真正的问题：

**pandas 的 `to_pandas()` 方法会将 HF Dataset 中的 `list` 类型字段转换为 `numpy.ndarray`，但 `dart_data.py` 中的 `build_lists` 函数只检查 `isinstance(ann, list)`，导致所有数据被跳过，最终返回空 DataFrame。**

### 调试过程

1. **初始现象**：训练时报 `num_samples=0`
2. **第一次调试**：发现并行处理没有错误输出，怀疑是缓存问题
3. **第二次调试**：清理缓存后，发现 `load_df()` 返回 0 行
4. **第三次调试**：手动测试 `build_lists` 逻辑，发现返回空字符串
5. **第四次调试**：对比两种加载方式，发现 `annotations` 是 `numpy.ndarray` 而不是 `list`
6. **最终定位**：`isinstance(ann, list)` 检查失败，导致数据被跳过

## ✅ 修复方案

### 修改文件：`mamba-peft/dataset/dart_data.py`

**第 183 行**，将：
```python
if isinstance(ann, list):
```

改为：
```python
if isinstance(ann, (list, np.ndarray)):
```

这样可以同时处理 Python 原生 `list` 和 pandas 转换后的 `numpy.ndarray`。

### 完整修改

```python
def build_lists(row):
    # Prefer standard annotations
    if "annotations" in row and row["annotations"] is not None:
        ann = row["annotations"]
        # Handle both list and numpy.ndarray (pandas may convert lists to arrays)
        if isinstance(ann, (list, np.ndarray)):  # ← 关键修改
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
        # ... 其余代码保持不变
```

## 📋 执行步骤

### 1. 上传修改后的文件到远程服务器

确保 `mamba-peft/dataset/dart_data.py` 已更新。

### 2. 验证修复（可选但推荐）

```bash
cd /home/user/mzs_h/code/zh-LAT-peft
python verify_dart_fix.py
```

预期输出：
```
✓ load_df() 返回: 30526 行
✓✓✓ 修复成功！数据加载正常！
✓ 初始化成功: 10 个样本
✓✓✓ 完整初始化成功！
所有测试通过！DART 数据集修复成功！
```

### 3. 清理旧缓存

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft
rm -f data/GEM_dart/cache_GEM_dart_train*.pkl
rm -f data/GEM_dart/parts/cache_GEM_dart_train_part_*.pkl
```

### 4. 重新运行训练

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

## 🎉 预期结果

修复后，训练启动时应该看到：

```
Loading GLA model: ...
Parallel processing: 0it [00:00, ?it/s]
Wrote data/GEM_dart/parts/cache_GEM_dart_train_part_000.pkl
Wrote data/GEM_dart/parts/cache_GEM_dart_train_part_001.pkl
...
Aggregating: 100%|██████████| 16/16 [00:00<00:00, ...]
Dropping last batch
Trainable parameters: ...
trainable params: 2,752,512 || all params: 1,368,266,752 || trainable%: 0.201...
Loaded model
[训练循环开始，不再报 num_samples=0]
```

## 🔍 技术细节

### 为什么会出现这个问题？

1. **HF Datasets 的行为**：`datasets.Dataset.to_pandas()` 会将嵌套的 `list` 字段转换为 `numpy.ndarray` 以提高性能
2. **类型检查的陷阱**：`isinstance(arr, list)` 对 `numpy.ndarray` 返回 `False`
3. **静默失败**：代码没有抛出异常，而是返回空列表，导致难以调试

### 为什么之前的测试脚本成功了？

在 `trace_build_lists.py` 中，我们直接使用 `load_dataset("json", ...)` 并立即转换为 pandas，这种情况下 pandas 保留了原始的 `list` 类型。但在 `DartDataset.load_hf_dataset_split()` 中，Dataset 可能经过了其他处理（如 `train_test_split`），导致类型转换。

### 其他可能受影响的数据集

这个问题可能也影响其他使用类似模式的数据集（SAMSum、Spider 等）。建议检查并应用相同的修复。

## 📝 相关文件

- **修复的文件**：`mamba-peft/dataset/dart_data.py`
- **改进的文件**：`mamba-peft/utils/parallel_processor_fs.py`（添加了错误处理）
- **验证脚本**：
  - `verify_dart_fix.py` - 快速验证修复
  - `test_dartdataset_full.py` - 完整测试
  - `compare_loading_methods.py` - 对比加载方式
  - `trace_build_lists.py` - 追踪 build_lists 执行

## 🚀 后续建议

1. **检查其他数据集**：SAMSum 和 Spider 可能有相同问题
2. **添加单元测试**：为 `build_lists` 添加测试，覆盖 `list` 和 `numpy.ndarray` 两种情况
3. **改进错误处理**：在 `build_lists` 中添加日志，记录处理的数据类型

## 🎓 经验教训

1. **类型检查要全面**：处理 pandas DataFrame 时，要考虑 `numpy.ndarray`
2. **调试要深入**：不要满足于表面现象，要追踪到根本原因
3. **测试要真实**：测试环境要尽可能接近实际运行环境
4. **错误处理要完善**：静默失败比显式错误更难调试

---

**修复完成！** 🎉

现在 DART 数据集应该可以正常训练了。如果还有问题，请查看日志中的详细错误信息（得益于改进的 `parallel_processor_fs.py`）。

