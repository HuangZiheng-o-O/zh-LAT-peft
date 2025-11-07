# DART Training Fix: num_samples=0 Error

## 问题描述

训练 DART 数据集时出现错误：
```
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

日志显示：
- ✓ 验证集缓存成功生成（`cache_GEM_dart_val_gen_part_*.pkl`）
- ✓ 训练集缓存文件成功写入（`cache_GEM_dart_train_part_*.pkl`）
- ✗ 但训练集最终为空（`num_samples=0`）

## 根本原因

在 `parallel_processor_fs.py` 的并行处理过程中：

1. **静默异常**：Worker 进程在处理样本时如果抛出异常，会导致该样本返回 `None`，但异常信息不会被打印
2. **过度过滤**：聚合时会过滤掉所有 `None` 值，如果所有样本都失败，最终数据集为空
3. **缺少诊断**：没有警告或错误信息说明有多少样本被过滤掉

可能导致样本返回 `None` 的原因：
- `get_input_label()` 抛出异常（如断言失败、KeyError 等）
- `preproc_input_label()` 处理失败
- `encode()` 编码失败
- 序列长度超过 `max_seqlen`（如果设置了）

## 解决方案

### 1. 改进 `parallel_processor_fs.py`

**添加异常处理和诊断信息**：

```python
# 在 _worker 方法中添加 try-except
try:
    out[idx] = self.func(idx)
except Exception as e:
    print(f"[Worker {worker_idx}] Error processing idx={idx}: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    out[idx] = None

# 在 aggregate_result 中添加诊断
none_count = sum(1 for o in output_all if o is None)
if none_count > 0:
    print(f"Warning: {none_count}/{self.size} samples returned None (will be filtered out)")

if len(output_all) == 0:
    print(f"ERROR: All {self.size} samples were filtered out (all returned None)")
    print(f"Check worker logs above for errors during processing")
```

### 2. 清理旧缓存

旧的缓存文件可能已损坏或包含空数据：

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft
rm -f data/GEM_dart/cache_GEM_dart_train*.pkl
rm -f data/GEM_dart/parts/cache_GEM_dart_train_part_*.pkl
```

### 3. 验证数据集加载

在重新训练前，先验证数据集可以正确加载：

```bash
# 使用提供的测试脚本
bash /tmp/fix_dart_training.sh
```

或手动测试：

```python
import os
os.environ["DART_LOCAL_DIR"] = "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart"

from transformers import AutoTokenizer
from dataset.dart_data import DartDataset

tokenizer = AutoTokenizer.from_pretrained(
    "/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B",
    trust_remote_code=True
)

# 测试训练集（不使用缓存）
ds_train = DartDataset(tokenizer, split="train", use_cache=False)
print(f"Train samples: {len(ds_train)}")

# 测试第一个样本
sample = ds_train[0]
print(f"input_ids shape: {sample['input_ids'].shape}")
print(f"label_ids shape: {sample['label_ids'].shape}")
```

### 4. 重新运行训练

清理缓存并验证数据集后，重新运行训练命令：

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

## 预期行为

修复后，训练启动时应该看到：

1. **缓存生成**：
   ```
   Wrote data/GEM_dart/parts/cache_GEM_dart_train_part_000.pkl
   Wrote data/GEM_dart/parts/cache_GEM_dart_train_part_001.pkl
   ...
   Aggregating: 100%|██████████| 16/16 [00:00<00:00, ...]
   ```

2. **如果有样本失败**（现在会显示）：
   ```
   [Worker 0] Error processing idx=123: AssertionError: ...
   Traceback (most recent call last):
     ...
   Warning: 5/30526 samples returned None (will be filtered out)
   ```

3. **训练开始**：
   ```
   trainable params: 2,752,512 || all params: 1,368,266,752 || trainable%: 0.201...
   Loaded model
   [训练循环开始，不再报 num_samples=0]
   ```

## 后续调试

如果问题仍然存在，新的错误信息会明确指出：
- 哪个 worker 处理哪个样本时失败
- 具体的异常类型和消息
- 完整的 traceback

这将帮助快速定位问题（如 tokenizer 配置、数据格式、内存限制等）。

## 相关文件

- `/Users/huangziheng/.../mamba-peft/utils/parallel_processor_fs.py` - 已修复
- `/Users/huangziheng/.../mamba-peft/dataset/dart_data.py` - DART 数据集实现
- `/Users/huangziheng/.../mamba-peft/dataset/base.py` - 基础数据集类
- `/tmp/fix_dart_training.sh` - 自动修复脚本

## 技术细节

### 为什么验证集成功但训练集失败？

可能的原因：
1. **模式不同**：训练集使用 `mode="lm"`，验证集使用 `mode="gen"`，处理逻辑不同
2. **数据分布**：训练集可能包含更多边缘情况或异常数据
3. **缓存时机**：验证集后处理，可能在训练集失败后环境已改变

### 并行处理的陷阱

`multiprocessing.Process` 的子进程：
- 不会自动传播异常到主进程
- `print()` 输出可能被缓冲或丢失
- 需要显式捕获和记录异常

修复后的代码确保：
- 所有异常都被捕获并打印
- 聚合时统计和报告失败样本数
- 提供清晰的错误消息指导调试

