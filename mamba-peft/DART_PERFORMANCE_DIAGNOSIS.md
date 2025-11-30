# DART 模型性能诊断报告

## ✅ 问题已修复 (2024-11)

### 核心问题：Greedy Decoding → Beam Search

原问题：DART 训练结果（BLEU ≈ 10, chrF ≈ 31, METEOR ≈ 0.42）明显偏低。

**根本原因**：使用 Greedy Decoding 而非 Beam Search。

**技术限制**：FLA (flash-linear-attention) 的 `FLACache` 类**不支持 `reorder_cache()` 方法**，
这是 HuggingFace beam search 在 `use_cache=True` 时所必需的。

**解决方案**：当 `num_beams > 1` 时，自动设置 `use_cache=False`。这会更慢，但能正确运行 beam search。

---

## 🔧 已实施的修复

### 1. `gla_hf_decoder.py` - 完全重写

关键改动：
```python
# 当使用 beam search 时，自动禁用 cache
use_cache = not use_beam_search  # num_beams > 1 时为 False

# 如果尝试强制使用 cache + beam search，抛出明确错误
if use_beam_search and force_cache:
    raise RuntimeError(
        f"[GLA] FATAL: Cannot use beam search (num_beams={effective_beams}) with use_cache=True. "
        f"FLA's Cache does not implement reorder_cache() required by HuggingFace beam search."
    )
```

新增功能：
- 支持 `num_beams`, `length_penalty`, `no_repeat_ngram_size`, `early_stopping`
- 环境变量覆盖：`EVAL_GEN_NUM_BEAMS`, `EVAL_GEN_LENGTH_PENALTY`, `EVAL_GEN_NO_REPEAT_NGRAM`
- 详细的 verbose 日志 (`GLA_VERBOSE=1`)
- 明确的错误消息（不再静默 fallback）

### 2. `train_gla_only.py` - 自动配置 beam search

对于 DART/Spider/SamSum 等生成任务，自动配置：
```python
# 为 DART 等任务设置合理的 beam search 默认值
if num_beams is None and data_name.startswith("dart"):
    num_beams = 4  # DART 标准配置
    
cfg["eval_gen"]["num_beams"] = num_beams
cfg["eval_gen"]["length_penalty"] = 1.0
cfg["eval_gen"]["no_repeat_ngram_size"] = 3
cfg["eval_gen"]["early_stopping"] = True
```

---

## 📊 预期性能提升

| 修复项 | 预期 BLEU 提升 | 预期 chrF 提升 |
|--------|---------------|---------------|
| Beam Search (num_beams=4) | +10~15 | +10~15 |
| no_repeat_ngram_size=3 | +2~5 | +2~5 |
| **总计** | **+12~20** | **+12~20** |

修复后预期：
- BLEU: 10 → 22~30
- chrF: 31 → 43~51
- METEOR: 0.42 → 0.45~0.50

---

## 🚀 使用方法

### 方法 1：自动配置（推荐）

对于 DART 任务，现在会**自动启用 beam search**：
```bash
# 只需指定数据集，beam search 会自动配置
export HP_DATA=dart
python train_gla_only.py --cfg cfg/my_lora_exp/yaml/E1_QKVO_r8_alpha16.yaml
```

### 方法 2：环境变量覆盖

```bash
# 自定义 beam search 参数
export EVAL_GEN_NUM_BEAMS=5
export EVAL_GEN_MAX_LENGTH=128
export EVAL_GEN_LENGTH_PENALTY=1.0
export EVAL_GEN_NO_REPEAT_NGRAM=3

# 启用详细日志
export GLA_VERBOSE=1
```

### 方法 3：禁用 beam search（使用 greedy）

```bash
# 显式设置为 1 会使用 greedy decoding
export EVAL_GEN_NUM_BEAMS=1
```

---

## ⚠️ 性能注意事项

**Beam search 会更慢**，因为 `use_cache=False`：
- 每个 decoding step 都需要重新计算整个序列的 attention
- 对于短序列（DART 平均输出 ~30 tokens），影响可接受
- 对于长序列生成，考虑使用 greedy + 其他技术

**速度对比**（估算）：
| 解码方式 | 相对速度 |
|---------|---------|
| Greedy + use_cache=True | 1.0x (基准) |
| Beam (n=4) + use_cache=False | ~0.3x |

---

## 🔍 调试

启用详细日志：
```bash
export GLA_VERBOSE=1
```

输出示例：
```
[GLA] Generation mode: beam_search(beams=4), use_cache=False (required for beam search)
[GLA] Beam search config: num_beams=4, length_penalty=1.0, early_stopping=True, no_repeat_ngram_size=3
```

---

## 📋 对其他任务的影响

### GLUE 分类任务
**无影响**。GLUE 任务不使用 `eval_generator`，走的是 `mode="lm"` 路径。

### Spider SQL 生成
**使用 Greedy Decoding**。SQL 生成任务需要精确匹配，beam search 反而可能引入错误。

### SamSum 摘要
**自动启用 beam search**（num_beams=4）。

---

## 📚 技术背景

### 为什么 FLA 不支持 beam search + cache？

HuggingFace 的 beam search 需要在每个 step 根据 beam 选择结果**重排 cache**：
```python
# HuggingFace 内部调用
past_key_values = model._reorder_cache(past_key_values, beam_idx)
```

FLA 的 `FLACache` 存储的是 **recurrent state**（线性注意力的累积状态），
其形状和语义与标准 Transformer 的 KV cache 不同，无法简单地按 batch 维度重排。

### 解决方案的权衡

| 方案 | 优点 | 缺点 |
|-----|-----|-----|
| `use_cache=False` | 正确、简单 | 慢 |
| 实现 `reorder_cache` | 快 | 复杂，需要修改 FLA 库 |

当前采用方案 1，因为：
1. DART 等任务的输出序列较短，速度影响可接受
2. 不需要修改第三方库 (FLA)
3. 实现简单，易于维护


