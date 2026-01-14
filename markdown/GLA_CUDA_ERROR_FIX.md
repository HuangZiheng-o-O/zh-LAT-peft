# GLA 模型训练中的 CUDA 错误修复文档

## 一、错误现象

### 1.1 错误信息
```
CUDA driver error: invalid argument
  File "/mnt/data4/mzs_h/code/zh-LAT-peft/mamba-peft/train_lat.py", line 370, in build_and_run_trainer_lat
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
  ...
  File "/mnt/data/cbm/software/anaconda3/envs/mzsz/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
```

### 1.2 发生时机
- **训练阶段正常**：前 5 个 step 训练损失正常下降（6.1894 → 4.8091）
- **首次评估时崩溃**：在 `HP_EVAL_STEPS=500` 触发评估时，执行 `lm_head` 线性层时报错

### 1.3 关键日志
```
{'loss': 4.8091, 'grad_norm': 1.828125, 'learning_rate': 1.697648756472286e-05, 'epoch': 0.01}
[gpu-memory] peak_alloc=8020.586 MiB, peak_reserved=9406.0 MiB
swanlab: Error happened while training
```

---

## 二、根本原因分析

### 2.1 核心问题

GLA（Gated Linear Attention）模型在评估阶段，默认启用了 **KV Cache** (`use_cache=True`)，与 **变长序列批处理** 的 unpadding/padding 策略冲突，导致张量形状不匹配。

### 2.2 三个关键机制的冲突

#### 机制 1：GLA 的 Unpadding 策略
为了提高计算效率，GLA 在处理批量数据时会：

```python
# fla/layers/gla.py:196-198
if attention_mask is not None:
    indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
    hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
```

**作用**：移除 padding tokens，只计算有效 token，节省显存和计算。

**示例**：
```
原始批次（batch_size=2）:
  seq1: [token1, token2, PAD, PAD]  # 实际长度 2
  seq2: [token1, token2, token3, PAD]  # 实际长度 3

Unpad 后（total_valid_tokens=5）:
  [token1, token2, token1, token2, token3]

cu_seqlens = [0, 2, 5]  # 累积序列长度，用于区分不同样本
```

#### 机制 2：GLA 的 KV Cache
生成任务中为了避免重复计算，GLA 维护一个递归状态：

```python
# fla/layers/gla.py:243-252
recurrent_state = last_state['recurrent_state'] if last_state is not None else None
if mode == 'fused_recurrent':
    o, recurrent_state = fused_recurrent_gla(
        q=q, k=k, v=v, gk=gk,
        initial_state=recurrent_state,  # 传入上一步的状态
        output_final_state=use_cache,   # 是否输出新状态
        cu_seqlens=cu_seqlens,
    )
```

#### 机制 3：GLA 配置的默认值
```python
# fla/models/gla/configuration_gla.py:35
use_cache: bool = True

# fla/models/gla/modeling_gla.py:209
use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
```

**行为**：
- 训练时 (`model.training=True`)：强制 `use_cache=False` ✓
- 评估时 (`model.eval()`)：默认 `use_cache=True` ✗

### 2.3 为什么会冲突？

在 **评估阶段** 的批处理中：

```
Step 1: Unpad
  Input: [batch=64, seq_len=128, hidden=2048]
  Output: [1, total_valid_tokens, hidden]  # total_valid_tokens 可能是 6000+

Step 2: Forward with Cache (错误！)
  - Cache 对象期望固定的 batch 维度
  - Unpad 后的形状 [1, 6000+, ...] 与 Cache 维护的形状不匹配

Step 3: Repad
  - 尝试将 [1, 6000+, hidden] 恢复到 [64, 128, hidden]
  - 但由于 Cache 造成的中间状态混乱，索引失效

Result: CUDA 内核收到非法参数 → "invalid argument"
```

---

## 三、技术细节深入

### 3.1 为什么训练阶段没问题？

```python
# transformers/trainer.py 的行为
def training_step():
    model.train()  # 设置 model.training = True
    loss = model(**inputs)  # GLA 内部强制 use_cache=False

def evaluation_loop():
    model.eval()  # 设置 model.training = False
    logits = model(**inputs)  # GLA 使用 config.use_cache = True (默认)
```

### 3.2 Trainer 的调用链

```
trainer.train()
  └─> _inner_training_loop()
      ├─> training_step() [前 5 步] → 成功
      └─> _maybe_log_save_evaluate()
          └─> evaluate() [第 500 步]
              └─> evaluation_loop()
                  └─> prediction_step()
                      └─> generic_lm_trainer._forward()
                          └─> model(input_ids, **add_inputs)  ← 没传 use_cache=False
```

### 3.3 为什么评估 batch 更大时更容易出错？

```bash
export HP_BATCH_SIZE=8          # 训练批次小
export HP_EVAL_BATCH_SIZE=64    # 评估批次大
```

- 批次越大 → Unpad 后的 `total_valid_tokens` 越多
- Cache 状态张量越大 → 索引越容易越界
- 显存压力更大 → CUDA 驱动更敏感

---

## 四、解决方案

### 4.1 修复代码

**文件**：`mamba-peft/trainer/generic_lm_trainer.py`

**位置**：`_forward` 方法

```python
def _forward(self, model, inputs):
    input_ids = inputs["input_ids"]
    label_ids = inputs["label_ids"]
    attention_mask = inputs.get("attention_mask")

    add_inputs = {}
    if attention_mask is not None:
        add_inputs["attention_mask"] = attention_mask

    # 处理 PEFT 模型
    if isinstance(model, PeftModel):
        base = model.base_model
        if "label_ids" in base.forward.__code__.co_varnames:
            add_inputs["label_ids"] = label_ids

    # ========== 修复：显式禁用 Cache ==========
    # 原因：GLA 的 use_cache=True 与 unpadding 策略冲突
    # 影响：仅影响 KV Cache，不影响数据集缓存
    add_inputs["use_cache"] = False
    # =========================================

    lm_logits = model(input_ids, **add_inputs).logits
    return input_ids, label_ids, lm_logits
```

### 4.2 为什么这样修复有效？

| 场景 | 原行为 | 新行为 |
|------|--------|--------|
| 训练 | `use_cache=False` (GLA 强制) | `use_cache=False` (显式传入) |
| 评估 | `use_cache=True` (配置默认) ✗ | `use_cache=False` (显式传入) ✓ |
| 生成 | 不受影响（通过 `generation_step`） | 不受影响 |

### 4.3 副作用检查

**Q: 会影响性能吗？**
A: 不会。训练/评估是全序列并行计算，不需要 KV Cache。

**Q: 会影响数据集加载吗？**
A: 不会。这是模型参数，与 `HF_DATASETS_CACHE` 无关。

**Q: 会影响生成任务吗？**
A: 不会。生成任务通过 `eval_generator` 单独处理，可以独立控制 `use_cache`。

---

## 五、概念辨析

### 5.1 三种"缓存"的区别

| 缓存类型 | 作用域 | 用途 | 相关参数 |
|---------|--------|------|---------|
| **KV Cache** | 模型前向传播 | 生成时复用历史 K/V | `model(use_cache=...)` |
| **数据集缓存** | 磁盘/内存 | 避免重复下载/预处理 | `HF_DATASETS_CACHE` |
| **梯度检查点** | 反向传播 | 权衡显存与计算 | `GRADIENT_CHECKPOINTING` |

### 5.2 KV Cache 的正确使用场景

#### 场景 1：自回归生成（需要 use_cache=True）
```python
# 逐个 token 生成，缓存之前的计算结果
for i in range(max_new_tokens):
    logits, cache = model(input_ids[:, -1:], past_key_values=cache, use_cache=True)
    next_token = logits.argmax(-1)
    input_ids = torch.cat([input_ids, next_token], dim=-1)
```

#### 场景 2：训练/评估（需要 use_cache=False）
```python
# 一次性处理整个序列，每个 batch 独立
for batch in dataloader:
    logits = model(batch["input_ids"], use_cache=False)
    loss = criterion(logits, batch["labels"])
```

---

## 六、验证方法

### 6.1 检查修复是否生效

重新运行训练命令：
```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_mnli" --gpus "0" --gpu-plan "1" --model-type gla
```

**预期输出**：
```
{'loss': 6.1894, 'grad_norm': 4.0, 'learning_rate': 3.39e-06, 'epoch': 0.0}
...
{'loss': 4.8091, 'grad_norm': 1.828125, 'learning_rate': 1.69e-05, 'epoch': 0.01}
{'eval_loss': 1.234, 'eval_accuracy': 0.456, 'epoch': 0.01}  ← 评估成功
{'loss': 4.512, 'grad_norm': 1.2, 'learning_rate': 2.03e-05, 'epoch': 0.02}
...
```

### 6.2 添加调试日志（可选）

如果想验证 `use_cache` 的传递：

```python
# 在 generic_lm_trainer.py 的 _forward 方法中
add_inputs["use_cache"] = False
print(f"[DEBUG] use_cache={add_inputs.get('use_cache')}, training={model.training}")
```

**预期输出**：
```
[DEBUG] use_cache=False, training=True   # 训练阶段
[DEBUG] use_cache=False, training=False  # 评估阶段
```

---

## 七、延伸思考

### 7.1 为什么 GLA 默认 use_cache=True？

这是为了与 HuggingFace Transformers 的标准接口保持一致：
```python
# transformers/models/gpt2/modeling_gpt2.py
class GPT2Config:
    use_cache = True  # 大多数自回归模型的默认值
```

### 7.2 未来改进方向

**方案 A**：在 GLA 模型层面自动检测
```python
# fla/models/gla/modeling_gla.py
def forward(self, ..., use_cache=None):
    # 如果有 attention_mask 且有 padding，自动禁用 cache
    if attention_mask is not None and (attention_mask == 0).any():
        use_cache = False
```

**方案 B**：在 Trainer 配置中暴露参数
```python
@dataclass
class GenericLMTrainingArguments(TrainingArguments):
    eval_use_cache: bool = False  # 评估时是否使用 KV Cache
```

---

## 八、总结

| 维度 | 内容 |
|------|------|
| **问题** | GLA 评估时 `use_cache=True` 与 unpadding 冲突 |
| **症状** | CUDA driver error: invalid argument |
| **根因** | 张量形状在 Cache + Unpad/Repad 流程中混乱 |
| **修复** | 显式设置 `add_inputs["use_cache"] = False` |
| **影响** | 无副作用，训练/评估均不需要 KV Cache |
| **验证** | 评估阶段不再报错，正常输出 metrics |

---

## 九、参考资料

1. **GLA 论文**：[Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)
2. **FLA 库源码**：`fla/layers/gla.py` 的 `forward` 方法
3. **Unpadding 策略**：`fla/layers/utils.py` 的 `get_unpad_data` 函数
4. **HuggingFace Trainer**：`transformers/trainer.py` 的 `evaluation_loop` 方法

---

 