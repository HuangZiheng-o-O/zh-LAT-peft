# SD-LoRA on GLA: 完整流程详解

> **SD-LoRA** = Sparse Dimension LoRA = 稀疏维度选择 + LoRA微调
>
> 本文档详细解释SD-LoRA在GLA (Gated Linear Attention) 模型上的完整工作流程。

---

## 一、核心思想：为什么需要SD-LoRA？

### 1.1 传统LoRA的局限

传统LoRA对**所有维度**一视同仁地进行低秩分解：
```
W' = W + BA    (B: d×r, A: r×k, r << min(d,k))
```

但研究发现：
- **不是所有维度都同等重要**
- 某些维度对任务几乎无贡献，可以直接置零
- 某些维度已经足够好，冻结即可
- 只有一小部分维度需要真正训练

### 1.2 SD-LoRA的核心策略

SD-LoRA将参数维度分为**三类**：

| 类型 | 比例示例 | 操作 | 原因 |
|------|----------|------|------|
| **Train (训练)** | 5% | 添加可训练adapter | 这些维度对任务最重要 |
| **Freeze (冻结)** | 95% | 保持原值不变 | 已经足够好，不需调整 |
| **Zero (置零)** | 0% | 强制设为极小值 | 对任务有害或无用 |

**关键问题**：如何知道哪些维度重要？→ **Warmup阶段**

---

## 二、两阶段训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    SD-LoRA 两阶段训练                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      自动切换        ┌──────────────────────┐ │
│  │  Phase 1     │  ──────────────────► │  Phase 2             │ │
│  │  WARMUP      │  (it_counter >       │  TRAINING            │ │
│  │  (100 steps) │   num_warmup_it)     │  (剩余全部epochs)     │ │
│  └──────────────┘                      └──────────────────────┘ │
│                                                                  │
│  目的: 收集梯度                         目的: 稀疏微调           │
│  计算维度重要性                         只训练重要维度           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 重要澄清：**不需要预先Full Fine-tune！**

SD-LoRA的Warmup阶段**不是**full fine-tuning，而是：
- 在**原始预训练模型**上直接开始
- 只运行**100个iteration**（不是100个epoch！）
- 目的是**收集梯度信息**，而不是真正更新模型

---

## 三、Phase 1: Warmup阶段详解

### 3.1 目标
找出哪些维度最重要，为后续稀疏训练做准备。

### 3.2 具体机制

```python
# gla_sd_lora.py: GlaSdLoraParameter.forward()

if self.sdlora_mode == "warmup":
    # 将梯度累加器加到原始权重上
    weight_new = weight + self.sdlora_alpha * self.sdlora_grad
    # sdlora_grad 初始化为零，但会在反向传播中累积梯度
```

**发生了什么：**

1. **初始化**：创建一个与权重同shape的零张量 `sdlora_grad`
2. **前向传播**：`weight_new = weight + sdlora_grad`（第一次等于原始weight）
3. **反向传播**：PyTorch自动计算 ∂Loss/∂sdlora_grad
4. **梯度累积**：`sdlora_grad` 不断更新，记录每个维度的梯度

### 3.3 重要性计算

```python
# gla_sd_lora.py: get_importances()

def get_importances(self, x, dim=0):
    """
    使用梯度的L2范数作为重要性指标
    梯度越大 → 该维度对Loss影响越大 → 越重要
    """
    norms = x.square().detach().sum(dim=1 if dim == 0 else 0)
    indices = torch.argsort(-norms)  # 降序排列
    return indices
```

**直觉解释**：
- 如果某个维度的梯度很大 → 这个维度对Loss影响大 → 需要训练
- 如果某个维度的梯度接近零 → 这个维度对Loss无影响 → 可以冻结/置零

### 3.4 Warmup持续多久？

```python
# 默认: num_warmup_it = 100
if self.it_counter > self.num_warmup_it:
    self.set_sdlora_mode("train")  # 自动切换到训练阶段
```

- **100个iteration**，不是100个epoch
- 假设batch_size=8，100 iterations = 800个样本
- 对于大多数任务，这足以估计维度重要性

---

## 四、Phase 1 → Phase 2: 维度选择

当Warmup结束时，根据累积的梯度信息进行维度划分：

### 4.1 划分逻辑

```python
# select_channels() 方法

importance_order = get_importances(sdlora_grad)  # 按重要性排序

# 假设 Train=5%, Freeze=95%, Zero=0%
# 假设总共有 1024 个channel

train_channels  = importance_order[0:51]      # 前5%最重要 → 训练
freeze_channels = importance_order[51:1024]   # 剩余95% → 冻结
zero_channels   = importance_order[1024:]     # 0% → 无
```

### 4.2 可视化

```
Channel重要性排序 (从高到低):
┌────────────────────────────────────────────────────────────────────┐
│ ████ TRAIN (5%)  │                    FREEZE (95%)                 │
│  最重要的51个     │                   剩余973个channel               │
└────────────────────────────────────────────────────────────────────┘
  ↑                                      ↑
  这些channel                            这些channel
  会被训练                               保持预训练值不变
```

---

## 五、Phase 2: Training阶段详解

### 5.1 机制

```python
# gla_sd_lora.py: forward() in train mode

elif self.sdlora_mode == "train":
    weight_new = self.build_train_param(weight, self.sdlora_adapter)
```

### 5.2 build_train_param() 详解

```python
def build_train_param(self, param, adapter):
    # 1. 首次调用时构建mask
    if self.train_mask is None:
        self.train_mask = self.get_mask("train")  # 标记哪些维度可训练
        self.zero_mask = self.get_mask("zero")    # 标记哪些维度置零

    # 2. 处理Zero维度：设为极大负值
    param_new = param.clone()
    if self.zero_mask.any():
        param_new = torch.where(
            self.zero_mask,
            torch.full_like(param, -100.0),  # ZERO_MASK_VALUE
            param_new
        )

    # 3. 处理Train维度：添加可训练adapter
    if self.train_mask.any():
        bias = torch.zeros_like(param)
        bias[self.train_mask] = adapter.flatten()
        param_new = param_new + self.sdlora_alpha * bias

    return param_new
```

### 5.3 为什么Zero用-100？

GLA的gate计算：
```python
gate = exp(logsigmoid(gk) / gate_logit_normalizer)  # normalizer=16
```

当 `gk = -100` 时：
```
logsigmoid(-100) ≈ -100
-100 / 16 = -6.25
exp(-6.25) ≈ 0.002  # 只保留0.2%的信息
```

这意味着这些维度的状态几乎完全衰减（被"遗忘"）。

---

## 六、LoRA组件：同时进行

SD-LoRA不只是稀疏维度选择，还同时在其他层上应用标准LoRA：

### 6.1 两种目标模块

| 模块类型 | 目标 | 技术 |
|----------|------|------|
| `target_modules` | `["gk_proj.1"]` | SDT (稀疏维度调优) |
| `lora_targets` | `["k_proj", "v_proj"]` 等 | 标准LoRA |

### 6.2 为什么需要LoRA？

SDT只作用于gate projection，但模型的其他投影层（Q/K/V/O）也需要适配下游任务。

```
GLA Attention Block:
┌─────────────────────────────────────────────────────┐
│                                                      │
│   q_proj ──────┐                                    │
│   k_proj ──────┼──► Attention ──► o_proj ──► output │
│   v_proj ──────┘         ↑                          │
│                          │                          │
│   gk_proj.1 ─────► gate ─┘  ← SDT作用于此            │
│                                                      │
└─────────────────────────────────────────────────────┘
         ↑
         │
    LoRA作用于这些
```

---

## 七、完整训练流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SD-LoRA 完整流程                             │
└─────────────────────────────────────────────────────────────────────┘

Step 0: 加载预训练GLA模型 (不需要任何预先微调!)
        │
        ▼
Step 1: 包装模型
        ├── gk_proj.1 → GlaSdLoraParameter (SDT)
        ├── k_proj    → LoraLinear (标准LoRA)
        ├── v_proj    → LoraLinear (标准LoRA)
        └── ...
        │
        ▼
Step 2: 开始训练 (mode = "warmup")
        │
        │   Iteration 1-100:
        │   ┌─────────────────────────────────────────────┐
        │   │ forward:  weight_new = weight + sdlora_grad │
        │   │ backward: 梯度累积到 sdlora_grad            │
        │   │ 同时: LoRA层也在正常训练                     │
        │   └─────────────────────────────────────────────┘
        │
        ▼
Step 3: 自动切换 (it_counter > 100)
        │
        │   计算维度重要性:
        │   importance = L2_norm(sdlora_grad, dim=input)
        │
        │   划分维度:
        │   ├── Train:  top 5% (最重要)
        │   ├── Freeze: 剩余95%
        │   └── Zero:   0% (本配置不使用)
        │
        ▼
Step 4: 继续训练 (mode = "train")
        │
        │   Iteration 101 ~ 结束:
        │   ┌─────────────────────────────────────────────┐
        │   │ forward:                                     │
        │   │   1. zero维度 → 设为-100                     │
        │   │   2. train维度 → 加上sdlora_adapter          │
        │   │   3. freeze维度 → 保持原值                   │
        │   │                                              │
        │   │ backward: 只有train维度的adapter有梯度       │
        │   │ 同时: LoRA层继续正常训练                     │
        │   └─────────────────────────────────────────────┘
        │
        ▼
Step 5: 训练结束
        │
        └── 保存: LoRA权重 + sdlora_adapter + 维度mask
```

---

## 八、配置文件详解

以 `gla_sdlora_kv_train05.json` 为例：

```json
{
    "peft_type": "GLA_SD_LORA",

    // SDT配置
    "select_mode": "CHANNELS_ONLY",        // 只在channel维度选择
    "target_modules": ["gk_proj.1"],        // SDT目标: gate key projection
    "num_zero": {"channel": 0.0},           // 0%置零
    "num_freeze": {"channel": 0.95},        // 95%冻结
    // → 隐含 num_train = 1 - 0 - 0.95 = 5% 训练

    "num_warmup_it": 100,                   // Warmup 100个iteration

    // LoRA配置
    "proj_lora_r": 8,                       // LoRA rank = 8
    "lora_targets": ["k_proj", "v_proj"],   // LoRA目标层

    // 缩放因子
    "sdlora_alpha": {
        "global": 1.0,
        "gk_proj.1": 1.0
    }
}
```

### 维度比例解读

| 配置名 | train_ratio | freeze_ratio | zero_ratio | 含义 |
|--------|-------------|--------------|------------|------|
| train01 | 1% | 99% | 0% | 极度保守，只训练1%最重要维度 |
| train05 | 5% | 95% | 0% | 保守策略 |
| train10 | 10% | 90% | 0% | 中等策略 |
| train20 | 20% | 80% | 0% | 较激进 |
| train30 | 30% | 70% | 0% | 激进策略 |

---

## 九、Epoch数量的影响

### 9.1 Warmup阶段不受epoch影响

Warmup只看iteration数（默认100），与epoch无关：
```python
if self.it_counter > self.num_warmup_it:  # 只看iteration
    self.set_sdlora_mode("train")
```

### 9.2 不同任务的建议epoch

| 任务类型 | 数据量 | 建议epoch | 原因 |
|----------|--------|-----------|------|
| CoLA | ~8.5k | 10 | 小数据需要更多迭代 |
| RTE | ~2.5k | 10 | 小数据需要更多迭代 |
| MRPC | ~3.7k | 10 | 小数据需要更多迭代 |
| SST-2 | ~67k | 4 | 中等数据标准配置 |
| QNLI | ~104k | 4 | 中等数据标准配置 |
| MNLI | ~392k | 3 | 大数据少epoch |
| QQP | ~363k | 5 | 大数据但有噪声 |

---

## 十、GLA模型特殊性

### 10.1 为什么SDT目标是 `gk_proj.1`？

GLA的核心是**Gated Linear Attention**：
```python
# GLA attention公式 (简化版)
gate = exp(logsigmoid(gk) / normalizer)  # gk来自gk_proj
output = gate * attention_output
```

`gk_proj` 控制信息的**门控衰减**：
- gate接近1 → 完全保留信息
- gate接近0 → 完全遗忘信息

通过调整`gk_proj`的特定维度，可以精准控制哪些"记忆通道"被使用。

### 10.2 GLA vs Mamba的SDT差异

| 方面 | Mamba SD-LoRA | GLA SD-LoRA |
|------|---------------|-------------|
| 目标模块 | A_log (离散化参数) | gk_proj.1 (gate投影) |
| 维度选择 | state + channel | channel only |
| Zero值 | +10 (大正值) | -100 (大负值) |
| 原因 | A_log用exp | gk用logsigmoid |

---

## 十一、训练时的自动切换机制

### 11.1 模型内部检测

```python
# GlaSdLoraModel.should_training_stop

@property
def should_training_stop(self):
    if self.last_mode == "warmup" and self.get_sdlora_mode() == "train":
        self.last_mode = "train"
        return True  # 告诉trainer暂停
    return False
```

### 11.2 Trainer响应

当 `should_training_stop=True` 时，训练器会：
1. 暂时停止当前训练循环
2. 让SD-LoRA完成维度选择
3. 然后继续剩余的训练

**用户无需手动干预！**

---

## 十二、完整代码调用链

```
用户运行: ./sdlora_spider.sh
          │
          ▼
lat_batch_tmux_sparse.sh
          │ 遍历 SEED:DATA pairs
          ▼
lat_round_sparse.sh
          │ 遍历 ROUND_SPARSE[0..14] (15个配置)
          │ 调用: python train_lat.py --peft xxx.json
          ▼
train_lat.py
          │ 解析 --peft 参数
          │ 调用 run_train(peft=path)
          ▼
lat_adapter.py::prepare_lat_model_and_tokenizer()
          │ _detect_peft_type() → "GLA_SD_LORA"
          │ _apply_sdlora_env_overrides()
          ▼
peft.get_peft_model(model, GlaSdLoraConfig(...))
          │
          ▼
gla_sd_lora.py::GlaSdLoraModel.__init__()
          │ 遍历所有层
          │ target_modules → 创建 GlaSdLoraParameter
          │ lora_targets   → 创建 LoraLinear
          ▼
开始训练循环
          │
          ├─ Iteration 1-100: WARMUP模式
          │   └─ GlaSdLoraParameter.forward() 累积梯度
          │
          ├─ Iteration 101: 自动切换
          │   └─ 计算重要性,构建mask
          │
          └─ Iteration 102+: TRAIN模式
              └─ GlaSdLoraParameter.forward() 稀疏更新
```

---

## 十三、常见问题

### Q1: SD-LoRA需要预先Full Fine-tune吗？
**不需要！** SD-LoRA直接在预训练模型上开始。Warmup阶段只是短暂收集梯度信息（100 iterations），不是真正的微调。

### Q2: Warmup阶段模型权重会变化吗？
Warmup期间，`sdlora_grad`会累积梯度，但**原始模型权重保持不变**。前向传播使用 `weight + sdlora_grad`，但weight本身不更新。

### Q3: 为什么Zero比例设为0？
当前保守策略认为：预训练模型的每个维度都有一定价值，直接置零可能损害性能。如果实验发现某些维度确实有害，可以增加Zero比例。

### Q4: 如何选择Train比例？
- 任务简单/数据少 → 小比例 (1-5%)
- 任务复杂/数据多 → 大比例 (20-30%)
- 建议从5%开始实验

### Q5: LoRA和SDT哪个更重要？
两者互补：
- SDT: 精准控制gate机制
- LoRA: 适配投影层到下游任务

通常两者结合效果最好。

---

## 十四、参考资料

1. **SD-LoRA Paper**: "SD-LoRA: Scalable and Deployable LoRA Fine-tuning for Large Language Models"
2. **GLA Paper**: "Gated Linear Attention Transformers with Hardware-Efficient Training"
3. **代码位置**:
   - 核心实现: `mamba_ssm_peft/peft/gla_sd_lora.py`
   - 基础SD-LoRA: `mamba_ssm_peft/peft/sd_lora.py`
   - 适配层: `lat_adapter.py`

 