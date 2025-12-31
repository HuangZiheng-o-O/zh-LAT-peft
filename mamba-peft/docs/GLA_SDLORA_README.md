# GLA SD-LoRA 实现文档

## 1. 什么是 SD-LoRA？

**SD-LoRA (Sparse Dimension LoRA)** 是一种针对状态空间模型的参数高效微调方法，结合了：

- **SDT (Sparse Dimension Tuning)**: 稀疏维度调优 - 只训练最重要的维度
- **LoRA**: 低秩适配 - 在线性层添加低秩矩阵

### 核心思想

```
传统全量微调:    训练所有参数 (100%)
LoRA:           训练低秩矩阵 (~1-5%)
SD-LoRA:        训练低秩矩阵 + 稀疏选择的关键维度 (~0.5-2%)
```

### 三类维度

SD-LoRA 将参数维度分为三类：

| 类别 | 处理方式 | 典型比例 |
|------|----------|----------|
| **Zero (剪枝)** | 设为极端值，等效于移除 | 30% |
| **Freeze (冻结)** | 保持原值不更新 | 30% |
| **Train (训练)** | 正常梯度更新 | 40% |

---

## 2. 为什么需要 GLA 专用适配？

### Mamba vs GLA 架构对比

```
┌─────────────────────────────────────────────────────────────┐
│                      Mamba SSM                               │
├─────────────────────────────────────────────────────────────┤
│  状态更新: h_t = Ā·h_{t-1} + B̄·x_t                          │
│  输出:     y_t = C·h_t + D·x_t                               │
│                                                              │
│  关键参数:                                                    │
│  ├── A_log (D×N): 状态衰减矩阵 ← SDT主要目标                  │
│  ├── B: 输入→状态映射                                        │
│  └── C: 状态→输出映射                                        │
│                                                              │
│  特点: A_log 是直接可训练的参数张量                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      GLA (Gated Linear Attention)            │
├─────────────────────────────────────────────────────────────┤
│  状态更新: S_t = Diag(α_t)·S_{t-1} + k_t^T·v_t              │
│  输出:     o_t = q_t·S_t                                     │
│                                                              │
│  关键参数:                                                    │
│  ├── gk_proj: 门控投影 (生成 α_t) ← SDT主要目标              │
│  ├── k_proj: 键投影 (类似B)                                  │
│  ├── q_proj: 查询投影 (类似C)                                │
│  └── v_proj: 值投影                                          │
│                                                              │
│  特点: α_t 是通过 gk_proj 投影计算的，不是直接参数            │
└─────────────────────────────────────────────────────────────┘
```

### 关键区别

| 方面 | Mamba | GLA | 影响 |
|------|-------|-----|------|
| **门控参数** | `A_log` 是直接参数 | `gk_proj` 是投影层 | SDT目标不同 |
| **状态形状** | 向量 (D×N) | 矩阵 (H×K×V) | 维度选择策略不同 |
| **零值掩码** | 设为10 (exp→0) | 设为-20 (logsigmoid→-∞) | 数值处理不同 |

**因此，不能直接复用 Mamba SD-LoRA 代码，需要针对 GLA 重新设计。**

---

## 3. GLA SD-LoRA 实现

### 3.1 文件结构

```
mamba-peft/
├── mamba_ssm_peft/peft/
│   ├── __init__.py              # 注册 GLA_SD_LORA 类型
│   ├── gla_base_tuner.py        # GLA 基础调优器
│   └── gla_sd_lora.py           # GLA SD-LoRA 核心实现
│       ├── GlaSdLoraConfig      # 配置类
│       ├── GlaSdLoraModel       # 模型包装器
│       └── GlaSdLoraParameter   # 参数包装器
│
├── train_gla_sdlora.py          # 训练入口脚本
│
└── configs/gla_sdlora/
    ├── default.json             # 默认配置
    ├── aggressive.json          # 激进剪枝配置
    └── glue_cola.yaml           # GLUE任务训练配置
```

### 3.2 核心类设计

#### GlaSdLoraConfig (配置)

```python
@dataclass
class GlaSdLoraConfig(PeftConfig):
    # SDT 配置
    target_modules: ["gk_proj.1"]     # SDT 目标: 门控投影第二层
    num_zero: {"channel": 0.3}        # 30% 维度设为零
    num_freeze: {"channel": 0.3}      # 30% 维度冻结
    num_warmup_it: 100                # 热身迭代次数

    # LoRA 配置
    lora_targets: ["q_proj", "k_proj", "v_proj", "o_proj"]
    proj_lora_r: 8                    # LoRA 秩
```

#### GlaSdLoraParameter (参数包装器)

```
┌──────────────────────────────────────────────────────────────┐
│                    GlaSdLoraParameter                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  base_layer (原始权重)                                        │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Warmup 模式 (前 N 次迭代)                               │ │
│  │  ────────────────────────────────────────────────────── │ │
│  │  weight_new = weight + α * sdlora_grad                  │ │
│  │                                                          │ │
│  │  • sdlora_grad: 累积梯度，用于评估维度重要性              │ │
│  │  • 所有维度都参与训练                                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│       │                                                       │
│       │ it_counter > num_warmup_it                           │
│       ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Train 模式 (热身后)                                     │ │
│  │  ────────────────────────────────────────────────────── │ │
│  │  1. 根据梯度重要性排序维度                                │ │
│  │  2. 构建 train_mask (训练) 和 zero_mask (剪枝)           │ │
│  │  3. weight_new = apply_masks(weight, sdlora_adapter)     │ │
│  │                                                          │ │
│  │  • 只有 train_mask 中的维度接收梯度                       │ │
│  │  • zero_mask 中的维度设为 -20 (门控失效)                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 两阶段训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 1: Warmup                             │
│                      (热身阶段)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  目的: 收集梯度信息，评估每个维度的重要性                          │
│                                                                  │
│  for step in range(num_warmup_it):                              │
│      loss = model(batch)                                         │
│      loss.backward()                                             │
│      # sdlora_grad 累积梯度变化                                  │
│                                                                  │
│  结束时: 保存 sdlora_grad 到文件 (用于恢复)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ should_training_stop = True
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 2: Training                           │
│                      (训练阶段)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 维度选择 (基于热身阶段收集的梯度)                              │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ 梯度L2范数排序:                                         │  │
│     │ [dim_5, dim_2, dim_8, ...., dim_3, dim_7, dim_1]      │  │
│     │  ↑高重要性                          低重要性↑            │  │
│     │                                                        │  │
│     │ 分配:                                                   │  │
│     │ [  Train (40%)  |  Freeze (30%)  |  Zero (30%)  ]     │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  2. 正常训练 (只更新 Train 维度)                                  │
│     for epoch in range(num_epochs):                             │
│         loss = model(batch)  # Zero维度被掩码，Freeze维度冻结    │
│         loss.backward()      # 只有Train维度接收梯度             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 零值掩码处理

GLA 的门控使用 `logsigmoid`:

```python
# GLA forward (fla/layers/gla.py:236)
gk = F.logsigmoid(gk) / gate_logit_normalizer

# 当 gk 很大的负数时:
#   logsigmoid(-20) ≈ -20
#   gate ≈ exp(-20/τ) ≈ 0
#   → 状态完全衰减，等效于"遗忘"
```

因此，GLA SD-LoRA 使用 `-20` 作为零值掩码（而非 Mamba 的 `10`）:

```python
# gla_sd_lora.py
ZERO_MASK_VALUE = -20.0

def build_train_param(self, param, adapter):
    # 对 zero_mask 中的维度设为 -20
    param_new = torch.where(
        self.zero_mask,
        torch.full_like(param, self.ZERO_MASK_VALUE),
        param
    )
```

---

## 4. 使用方法

### 4.1 基本训练

```bash
# 使用默认配置训练
python train_gla_sdlora.py --cfg configs/gla_sdlora/glue_cola.yaml

# 使用激进剪枝配置
python train_gla_sdlora.py \
    --cfg configs/gla_sdlora/glue_cola.yaml \
    --peft configs/gla_sdlora/aggressive.json
```

### 4.2 环境变量覆盖

```bash
# 调整热身迭代次数
HP_WARMUP_IT=200 python train_gla_sdlora.py --cfg configs/gla_sdlora/glue_cola.yaml

# 调整剪枝比例
HP_ZERO_RATIO=0.4 HP_FREEZE_RATIO=0.4 python train_gla_sdlora.py ...
```

### 4.3 配置文件示例

**SD-LoRA 配置 (JSON)**:
```json
{
    "peft_type": "GLA_SD_LORA",
    "target_modules": ["gk_proj.1"],
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "proj_lora_r": 8,
    "num_zero": {"channel": 0.3},
    "num_freeze": {"channel": 0.3},
    "num_warmup_it": 100
}
```

**训练配置 (YAML)**:
```yaml
model: "fla-hub/gla-1.3B-100B"
data: "glue-tvt_cola"
num_epochs: 10
batch_size: 8
learning_rate: 5e-4
prec: "bf16"
seed: 42
```

---

## 5. 与 Mamba SD-LoRA 的对应关系

| 组件 | Mamba SD-LoRA | GLA SD-LoRA | 说明 |
|------|---------------|-------------|------|
| **SDT目标** | `A_log` | `gk_proj.1` | 都是控制状态衰减的核心参数 |
| **维度选择** | Channel + State | Channel only | GLA 矩阵状态结构不同 |
| **零值掩码** | `10` | `-20` | 数值语义不同 (exp vs logsigmoid) |
| **基础调优器** | `MambaBaseTuner` | `GLABaseTuner` | 模型结构不同 |
| **配置类** | `SdLoraConfig` | `GlaSdLoraConfig` | 参数略有不同 |
| **参数包装** | `SdLoraParameter` | `GlaSdLoraParameter` | 核心逻辑相似，细节不同 |

### 代码复用情况

```
从 Mamba SD-LoRA 复用的逻辑:
├── 两阶段训练框架 (warmup → train)
├── 梯度累积与重要性评估
├── 掩码构建与稀疏训练
├── 配置保存/加载机制
└── should_training_stop 机制

针对 GLA 重新实现的部分:
├── 目标模块识别 (gk_proj vs A_log)
├── 维度解析 (只有 channel，无 state)
├── 零值掩码数值 (-20 vs 10)
├── 块级定位 (GatedLinearAttention vs Mamba mixer)
└── 基础调优器 (GLABaseTuner)
```

---

## 6. 依赖关系

所有依赖都在 `mamba-peft` 文件夹内，无外部依赖于 `mamba-peft-sd_lora`:

```
mamba-peft/ (自包含)
├── utils/utils.py                    → find_layer_by_name, find_module_parent
├── trainer/generic_lm_trainer.py     → SD-LoRA 训练支持已内置
├── dataset/                          → 数据集加载
├── mamba_ssm_peft/utils/             → 模型加载、解码器
└── mamba_ssm_peft/peft/              → GLA SD-LoRA 实现
```

**GenericLMTrainer 已内置 SD-LoRA 支持**:
- 初始化时调用 `model.load_config()` 恢复状态
- 训练时检查 `model.should_training_stop` 触发阶段切换
- 阶段切换时调用 `model.save_config()` 保存状态
