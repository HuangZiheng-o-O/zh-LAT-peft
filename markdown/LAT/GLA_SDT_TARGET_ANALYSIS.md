# GLA SD-LoRA目标选择深度分析

## 核心问题

当前GLA SD-LoRA实现中，SDT目标设置为`gk_proj.1`。本文从GLA架构原理出发，系统性分析：
1. 这个选择是否正确？
2. 为什么不是`gk_proj.0`或整个`gk_proj`？
3. 是否存在更优的目标选择？

---

## 1. GLA层架构完整解析

### 1.1 组件结构

```python
class GatedLinearAttention(nn.Module):
    def __init__(self, hidden_size=2048, expand_k=0.5, expand_v=1.0,
                 num_heads=4, gate_low_rank_dim=16, ...):

        # 维度计算
        key_dim = int(hidden_size * expand_k)           # 2048 * 0.5 = 1024
        value_dim = int(hidden_size * expand_v)         # 2048 * 1.0 = 2048
        key_dim_per_group = key_dim // num_kv_groups    # 1024 / 4 = 256
        value_dim_per_group = value_dim // num_kv_groups # 2048 / 4 = 512

        # Query投影
        self.q_proj = nn.Linear(hidden_size, key_dim)           # (2048, 1024)

        # Key投影 (可能分组)
        self.k_proj = nn.Linear(hidden_size, key_dim_per_group) # (2048, 256)

        # Value投影 (可能分组)
        self.v_proj = nn.Linear(hidden_size, value_dim_per_group) # (2048, 512)

        # 门控投影 ← SDT核心目标
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),    # .0: (2048, 16)
            nn.Linear(gate_low_rank_dim, key_dim_per_group, bias=True) # .1: (16, 256)
        )

        # 输出门控 (可选)
        self.g_proj = nn.Linear(hidden_size, value_dim)          # (2048, 2048)

        # 输出投影
        self.o_proj = nn.Linear(value_dim, hidden_size)          # (2048, 2048)
```

### 1.2 前向传播中的门控计算

```python
def forward(self, hidden_states):
    # hidden_states: (B, T, 2048)

    # Step 1: 计算Q/K/V
    q = self.q_proj(hidden_states)  # (B, T, 1024)
    k = self.k_proj(hidden_states)  # (B, T, 256)
    v = self.v_proj(hidden_states)  # (B, T, 512)

    # Step 2: 计算门控值 ← 关键!
    gk = self.gk_proj(hidden_states)  # (B, T, 256)

    # Step 3: 门控变换
    gk = F.logsigmoid(gk) / gate_logit_normalizer  # logsigmoid后除以16

    # Step 4: 线性注意力递归 (概念化)
    # S_t = Diag(exp(gk_t)) * S_{t-1} + k_t^T * v_t
    # 其中 exp(gk) 是每个key维度的衰减因子

    return o
```

### 1.3 数据流图

```
                    GLA 数据流
                    ══════════

hidden_states ────────────────────────────────────────────────────────┐
(B, T, 2048)                                                          │
     │                                                                │
     ├──────────────┬──────────────┬──────────────┬──────────────────┤
     │              │              │              │                   │
     ▼              ▼              ▼              ▼                   ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────────────┐  ┌─────────┐
│ q_proj  │  │ k_proj  │  │ v_proj  │  │     gk_proj       │  │ g_proj  │
│(2048→   │  │(2048→   │  │(2048→   │  │   (Sequential)    │  │(2048→   │
│ 1024)   │  │  256)   │  │  512)   │  │                   │  │ 2048)   │
└────┬────┘  └────┬────┘  └────┬────┘  │ ┌───────────────┐ │  └────┬────┘
     │            │            │       │ │    .0         │ │       │
     │            │            │       │ │ (2048→16)     │ │       │
     │            │            │       │ └───────┬───────┘ │       │
     │            │            │       │         │         │       │
     │            │            │       │         ▼         │       │
     │            │            │       │ ┌───────────────┐ │       │
     │            │            │       │ │    .1         │ │       │
     │            │            │       │ │  (16→256)     │ │       │
     │            │            │       │ └───────┬───────┘ │       │
     │            │            │       └─────────┼─────────┘       │
     │            │            │                 │                  │
     ▼            ▼            ▼                 ▼                  │
   Q (1024)    K (256)      V (512)         gk (256)               │
     │            │            │                 │                  │
     │            │            │                 ▼                  │
     │            │            │        ┌───────────────┐           │
     │            │            │        │  logsigmoid   │           │
     │            │            │        │    / τ        │           │
     │            │            │        └───────┬───────┘           │
     │            │            │                │                   │
     │            │            │                ▼                   │
     └────────────┴────────────┴───────► Linear Attention ◄─────────┘
                                          Recurrence
                                              │
                                              ▼
                                           Output
```

---

## 2. 所有可能的SDT目标逐一分析

### 2.1 gk_proj.0 (第一层投影)

```
权重形状: (gate_low_rank_dim, hidden_size) = (16, 2048)
偏置: 无

作用: 将2048维hidden states压缩到16维门控表示

数据流:
  hidden_states (B, T, 2048)
         │
         ▼ W[16, 2048] × x^T
  compressed (B, T, 16)  ← 极度压缩!
```

**SDT分析:**

| SDT方向 | 维度 | 语义 | 可行性 |
|---------|------|------|--------|
| **输出维度** (16) | 选择哪些"门控概念"激活 | 控制信息压缩的哪些通道 | ⚠️ 粒度太粗，仅16维 |
| **输入维度** (2048) | 选择哪些hidden特征参与门控 | 控制输入信息流 | ✅ 细粒度，但语义间接 |

**问题:**
- 输出仅16维，SDT粒度太粗
- 零掩码一个输出维度会影响所有256个最终门控通道
- 不适合作为主要SDT目标

### 2.2 gk_proj.1 (第二层投影) ← 当前选择

```
权重形状: (key_dim_per_group, gate_low_rank_dim) = (256, 16)
偏置: (256,)

作用: 将16维压缩表示展开为256维门控值

数据流:
  compressed (B, T, 16)
         │
         ▼ W[256, 16] × x^T + b[256]
  gk (B, T, 256)  ← 每个key维度的门控值
```

**SDT分析:**

| SDT方向 | 维度 | 语义 | 可行性 |
|---------|------|------|--------|
| **输出维度** (256) | 控制每个key维度的门控 | 直接对应状态矩阵的列 | ✅ 语义清晰，最佳选择 |
| **输入维度** (16) | 控制哪些压缩特征参与 | 间接控制 | ⚠️ 维度太少 |

**关键洞察:**

```python
# 零掩码的效果链 (已修正):
gk_proj.1.weight[dim, :] = 0, gk_proj.1.bias[dim] = -100
    ↓
gk[..., dim] = -100
    ↓
logsigmoid(-100) / 16 ≈ -100 / 16 = -6.25
    ↓
exp(-6.25) ≈ 0.002  # 接近完全遗忘 ✓
```

**✓ 已修正: 零掩码值已从-20更新为-100 (2025-12-31)**

### 2.3 完整gk_proj (两层联合)

```
权重: .0 (16, 2048) + .1 (256, 16)
等效: (256, 2048) 的低秩分解

完整映射:
  hidden_states → compressed → gk
  (2048)        → (16)       → (256)
```

**SDT策略选项:**

| 策略 | 描述 | 优缺点 |
|------|------|--------|
| **分层SDT** | .0选输入，.1选输出 | 复杂但全面 |
| **2D块选择** | 类似SMT，选(input, output)块 | 可借鉴SMT思想 |
| **仅.1输出** | 当前实现 | 简单直接 |

### 2.4 q_proj, k_proj, v_proj

```
权重形状:
  q_proj: (1024, 2048)
  k_proj: (256, 2048)
  v_proj: (512, 2048)

作用: 生成Query/Key/Value向量
```

**SDT分析:**

| 投影 | SDT语义 | 与记忆机制关系 | 推荐 |
|------|---------|---------------|------|
| q_proj | 控制查询模式 | 间接，影响检索 | LoRA更合适 |
| k_proj | 控制键模式 | 中等，影响存储 | 可考虑SDT |
| v_proj | 控制值内容 | 间接，影响内容 | LoRA更合适 |

**k_proj的特殊性:**
- k_proj的输出维度与gk_proj.1相同 (256)
- k和gk共同决定状态矩阵的更新: `S_t += k_t^T * v_t`
- 可能值得作为次级SDT目标

### 2.5 g_proj (输出门控)

```
权重形状: (2048, 2048)
作用: 输出门控，控制信息传递到下一层
```

**SDT分析:**
- 位于状态计算之后，不影响"记忆"机制
- 影响信息向后传播
- 不推荐作为SDT主目标

### 2.6 o_proj (输出投影)

```
权重形状: (2048, 2048)
作用: 将注意力输出投影回hidden_size
```

**SDT分析:**
- 最下游模块，对记忆机制无直接影响
- 标准LoRA即可

---

## 3. 与Mamba对比验证

### 3.1 Mamba的SDT目标

```python
# Mamba结构:
class MambaLayer:
    # 状态更新: h_t = Ā * h_{t-1} + B̄ * x_t
    # 其中: Ā = exp(Δ * A_log)

    self.A_log = nn.Parameter(...)  # (D, N) 直接参数
    self.in_proj = nn.Linear(...)   # 输入投影
    self.x_proj = nn.Linear(...)    # 生成B, C, Δ
```

**Mamba的SDT目标: A_log**
- 直接参数，非投影
- 控制状态衰减
- 维度: (D, N) = (channel, state)

### 3.2 GLA与Mamba的对应关系

| Mamba | GLA | 说明 |
|-------|-----|------|
| A_log | gk_proj输出 | 都控制状态衰减 |
| 直接参数 | 投影输出 | GLA是计算得到的 |
| (D, N) | (H, K) | 维度结构不同 |
| exp(Δ*A) | exp(logsigmoid(gk)/τ) | 激活函数不同 |

**关键差异:**
- Mamba可以直接mask A_log参数
- GLA需要通过mask gk_proj权重来间接影响gk值

---

## 4. gk_proj.1 vs 其他选择的最终评估

### 4.1 评估矩阵

| 目标 | 语义清晰度 | 粒度 | 与记忆机制关系 | 实现复杂度 | 推荐度 |
|------|-----------|------|---------------|-----------|-------|
| **gk_proj.1输出** | ★★★★★ | 256维 | 直接 | 低 | **推荐** |
| gk_proj.0输出 | ★★★☆☆ | 16维 | 间接 | 低 | 不推荐 |
| gk_proj.0输入 | ★★★☆☆ | 2048维 | 间接 | 中 | 可选补充 |
| 完整gk_proj 2D | ★★★★☆ | 块级 | 直接 | 高 | 高级选项 |
| k_proj | ★★★☆☆ | 256维 | 中等 | 低 | 可选补充 |
| q/v/g/o_proj | ★★☆☆☆ | 各异 | 间接 | 低 | LoRA足够 |

### 4.2 推荐SDT目标配置

**主配置 (当前实现):**
```json
{
    "target_modules": ["gk_proj.1"],
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```

**增强配置 (实验性):**
```json
{
    "sdt_targets": {
        "primary": ["gk_proj.1"],
        "secondary": ["k_proj"]
    },
    "lora_targets": ["q_proj", "v_proj", "o_proj"]
}
```

---

## 5. 零掩码值分析与修正

### 5.1 问题与修正 (已解决)

```python
# 旧实现 (有问题):
ZERO_MASK_VALUE = -20.0  # 衰减不够彻底
# gk = -20 → exp(-1.25) ≈ 0.29 (29%信息保留!)

# 新实现 (已修正):
ZERO_MASK_VALUE = -100.0  # 确保彻底遗忘
# gk = -100 → exp(-6.25) ≈ 0.002 (仅0.2%保留) ✓
```

**✓ 已修正 (2025-12-31): 零掩码值从-20更新为-100**

### 5.2 修正建议

```python
# 已修正实现 (gla_sd_lora.py):
ZERO_MASK_VALUE = -100.0

# 验证:
gk = -100
normalized_gk = logsigmoid(-100) / 16 ≈ -100 / 16 = -6.25
decay = exp(-6.25) ≈ 0.002  # 仅0.2%信息保留 ✓
```

### 5.3 修改历史

| 日期 | 修改 | 原因 |
|------|------|------|
| 2025-12-31 | ZERO_MASK_VALUE: -20 → -100 | 原值导致29%信息保留，不够彻底 |

---

## 6. 结论

### 6.1 目标选择验证

**gk_proj.1作为SDT目标是正确的**，原因：

1. **语义对应**: 输出维度直接对应key维度，即状态矩阵的列
2. **效果直接**: 零掩码可直接使该维度的门控失效
3. **Mamba类比**: 类似于Mamba中对A_log的操作
4. **粒度适中**: 256维提供足够的细粒度控制

### 6.2 为什么不选其他

| 不选择 | 原因 |
|--------|------|
| gk_proj.0 | 仅16输出维度，粒度太粗 |
| 完整gk_proj | 增加复杂度，收益不明确 |
| q/v/o_proj | 与记忆机制关系间接，LoRA足够 |
| k_proj | 可作为补充，但非主要目标 |

### 6.3 改进建议

1. **✓ 零掩码值已修正**: -20 → -100 (已于2025-12-31完成)
2. **可选增强**: 考虑k_proj作为次级SDT目标
3. **未来探索**: 2D块选择 (结合SMT思想)

---

## 附录: gate_logit_normalizer的影响

```python
# GLA中gate_logit_normalizer默认为16
gk = F.logsigmoid(gk) / self.gate_logit_normalizer

# 零掩码值计算 (已修正为-100):
# gk=-100 → normalized=-6.25 → decay=0.002 (仅0.2%保留) ✓
```

**当前实现: ZERO_MASK_VALUE = -100.0 (已于2025-12-31修正)**
