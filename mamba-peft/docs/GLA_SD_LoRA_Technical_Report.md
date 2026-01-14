# GLA SD-LoRA 深度技术报告

## 目录
1. [引言与背景](#1-引言与背景)
2. [线性注意力理论基础](#2-线性注意力理论基础)
3. [门控线性注意力 (GLA) 详解](#3-门控线性注意力-gla-详解)
4. [SD-LoRA 方法论](#4-sd-lora-方法论)
5. [代码实现深度解析](#5-代码实现深度解析)
6. [配置系统与训练流程](#6-配置系统与训练流程)
7. [数学推导附录](#7-数学推导附录)

---

## 1. 引言与背景

### 1.1 问题背景

传统的 Transformer 模型使用 Softmax 注意力机制，其计算复杂度为 $O(L^2 d)$，其中 $L$ 是序列长度，$d$ 是隐藏维度。这导致长序列处理时存在严重的效率瓶颈。

线性注意力机制通过将 Softmax 替换为核函数，将复杂度降低到 $O(L d^2)$，但通常会牺牲模型性能。门控线性注意力 (GLA) 通过引入数据依赖的遗忘门 (data-dependent gating mechanism) 来弥补这一差距。

### 1.2 研究动机

参数高效微调 (PEFT) 方法如 LoRA 在 Transformer 模型上取得了巨大成功，但这些方法在状态空间模型 (SSM) 和线性注意力模型上的效果并不理想。原因在于：

1. **SSM 模块的特殊性**: SSM 使用对角化的状态转移矩阵，LoRA 的低秩分解无法有效捕捉这种结构
2. **维度选择的重要性**: 不同维度对下游任务的贡献差异巨大
3. **门控机制**: GLA 中的遗忘门需要特殊处理

**SD-LoRA (Sparse Dimension LoRA)** 是专门为 SSM/GLA 模型设计的 PEFT 方法，核心思想是**稀疏维度调优 (Sparse Dimension Tuning, SDT)**。

### 1.3 参考论文

| 论文 | 内容 | 链接 |
|------|------|------|
| Gated Linear Attention Transformers | GLA 原始论文 (ICML 2024) | [arXiv:2312.06635](https://arxiv.org/abs/2312.06635) |
| Parameter-Efficient Fine-Tuning of State Space Models | SD-LoRA 方法 (ICML 2025) | [arXiv:2410.09016](https://arxiv.org/abs/2410.09016) |

---

## 2. 线性注意力理论基础

### 2.1 标准 Softmax 注意力

给定输入序列 $\mathbf{X} \in \mathbb{R}^{L \times d}$，标准注意力计算如下：

$$
\mathbf{Q}, \mathbf{K}, \mathbf{V} = \mathbf{X} \boldsymbol{W}_{Q}, \mathbf{X} \boldsymbol{W}_{K}, \mathbf{X} \boldsymbol{W}_{V}
$$

$$
\mathbf{O} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^{\mathsf{T}}}{\sqrt{d}} \odot \mathbf{M} \right) \mathbf{V}
$$

其中 $\mathbf{M} \in \{-\infty, 1\}^{L \times L}$ 是因果掩码，$\mathbf{M}_{ij} = 1$ 当 $i \geq j$，否则 $\mathbf{M}_{ij} = -\infty$。

**并行形式 (Parallel Form)** 的计算复杂度为 $O(L^2 d)$。

### 2.2 递归形式

在推理时，Transformer 必须使用递归形式：

$$
\boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t = \boldsymbol{x}_t \boldsymbol{W}_{Q}, \boldsymbol{x}_t \boldsymbol{W}_{K}, \boldsymbol{x}_t \boldsymbol{W}_{V}
$$

$$
\boldsymbol{o}_t = \frac{\sum_{i=1}^t \exp(\boldsymbol{q}_t \boldsymbol{k}_i^{\mathsf{T}}) \boldsymbol{v}_i}{\sum_{i=1}^t \exp(\boldsymbol{q}_t \boldsymbol{k}_i^{\mathsf{T}})}
$$

这需要维护不断增长的 KV 缓存 $\{\boldsymbol{k}_1,...,\boldsymbol{k}_t\}$ 和 $\{\boldsymbol{v}_1,...,\boldsymbol{v}_t\}$。

### 2.3 线性注意力

线性注意力用核函数 $k(\boldsymbol{x}, \boldsymbol{y}) = \langle \phi(\boldsymbol{x}), \phi(\boldsymbol{y}) \rangle$ 替换 $\exp(\boldsymbol{q}_t \boldsymbol{k}_i^{\mathsf{T}})$：

$$
\boldsymbol{o}_{t} = \frac{\sum_{i=1}^{t} \phi(\boldsymbol{q}_{t}) \phi(\boldsymbol{k}_{i})^{\mathsf{T}} \boldsymbol{v}_{i}}{\sum_{i=1}^{t} \phi(\boldsymbol{q}_{t}) \phi(\boldsymbol{k}_{i})^{\mathsf{T}}} = \frac{\phi(\boldsymbol{q}_{t}) \sum_{i=1}^{t} \phi(\boldsymbol{k}_{i})^{\mathsf{T}} \boldsymbol{v}_{i}}{\phi(\boldsymbol{q}_{t}) \sum_{i=1}^{t} \phi(\boldsymbol{k}_{i})^{\mathsf{T}}}
$$

定义：
- $\mathbf{S}_{t} = \sum_{i=1}^{t} \phi(\boldsymbol{k}_{i})^{\mathsf{T}} \boldsymbol{v}_{i} \in \mathbb{R}^{d \times d}$ (隐藏状态矩阵)
- $\boldsymbol{z}_{t} = \sum_{i=1}^{t} \phi(\boldsymbol{k}_{i})^{\mathsf{T}} \in \mathbb{R}^{d \times 1}$ (归一化因子)

可以重写为 **RNN 形式**：

$$
\mathbf{S}_{t} = \mathbf{S}_{t-1} + \phi(\boldsymbol{k}_{t})^{\mathsf{T}} \boldsymbol{v}_{t}
$$

$$
\boldsymbol{z}_{t} = \boldsymbol{z}_{t-1} + \phi(\boldsymbol{k}_{t})^{\mathsf{T}}
$$

$$
\boldsymbol{o}_{t} = \frac{\phi(\boldsymbol{q}_{t})\mathbf{S}_{t}}{\phi(\boldsymbol{q}_{t})\boldsymbol{z}_{t}}
$$

### 2.4 无归一化线性注意力

实践中发现，使用线性核 ($\phi$ 为恒等函数) 且不使用归一化效果更好：

$$
\mathbf{S}_{t} = \mathbf{S}_{t-1} + \boldsymbol{k}_{t}^{\mathsf{T}} \boldsymbol{v}_{t}
$$

$$
\boldsymbol{o}_{t} = \boldsymbol{q}_{t} \mathbf{S}_{t}
$$

**关键洞察**: 线性注意力本质上是具有**矩阵值隐藏状态** $\mathbf{S}_t$ 的线性递归层，通过外积 $\boldsymbol{k}_{t}^{\mathsf{T}} \boldsymbol{v}_{t}$ 进行更新。

### 2.5 分块并行形式 (Chunkwise Parallel Form)

分块并行形式在并行和递归形式之间取得平衡，实现次二次复杂度的部分并行训练。

将输入 $\mathbf{X}$ 分成长度为 $C$ 的非重叠块，定义：
- $\mathbf{S}_{[i]} := \mathbf{S}_{iC}$ (处理 $i$ 个块后的隐藏状态)
- $\mathbf{Q}_{[i]} := \mathbf{Q}_{iC+1:(i+1)C+1} \in \mathbb{R}^{C \times d}$

**块间递归 (Inter-chunk recurrence)**:

$$
\mathbf{S}_{[i+1]} = \mathbf{S}_{[i]} + \underbrace{\sum_{j=iC+1}^{(i+1)C} \boldsymbol{k}_{j}^{\mathsf{T}} \boldsymbol{v}_{j}}_{\mathbf{K}_{[i]}^{\mathsf{T}} \mathbf{V}_{[i]}} \in \mathbb{R}^{d \times d}
$$

**块内并行计算 (Intra-chunk parallel)**:

$$
\mathbf{O}_{[i+1]} = \underbrace{\mathbf{Q}_{[i+1]} \mathbf{S}_{[i]}}_{\text{inter-chunk: } \mathbf{O}^{\text{inter}}_{[i+1]}} + \underbrace{((\mathbf{Q}_{[i+1]} \mathbf{K}_{[i+1]}^{\mathsf{T}}) \odot \mathbf{M}) \mathbf{V}_{[i+1]}}_{\text{intra-chunk: } \mathbf{O}^{\text{intra}}_{[i+1]}}
$$

**复杂度分析**: $O\left(\frac{L}{C}(C^2d + Cd^2)\right) = O(LCd + Ld^2)$

- 当 $C = L$ 时，恢复并行形式
- 当 $C = 1$ 时，恢复递归形式

---

## 3. 门控线性注意力 (GLA) 详解

### 3.1 遗忘门的必要性

线性递归 $\mathbf{S}_{t} = \mathbf{S}_{t-1} + \boldsymbol{k}_{t}^{\mathsf{T}} \boldsymbol{v}_{t}$ 缺少衰减项或遗忘门，这在 RNN 中被证明是至关重要的 (LSTM, GRU)。缺少衰减项使模型难以"遗忘"信息，导致在长上下文任务中出现不稳定性。

### 3.2 数据依赖的门控机制

GLA 引入了随时间变化的 2D 遗忘门 $\mathbf{G}_t \in (0,1)^{d_k \times d_v}$：

$$
\mathbf{S}_{t} = \mathbf{G}_{t} \odot \mathbf{S}_{t-1} + \boldsymbol{k}_{t}^{\mathsf{T}} \boldsymbol{v}_{t}
$$

其中 $\odot$ 表示逐元素乘法 (Hadamard product)。

### 3.3 门控矩阵的参数化

不同的 $\mathbf{G}_t$ 参数化方案涵盖了多种最新 RNN 模型：

| 模型 | $\mathbf{G}_t$ 参数化 |
|------|----------------------|
| RetNet | $\gamma \mathbf{1}^{\mathsf{T}} \mathbf{1}$ (全局常量) |
| Mamba | $\text{Diag}(\boldsymbol{\alpha}) \cdot \mathbf{A}$ (混合) |
| Mamba-2 | $\gamma_t \mathbf{1}^{\mathsf{T}} \mathbf{1}$ (标量数据依赖) |
| **GLA** | $\boldsymbol{\alpha}_t^{\mathsf{T}} \mathbf{1}$ (向量数据依赖) |

GLA 选择中间方案 $\mathbf{G}_t = \boldsymbol{\alpha}_t^{\mathsf{T}} \mathbf{1}$，得到：

$$
\mathbf{S}_{t} = (\boldsymbol{\alpha}_{t}^{\mathsf{T}} \mathbf{1}) \odot \mathbf{S}_{t-1} + \boldsymbol{k}_{t}^{\mathsf{T}} \boldsymbol{v}_{t} = \text{Diag}(\boldsymbol{\alpha}_{t}) \mathbf{S}_{t-1} + \boldsymbol{k}_{t}^{\mathsf{T}} \boldsymbol{v}_{t}
$$

其中 $\boldsymbol{\alpha}_t \in (0,1)^{d_k}$ 是通过低秩线性层和 sigmoid 从 $\boldsymbol{x}_t$ 计算得到的。

### 3.4 并行形式推导

展开递归公式：

$$
\mathbf{S}_{t} = \sum_{i=1}^{t} \left( \left( \prod_{j=i+1}^{t} \boldsymbol{\alpha}_{j}^{\mathsf{T}} \mathbf{1} \right) \odot \boldsymbol{k}_{i}^{\mathsf{T}} \boldsymbol{v}_{i} \right)
$$

令 $\boldsymbol{b}_t := \prod_{j=1}^t \boldsymbol{\alpha}_j$ (累积乘积)，则：

$$
\boldsymbol{o}_{t} = \boldsymbol{q}_{t} \mathbf{S}_{t} = \boldsymbol{q}_{t} \sum_{i=1}^{t} \left( \left( \frac{\boldsymbol{b}_{t}}{\boldsymbol{b}_{i}} \right)^{\mathsf{T}} \mathbf{1} \right) \odot \boldsymbol{k}_{i}^{\mathsf{T}} \boldsymbol{v}_{i}
$$

$$
= \sum_{i=1}^{t} \left( \boldsymbol{q}_{t} \odot \boldsymbol{b}_{t} \right) \left( \frac{\boldsymbol{k}_{i}}{\boldsymbol{b}_{i}} \right)^{\mathsf{T}} \boldsymbol{v}_{i}
$$

令 $\mathbf{B} \in (0,1)^{L \times d}$ 为 $\boldsymbol{b}_t$ 堆叠形成的矩阵，并行形式为：

$$
\mathbf{O} = \left( \left( \underbrace{(\mathbf{Q} \odot \mathbf{B}) \left( \frac{\mathbf{K}}{\mathbf{B}} \right)^{\mathsf{T}}}_{\mathbf{P}} \right) \odot \mathbf{M} \right) \mathbf{V}
$$

### 3.5 数值稳定性处理

由于 $\boldsymbol{b}_t = \prod_{j=1}^t \boldsymbol{\alpha}_j$ 是 $(0,1)$ 值的累积乘积，当 $t$ 较大时会变得极小，导致 $\frac{\mathbf{K}}{\mathbf{B}}$ 爆炸。

**解决方案**: 在对数空间计算

$$
\mathbf{P}_{ij} = \sum_{k=1}^{d} \mathbf{Q}_{ik} \mathbf{K}_{jk} \exp(\log \mathbf{B}_{ik} - \log \mathbf{B}_{jk}), \quad i \geq j
$$

### 3.6 GLA 门控投影实现

在代码中，门控投影 `gk_proj` 的实现使用低秩参数化：

```python
# fla/layers/gla.py:150-151
self.gk_proj = nn.Sequential(
    nn.Linear(hidden_size, gate_low_rank_dim, bias=False),  # 默认 16
    nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True)
)
```

门控值的计算：

```python
# fla/layers/gla.py:226, 236-238
gk = self.gk_proj(hidden_states)
# ...
gk = F.logsigmoid(gk) / self.gate_logit_normalizer  # 默认除以 16
if self.clamp_min is not None:
    gk = torch.clamp_min(gk, self.clamp_min)
```

**公式解释**:

$$
\boldsymbol{g}_t = \frac{\text{logsigmoid}(\text{gk\_proj}(\boldsymbol{x}_t))}{\tau}
$$

其中 $\tau = 16$ 是温度参数 (gate_logit_normalizer)，用于控制遗忘速率。

实际的遗忘门值为：

$$
\boldsymbol{\alpha}_t = \exp(\boldsymbol{g}_t) = \exp\left(\frac{\log\sigma(\text{gk\_proj}(\boldsymbol{x}_t))}{\tau}\right) = \sigma(\text{gk\_proj}(\boldsymbol{x}_t))^{1/\tau}
$$

---

## 4. SD-LoRA 方法论

### 4.1 核心思想：稀疏维度调优 (SDT)

SD-LoRA 的核心创新是**稀疏维度调优 (Sparse Dimension Tuning, SDT)**，其理论基础来自对 SSM 模型的分析：

**关键洞察**: 在微调过程中，不同维度对下游任务的贡献差异巨大：
- 部分维度对目标函数贡献极小，可以完全**剪枝 (zero/prune)**
- 部分维度已与目标功能对齐，无需更新，可以**冻结 (freeze)**
- 只有少数维度需要**训练 (train)**

### 4.2 三分类维度划分

SDT 将隐藏维度分为三类：

| 类别 | 说明 | 处理方式 |
|------|------|----------|
| **Zero (剪枝)** | 对输出贡献为零的维度 | 将权重设为强制零值 |
| **Freeze (冻结)** | 已对齐目标功能的维度 | 保持原始权重不变 |
| **Train (训练)** | 需要适配的关键维度 | 允许梯度更新 |

### 4.3 两阶段训练

SD-LoRA 采用两阶段训练策略：

```
┌─────────────────────────────────────────────────────────────┐
│                     Phase 1: Warmup                          │
│                                                              │
│  1. 在所有维度上运行前向传播                                  │
│  2. 累积梯度 (不更新权重)                                     │
│  3. 保存梯度累积结果 (.pkl 文件)                              │
│                                                              │
│  迭代次数: num_warmup_it (默认 100)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Phase 2: Training                          │
│                                                              │
│  1. 加载 Phase 1 的梯度数据                                   │
│  2. 基于梯度范数计算维度重要性                                │
│  3. 划分 Train/Freeze/Zero 维度                              │
│  4. 只训练 Train 维度，应用 Zero 掩码                        │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 重要性计算

维度重要性基于梯度的 L2 范数：

$$
\text{importance}(d) = \|\nabla_{W_d} \mathcal{L}\|_2^2
$$

对于权重矩阵 $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$：

$$
\text{importance}(i) = \sum_{j=1}^{d_{in}} |\nabla_{W_{ij}} \mathcal{L}|^2
$$

代码实现 (`gla_sd_lora.py:770-778`):

```python
def get_importances(self, x, dim=0):
    """
    Compute importance scores for each channel.
    Uses L2 norm of gradient as importance metric.
    """
    norms = x.square().detach().sum(dim=1 if dim == 0 else 0)
    indices = torch.argsort(-norms)  # Sort descending
    return indices
```

### 4.5 维度选择策略

根据重要性排序后进行划分：

```
重要性排序: [高 ──────────────────────────────────────→ 低]
            │                                          │
            │ Train │      Freeze        │    Zero     │
            │ dims  │      dims          │    dims     │
            ├───────┼───────────────────┼─────────────┤
            │  n_t  │       n_f         │     n_z     │
```

代码实现 (`gla_sd_lora.py:780-802`):

```python
def select_channels(self, importance_order, channel_type):
    num_train = self.num_train["channel"]
    num_freeze = self.num_freeze["channel"]
    num_zero = self.num_zero["channel"]

    if channel_type == "train":
        return importance_order[:num_train]
    elif channel_type == "freeze":
        return importance_order[num_train:num_train + num_freeze]
    elif channel_type == "zero":
        return importance_order[num_train + num_freeze:num_train + num_freeze + num_zero]
```

### 4.6 GLA 门控维度的特殊处理

在 GLA 中，SD-LoRA 主要作用于门控投影 `gk_proj`。要"零化"一个门控通道，需要确保该通道的输出始终是一个大的负值（使遗忘门接近 1，即完全遗忘）。

**零掩码处理** (`gla_sd_lora.py:486-494`):

```python
# Large negative value for zeroing gate dimensions
# In GLA: gate = exp(logsigmoid(gk) / gate_logit_normalizer)
# where gate_logit_normalizer = 16 (default)
#
# To achieve near-zero decay (complete forgetting):
#   gk = -100 → logsigmoid(-100)/16 ≈ -6.25 → exp(-6.25) ≈ 0.002 (0.2% retained)
ZERO_MASK_VALUE = -100.0
```

**数学分析**:

$$
\text{gate} = \exp\left(\frac{\log\sigma(\text{gk})}{\tau}\right)
$$

当 $\text{gk} = -100$ 时：
$$
\log\sigma(-100) = \log\frac{1}{1+e^{100}} \approx -100
$$
$$
\text{gate} = \exp\left(\frac{-100}{16}\right) = \exp(-6.25) \approx 0.002
$$

即只保留 0.2% 的历史信息，接近完全遗忘。

---

## 5. 代码实现深度解析

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       代码架构总览                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────────────────────┐   │
│  │  lat_adapter.py │────▶│  prepare_lat_model_and_tokenizer │   │
│  └─────────────────┘     └─────────────────────────────────┘   │
│           │                          │                          │
│           │ _detect_peft_type()      │                          │
│           ▼                          ▼                          │
│  ┌─────────────────┐     ┌─────────────────────────────────┐   │
│  │ GLA_SD_LORA     │     │     标准 LoRA                    │   │
│  └─────────────────┘     └─────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               gla_sd_lora.py                             │   │
│  │  ┌─────────────────┐  ┌────────────────────────────┐    │   │
│  │  │ GlaSdLoraConfig │  │    GlaSdLoraModel          │    │   │
│  │  └─────────────────┘  │  ┌────────────────────┐    │    │   │
│  │                       │  │ GlaSdLoraParameter │    │    │   │
│  │                       │  └────────────────────┘    │    │   │
│  │                       └────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  fla/layers/gla.py                       │   │
│  │              GatedLinearAttention 层                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 配置类 `GlaSdLoraConfig`

**文件位置**: `mamba-peft/mamba_ssm_peft/peft/gla_sd_lora.py:45-160`

```python
@register_peft_config(MambaPeftType.GLA_SD_LORA)
@dataclass
class GlaSdLoraConfig(PeftConfig):
    """
    GLA SD-LoRA 配置类

    核心配置项:
    - select_mode: 维度选择模式 (目前仅支持 CHANNELS_ONLY)
    - proj_lora_r: 投影层 LoRA 秩
    - num_zero: 零化维度比例/数量 {"channel": 0.1}
    - num_freeze: 冻结维度比例/数量 {"channel": 0.5}
    - num_warmup_it: warmup 迭代次数
    - target_modules: SDT 目标模块 (如 ["gk_proj.1"])
    - lora_targets: LoRA 目标模块 (如 ["q_proj", "k_proj", ...])
    """
    select_mode: GLASelectMode = field(default=GLASelectMode.CHANNELS_ONLY)
    proj_lora_r: int = field(default=None)
    num_zero: Dict = field(default=None)
    num_freeze: Dict = field(default=None)
    num_warmup_it: int = field(default=None)
    target_modules: List[str] = field(default=None)
    lora_targets: List[str] = field(default=None)
    finetune_parameters: List[str] = field(default=None)
    sdlora_alpha: Dict = field(default=None)
    proj_lora_alpha: Optional[float] = field(default=None)
    proj_lora_dropout: float = field(default=0.1)
```

**配置验证逻辑** (`__post_init__`):

```python
def __post_init__(self):
    self.peft_type = MambaPeftType.GLA_SD_LORA

    # GLA 的 gk_proj 输出通道由 KV 组共享，无法进行 per-head 稀疏化
    if self.select_mode != GLASelectMode.CHANNELS_ONLY:
        raise ValueError(
            "GLA SD-LoRA currently supports only select_mode='CHANNELS_ONLY'."
        )

    # 验证 Train + Freeze + Zero ≤ 1.0
    if isinstance(self.num_zero["channel"], float) and \
       isinstance(self.num_freeze["channel"], float):
        total_ratio = self.num_zero["channel"] + self.num_freeze["channel"]
        assert total_ratio <= 1.0, f"num_zero + num_freeze must be <= 1.0"
```

### 5.3 模型包装类 `GlaSdLoraModel`

**文件位置**: `mamba-peft/mamba_ssm_peft/peft/gla_sd_lora.py:162-468`

#### 5.3.1 核心方法：`_create_new_module`

根据目标模块类型创建不同的适配器：

```python
def _create_new_module(self, peft_config, adapter_name, target, target_name):
    # 获取模块完整名称
    matching_names = [n for n, m in self.model.named_modules() if m is target]
    module_name = matching_names[0]

    # 判断是 LoRA 目标还是 SDT 目标
    lora_targets = peft_config.lora_targets or []

    if target_name in lora_targets and peft_config.proj_lora_r is not None:
        # 为线性投影层创建标准 LoRA
        new_module = LoraLinear(
            target, adapter_name,
            r=peft_config.proj_lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
    else:
        # 为门控投影创建 SD-LoRA 参数包装器
        block = self._find_gla_block_for_module(module_name)
        new_module = GlaSdLoraParameter(
            target, adapter_name, module_name, block,
            peft_config.select_mode,
            num_zero=peft_config.num_zero,
            num_freeze=peft_config.num_freeze,
            num_warmup_it=peft_config.num_warmup_it,
            sdlora_alpha=sdlora_alpha
        )

    return new_module
```

#### 5.3.2 训练模式管理

```python
@property
def should_training_stop(self):
    """检查是否应该停止训练 (warmup → train 转换)"""
    if self.last_mode == "warmup" and self.get_sdlora_mode() == "train":
        self.last_mode = "train"
        return True

    if self.last_mode is None:
        self.last_mode = self.get_sdlora_mode()

    return False
```

#### 5.3.3 配置保存与加载

```python
def save_config(self, path):
    """保存 warmup 阶段的梯度累积数据"""
    for m in self._get_sdlora_params():
        m.save_config(path)  # 保存为 .pkl 文件

def load_config(self, path, required=False):
    """加载梯度数据并切换到训练模式"""
    for m in self._get_sdlora_params():
        success = m.load_config(path, required=required)
        # 加载成功后会自动切换到 "train" 模式
```

### 5.4 参数包装类 `GlaSdLoraParameter`

**文件位置**: `mamba-peft/mamba_ssm_peft/peft/gla_sd_lora.py:470-969`

这是 SD-LoRA 的核心实现类，包装单个模块（如 `gk_proj.1`）。

#### 5.4.1 初始化

```python
class GlaSdLoraParameter(nn.Module, BaseTunerLayer):
    ZERO_MASK_VALUE = -100.0  # 用于强制关闭门控通道

    def __init__(self, base_layer, adapter_name, module_name, block,
                 select_mode, num_zero, num_freeze, num_warmup_it, sdlora_alpha=1):
        super().__init__()

        self.base_layer = base_layer  # 原始 nn.Linear 层
        self.module_name = module_name.replace(".", "_")
        self.select_mode = select_mode

        # 解析维度配置
        self.num_zero = self._parse_dims(num_zero)
        self.num_freeze = self._parse_dims(num_freeze)
        self.num_train = self._compute_num_train()

        # 创建梯度累积器和适配器参数
        self.sdlora_grad = self._create_grad_param()     # 全尺寸，用于 warmup
        self.sdlora_adapter = self._create_adapter_param()  # 只包含可训练维度

        # 设置初始模式
        self.set_sdlora_mode("warmup" if self.training and num_warmup_it >= 0 else "train")
```

#### 5.4.2 维度解析

```python
def _parse_dims(self, dims):
    """解析维度配置，支持分数和绝对值"""
    if dims is None:
        return {"channel": 0}

    param_info = self.get_model_param_info()
    channel_dim = dims.get("channel", 0)

    # 如果是分数，转换为绝对数量
    if isinstance(channel_dim, float):
        channel_dim = int(round(channel_dim * param_info.out_features))

    return {"channel": channel_dim}

def _compute_num_train(self):
    """计算可训练维度数量"""
    param_info = self.get_model_param_info()
    total_channels = param_info.out_features
    train_channels = total_channels - self.num_zero["channel"] - self.num_freeze["channel"]
    return {"channel": max(0, train_channels)}
```

#### 5.4.3 前向传播

```python
def forward(self, x):
    """
    前向传播，根据当前模式执行不同逻辑

    Warmup 模式: 在全参数上添加梯度累积器
    Train 模式: 应用稀疏适配器和零掩码
    """
    if self.sdlora_mode == "warmup" and self.it_counter > self.num_warmup_it:
        self.set_sdlora_mode("train")

    if self.is_layer:
        weight = self.base_layer.weight
        bias = self.base_layer.bias

        if self.sdlora_mode == "warmup":
            # Warmup: 添加全尺寸梯度累积器
            weight_new = weight + self.sdlora_alpha * self.sdlora_grad
            bias_new = bias
        elif self.sdlora_mode == "train":
            # Train: 应用稀疏适配器
            weight_new, bias_new = self.build_train_param(weight, bias, self.sdlora_adapter)

        self.it_counter += 1
        return F.linear(x, weight_new, bias_new)
```

#### 5.4.4 训练参数构建

```python
def build_train_param(self, weight, bias, adapter):
    """
    构建训练参数，应用稀疏适配器和零掩码

    零掩码逻辑（关键修正）：
        要真正"零化"一个门控通道，必须确保输出是大负数常量，
        不受输入影响。对于 Linear 层: output = W @ x + b

        错误方法（之前）: 将 W 行设为 -100，保持 b 不变
          → 如果 x 有负值，W @ x 可能为正，门仍然打开！

        正确方法: 将 W 行设为 0，将 b 设为 -100
          → output = 0 @ x + (-100) = -100，门保证关闭
    """
    # 构建掩码
    if self.train_mask is None:
        self.train_mask = self.get_mask("train")
    if self.zero_mask is None:
        self.zero_mask = self.get_mask("zero")

    weight_new = weight.clone()
    bias_new = bias.clone() if bias is not None else None

    # 应用零掩码
    if self.zero_mask.any():
        zero_channel_mask = self.zero_mask.any(dim=1)
        weight_new[zero_channel_mask] = 0.0  # 权重设为0
        if bias_new is not None:
            bias_new[zero_channel_mask] = self.ZERO_MASK_VALUE  # 偏置设为-100

    # 应用训练适配器 (使用 masked_scatter 保持梯度流)
    if self.train_mask.any():
        adapter_bias = torch.masked_scatter(
            torch.zeros_like(weight),
            self.train_mask,
            adapter.flatten()
        )
        weight_new = weight_new + self.sdlora_alpha * adapter_bias

    return weight_new, bias_new
```

#### 5.4.5 掩码生成

```python
def get_mask(self, mask_type):
    """
    基于梯度重要性构建 train/zero 掩码

    Args:
        mask_type: "train" 或 "zero"

    Returns:
        布尔掩码张量
    """
    grad = self.sdlora_grad.data

    param_info = self.get_model_param_info()
    mask = torch.zeros(param_info.shape, device=param_info.device, dtype=torch.bool)

    # 获取通道重要性排序
    importance_order = self.get_importances(grad, dim=0)

    # 选择通道
    channel_indices = self.select_channels(importance_order, mask_type)

    if len(channel_indices) > 0:
        mask.index_fill_(0, channel_indices, True)

    return mask
```

### 5.5 适配器层 `lat_adapter.py`

**文件位置**: `mamba-peft/lat_adapter.py`

#### 5.5.1 PEFT 类型检测

```python
def _detect_peft_type(peft_json: Dict[str, Any]) -> str:
    """
    检测 PEFT 类型

    优先级:
    1. HP_PEFT_TYPE 环境变量
    2. 配置文件中的 peft_type 字段
    3. 默认为 "LORA"
    """
    # 环境变量优先
    env_type = os.environ.get("HP_PEFT_TYPE", "").strip().lower()
    if env_type in ("sdlora", "sd_lora", "gla_sd_lora", "gla_sdlora"):
        return "GLA_SD_LORA"
    if env_type in ("lora",):
        return "LORA"

    # 检查配置文件
    config_type = peft_json.get("peft_type", "LORA")
    if isinstance(config_type, str):
        config_type_upper = config_type.upper().replace("-", "_")
        if config_type_upper in ("GLA_SD_LORA", "SDLORA", "SD_LORA"):
            return "GLA_SD_LORA"

    return "LORA"
```

#### 5.5.2 SD-LoRA 环境变量覆盖

```python
def _apply_sdlora_env_overrides(peft_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    应用 SD-LoRA 特定的环境变量覆盖

    维度比例逻辑 (Train + Freeze + Zero = 100%):
    - 默认: Train=40%, Freeze=50%, Zero=10%
    - 如果设置 HP_TRAIN_RATIO，Zero 自动计算: Zero = 1 - Train - Freeze
    """
    # 首先应用标准 LoRA 覆盖
    peft_json = _apply_lora_env_overrides(peft_json)

    # SD-LoRA 特定覆盖
    warmup_it = _env_int("HP_WARMUP_IT", peft_json.get("num_warmup_it", 100))
    peft_json["num_warmup_it"] = warmup_it

    # 默认比例
    default_train = 0.4
    default_freeze = 0.5
    default_zero = 0.1

    # 处理环境变量覆盖
    train_ratio_env = os.environ.get("HP_TRAIN_RATIO")
    freeze_ratio_env = os.environ.get("HP_FREEZE_RATIO")
    zero_ratio_env = os.environ.get("HP_ZERO_RATIO")

    # 如果设置了 HP_TRAIN_RATIO 但没有 HP_ZERO_RATIO，自动计算
    if train_ratio_env is not None and zero_ratio_env is None:
        train_ratio = float(train_ratio_env)
        zero_ratio = max(0.0, 1.0 - train_ratio - freeze_ratio)
        print(f"[SD-LoRA] HP_TRAIN_RATIO={train_ratio:.2f} set, auto-computed "
              f"zero_ratio={zero_ratio:.2f}")

    return peft_json
```

### 5.6 FLA 库的 GLA 层

**文件位置**: `3rdparty/flash-linear-attention/fla/layers/gla.py`

#### 5.6.1 GatedLinearAttention 类

```python
class GatedLinearAttention(nn.Module):
    """
    门控线性注意力层

    关键参数:
    - mode: GLA 内核类型 ('chunk', 'fused_recurrent', 'fused_chunk')
    - hidden_size: 输入隐藏维度
    - expand_k: key 维度扩展比例 (默认 0.5)
    - expand_v: value 维度扩展比例 (默认 1.0)
    - num_heads: 注意力头数
    - gate_logit_normalizer: 门控值归一化因子 (默认 16)
    - gate_low_rank_dim: 门控投影低秩维度 (默认 16)
    """

    def __init__(self, ...):
        # 投影层
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)

        # 门控投影 (SD-LoRA 主要目标)
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),  # 降维
            nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True)  # 升维
        )

        # 输出投影和门控
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
```

#### 5.6.2 前向传播

```python
def forward(self, hidden_states, ...):
    # 计算 Q, K, V
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # 计算门控值
    gk = self.gk_proj(hidden_states)  # SD-LoRA 作用于此

    # 重塑为多头格式
    q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
    # ...

    # 应用 logsigmoid 和归一化
    gk = F.logsigmoid(gk) / self.gate_logit_normalizer

    # 选择计算模式
    if mode == 'fused_recurrent':
        o, recurrent_state = fused_recurrent_gla(q, k, v, gk, ...)
    elif mode == 'fused_chunk':
        o, recurrent_state = fused_chunk_gla(q, k, v, g=gk, ...)
    elif mode == 'chunk':
        o, recurrent_state = chunk_gla(q, k, v, g=gk, ...)

    # 输出门控和投影
    if self.use_output_gate:
        g = self.g_proj(hidden_states)
        o = self.g_norm_swish_gate(o, g)
    o = self.o_proj(o)

    return o, None, past_key_values
```

### 5.7 Chunk GLA 核心算法

**文件位置**: `3rdparty/flash-linear-attention/fla/ops/gla/chunk.py`

#### 5.7.1 算法概述

Chunk GLA 使用 Triton 实现硬件高效的分块并行计算：

```python
def chunk_gla_fwd(q, k, v, g, g_cumsum, scale, initial_state, output_final_state, ...):
    """
    GLA 前向传播

    步骤:
    1. 计算门控值的累积和 (chunk_local_cumsum)
    2. 计算块间隐藏状态 (chunk_fwd_h)
    3. 计算块内注意力矩阵 (chunk_gla_fwd_intra_gk)
    4. 计算最终输出 (chunk_gla_fwd_o_gk)
    """
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, chunk_size, cu_seqlens=cu_seqlens)

    # 块间隐藏状态
    h, ht = chunk_fwd_h(k=k, v=v, gk=g_cumsum, ...)

    # 块内注意力
    A = chunk_gla_fwd_intra_gk(q=q, k=k, g=g_cumsum, scale=scale, ...)

    # 最终输出
    o = chunk_gla_fwd_o_gk(q=q, v=v, g=g_cumsum, A=A, h=h, scale=scale, ...)

    return g_cumsum, A, h, ht, o
```

#### 5.7.2 块内 Inter-Sub-Chunk 计算

```python
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(...):
    """
    计算块内 sub-chunk 间的注意力
    使用半精度矩阵乘法加速

    计算: P[i][j] = (Q[i] ⊙ Λ[i]) @ (K[j] ⊙ Γ[j] ⊙ b[iC]/b[(j+1)C])^T
    """
    # 加载门控值
    b_gn = tl.load(p_gn, mask=m_k, other=0)

    # 加载并缩放 Q
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_qg = b_q * exp(b_g - b_gn[None, :]) * scale

    # 加载并缩放 K
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_kg = b_k * exp(b_gn[:, None] - b_gk)

    # 矩阵乘法 (使用 tf32 提高精度)
    b_A += tl.dot(b_qg, b_kg)
```

#### 5.7.3 块内 Intra-Sub-Chunk 计算

```python
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(...):
    """
    计算块内 sub-chunk 内的注意力
    需要全精度计算以保持数值稳定性

    计算: P[i,j] = Σ_k Q[i,k] * K[j,k] * exp(log B[i,k] - log B[j,k])
    """
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)

        # 全精度计算
        b_A = tl.sum(b_q * b_k[None, :] * exp(b_g - b_gk[None, :]), 1) * scale

        tl.store(A + o_A + j, b_A, mask=m_A)
```

---

## 6. 配置系统与训练流程

### 6.1 配置文件结构

**示例配置** (`gla_sdlora_kv_train05.json`):

```json
{
    "peft_type": "GLA_SD_LORA",
    "select_mode": "CHANNELS_ONLY",
    "proj_lora_r": 8,
    "num_zero": {
        "channel": 0.0
    },
    "num_freeze": {
        "channel": 0.95
    },
    "num_warmup_it": 100,
    "target_modules": [
        "gk_proj.1",
        "k_proj",
        "v_proj"
    ],
    "lora_targets": [
        "k_proj",
        "v_proj"
    ],
    "finetune_parameters": null,
    "sdlora_alpha": {
        "global": 1.0,
        "gk_proj.1": 1.0
    },
    "_comment": "GLA SD-LoRA: Train=5%, Freeze=95%, Zero=0%. LoRA on: k_proj, v_proj"
}
```

**配置解释**:

| 字段 | 说明 |
|------|------|
| `peft_type` | PEFT 类型标识 |
| `select_mode` | 维度选择模式，当前仅支持 `CHANNELS_ONLY` |
| `proj_lora_r` | 投影层 LoRA 秩 |
| `num_zero.channel` | 零化通道比例 (0.0 = 0%) |
| `num_freeze.channel` | 冻结通道比例 (0.95 = 95%) |
| `num_warmup_it` | warmup 迭代次数 |
| `target_modules` | SDT 目标模块列表 |
| `lora_targets` | LoRA 目标模块列表 |
| `sdlora_alpha` | SDT 缩放因子 |

### 6.2 配置生成脚本

**文件位置**: `mamba-peft/generate_gla_sdlora_configs.py`

```python
# LoRA 目标组合
LORA_TARGETS_CONFIGS = {
    "kv": ["k_proj", "v_proj"],
    "v": ["v_proj"],
    "vo": ["v_proj", "o_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qkvog": ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj"],
    "qkvo_plus_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "mlp.up_proj", "mlp.down_proj"],
    "omlp": ["o_proj", "mlp.up_proj", "mlp.down_proj"],
}

# Train 比例配置
FULL_TRAIN_RATIOS = [1, 5, 10, 20, 30]  # 用于 KV 和 QKVO
SINGLE_TRAIN_RATIO = [5]  # 其他配置只使用 5%

# 始终使用 Zero=0%
ZERO_RATIO = 0.0
NUM_WARMUP_IT = 100
```

### 6.3 环境变量控制

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `HP_PEFT_TYPE` | PEFT 类型 (`lora`, `sdlora`) | 从配置文件读取 |
| `HP_PEFT_R` | LoRA/proj_lora_r 秩 | 配置文件值 |
| `HP_WARMUP_IT` | warmup 迭代次数 | 100 |
| `HP_TRAIN_RATIO` | 训练维度比例 | 0.4 |
| `HP_FREEZE_RATIO` | 冻结维度比例 | 0.5 |
| `HP_ZERO_RATIO` | 零化维度比例 | 0.1 (或自动计算) |

### 6.4 两阶段训练实现

#### Phase 1: Warmup

```python
# 在训练循环中
model.train()
for step, batch in enumerate(dataloader):
    # 前向传播 (GlaSdLoraParameter 会累积梯度)
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # 检查是否完成 warmup
    if model.should_training_stop:
        # 保存梯度累积数据
        model.save_config(config_dir)
        break

    optimizer.step()
    optimizer.zero_grad()
```

#### Phase 2: Training

```python
# 加载 warmup 梯度数据
model.load_config(config_dir, required=True)

# 验证所有模块都处于训练模式
model.verify_train_mode()

# 正常训练
for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
```

### 6.5 完整调用链

```
用户命令
    │
    ▼
train_lat.py (--peft sparse_peft/xxx.json)
    │
    ├─▶ lat_adapter.py::prepare_lat_model_and_tokenizer()
    │       │
    │       ├─▶ _detect_peft_type() → "GLA_SD_LORA"
    │       │
    │       ├─▶ _apply_sdlora_env_overrides()
    │       │
    │       ├─▶ GlaSdLoraConfig(**filtered_cfg)
    │       │
    │       └─▶ peft.get_peft_model(model, peft_cfg)
    │               │
    │               └─▶ GlaSdLoraModel.__init__()
    │                       │
    │                       └─▶ _create_new_module() for each target
    │                               │
    │                               ├─▶ LoraLinear (for lora_targets)
    │                               │
    │                               └─▶ GlaSdLoraParameter (for gk_proj.1)
    │
    ├─▶ Phase 1: Warmup Training
    │       │
    │       ├─▶ Forward: GlaSdLoraParameter.forward() [warmup mode]
    │       │       weight_new = weight + sdlora_alpha * sdlora_grad
    │       │
    │       ├─▶ model.save_config(config_dir)
    │       │       保存 sdlora_grad 到 .pkl 文件
    │       │
    │       └─▶ should_training_stop → True
    │
    └─▶ Phase 2: Main Training
            │
            ├─▶ model.load_config(config_dir, required=True)
            │       加载梯度数据，计算重要性，生成掩码
            │
            └─▶ Forward: GlaSdLoraParameter.forward() [train mode]
                    │
                    └─▶ build_train_param()
                            │
                            ├─▶ 应用 zero_mask: weight=0, bias=-100
                            │
                            └─▶ 应用 train_mask: masked_scatter adapter
```

---

## 7. 数学推导附录

### 7.1 GLA 递归展开

从递归形式：
$$
\mathbf{S}_{t} = \text{Diag}(\boldsymbol{\alpha}_{t}) \mathbf{S}_{t-1} + \boldsymbol{k}_{t}^{\mathsf{T}} \boldsymbol{v}_{t}
$$

展开到时刻 $t$：
$$
\mathbf{S}_{t} = \sum_{i=1}^{t} \left( \prod_{j=i+1}^{t} \text{Diag}(\boldsymbol{\alpha}_{j}) \right) \boldsymbol{k}_{i}^{\mathsf{T}} \boldsymbol{v}_{i}
$$

由于对角矩阵乘法可交换：
$$
\prod_{j=i+1}^{t} \text{Diag}(\boldsymbol{\alpha}_{j}) = \text{Diag}\left(\prod_{j=i+1}^{t} \boldsymbol{\alpha}_{j}\right) = \text{Diag}\left(\frac{\boldsymbol{b}_{t}}{\boldsymbol{b}_{i}}\right)
$$

其中 $\boldsymbol{b}_t = \prod_{j=1}^{t} \boldsymbol{\alpha}_j$。

### 7.2 输出计算

$$
\boldsymbol{o}_{t} = \boldsymbol{q}_{t} \mathbf{S}_{t} = \boldsymbol{q}_{t} \sum_{i=1}^{t} \text{Diag}\left(\frac{\boldsymbol{b}_{t}}{\boldsymbol{b}_{i}}\right) \boldsymbol{k}_{i}^{\mathsf{T}} \boldsymbol{v}_{i}
$$

$$
= \sum_{i=1}^{t} \boldsymbol{q}_{t} \text{Diag}\left(\frac{\boldsymbol{b}_{t}}{\boldsymbol{b}_{i}}\right) \boldsymbol{k}_{i}^{\mathsf{T}} \boldsymbol{v}_{i}
$$

$$
= \sum_{i=1}^{t} \left(\boldsymbol{q}_{t} \odot \frac{\boldsymbol{b}_{t}}{\boldsymbol{b}_{i}}\right) \boldsymbol{k}_{i}^{\mathsf{T}} \boldsymbol{v}_{i}
$$

$$
= \sum_{i=1}^{t} \left(\boldsymbol{q}_{t} \odot \boldsymbol{b}_{t}\right) \left(\frac{\boldsymbol{k}_{i}}{\boldsymbol{b}_{i}}\right)^{\mathsf{T}} \boldsymbol{v}_{i}
$$

### 7.3 对数空间数值稳定性

注意力权重可以写为：
$$
\mathbf{P}_{ij} = \sum_{k=1}^{d} (\boldsymbol{q}_{i} \odot \boldsymbol{b}_{i})_k \left(\frac{\boldsymbol{k}_{j}}{\boldsymbol{b}_{j}}\right)_k
$$

$$
= \sum_{k=1}^{d} \mathbf{Q}_{ik} \mathbf{K}_{jk} \frac{\mathbf{B}_{ik}}{\mathbf{B}_{jk}}
$$

在对数空间：
$$
= \sum_{k=1}^{d} \mathbf{Q}_{ik} \mathbf{K}_{jk} \exp(\log \mathbf{B}_{ik} - \log \mathbf{B}_{jk})
$$

### 7.4 门控梯度的闭式解

对于 $\mathbf{d}\log\boldsymbol{\alpha}_t$ 的计算，可以通过对方程求导得到闭式解：

$$
\mathbf{d}\log\boldsymbol{b}_{t} = \boldsymbol{q}_{t} \odot \mathbf{d}\boldsymbol{q}_{t} - \boldsymbol{k}_{t} \odot \mathbf{d}\boldsymbol{k}_{t}
$$

$$
\mathbf{d}\log\boldsymbol{\alpha}_{t} = \sum_{i=t}^{L} \mathbf{d}\log\boldsymbol{b}_{i}
$$

这避免了需要存储所有时间步的隐藏状态 $\mathbf{S}_t$，大大节省了内存。

### 7.5 SDT 理论基础

根据 Lemma 2 (SD-LoRA 论文)，最小可调参数数量为：

$$
\min = \|\text{diag}(\bar{\mathbf{A}}) \odot \bar{\mathbf{B}} \odot \mathbf{C}^{\mathsf{T}}\|_{(H^*+1):H}\|_0
$$
$$
+ \|[\bar{\mathbf{A}}]_{(1:H^*,1:H^*)} - \bar{\mathbf{A}}^*\|_0
$$
$$
+ \|[\bar{\mathbf{B}} \odot \mathbf{C}^{\mathsf{T}}]_{(1:H^*)} - \bar{\mathbf{B}}^* \odot \mathbf{C}^{*\mathsf{T}}\|_0
$$

其中：
- $H^*$ 是目标模型的状态维度
- $H$ 是当前模型的状态维度
- 第一项对应可剪枝的维度
- 第二、三项对应需要训练的维度

### 7.6 重要性分数计算

对于参数 $\theta$，其重要性分数定义为：

$$
I(\theta) = \mathbb{E}\left[\left\|\frac{\partial \mathcal{L}}{\partial \theta}\right\|_2^2\right]
$$

在 warmup 阶段通过累积估计：

$$
\hat{I}(\theta) = \sum_{t=1}^{T_{warmup}} \left\|\frac{\partial \mathcal{L}_t}{\partial \theta}\right\|_2^2
$$

---

## 总结

GLA SD-LoRA 是一种专门为门控线性注意力模型设计的参数高效微调方法，其核心创新在于：

1. **稀疏维度调优 (SDT)**: 基于梯度重要性将维度划分为 Train/Freeze/Zero 三类
2. **两阶段训练**: Warmup 阶段收集梯度信息，Training 阶段选择性更新
3. **门控特殊处理**: 通过将权重设为 0、偏置设为大负值来"关闭"不重要的门控通道
4. **混合策略**: 对门控投影使用 SDT，对其他投影层使用标准 LoRA

这种方法在保持参数效率的同时，能够更好地适配 GLA 模型的特殊结构，实现更好的下游任务性能。

---

## 参考资料

1. [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) (ICML 2024)
2. [Parameter-Efficient Fine-Tuning of State Space Models](https://arxiv.org/abs/2410.09016) (ICML 2025)
3. [Flash Linear Attention GitHub](https://github.com/sustcsonglin/flash-linear-attention)
4. [SSM-PEFT GitHub](https://github.com/furiosa-ai/ssm-peft)
