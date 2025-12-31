# GLA模型的LoRA vs SDT分工分析

基于"LoRA负责线性投影，SDT负责动力学参数"的分工原则，本文深入分析GLA (Gated Linear Attention) 模型中哪些参数更适合LoRA，哪些更适合SDT。

## 概述：GLA的参数结构

GLA核心在于一个**分层的参数设计**：
- **线性投影层**：处理特征空间变换
- **门控投影层**：控制SSM状态的衰减与遗忘

```
Input (hidden_size)
    │
    ├─→ q_proj (dense) → key_dim              [Linear Projection]
    ├─→ k_proj (dense) → key_dim_per_group    [Linear Projection]
    ├─→ v_proj (dense) → value_dim_per_group  [Linear Projection]
    ├─→ g_proj (dense) → value_dim            [Output Gate Projection]
    ├─→ gk_proj (two-layer low-rank) → key_dim_per_group  [Dynamics Gate]
    │
    └─→ [GLA Core: S_t = (α_t^T · 1) ⊙ S_{t-1} + k_t^T v_t]
        │
        └─→ o_proj (dense) → hidden_size     [Output Projection]
```

---

## 第一部分：线性投影层（适合LoRA）

### 1.1 Q投影 (q_proj)
**参数形状**: hidden_size → key_dim

**语义**:
- 将输入特征映射到查询空间
- 与注意力机制的"查询"相同，用于特征提取

**为什么适合LoRA**:
1. **任务适配性强**：不同任务需要不同的特征表示维度和方向
2. **低秩变化**：微调时，Q的适配通常表现为"整体但有限的秩"的权重变化
3. **独立于动力学**：Q的变化不影响SSM的时间动态，只影响特征提取
4. **与Transformer一致**：在Transformer中，Q/K/V投影已被验证为低秩适配的有效位置

**LoRA配置建议**:
```json
{
  "target_modules": ["q_proj"],
  "r": 8,           // 保守配置，GLA的key_dim=256时合理
  "alpha": 16,      // alpha=2*r
  "dropout": 0.1
}
```

---

### 1.2 K投影 (k_proj)
**参数形状**: hidden_size → key_dim_per_group

**语义**:
- 将输入映射到键空间
- 与线性注意力的"键"对应，用于特征相似度计算

**为什么适合LoRA**:
1. **与Q互补**：Q和K共同定义注意力空间，都需要任务适配
2. **低秩表示**：键的映射通常可以用低秩矩阵有效表示
3. **无需关心门控**：K的变化不涉及α_t这样的动力学参数

**LoRA配置建议**:
```json
{
  "target_modules": ["k_proj"],
  "r": 8,
  "alpha": 16
}
```

---

### 1.3 V投影 (v_proj)
**参数形状**: hidden_size → value_dim_per_group

**语义**:
- 将输入映射到值空间
- 与线性注意力的"值"对应，用于状态矩阵更新

**为什么适合LoRA**:
1. **特征表示**：V投影决定了信息的编码方式
2. **任务特异性**：不同任务需要不同的值表示
3. **直接与S交互**：V通过外积 k_t^T v_t 更新状态，但这本身仍是低秩变化
4. **SMT论文验证**：在Transformer中，V向量是梯度最大的部分，表明其适应性强

**LoRA配置建议**:
```json
{
  "target_modules": ["v_proj"],
  "r": 8,
  "alpha": 16
}
```

**特别说明**:
- 根据SMT论文，V向量梯度是Q/K的5-10倍
- 可考虑为V分配更大的LoRA秩：`r=12`

---

### 1.4 G投影 (g_proj)
**参数形状**: hidden_size → value_dim

**语义**:
- 计算输出门 r_t = Swish(x_t W_r)
- 用于信息流量控制

**为什么适合LoRA**:
1. **任务相关的门控**：不同任务需要不同的信息流量
2. **非动力学参数**：不影响SSM状态更新
3. **与输出相关**：类似Transformer的输出投影，需要任务适配

**LoRA配置建议**:
```json
{
  "target_modules": ["g_proj"],
  "r": 8,
  "alpha": 16
}
```

---

### 1.5 O投影 (o_proj)
**参数形状**: value_dim → hidden_size

**语义**:
- 将多头GLA输出投影回原始隐藏维度
- 用于维度恢复和信息聚合

**为什么适合LoRA**:
1. **任务适配关键位置**：类似Transformer中的o_proj
2. **信息聚合**：需要重新组织多个特征空间中的信息
3. **独立于时间动态**：O投影不影响状态递推

**LoRA配置建议**:
```json
{
  "target_modules": ["o_proj"],
  "r": 8,
  "alpha": 16
}
```

---

### 1.6 短卷积投影 (q_conv1d, k_conv1d, v_conv1d)
**可选的短时卷积特征提取**

**为什么适合LoRA**:
- 这些是在q/k/v之前的局部特征提取
- 本质上仍是参数化的"特征重编码"
- 低秩适配适用于卷积权重的任务转移

---

## 第二部分：动力学参数（适合SDT）

### 2.1 GK投影 (gk_proj)：核心动力学参数
**参数形状**: Sequential(
  - Linear(hidden_size, gate_low_rank_dim, bias=False),     # .0: hidden→16
  - Linear(gate_low_rank_dim, key_dim_per_group, bias=True) # .1: 16→256
)

**语义**：
根据GLA论文第4.1-4.4节的设计：
```
α_t = Sigmoid(gk_proj(x_t)) / gate_logit_normalizer
S_t = Diag(α_t) · S_{t-1} + k_t^T v_t
```

即：gk_proj直接参数化SSM的**遗忘门**，控制每个通道的状态衰减。

### 2.2 为什么gk_proj适合SDT而不是LoRA

#### 原因1：非低秩语义
```
低秩假设（LoRA）：
  W'[i,:] = W[i,:] + (A·B^T)[i,:]  ← W[i,:]沿着低秩方向变化

动力学调整（SDT）：
  α_t[j] = sigmoid(gk_proj(x_t)[j])  ← 通道j的衰减因子

假如只训练40%维度：
  α_t[重要通道] = 学习新的衰减
  α_t[冻结通道] = 原始衰减（保留预训练）
  α_t[零通道] = exp(-100)/16 ≈ 0  → 完全遗忘

这不是"低秩扰动"，而是"通道级的显式控制"。
```

#### 原因2：结构化的参数空间
```
LoRA框架期望：
  参数矩阵的行向量能用低秩表示
  例：q_proj的所有行都通过相同的低秩基变化

SSM动力学参数期望：
  每个通道有自己的衰减因子
  α_t[j] 是通道j的二值特征（重要/冻结/遗忘）

gk_proj不需要"所有通道同时低秩变化"，
它需要"选择哪些通道改变行为"。
```

#### 原因3：与预训练知识的关系
```
Q/K/V投影：
  原始权重W学习了"通用特征提取"
  LoRA加法△W适应"新任务的特征重加权"
  两者可以叠加

gk_proj（α_t）：
  原始权重学习了"该任务的遗忘模式"
  SDT修改意图："这个通道在新任务中应该遗忘更快/更慢"

维度零化（α_t ≈ 0）意味着：
  "该维度在新任务中完全无用，不需要记忆"
  这是对通道重要性的**显式判断**，而非"渐进式权重混合"
```

### 2.3 GLA中的两层gk_proj设计分析

GLA使用低秩设计：`gk_proj = [hidden→16→key_dim]`

```python
self.gk_proj = nn.Sequential(
    nn.Linear(hidden_size, gate_low_rank_dim, bias=False),     # .0
    nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True) # .1
)
```

**第一层** (hidden_size→16)：
- 目的：参数高效地压缩隐藏状态信息
- 本质：特征选择（从hidden_size维筛选出16个关键特征）
- 适配策略：**可考虑LoRA**（如果要微调特征选择方式）

**第二层** (16→key_dim)：
- 目的：将16维压缩特征扩展到每个通道的门控值
- 本质：学习"16个基特征如何组合成256个通道的遗忘因子"
- 适配策略：**必须SDT**（直接控制α_t，决定通道的生死）

---

## 第三部分：分工实施方案

### 3.1 推荐配置

```python
{
  # LoRA目标：所有线性投影
  "lora_targets": [
    "q_proj",       # 查询投影 (r=8)
    "k_proj",       # 键投影 (r=8)
    "v_proj",       # 值投影 (r=8 or 12)
    "g_proj",       # 输出门 (r=8)
    "o_proj"        # 输出投影 (r=8)
  ],

  # SDT目标：仅gk_proj.1（第二层）
  # gk_proj.0（第一层）可选：也用LoRA或保持冻结
  "sdt_targets": ["gk_proj.1"],
  "sdt_config": {
    "num_zero": {"channel": 0.1},      # 10% 维度彻底遗忘
    "num_freeze": {"channel": 0.5},    # 50% 维度冻结不变
    "num_train": {"channel": 0.4}      # 40% 维度学习新衰减
  },

  # LoRA配置
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.1,

  # SDT配置
  "warmup_it": 100,        # 梯度累积评估重要性
  "zero_mask_value": -100  # 彻底遗忘
}
```

### 3.2 微调策略

#### 方案A：纯LoRA（保守，推荐）
```bash
# 仅对所有投影层应用LoRA
HP_PEFT_TYPE=lora ./lat_batch_tmux.sh --pairs "87:glue-tvt_cola"
```
**优点**：
- 参数少（~5%）
- 稳定性高
- 保留完整的动力学系统

**缺点**：
- 无法重设SSM的遗忘模式
- 对SSM特定任务的适配有上限

#### 方案B：LoRA + SDT（推荐）
```bash
# 对投影层LoRA，对gk_proj.1做SDT
HP_PEFT_TYPE=sdlora HP_TRAIN_RATIO=0.4 HP_FREEZE_RATIO=0.5 HP_ZERO_RATIO=0.1 \
  ./lat_batch_tmux.sh --pairs "87:glue-tvt_cola"
```
**优点**：
- 同时适配特征空间和动力学系统
- 参数量适中（~8-10%）
- 充分发挥GLA的SSM特性

**推荐的超参数**：
- **Train:40%** - 允许足够的通道适应新任务的遗忘模式
- **Freeze:50%** - 保留预训练的关键遗忘策略
- **Zero:10%** - 只完全遗忘最不重要的通道

#### 方案C：完全LoRA（仅对gk_proj）
```bash
# gk_proj完全用LoRA（不做SDT）
HP_PEFT_TYPE=lora \
  HP_LORA_TARGETS="q_proj,k_proj,v_proj,g_proj,o_proj,gk_proj" \
  ./lat_batch_tmux.sh --pairs "87:glue-tvt_cola"
```
**缺点**：
- gk_proj用LoRA不是最优的（不符合动力学参数的特性）

---

## 第四部分：实验设计与验证

### 4.1 消融实验框架

| 方案 | LoRA目标 | SDT目标 | 参数% | 预期性能 |
|------|---------|--------|-------|----------|
| **Baseline** | - | - | 100% | 100% |
| **Pure LoRA** | 所有投影 | - | 5% | 85-95% |
| **gk只LoRA** | 所有+gk | - | 8% | 82-90% |
| **Mixed (推荐)** | 投影层 | gk.1(40/50/10) | 8-10% | 90-98% |
| **LoRA+更大SDT** | 投影层 | gk.1(60/30/10) | 8-10% | 88-96% |

### 4.2 验证gk_proj为动力学参数的实验

**实验A：通道梯度方向分析**
```python
# 在warmup阶段，计算各层的梯度方向相关性
for layer in model.layers:
    # 投影层梯度（应该接近低秩）
    q_grad = layer.q_proj.weight.grad  # [key_dim, hidden_size]
    k_grad = layer.k_proj.weight.grad

    # 计算SVD秩
    U, S, Vt = torch.linalg.svd(q_grad, full_matrices=False)
    q_effective_rank = torch.sum(S > 0.01 * S[0]).item()  # 相对秩

    # 动力学参数梯度（应该是通道方向）
    gk_grad = layer.gk_proj[1].weight.grad  # [key_dim_per_group, 16]
    # gk_grad的行向量应该具有"选择性"（稀疏/二值化倾向）
    gk_row_l1 = gk_grad.abs().sum(dim=1)  # 每行的L1范数

    print(f"q_proj effective rank: {q_effective_rank} / {min(q_grad.shape)}")
    print(f"gk_proj row sparsity: {(gk_row_l1 < 0.1).float().mean():.2%}")
```

**预期结果**：
- q/k/v/g/o_proj: effective_rank 很小 (<<d) → 低秩性确认
- gk_proj: 行的L1范数分化明显 → 通道选择性确认

**实验B：通道重要性排序的稳定性**
```python
# 对不同数据集，评估SDT选出的重要通道集合的Jaccard相似度
train_on_cola()    # 在COLA上SDT，记录重要通道集合I_cola
train_on_mrpc()    # 在MRPC上SDT，记录I_mrpc

jaccard = len(I_cola & I_mrpc) / len(I_cola | I_mrpc)
# 如果gk_proj确实控制动力学，jaccard应该较高
# 如果gk_proj只是"任意参数"，jaccard会很低
```

---

## 第五部分：理论总结

### 5.1 GLA中的参数类型

| 参数 | 矩阵形状 | 语义 | 适配方式 | 理由 |
|------|---------|------|---------|------|
| q_proj | d×d_k | 查询特征 | LoRA | 特征提取，低秩适配 |
| k_proj | d×d_k | 键特征 | LoRA | 特征提取，低秩适配 |
| v_proj | d×d_v | 值特征 | LoRA | 特征表示，低秩适配 |
| g_proj | d×d_v | 输出门 | LoRA | 特征流量控制，低秩 |
| o_proj | d_v×d | 输出线性 | LoRA | 维度恢复，低秩 |
| **gk_proj.1** | **d_k×d_r** | **衰减因子** | **SDT** | **通道级动力学**，非低秩 |

### 5.2 核心区别

**LoRA适用的条件**：
1. 参数矩阵具有"整体但有限秩"的变化空间
2. 微调是对基础特征提取的"重加权"
3. 无需改变参数的结构化语义

**SDT适用的条件**：
1. 参数直接控制模型的动力学行为
2. 微调需要"选择哪些通道改变行为"
3. 参数具有明确的"通道/维度"语义

**GLA的gk_proj**：
- 直接参数化每个通道的遗忘因子 α_t
- 不同通道应该有**不同的适应策略**（重要的训练，预训练的冻结，无用的遗忘）
- 这正是SDT的设计初衷

### 5.3 与Mamba的对比

| 模型 | 动力学参数 | 适配方式 | 实现 |
|------|-----------|---------|------|
| Mamba | A_log (全秩矩阵) | SDT | 每个通道独立选择 |
| GLA | gk_proj → α_t (向量) | SDT | 通过梯度选择重要通道 |
| Transformer | 无（全注意力）| LoRA | 对Q/K/V/O投影做LoRA |

GLA和Mamba都需要SDT的核心原因是：**它们的"记忆参数"不是投影矩阵，而是通道级的动力学系数**。

---

## 实施建议

### 即刻可行
1. ✅ 当前已实现：gk_proj.1做SDT，其他投影LoRA
2. ✅ 超参数已正确设置：Train=40%, Freeze=50%, Zero=10%
3. ✅ 代码已集成在 lat_adapter.py 中

### 未来优化方向
1. **gk_proj.0也用LoRA**（可选）
   - 第一层投影可考虑LoRA，原因同q/k/v
   - 但由于维度只有16→256，限制不大

2. **V向量优先级调整**
   - SMT论文显示V梯度最大（5-10倍）
   - 可为v_proj分配更大的LoRA秩：r=12-16

3. **全局Top-K选择**（高级）
   - 而不是per-layer固定比例
   - 根据梯度全局排序，自动分配Zero/Train/Freeze

4. **通道级梯度策略对比**
   - 验证mean(|grad|)是否是最佳选择
   - 尝试L2范数、方差等其他度量

---

## 参考文献

1. Yang, S. et al. "Gated Linear Attention Transformers with Hardware-Efficient Training." ICML 2024.
2. Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
3. SMT ICLR 2025 论文：Sparse matrix tuning的发现与启示
4. GLA源码：fla/layers/gla.py
