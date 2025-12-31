# GLA中LoRA与SDT的正确分工：基于线性投影vs动力学参数的理论框架

## 1. 前言

在PEFT方法对GLA（Gated Linear Attention）的适配中，存在一个关键设计选择：为什么要采用"**投影层LoRA + gk_proj SDT**"，而不是全部用LoRA或全部用SDT？

本文基于GLA论文核心动力学设计和Mamba PEFT的理论洞察，系统阐释这一选择背后的根本原因：**线性投影矩阵和SSM动力学参数具有本质不同的任务适配特性**。

---

## 2. GLA的核心动力学设计

### 2.1 递推形式

GLA的递推形式由论文第269-273行定义：

$$\mathbf{S}_{t} = \text{Diag}(\mathbf{\alpha}_{t}) \mathbf{S}_{t-1} + \mathbf{k}_{t}^{\top} \mathbf{v}_{t}$$

其中：
- $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$：递推状态（矩阵）
- $\mathbf{\alpha}_t \in \mathbb{R}^{d_k}$：**通道维度的衰减因子**（向量）
- $\mathbf{k}_t, \mathbf{v}_t$：当前时刻的key和value
- $\text{Diag}(\mathbf{\alpha}_{t})$：对角矩阵，将α_t应用到S的每一行

**关键理解**：α_t不是一个标量（如Mamba-2），而是一个**向量**，维度等于key维度。这意味着**每个通道都有自己独立的衰减因子**。

### 2.2 α_t的参数化

论文第352-356行详细说明了α_t的参数化方式：

$$\mathbf{\alpha}_{t} = \sigma\left(\left(\mathbf{x}_{t} \mathbf{W}_{\alpha}^1 \mathbf{W}_{\alpha}^2 + \mathbf{b}_{\alpha}\right)\right)^{1/\tau}$$

其中：
- $\mathbf{W}_{\alpha}^1 \in \mathbb{R}^{d \times 16}$：**第一层（低秩瓶颈）**
- $\mathbf{W}_{\alpha}^2 \in \mathbb{R}^{16 \times d_k}$：**第二层（展开回key维度）**
- $\tau = 16$：温度系数
- $\sigma$：sigmoid激活函数

在代码中（gla.py第150-151行），这对应gk_proj的两层Sequential：

```python
self.gk_proj = nn.Sequential(
    nn.Linear(hidden_size, gate_low_rank_dim, bias=False),  # W_α^1
    nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True)  # W_α^2
)
```

**设计思想**：第一层作为"瓶颈"压缩信息（hidden_size → 16维），第二层再展开回原始key维度。这种**显式的低秩结构**保证了参数效率，同时第二层的输出与α_t的各维度有**直接的1:1对应关系**。

### 2.3 投影层的设计

论文第336-341行定义的GLA Transformer中，投影层为：

- $\mathbf{q}_{t}^{h} = \mathbf{x}_{t} \mathbf{W}_Q$
- $\mathbf{k}_{t}^{h} = \mathbf{x}_{t} \mathbf{W}_K$
- $\mathbf{v}_{t}^{h} = \mathbf{x}_{t} \mathbf{W}_V$
- $\mathbf{r}_{t} = \text{Swish}(\mathbf{x}_{t} \mathbf{W}_{r} + \mathbf{b}_{r})$
- $\mathbf{y}_{t} = (\mathbf{r}_{t} \odot \mathbf{o}_{t}^{\prime}) \mathbf{W}_{O}$

代码中（gla.py第123-128行）对应：

```python
self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
if self.use_output_gate:
    self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
```

**关键特征**：这些都是**标准的全秩线性层**，没有任何低秩约束。它们的作用是**跨通道的全连接混合**，将hidden_size维的输入映射到不同的表示空间（key/value维度）。

---

## 3. 线性投影 vs 动力学参数：本质区别

### 3.1 投影层的任务适配特性

**定义**：投影层（Q/K/V/O/G）是**维度转换和跨通道混合**的密集矩阵乘法。

**任务适配表现**：当模型适应新任务时，这些投影矩阵通常需要进行"**低秩重加权**"：
- 某些特征维度的重要性变化
- 特征间的相互作用模式调整
- 但整体的维度和通道数不变

**理论依据**：Mamba PEFT论文的**Lemma 1**指出，仅微调投影矩阵（即使不调整SSM内部参数）已经足以覆盖SSM参数调整所能达到的表达力。这说明投影层已经是信息瓶颈，低秩调整足以适应任务变化。

**数据支持**：论文Table 1的消融实验显示，仅微调投影层的LoRA性能几乎等同于"投影层+SSM都用LoRA"的设置，边际收益极低。

### 3.2 动力学参数的任务适配特性

**定义**：α_t（通过gk_proj生成）是**每通道的衰减因子向量**，直接控制递推状态的遗忘机制。

**任务适配表现**：不同任务对**不同通道的遗忘策略的需求差异很大**：
- 某些通道需要长期记忆（训练Train），α_t应该接近1
- 某些通道可以快速衰减（冻结Freeze），α_t应该接近0
- 某些通道完全不必要（零化Zero），可以置为0

**与线性投影的根本区别**：
- 线性投影："维度转换"，没有显式的语义分工
- α_t："通道选择"，每个维度有明确的衰减语义

### 3.3 为什么投影层适合LoRA

1. **低秩加法假设成立**：投影矩阵的任务适配本质上是低秩扰动（△W是低秩的）
2. **参数效率**：LoRA显著减少了可训练参数数量（从O(d²)降至O(d·r)）
3. **实验验证**：在投影层上应用LoRA已经达到接近全微调的性能

**代码实现原理**：
- W_Q, W_K, W_V, W_O都是full-rank的cross-channel混合
- LoRA在这些层上添加low-rank修正：W_Q ← W_Q + W_Q_lora_down @ W_Q_lora_up
- 这完全匹配了任务适配的"低秩重加权"特性

### 3.4 为什么gk_proj适合SDT

1. **显式的通道语义**：gk_proj第二层的每个输出维度对应α_t的一个维度，即一个通道的衰减因子
2. **维度选择而非调整**：任务适配需要的不是"调整α_t的值"，而是"决定哪些通道训练、哪些冻结、哪些归零"
3. **SDT的归纳偏置**：通道级选择（channel-wise selection）直接对症，避免了破坏通道的显式衰减语义

**为什么不是全体LoRA**：
- 虽然gk_proj包含低秩结构（第一层的瓶颈），但继续在其上添加LoRA意义不大
- 低秩加法假设（△W是低秩）在动力学参数上常常不成立，因为α_t的变化往往是"某些维度需要变，某些维度不需要变"，这是**稀疏选择**而非**低秩调整**
- 实验evidence：Mamba PEFT论文中，SSM动力学参数上的LoRA几乎没有边际收益

---

## 4. SD-LoRA的设计：投影层LoRA + gk_proj SDT

### 4.1 组件分工

**投影层LoRA**：
- 应用对象：q_proj, k_proj, v_proj, g_proj, o_proj
- 方式：在full-rank线性层上应用LoRA
- 参数数量：每层从O(d²)降至O(d·r)，其中r是LoRA rank
- 理由：低秩重加权假设，实验验证边际效应小

**gk_proj SDT**：
- 应用对象：gk_proj的第二层（16 → key_dim）
- 方式：通道级维度选择（channel-wise dimension selection）
- Train/Freeze/Zero划分：
  - Train（40%）：通道的衰减因子可训练，α_t可调
  - Freeze（50%）：通道的衰减因子冻结，α_t固定
  - Zero（10%）：通道的衰减因子置零，该通道快速衰减，等效剪枝
- 理由：通道显式语义，决定遗忘策略

### 4.2 为什么gk_proj只需SDT

gk_proj的特殊性在于其**显式的两层结构**：
- 第一层（hidden_size → 16）：压缩信息，学习通用的gate生成方式
- 第二层（16 → key_dim）：展开并映射到具体通道，每个输出维度对应一个α_t分量

**应用SDT的位置**：通常在第二层，因为：
1. 第二层输出维度就是α_t的维度（key_dim），有明确的通道对应
2. 选择哪些通道被激活（Train）、保持不变（Freeze）或关闭（Zero），直接对应选择哪些通道有快速衰减能力
3. 这避免了LoRA可能带来的"低秩混合"破坏通道语义的问题

### 4.3 与SM-LoRA（Mamba PEFT）的对应关系

Mamba中SSM参数（A矩阵）的处理与GLA的gk_proj类似：
- Mamba：A矩阵是对角的，每个元素控制一个通道的衰减
- GLA：α_t向量，每个元素控制一个通道的衰减
- 两者都用SDT：因为都是"通道维度的衰减因子"

---

## 5. 为什么不全部SDT？

### 5.1 SDT的归纳偏置限制

SDT（Selective Dimension Tuning）的工作假设：
- 被选中的维度进行参数更新
- 被冻结的维度保持原值
- 被归零的维度彻底删除

这个假设对**有明确通道/维度语义的层**有效：
- SSM的对角A矩阵：每个对角元素控制一个state channel的衰减
- GLA的α_t向量：每个分量控制一个key channel的衰减

但对**密集线性层无效**：
- Q/K/V投影是full-rank cross-channel混合
- 每个输出维度是所有输入维度的组合
- 单独"选择"某些维度会破坏这种混合，导致信息丢失

### 5.2 LoRA的优势

LoRA的设计正好补偿了SDT在密集层上的劣势：
- 通过秩约束（rank ≪ dimension）而不是二值选择（Train/Zero）
- 保留了cross-channel的混合能力
- 在投影层的"低秩重加权"假设下非常高效

---

## 6. 为什么不全部LoRA？

### 6.1 低秩加法假设的限制

LoRA基于假设：任务适配对应于向W矩阵添加**低秩矩阵**△W。

这在投影层上成立，但在动力学参数上常常不成立：
- 投影层的任务适配：某些特征维度的重要性变化（全局，continuous）
- 动力学参数的任务适配：某些通道的遗忘策略变化（局部，discrete）

### 6.2 实验证据

Mamba PEFT论文的Table 1比较了不同PEFT方案：
- SSM参数+LoRA：相比仅投影层LoRA，边际收益极小（< 0.1%）
- SSM参数+SDT：对recall-intensive任务有明显改进

这表明：
1. 仅微调投影层（LoRA）已经足够应对大多数任务
2. 在SSM动力学上继续叠LoRA收益递减
3. SDT的"维度选择"更契合动力学参数的适配方式

### 6.3 参数效率考虑

- 仅投影层LoRA + gk_proj SDT：参数量最少，性能最优
- 投影层LoRA + 全部层LoRA：参数量多，但性能增益不足以抵消
- SDT的通道选择相比LoRA更"极端"（要么Train要么Zero），因此参数效率更高

---

## 7. GLA代码中的实现细节

### 7.1 投影层（支持LoRA）

```python
# gla.py, lines 123-128
self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
if self.use_output_gate:
    self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
```

这些都是标准nn.Linear，LoRA包装器可以直接应用。

### 7.2 动力学参数gk_proj（支持SDT）

```python
# gla.py, lines 150-151
self.gk_proj = nn.Sequential(
    nn.Linear(hidden_size, gate_low_rank_dim, bias=False),  # gk_proj.0
    nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True)  # gk_proj.1
)
```

SDT应用在gk_proj.1（16 → key_dim_per_group）：
- 此层的每个输出对应α_t的一个维度
- 通道级选择（Train/Freeze/Zero）直接对应α_t维度的可训练性

### 7.3 前向传播中的应用

```python
# gla.py, line 226
gk = self.gk_proj(hidden_states)

# lines 236-238
gk = F.logsigmoid(gk) / self.gate_logit_normalizer
if self.clamp_min is not None:
    gk = torch.clamp_min(gk, self.clamp_min)
```

此gk即为α_t（reshape后），通过SDT选择的维度决定其可训练性。

---

## 8. 实验配置：Train=40%, Freeze=50%, Zero=10%

### 8.1 设置的理由

在LAT框架中，gk_proj.1的key_dim_per_group个通道被划分为：
- **Train（40%）**：这些通道的衰减因子可全量调整，适应新任务的长期记忆需求
- **Freeze（50%）**：大多数通道保持原始衰减能力，保留预训练知识
- **Zero（10%）**：极小部分通道快速衰减或直接置零，进行"动态剪枝"

### 8.2 为什么不是其他比例？

- **Train不宜过高**（< 50%）：过多的可训练参数容易过拟合，且预训练知识丧失
- **Freeze不宜过低**（> 30%）：预训练知识保留不足
- **Zero占比小（~10%）**：大多数通道都是有用的，只需要剪枝明显冗余的维度

---

## 9. 总结

### 9.1 核心论点

GLA中"投影层LoRA + gk_proj SDT"的设计源于**对任务适配特性的正确认识**：

1. **线性投影（Q/K/V/O/G）**：
   - 跨通道的全连接混合
   - 任务适配 = 低秩重加权
   - 适合LoRA

2. **动力学参数（α_t通过gk_proj）**：
   - 每通道的衰减因子向量
   - 任务适配 = 通道选择与策略变化
   - 适合SDT

3. **为什么不全SDT**：
   - 密集线性层无显式的通道语义
   - SDT会破坏cross-channel混合能力
   - LoRA在这些层上更高效

4. **为什么不全LoRA**：
   - 动力学参数的低秩假设不成立（需要的是离散选择，不是连续调整）
   - 实验证明SSM参数上的LoRA边际收益极小
   - SDT的通道选择对症SSM动力学的特性

### 9.2 设计的一致性

这套设计与Mamba PEFT论文的理论框架完全一致，充分体现了**在参数受限条件下，针对不同类型的模块采用差异化PEFT策略**的设计哲学。

