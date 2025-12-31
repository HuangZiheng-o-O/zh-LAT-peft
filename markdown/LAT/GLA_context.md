# Gated Linear Attention (GLA) - 核心实现原理与架构

## 论文信息
- **标题**: Gated Linear Attention Transformers with Hardware-Efficient Training
- **作者**: Songlin Yang, Bailin Wang, et al. (MIT)
- **发表**: ICML 2024
- **arXiv**: 2312.06635v6
- **代码**: https://github.com/sustcsonglin/flash-linear-attention

---

## 一、核心动机与创新

### 1.1 问题背景

**Softmax Attention的局限**:
- 训练时并行高效,但复杂度 O(L²d) 对长序列不友好
- 推理时需要维护完整KV cache,内存占用随序列长度线性增长

**Linear Attention的挑战**:
- 理论上可以降低复杂度到 O(Ld²),但性能通常低于Softmax Attention
- 缺乏数据依赖的遗忘机制(forget gate)
- 现有实现不具备I/O感知能力,实际速度慢于优化的FlashAttention

### 1.2 GLA的三大创新

1. **硬件高效算法 (FlashLinearAttention)**
   - I/O感知的chunk-based并行训练算法
   - 在1K序列长度上已快于FlashAttention-2

2. **数据依赖门控机制 (Data-Dependent Gating)**
   - 引入细粒度、数据驱动的遗忘门 α_t ∈ (0,1)^{d_k}
   - 相比RetNet的全局衰减γ,更具表现力

3. **兼顾性能与效率**
   - 训练吞吐量高于Mamba (尤其是长序列)
   - 长度外推能力强 (2K训练 → 20K推理)
   - 在recall密集型任务上表现优于其他subquadratic模型

---

## 二、Linear Attention基础

### 2.1 并行形式 vs 循环形式

**标准Linear Attention**:

循环形式 (推理):
```
S_t = S_{t-1} + k_t^T v_t     (状态更新)
o_t = q_t S_t                   (输出计算)
```
- **S_t ∈ ℝ^{d×d}**: 矩阵值隐状态 (类似"快速权重")
- **复杂度**: O(Ld²) 时间, O(d²) 空间 (单步)

并行形式 (训练):
```
O = ((QK^T) ⊙ M) V
```
- **M**: 因果掩码 (下三角矩阵)
- **复杂度**: O(L²d) - 仍然是二次的!

### 2.2 Chunkwise并行形式 (关键优化)

**核心思想**: 将序列分成长度为C的chunk,在chunk间串行递归,chunk内并行计算

**Inter-chunk递归** (chunk级别状态传递):
```
S_{[i+1]} = S_{[i]} + K_{[i]}^T V_{[i]}
```

**Intra-chunk并行** (chunk内attention):
```
O_{[i+1]} = Q_{[i+1]} S_{[i]}           (inter-chunk贡献)
          + ((Q_{[i+1]} K_{[i+1]}^T) ⊙ M) V_{[i+1]}  (intra-chunk贡献)
```

**复杂度分析**:
- 总复杂度: O(LCd + Ld²)
- C=L → 退化为并行形式 O(L²d)
- C=1 → 退化为循环形式 O(Ld²)
- **最优选择**: C ≈ √(d/2) 时达到理论最优

---

## 三、Gated Linear Attention核心机制

### 3.1 循环形式

**标准GLA更新方程**:
```
S_t = (α_t^T 1) ⊙ S_{t-1} + k_t^T v_t
    = Diag(α_t) S_{t-1} + k_t^T v_t

o_t = q_t S_t
```

**关键组件**:
- **α_t ∈ (0,1)^{d_k}**: 数据依赖的遗忘门 (per-dimension)
- **α_t^T 1**: 向量外积,产生对角矩阵形式的门控
- **Diag(α_t)**: 对角矩阵,逐维度衰减

**门控参数化** (低秩设计):
```python
α_t = sigmoid(x_t W_α^1 W_α^2 + b_α)^{1/τ}
# W_α^1 ∈ ℝ^{d×16}, W_α^2 ∈ ℝ^{16×d_k}
# τ = 16 (温度项,鼓励慢遗忘)
```

### 3.2 并行形式 (数值稳定版本)

**朴素并行展开** (会数值爆炸):
```
S_t = Σ_{i=1}^t [(Π_{j=i+1}^t α_j^T 1) ⊙ k_i^T v_i]

令 b_t = Π_{j=1}^t α_j (累积门控)
o_t = Σ_{i=1}^t (q_t ⊙ b_t) (k_i / b_i)^T v_i
```
**问题**: b_t → 0 exponentially,导致 k_i/b_i → ∞

**对数空间稳定计算**:
```
P_{ij} = Σ_k Q_{ik} K_{jk} exp(log B_{ik} - log B_{jk})    (i ≥ j)
```
- **B**: 累积门控矩阵 (stacking b_t)
- **log B**: 在对数空间累加,避免underflow

### 3.3 Chunkwise形式 (GLA完整版本)

**Inter-chunk状态传递**:
```
Λ_{iC+j} = b_{iC+j} / b_{iC}       (chunk起始到当前的累积衰减)
Γ_{iC+j} = b_{(i+1)C} / b_{iC+j}   (当前到chunk结束的累积衰减)
γ_{i+1}  = b_{(i+1)C} / b_{iC}     (整个chunk的累积衰减)

S_{[i+1]} = (γ_{i+1}^T 1) ⊙ S_{[i]} + (K_{[i+1]} ⊙ Γ_{[i+1]})^T V_{[i+1]}
O_{[i+1]}^{inter} = (Q_{[i+1]} ⊙ Λ_{[i+1]}) S_{[i]}
```

**Intra-chunk计算**:
- 使用对数空间公式 (Eq. 4)
- 二级chunking优化 (见下节)

---

## 四、FlashLinearAttention硬件优化

### 4.1 硬件考虑因素

**三大原则**:
1. **Occupancy**: 充分利用GPU SM (Streaming Multiprocessors)
2. **Tensor Cores**: 利用专用矩阵乘法单元 (半精度matmul快16×)
3. **Memory Hierarchy**: 优化SRAM/HBM访问,减少I/O

**Linear Attention的硬件挑战**:
- **循环形式**: 元素级操作无法用tensor cores,低arithmetic intensity
- **并行形式**: O(L²d) FLOPs过高
- **Chunkwise形式**: 现有实现不I/O-aware

### 4.2 FlashLinearAttention算法

**两个版本**:

1. **Non-Materialization版本** (内存优化):
   - 顺序处理chunk,S_{[n]}只保存在SRAM
   - 并行维度: batch size × num_heads × head_dim
   - **适用场景**: 大batch size场景

2. **Materialization版本** (并行优化):
   - 第一遍: 计算并保存所有S_{[n]}到HBM
   - 第二遍: 并行计算所有O_{[n]}
   - **Recomputation**: backward时重算S_{[n]},节省10-20%内存
   - **适用场景**: 小batch size长序列 (如大模型训练)

**关键优化**:
- **Tiling**: 分块加载Q/K/V,在SRAM上复用
- **Fusion**: Q_{[n]}加载一次后计算Q S + (QK^T⊙M)V,避免重复I/O
- **On-chip mask**: 因果掩码M在SRAM预先构造

### 4.3 GLA特有优化: 二级Chunking

**问题**: Intra-chunk的对数空间计算 (Eq. 4) 无法使用半精度matmul

**解决方案**: Sub-chunk tiling
```
Chunk → Sub-chunks (secondary tiling)

Inter-sub-chunk: 半精度matmul (利用tensor cores)
P_{[i][j]} = (Q_{[i]} ⊙ Λ_{[i]}) (K_{[j]} ⊙ Γ_{[j]} ⊙ b_{iC}/b_{(j+1)C})^T

Intra-sub-chunk: 全精度对数空间计算 (Eq. 4)
```

**效果**:
- 大部分计算 (橙色tile) 使用tensor cores
- 只有diagonal附近的小块 (粉色tile) 用全精度
- 显著提升wall-clock速度

### 4.4 内存高效的dα_t计算

**问题**: 之前工作认为需要保存所有L个S_t才能计算梯度

**GLA闭式解**:
```
d log b_t = q_t ⊙ dq_t - k_t ⊙ dk_t
d log α_t = Σ_{i=t}^L d log b_i
```
- 只需要dq_t和dk_t,无需保存所有隐状态
- 大幅降低backward pass的内存占用

---

## 五、GLA Transformer架构

### 5.1 Multi-Head GLA Layer

**单头更新** (h ∈ [1,H]):
```python
S_t^h = (α_t^h)^T 1 ⊙ S_{t-1}^h + k_t^{h T} v_t^h  ∈ ℝ^{d'_k × d'_v}
o_t^h = q_t^h S_t^h                                  ∈ ℝ^{1 × d'_v}

# d'_k = d_k/H, d'_v = d_v/H (per-head维度)
```

**输出合成**:
```python
o_t' = concat(LN(o_t^1), ..., LN(o_t^H))  ∈ ℝ^{1 × d_v}
r_t = Swish(x_t W_r + b_r)                 ∈ ℝ^{1 × d_v}  (输出门)
y_t = (r_t ⊙ o_t') W_O                     ∈ ℝ^{1 × d}
```

**关键设计**:
- 每个头后接LayerNorm (类似RetNet)
- 输出门r_t控制信息流
- d_k = d/2, d_v = d (默认配置)

### 5.2 完整Block结构

```python
# Layer l的前向传播
Y^{(l)} = GLA(LN(X^{(l)})) + X^{(l)}           (GLA sublayer)
X^{(l+1)} = SwiGLU(LN(Y^{(l)})) + Y^{(l)}      (FFN sublayer)

# SwiGLU FFN
SwiGLU(Z) = (Swish(ZW_1) ⊙ ZW_2) W_3
```

**参数分配** (≈4d² per layer,与标准Attention相同):
- W_Q, W_K, W_V, W_O, W_r: 全秩 ∈ ℝ^{d×d}
- W_α: 低秩 W_α^1 (d×16) + W_α^2 (16×d_k)

### 5.3 配置参数详解

```python
class GLAConfig:
    hidden_size: int = 2048           # 隐藏维度
    expand_k: float = 0.5             # key维度扩展比例 (d_k = d×0.5)
    expand_v: float = 1.0             # value维度扩展比例 (d_v = d×1.0)
    num_heads: int = 4                # 注意力头数
    num_kv_heads: int | None = None   # MQA/GQA支持

    # 门控机制
    use_gk: bool = True               # 启用key门控 (标准GLA)
    use_gv: bool = False              # 启用value门控 (可选)
    clamp_min: float | None = None    # 门控下界 (防止过度遗忘)

    # 卷积增强 (可选)
    use_short_conv: bool = False      # 启用短卷积
    conv_size: int = 4                # 卷积核大小

    # 训练模式
    attn_mode: str = "chunk"          # chunk | fused_recurrent | fused_chunk
    fuse_norm: bool = True            # 融合RMSNorm (内存优化)
    fuse_swiglu: bool = True          # 融合SwiGLU (速度优化)
```

---

## 六、代码实现核心逻辑

### 6.1 GatedLinearAttention Layer

**文件**: `fla/layers/gla.py`

```python
class GatedLinearAttention(nn.Module):
    def __init__(self, mode='chunk', hidden_size=1024,
                 expand_k=0.5, expand_v=1.0, num_heads=4, ...):
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # 投影层
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)

        # 门控投影 (低秩)
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),  # 16维瓶颈
            nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True)
        )

        # 输出门 (可选)
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # 短卷积 (可选)
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size)
            self.k_conv1d = ShortConvolution(self.key_dim_per_group, kernel_size=conv_size)
            self.v_conv1d = ShortConvolution(self.value_dim_per_group, kernel_size=conv_size)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, ...):
        # 1. 投影
        q = self.q_proj(hidden_states)  # [B, L, key_dim]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)

        # 2. 可选短卷积
        if self.use_short_conv:
            q, _ = self.q_conv1d(q, cache=conv_state_q, ...)
            k, _ = self.k_conv1d(k, cache=conv_state_k, ...)
            v, _ = self.v_conv1d(v, cache=conv_state_v, ...)

        # 3. Reshape到multi-head形式
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        gk = rearrange(gk, '... (h d) -> ... h d', d=self.head_k_dim)

        # 4. 门控归一化 (log-sigmoid + scaling)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer  # τ=16
        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        # 5. 选择kernel模式
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        # 6. 核心计算
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(q, k, v, gk, ...)
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(q, k, v, gk, ...)
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(q, k, v, gk, ...)

        # 7. 输出门 + 归一化
        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)  # 融合版本
                o = rearrange(o, '... h d -> ... (h d)')
            else:
                o = rearrange(self.g_norm(o), '... h d -> ... (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), '... h d -> ... (h d)')

        # 8. 输出投影
        o = self.o_proj(o)

        return o, None, past_key_values
```

### 6.2 Chunk GLA Kernel (核心算子)

**文件**: `fla/ops/gla/chunk.py`

**关键函数**:
```python
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(...):
    """
    计算inter-sub-chunk的注意力矩阵块
    使用半精度matmul (tensor cores)
    """
    # 加载Q, K块
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    # 加载门控g
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))

    # 应用门控 (对数空间)
    b_qg = b_q * exp(b_g - b_gn[None, :]) * scale
    b_kg = b_k * exp(b_gn[:, None] - b_gk)

    # 矩阵乘法 (利用tensor cores)
    b_A += tl.dot(b_qg, b_kg)

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(...):
    """
    计算intra-sub-chunk的注意力矩阵
    使用全精度对数空间计算 (数值稳定)
    """
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)

        # 对数空间attention (Eq. 4)
        b_A = tl.sum(b_q * b_k[None, :] * exp(b_g - b_gk[None, :]), 1) * scale

        tl.store(A + o_A + j, b_A, mask=m_A)
```

### 6.3 Fused Recurrent GLA

**文件**: `fla/ops/gla/fused_recurrent.py`

```python
def fused_recurrent_gla(q, k, v, gk, gv=None, scale=None,
                        initial_state=None, output_final_state=False, ...):
    """
    循环形式GLA (推理或短序列)

    Args:
        q: [B, T, H, K] - queries
        k: [B, T, H, K] - keys
        v: [B, T, H, V] - values
        gk: [B, T, H, K] - key forget gates (log-sigmoid)
        gv: [B, T, H, V] - value forget gates (可选)
        initial_state: [B, H, K, V] - 初始隐状态

    Returns:
        o: [B, T, H, V] - outputs
        final_state: [B, H, K, V] - 最终隐状态
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # 调用通用fused_recurrent,传递门控参数
    o, final_state = fused_recurrent(
        q=q, k=k, v=v,
        g=None,           # 不使用scalar gate
        gk=gk,            # 使用per-dimension key gate
        gv=gv,            # 可选value gate
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        ...
    )
    return o, final_state
```

---

## 七、关键差异对比

### 7.1 GLA vs RetNet

| 维度 | RetNet | GLA |
|------|--------|-----|
| **门控类型** | 全局标量 γ ∈ (0,1) | Per-dimension向量 α_t ∈ (0,1)^{d_k} |
| **数据依赖** | 无 (位置相关) | 有 (内容相关) |
| **表达能力** | S_t = γ S_{t-1} + k_t^T v_t | S_t = Diag(α_t) S_{t-1} + k_t^T v_t |
| **并行训练** | 简单 (ALiBi风格) | 需要对数空间稳定 (Eq. 4) |
| **Recall任务** | 较弱 | 较强 |

### 7.2 GLA vs Mamba

| 维度 | Mamba | GLA |
|------|-------|-----|
| **隐状态结构** | SISO扩展 (N个独立1D状态) | 2D矩阵 (d_k × d_v) |
| **状态维度** | 扩展率≤16 (SRAM限制) | 无限制 (HBM存储) |
| **训练算法** | Parallel scan (自定义CUDA) | Chunk-based (Triton) |
| **Tensor Cores** | Mamba-2可用 (scalar gate) | 完全利用 (半精度matmul) |
| **张量并行** | 不支持 (非多头) | 原生支持 (多头架构) |
| **长序列吞吐** | 较低 | 较高 (尤其>4K) |

### 7.3 GLA vs Softmax Attention

| 维度 | Softmax Attention | GLA |
|------|-------------------|-----|
| **训练复杂度** | O(L²d) | O(LCd + Ld²) ≈ O(Ld²) when C≈√d |
| **推理复杂度** | O(L) per token (KV cache) | O(1) per token (固定状态) |
| **KV cache大小** | O(Ld) | O(d²) |
| **长度外推** | 困难 (需特殊位置编码) | 自然 (RNN特性) |
| **Recall任务** | 最强 (完整历史) | 次之 (有限状态容量) |
| **训练速度** | FlashAttention-2基线 | 更快 (1K+序列) |

---

## 八、实验结果关键发现

### 8.1 性能对比 (340M/1.3B模型)

**Language Modeling** (SlimPajama 100B tokens):
- GLA ≈ Transformer++ (LLaMA架构)
- GLA > RetNet (所有任务)
- GLA ≈ Mamba (整体)

**Recall-Intensive Tasks**:
```
MQAR (合成任务):
  GLA > RetNet > Mamba > Hyena/RWKV

FDA (信息抽取):
  Transformer++ (81.1) >> GLA (68.7) > Mamba (64.2) > RetNet (60.7)

SWDE (信息抽取):
  Transformer++ (31.8) >> GLA (20.5) > Mamba (17.2) > RetNet (15.6)
```
**结论**: GLA在subquadratic模型中recall能力最强 (得益于更大状态容量)

### 8.2 长度外推能力

**训练2K → 测试20K**:
- GLA: 平滑外推,困惑度稳定上升
- Mamba: 4K后性能急剧下降
- RetNet: 18K后才下降
- Transformer++: 完全无法外推 (即使用RoPE)

**8K直接训练 vs 24K TBPTT**:
- GLA两种方式困惑度相近 → TBPTT经济实用
- Mamba在8K训练后显著提升

### 8.3 训练效率 (1.3B模型, H100)

**Throughput (tokens/sec)**:
```
序列长度     GLA    Transformer++   Mamba
1K          ~45K      ~50K          ~40K
2K          ~30K      ~28K          ~22K
4K          ~18K      ~14K          ~12K
8K          ~10K      ~7K           ~6K
```
**结论**: GLA在4K+序列长度上吞吐量超过Transformer++和Mamba

**GPU Memory**:
- 三者内存占用相近 (均为线性复杂度)
- GLA materialization版本多10-20%,但通过recomputation补偿

---

## 九、应用指南

### 9.1 何时使用GLA

**推荐场景**:
1. **长序列建模** (>4K): 训练效率高于Mamba
2. **长度外推**: 需要2K训练→10K+推理
3. **Recall任务**: 信息抽取、问答等
4. **大模型训练** (>7B): 兼容张量并行

**不推荐场景**:
1. **超短序列** (<512): Transformer++更快
2. **极致recall**: Softmax Attention仍最优
3. **受限硬件**: Mamba对SRAM要求更低

### 9.2 超参数调优建议

**默认配置** (经论文验证):
```python
GLAConfig(
    hidden_size=2048,
    num_heads=4,              # Ablation显示4头最优
    expand_k=0.5,             # d_k = d/2
    expand_v=1.0,             # d_v = d
    attn_mode="chunk",        # 训练用chunk
    use_short_conv=False,     # 默认不用卷积
    use_output_gate=True,     # 输出门重要
    clamp_min=None,           # 无下界约束
    gate_logit_normalizer=16, # τ=16
)
```

**Ablation结果**:
- **门控细粒度**: Per-dimension > Scalar >> None
- **数据依赖**: 数据依赖 >> 位置依赖
- **头数**: 1头最优(marginal),但4头节省GPU内存

### 9.3 推理优化

**模式选择**:
```python
# 短序列 (<64 tokens): fused_recurrent
mode = 'fused_recurrent' if seq_len <= 64 else 'chunk'

# 长序列batch推理: chunk with small C
# 单样本自回归: fused_recurrent
```

**KV Cache替代**:
- GLA只需保存 S_t ∈ ℝ^{H×d_k×d_v}
- Transformer需要保存 (K, V) ∈ ℝ^{L×d}
- **节省**: ~L/d 倍 (当L>>d时显著)

---

## 十、代码文件组织

### 10.1 核心文件列表

```
fla/
├── models/gla/
│   ├── modeling_gla.py          # GLAForCausalLM, GLAModel, GLABlock
│   ├── configuration_gla.py     # GLAConfig
│   └── __init__.py
│
├── layers/
│   ├── gla.py                   # GatedLinearAttention layer
│   └── simple_gla.py            # 简化版本
│
├── ops/gla/
│   ├── chunk.py                 # Triton chunk kernels (核心!)
│   ├── fused_chunk.py           # 融合版本chunk
│   ├── fused_recurrent.py       # 循环版本
│   └── naive.py                 # PyTorch参考实现
│
└── modules/
    ├── fused_rms_norm_gate.py   # 融合RMSNorm+Gate
    └── short_conv.py            # 短卷积模块
```

### 10.2 关键类与函数

**高层API**:
- `GLAForCausalLM`: HuggingFace兼容的因果语言模型
- `GatedLinearAttention`: 可复用的GLA层
- `GLAConfig`: 配置类

**底层Kernel**:
- `chunk_gla(q, k, v, g, ...)`: Chunk-based训练
- `fused_recurrent_gla(q, k, v, gk, ...)`: 循环推理
- `fused_chunk_gla(...)`: 融合优化版本

**Triton Kernels** (chunk.py):
- `chunk_gla_fwd_A_kernel_intra_sub_inter`: Inter-sub-chunk计算
- `chunk_gla_fwd_A_kernel_intra_sub_intra`: Intra-sub-chunk计算
- `chunk_gla_bwd_dqkvg_kernel`: Backward pass梯度计算

---

## 十一、理论洞察与未来方向

### 11.1 为何GLA有效

**1. 数据依赖门控的必要性**:
- 1D RNN中forget gate已被证明关键 (LSTM/GRU)
- 2D隐状态同样需要选择性遗忘
- 实验证明: 数据依赖 >> 位置依赖 >> 无门控

**2. 细粒度 vs 粗粒度**:
- Scalar gate (Mamba-2, RWKV-v6): 简单但表达受限
- Per-dimension gate (GLA): 平衡表达与效率
- Full matrix gate: 参数过多,难以训练

**3. 硬件co-design的重要性**:
- 算法设计必须考虑硬件特性 (tensor cores, memory hierarchy)
- FlashLinearAttention是GLA成功的关键 (否则比Mamba还慢)

### 11.2 开放问题

**1. 更大规模的扩展性** (>7B):
- 论文实验限于340M/1.3B
- 更大模型是否保持优势? (理论上GLA更友好)

**2. Hybrid架构**:
- GLA + Softmax Attention混合 (如NVIDIA的Hymba)
- 不同层使用不同机制

**3. 其他模态**:
- Vision: 长序列图像处理
- Audio: 音频建模
- Multimodal: 跨模态理解

**4. 理论分析**:
- GLA的表达能力边界
- Recall能力的理论上限
- 最优状态维度选择

---

## 十二、快速参考

### 12.1 GLA核心公式

```
# 循环形式
S_t = Diag(α_t) S_{t-1} + k_t^T v_t
o_t = q_t S_t

# α_t参数化
α_t = sigmoid(x_t W_α^1 W_α^2 + b_α)^{1/16}

# Chunkwise inter-chunk
S_{[i+1]} = (γ_{i+1}^T 1) ⊙ S_{[i]} + (K_{[i+1]} ⊙ Γ_{[i+1]})^T V_{[i+1]}

# 对数空间稳定计算
P_{ij} = Σ_k Q_{ik} K_{jk} exp(log B_{ik} - log B_{jk})

# 梯度闭式解
d log α_t = Σ_{i=t}^L (q_i ⊙ dq_i - k_i ⊙ dk_i)
```

### 12.2 性能基准 (1.3B, H100)

| 序列长度 | GLA吞吐量 | vs Mamba | vs Transformer++ |
|----------|-----------|----------|------------------|
| 1K       | 45K tok/s | +12%     | -10%             |
| 2K       | 30K tok/s | +36%     | +7%              |
| 4K       | 18K tok/s | +50%     | +29%             |
| 8K       | 10K tok/s | +67%     | +43%             |

### 12.3 常用配置模板

**标准LLM**:
```python
GLAConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_hidden_layers=24,
    num_heads=4,
    expand_k=0.5,
    expand_v=1.0,
    max_position_embeddings=2048,
    attn_mode="chunk",
    use_output_gate=True,
)
```

**长序列优化**:
```python
# 训练时materialization + recomputation
attn_mode="chunk"
# 推理时自动切换
# seq_len <= 64 → fused_recurrent
# seq_len > 64 → chunk
```

---

## 参考文献

1. **GLA原论文**: Yang & Wang et al. (2024), "Gated Linear Attention Transformers with Hardware-Efficient Training", ICML 2024
2. **Linear Attention**: Katharopoulos et al. (2020), "Transformers are RNNs"
3. **RetNet**: Sun et al. (2023), "Retentive Network"
4. **Mamba**: Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
5. **FlashAttention**: Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention"

---

**文档版本**: v1.0
**最后更新**: 2025-01-XX
**维护者**: Claude Code (基于fla库和论文2312.06635v6)
