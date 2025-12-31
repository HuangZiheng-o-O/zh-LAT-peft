# GLA高级PEFT优化路线图

基于当前"投影层LoRA + gk_proj.1 SDT"的设计，本文档提出基于SMT和学术最新进展的高级优化方向。

---

## 一、当前设计的理论基础

### 当前分工

```
GLA模型
├─ 特征投影层 (Q/K/V/O/G)
│  └─ 适配方式：LoRA
│     理由：低秩特征重编码，参数效率高
│
└─ 动力学参数 (gk_proj)
   └─ 适配方式：SDT
      理由：通道级选择，改变状态衰减机制
```

### 为什么有效

```
LoRA的成功条件：
  W_new = W + ΔW，其中rank(ΔW) << rank(W)

GLA投影层满足此条件：
  • q_proj: 查询特征，任务转移通常是"低秩重加权"
  • k_proj: 键特征，同上
  • v_proj: 值特征，同上
  • o_proj: 输出投影，同上

SDT的成功条件：
  参数具有"通道/维度"的结构化语义

GLA gk_proj满足此条件：
  • gk_proj.1 → α_t 是每个通道的衰减因子
  • 不同通道应该有不同的适应策略（Train/Freeze/Zero）
  • 修改不是"权重扰动"而是"通道选择"
```

---

## 二、SMT论文给GLA的启示

### 2.1 GW-Selection优于AW-Selection

**SMT的发现**：
- 梯度加权选择（GW）：78.7% 相对性能
- 激活加权选择（AW）：53.2% 相对性能
- **结论**：梯度信息比激活信息更重要

**GLA当前实现**：
- ✅ 使用梯度累积评估重要性（warmup阶段）
- ✅ 已符合GW-Selection的最佳实践

### 2.2 全局Top-K优于per-layer固定比例

**SMT的发现**：
```
Per-layer固定：Train=40%, Freeze=50%, Zero=10%（每层相同）
  └─ 限制：无法适应不同层的特性差异

全局Top-K：根据梯度全局排序，自动分配
  └─ 优势：
    • Attention层可能需要更多Train维度
    • MLP层可能更容易被剪枝
    • 自适应最优分配
```

**当前GLA实现**：
- ⚠️ 仍使用per-layer固定比例
- 🎯 优化方向：实现全局Top-K

### 2.3 梯度度量：mean(|grad|) vs L2 vs 方差

**SMT的发现**：
```
重要性度量对比：
  L2范数：    Σ sqrt(grad²)  ← Frobenius norm
  mean_abs：  mean(|grad|)  ← SMT推荐 ✓
  max_abs：   max(|grad|)   ← 异常值敏感
  variance：  E[grad²] - E[grad]²  ← 不稳定
```

**当前GLA实现**：
- 使用L2范数（沿梯度轴求和）
- 🎯 待验证：mean(|grad|)是否更优

### 2.4 V向量优先级

**SMT对Transformer的发现**：
```
层内梯度分布（Attention块）：
  Q向量梯度：100%  (基线)
  K向量梯度：100%
  V向量梯度：500-1000%  ← 最大！
```

**含义**：
- V向量变化最快，最容易被过拟合
- 但同时也说明V对任务适配最关键

**当前GLA实现**：
- v_proj用LoRA秩r=8
- 🎯 优化：考虑r=12或r=16

---

## 三、分阶段优化路线

### Phase 1：验证基础假设（第1-2周）

#### 任务：验证gk_proj确实是"动力学参数"

```python
# 实验A：梯度方向分析
def verify_lora_vs_sdt_difference():
    """
    验证投影层梯度具有低秩性，而gk_proj梯度具有通道选择性。
    """
    # 在warmup阶段，对所有参数计算梯度

    # 对于投影层（q/k/v/o/g）：
    for layer in model.layers:
        q_grad = layer.q_proj.weight.grad  # [d_k, d_h]
        U, S, Vt = torch.linalg.svd(q_grad, full_matrices=False)

        # 计算有效秩
        threshold = 0.01 * S[0]
        effective_rank = (S > threshold).sum().item()

        print(f"q_proj effective rank: {effective_rank} / {min(q_grad.shape)}")
        # 预期：effective_rank << min(q_grad.shape)  → 低秩 ✓

    # 对于gk_proj.1：
    for layer in model.layers:
        gk1_grad = layer.gk_proj[1].weight.grad  # [d_k, 16]
        row_importance = gk1_grad.abs().sum(dim=1)  # [d_k]

        # 检查行重要性的分化程度
        row_importance_normalized = row_importance / row_importance.sum()
        entropy = -torch.sum(row_importance_normalized *
                             torch.log(row_importance_normalized + 1e-8))

        # 计算Gini系数（不等性指标）
        gini = compute_gini(row_importance)

        print(f"gk_proj.1 row Gini: {gini:.3f}")  # 预期：高Gini → 通道分化 ✓
```

#### 代码位置
- `mamba_ssm_peft/peft/gla_sd_lora.py`：添加调试输出
- `scripts/debug/analyze_gradients.py`：新建分析脚本

### Phase 2：梯度度量优化（第2-3周）

#### 任务：比较不同的重要性度量

```python
# 当前实现（warmup阶段）
importance = gradients.abs().sum(dim=-1)  # L2范数沿最后维累加

# 优化选项
class ImportanceMetric:
    @staticmethod
    def l2_norm(grad):
        """当前方式"""
        return (grad ** 2).sum(dim=-1) ** 0.5

    @staticmethod
    def mean_abs(grad):
        """SMT推荐"""
        return grad.abs().mean(dim=-1)

    @staticmethod
    def max_abs(grad):
        """最大值（可能敏感）"""
        return grad.abs().max(dim=-1)[0]

    @staticmethod
    def percentile_90(grad):
        """90分位数（鲁棒）"""
        return torch.quantile(grad.abs(), 0.9, dim=-1)
```

#### 实验设计
```bash
# 对比四种度量方式在GLUE任务上的性能
for metric in l2_norm, mean_abs, max_abs, percentile_90:
    HP_GRADIENT_METRIC=$metric python train_lat.py --cfg configs/gla_sdlora.yaml
    # 记录 Train loss, Val accuracy, 参数量
```

#### 代码位置
- `mamba_ssm_peft/peft/gla_sd_lora.py`：`_compute_importance()`方法

### Phase 3：全局Top-K实现（第3-4周）

#### 当前方式（per-layer固定）
```python
# 当前实现
for layer in model.layers:
    # 每层都独立应用：Train 40%, Freeze 50%, Zero 10%
    layer.gk_proj[1].importance_rank = sort(importance)
    layer.gk_proj[1].train_mask = importance_rank[:0.4*d_k]
    layer.gk_proj[1].freeze_mask = importance_rank[0.4*d_k:0.9*d_k]
    layer.gk_proj[1].zero_mask = importance_rank[0.9*d_k:]
```

#### 优化方式（全局Top-K）
```python
# 优化实现
all_importances = []
for layer in model.layers:
    all_importances.append(layer.gk_proj[1].importance)

all_importances_flat = torch.cat(all_importances)  # [num_layers * d_k]
global_rank = torch.argsort(all_importances_flat, descending=True)

# 全局分配
total_dim = all_importances_flat.shape[0]
num_train = int(total_dim * 0.4)
num_freeze = int(total_dim * 0.5)

for layer_idx, layer in enumerate(model.layers):
    offset = layer_idx * layer.gk_proj[1].weight.shape[0]

    for ch_idx in range(layer.gk_proj[1].weight.shape[0]):
        global_idx = offset + ch_idx

        if global_idx in global_rank[:num_train]:
            layer.train_mask[ch_idx] = True
        elif global_idx in global_rank[num_train:num_train+num_freeze]:
            layer.freeze_mask[ch_idx] = True
        else:
            layer.zero_mask[ch_idx] = True
```

#### 预期改进
- **Before**: 所有层都是40/50/10
- **After**: 根据梯度自动调整（可能成为70/20/10, 30/60/10等）

#### 代码位置
- `mamba_ssm_peft/peft/gla_sd_lora.py`：新增`GlobalTopKSelector`类
- `mamba_ssm_peft/peft/sd_lora.py`：参考已有的全局选择实现

### Phase 4：V向量优先级调整（第4周）

#### 当前配置
```json
{
    "lora_targets": [
        "q_proj",    // r=8
        "k_proj",    // r=8
        "v_proj",    // r=8  ← 应该更大
        "o_proj"     // r=8
    ]
}
```

#### 优化方案
```python
# 方案：根据梯度给不同投影层分配不同秩
def get_adaptive_lora_config(model):
    """
    根据warmup梯度，为每层分配最适的LoRA秩。
    """
    config = {}

    for name, param in model.named_parameters():
        if 'proj.weight' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()

            # 梯度大 → 需要更多表达能力 → 更大的秩
            if 'v_proj' in name or 'g_proj' in name:
                rank = 12  # V/G向量梯度通常最大
            elif 'q_proj' in name or 'k_proj' in name:
                rank = 8   # Q/K梯度中等
            else:  # o_proj
                rank = 8

            config[name] = rank

    return config
```

#### 简单方案（立即可实施）
```json
{
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj"],
    "lora_rank_default": 8,
    "lora_rank_for_v": 12    // 新增：为v_proj单独配置更大秩
}
```

---

## 四、gk_proj.0的处理

### 当前：gk_proj.0被冻结

```python
# gk_proj = Sequential(
#     Linear(hidden_size, 16),     # .0 - 当前被冻结
#     Linear(16, key_dim_per_group) # .1 - SDT目标
# )
```

### 三个选择

#### 选项A：保持冻结（当前方案）
```python
# gk_proj.0保持不变，只对.1做SDT
target_modules = ["gk_proj.1"]

优点：简单，避免过度适配
缺点：.0学到的"16维压缩表示"无法适应新任务
```

#### 选项B：对gk_proj.0也用LoRA
```python
# gk_proj.0也用LoRA（与q/k/v同级）
lora_targets = ["q_proj", "k_proj", "v_proj", "gk_proj.0", "o_proj"]
target_modules = ["gk_proj.1"]  # 仍然SDT

逻辑：
  • gk_proj.0是特征压缩（h→16），可用LoRA
  • gk_proj.1是通道衰减（16→d_k），必须SDT

优点：更灵活的特征选择
缺点：参数增加，可能过拟合
```

#### 选项C：gk_proj整体用LoRA（不做SDT）
```python
# 整个gk_proj用LoRA，不做SDT
lora_targets = ["q_proj", "k_proj", "v_proj", "gk_proj", "o_proj"]
target_modules = []  # 不用SDT

缺点：不符合"动力学参数用SDT"的原则 ✗
```

### 建议
- 🎯 **推荐选项B**：gk_proj.0用LoRA，gk_proj.1用SDT
- 实施难度：中等
- 预期改进：+2-3%

---

## 五、高级实验设计

### 5.1 完整的消融实验矩阵

```
实验名 | LoRA目标 | SDT目标 | v秩 | 全局Top-K | 重要性度量 | 参数% | 预期性能
-------|---------|--------|-----|---------|-----------|-------|--------
Base   | -       | -      | -   | -       | -         | 100%  | 100%
A1     | Q/K/V/O | -      | 8   | ✗       | -         | 5%    | 85-92%
A2     | Q/K/V/O | gk.1   | 8   | ✗       | L2norm    | 8%    | 90-97%
A3     | Q/K/V/O | gk.1   | 12  | ✗       | L2norm    | 8%    | 91-97%
A4     | Q/K/V/O | gk.1   | 12  | ✓       | L2norm    | 8%    | 92-98% ← 目标
A5     | ...+G   | gk.1   | 12  | ✓       | mean_abs  | 9%    | 92-98%
```

### 5.2 多任务评估

```bash
# 在多个GLUE任务上评估
for task in cola mrpc qnli qqp rte sst2 stsb wnli; do
    # 配置：LoRA + SDT (全局Top-K, mean_abs)
    HP_PEFT_TYPE=sdlora \
        HP_TRAIN_RATIO=0.4 \
        python train_lat.py --cfg configs/gla_sdlora.yaml --task $task
done

# 记录：
# - Train loss curve
# - Validation accuracy
# - Inference latency
# - Peak memory usage
```

### 5.3 超参数搜索

```python
# 搜索空间
search_space = {
    'train_ratio': [0.3, 0.4, 0.5, 0.6],      # Train维度比例
    'zero_ratio': [0.05, 0.1, 0.15, 0.2],    # Zero维度比例
    'lora_rank': [4, 8, 12, 16],              # LoRA秩
    'warmup_it': [50, 100, 200, 500],        # Warmup轮数
}

# 在验证集上grid search，找最优超参
best_config = grid_search(search_space, val_metric='accuracy')
```

---

## 六、长期研究方向

### 6.1 通道与块的混合策略

**启发来源**：SMT的块选择 + SD-LoRA的维度选择

```
当前方式（维度）：
  Train[0:102], Freeze[102:179], Zero[179:256]

混合方式（块+维度）：
  将256维分成16个16维的块
  → 先按块选择（高效）
  → 再按维度微调（精细）
```

### 6.2 动态比例调整

**概念**：根据任务复杂度自动调整Train/Freeze/Zero比例

```python
def adaptive_ratio(task_difficulty):
    """
    简单任务（如CoLA）：更激进的剪枝
      Train: 30%, Freeze: 60%, Zero: 10%

    复杂任务（如QA）：保留更多维度
      Train: 50%, Freeze: 40%, Zero: 10%
    """
    ...
```

### 6.3 多模型对比

当前仅支持GLA，未来扩展到：
- **RetNet**：分析其衰减率参数
- **Mamba**：对比A_log和gk_proj的差异
- **Transformer**：验证LoRA在纯注意力中的表现

---

## 七、实施时间表

```
第1-2周：验证基础假设
  □ 实现梯度分析脚本
  □ 确认低秩vs通道选择的差异
  □ 文档记录

第2-3周：梯度度量优化
  □ 实现四种度量方式
  □ 消融实验对比
  □ 选择最优度量

第3-4周：全局Top-K
  □ 实现GlobalTopKSelector
  □ 验证性能改进
  □ 参数搜索

第4周：V向量秩调整
  □ 调整配置
  □ 快速验证

第5周：综合优化版本
  □ 集成所有改进
  □ 多任务评估
  □ 文档总结
```

---

## 八、评估指标

| 指标 | 当前方案 | 目标 |
|------|---------|------|
| **参数% (相对base)** | 8% | 8% (保持) |
| **GLUE avg 性能** | 92% | 95%+ |
| **最坏任务性能** | 85% | 90%+ |
| **推理速度** | 95% baseline | 98%+ baseline |
| **内存占用** | -5% | -10%+ |

---

## 参考代码结构

```
优化工作的代码应该放在：

mamba_ssm_peft/peft/
├── gla_sd_lora.py           (现有，主要修改处)
│   ├── GlaSdLoraConfig      (现有)
│   ├── GlaSdLoraModel       (现有)
│   └── GlaSdLoraParameter   (现有)
│       ├── _compute_importance()    ← Phase 2修改（梯度度量）
│       ├── _select_important_dims() ← Phase 3修改（全局Top-K）
│       └── forward()                ← 现有，无需改动
│
└── selectors/               ← Phase 3新增
    └── global_topk_selector.py
        └── GlobalTopKSelector class
```

---

## 总结

当前GLA PEFT的设计已经是**理论上合理、实施上可行**的方案。未来的优化主要沿着三个方向：

1. **梯度度量优化**：找到比L2范数更好的重要性指标
2. **全局优化**：从per-layer固定比例→全局动态分配
3. **精细调参**：根据不同投影层的特性调整LoRA秩

这些优化预期可以将性能从92%提升到95%+，同时保持参数预算不变。
