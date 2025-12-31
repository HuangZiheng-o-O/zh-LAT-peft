# GLA SD-LoRA 改进建议与实施路线图

> **基于**: SMT论文和代码分析
> **日期**: 2025-12-31
> **目标**: 为GLA SD-LoRA项目提供可执行的改进方案

---

## 1. 当前实现状态回顾

### 1.1 默认配置 (已调整)

```
当前SD-LoRA默认配置:
┌─────────────────────────────────────────────────────────────────┐
│  参数                │  值    │  说明                          │
│  ────────────────────┼────────┼──────────────────────────────│
│  Train Ratio        │  40%   │  正常梯度更新                  │
│  Freeze Ratio       │  50%   │  保持原值                      │
│  Zero Ratio         │  10%   │  设为-100 (遗忘)              │
│  Warmup Iterations  │  100   │  梯度收集阶段                  │
│  Block Size         │  N/A   │  Channel级别，非块级别        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 目标模块

```python
# 当前SD-LoRA的目标模块 (GLA)
target_modules = ["gk_proj.1"]        # GLA的门控投影第二层
lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

# SMT发现的高重要性目标
# - V向量: 占选中参数的95.17%
# - Attention > MLP
```

---

## 2. 高优先级改进

### 2.1 引入全局Top-K选择

**问题**: 当前每层使用相同的Train/Freeze/Zero比例

**改进方案**: 实现全局channel排序，让重要层获得更多可训练维度

```python
# mamba_ssm_peft/peft/gla_sd_lora.py

def _select_channels_global(self, num_zero: Dict, num_freeze: Dict):
    """
    全局channel选择，替代per-layer固定比例

    原方法: 每层独立选择Train/Freeze/Zero channels
    新方法: 所有layers的channels一起排序，全局分配
    """
    all_channel_importance = []  # (importance, module_key, channel_idx)

    # 收集所有channel的梯度信息
    for module_key, param in self.named_parameters():
        if hasattr(param, 'sdlora_grad'):
            grad = param.sdlora_grad  # [out_features, in_features]

            # 计算每个channel的重要性 (使用L2 norm)
            if grad.dim() == 2:
                channel_importance = grad.norm(dim=0)  # 每列的L2 norm
            else:
                channel_importance = grad.abs()

            for ch_idx, imp in enumerate(channel_importance):
                all_channel_importance.append((imp.item(), module_key, ch_idx))

    # 全局排序
    all_channel_importance.sort(reverse=True, key=lambda x: x[0])

    # 计算全局阈值
    total_channels = len(all_channel_importance)
    zero_threshold = all_channel_importance[int(total_channels * self.zero_ratio)][0]
    freeze_threshold = all_channel_importance[int(total_channels * (self.zero_ratio + self.freeze_ratio))][0]

    # 分配到三类
    zero_channels = {}    # {module_key: [ch_idx, ...]}
    freeze_channels = {}  # {module_key: [ch_idx, ...]}
    train_channels = {}   # {module_key: [ch_idx, ...]}

    for imp, module_key, ch_idx in all_channel_importance:
        if imp <= zero_threshold:
            zero_channels.setdefault(module_key, []).append(ch_idx)
        elif imp <= freeze_threshold:
            freeze_channels.setdefault(module_key, []).append(ch_idx)
        else:
            train_channels.setdefault(module_key, []).append(ch_idx)

    return train_channels, freeze_channels, zero_channels


# 或者使用Min-Heap优化 (SMT方法)
import heapq

def _select_channels_global_efficient(self, num_zero: Dict, num_freeze: Dict):
    """
    使用min-heap高效实现全局Top-K选择
    时间复杂度: O(N log K) vs O(N log N) for sorting
    """
    total_channels = self._count_total_channels()
    n_zero = int(total_channels * self.zero_ratio)
    n_freeze = int(total_channels * self.freeze_ratio)
    n_train = total_channels - n_zero - n_freeze

    # 使用三个heap维护Top-K
    train_heap = []  # min-heap, size = n_train
    freeze_heap = []  # min-heap, size = n_freeze
    zero_heap = []    # min-heap, size = n_zero

    for module_key, param in self.named_parameters():
        if hasattr(param, 'sdlora_grad'):
            grad = param.sdlora_grad
            channel_importance = grad.norm(dim=0) if grad.dim() == 2 else grad.abs()

            for ch_idx, imp in enumerate(channel_importance):
                imp_val = imp.item()

                # 先尝试放入train_heap
                if len(train_heap) < n_train:
                    heapq.heappush(train_heap, (imp_val, module_key, ch_idx))
                elif imp_val > train_heap[0][0]:
                    popped = heapq.heapreplace(train_heap, (imp_val, module_key, ch_idx))
                    # 被挤出的可能进入freeze_heap
                    if len(freeze_heap) < n_freeze:
                        heapq.heappush(freeze_heap, popped)
                    elif popped[0] > freeze_heap[0][0]:
                        popped2 = heapq.heapreplace(freeze_heap, popped)
                        # 被挤出的可能进入zero_heap
                        if len(zero_heap) < n_zero:
                            heapq.heappush(zero_heap, popped2)
                        # else: 丢弃

    return self._heaps_to_dict(train_heap, freeze_heap, zero_heap)
```

**预期效果**:
- 重要层(如后期attention层)获得更多Train维度
- 不重要层可能大部分被Freeze或Zero
- 类似SMT中95%参数自动分配给V向量的效果

### 2.2 V向量优先级

**SMT发现**: V向量的梯度是Q/K的5-10倍，应优先分配可训练参数

```python
# 针对GLA的特殊处理: 检测类似V的模块

def _detect_value_like_modules(self):
    """
    识别GLA中类似V向量的关键模块

    在Transformer中，V向量直接参与输出计算，梯度最大
    在GLA中，gk_proj控制状态衰减，可能具有类似特性
    """
    value_modules = []

    for name, module in self.model.named_modules():
        # GLA的gk_proj.1是门控投影，类似V的作用
        if 'gk_proj' in name and '.1' in name:
            value_modules.append(name)
        # 标准的v_proj
        elif 'v_proj' in name:
            value_modules.append(name)

    return value_modules


def _apply_value_boost(self, train_channels, value_modules=None, boost_factor=1.5):
    """
    为V类模块分配额外的Train维度

    Args:
        train_channels: 原始分配的train channels
        value_modules: V类模块列表
        boost_factor: 额外分配倍数
    """
    if value_modules is None:
        value_modules = self._detect_value_like_modules()

    boosted_channels = {}
    for module_key, channels in train_channels.items():
        if any(vm in module_key for vm in value_modules):
            # V类模块: 分配更多Train维度
            n_extra = int(len(channels) * (boost_factor - 1))
            # 从Freeze中借用维度
            if module_key in self.freeze_channels:
                borrowed = self.freeze_channels[module_key][:n_extra]
                boosted_channels[module_key] = channels + borrowed
                self.freeze_channels[module_key] = self.freeze_channels[module_key][n_extra:]
            else:
                boosted_channels[module_key] = channels
        else:
            boosted_channels[module_key] = channels

    return boosted_channels
```

### 2.3 动态比例调整

**问题**: 固定40/50/10可能不是所有任务的最优值

**改进方案**: 基于梯度分布动态调整比例

```python
def _compute_adaptive_ratios(self, gradient_stats):
    """
    基于梯度分布自适应调整Train/Freeze/Zero比例

    思路: 如果梯度分布很尖锐，增加Zero比例；如果平坦，减少Zero比例
    """
    import torch

    # 收集所有梯度值
    all_grads = []
    for grad_dict in gradient_stats.values():
        for grad in grad_dict.values():
            all_grads.extend(grad.flatten().tolist())

    all_grads = torch.tensor(all_grads)

    # 计算梯度分布统计
    grad_mean = all_grads.mean()
    grad_std = all_grads.std()
    grad_min = all_grads.min()
    grad_max = all_grads.max()

    # 计算变异系数 (CV = std/mean)
    cv = grad_std / (grad_mean + 1e-8)

    # 根据CV调整比例
    if cv > 2.0:  # 梯度分布很尖锐
        # 增加Zero比例，更激进地剪枝
        zero_ratio = 0.2
        freeze_ratio = 0.3
        train_ratio = 0.5
    elif cv > 1.0:
        # 默认配置
        zero_ratio = 0.1
        freeze_ratio = 0.5
        train_ratio = 0.4
    else:  # 梯度分布平坦
        # 减少Zero比例，更多维度参与训练
        zero_ratio = 0.05
        freeze_ratio = 0.45
        train_ratio = 0.5

    return {
        'zero_ratio': zero_ratio,
        'freeze_ratio': freeze_ratio,
        'train_ratio': train_ratio,
        'cv': cv
    }
```

---

## 3. 中优先级改进

### 3.1 部分反向传播优化

**SMT优势**: 只计算选中参数的梯度

**SD-LoRA实现**:

```python
# mamba_peft/peft/gla_sd_lora.py

import torch
from torch import nn

class GlaSdLoraParameter(nn.Parameter):
    """
    支持部分反向传播的参数包装器

    仅对Train维度计算梯度，Freeze/Zero维度跳过
    """

    def __new__(cls, data, requires_grad=True, train_mask=None):
        param = super().__new__(cls, data, requires_grad=requires_grad)
        param.train_mask = train_mask  # [d_out] boolean tensor
        return param

    @classmethod
    def from_parameter(cls, param, train_channels):
        """从普通Parameter转换"""
        train_mask = torch.zeros(param.shape[0], dtype=torch.bool, device=param.device)
        train_mask[train_channels] = True
        return cls(param.data, requires_grad=True, train_mask=train_mask)


class ChannelSparseLinearFunction(torch.autograd.Function):
    """
    支持channel级稀疏的Linear层autograd函数

    参考: SMT的linearZ实现
    """

    @staticmethod
    def forward(ctx, input, weight, bias, train_mask):
        """
        Args:
            input: [batch, seq_len, in_features]
            weight: [out_features, in_features]
            train_mask: [out_features] boolean, True=该行需要梯度
        """
        ctx.save_for_backward(weight, train_mask)
        # 仅保存需要的列 (对应train mask的行，但在输入是列)
        # 这里简化处理，保存完整输入
        ctx.input_shape = input.shape

        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, train_mask = ctx.saved_tensors

        # 输入梯度需要完整计算 (上游需要)
        grad_input = torch.matmul(grad_output, weight)

        # 权重梯度: 仅计算train_mask=True的行
        grad_weight = torch.zeros_like(weight)

        if train_mask.any():
            # ∂L/∂W = (∂L/∂Z)^T @ X
            # 仅计算选中行
            train_rows = train_mask.nonzero().squeeze()
            grad_weight[train_rows, :] = torch.matmul(
                grad_output.permute(0, 2, 1)[:, train_rows, :],
                ctx.saved_tensors[0]  # 需要保存input
            ).sum(dim=0)

        return grad_input, grad_weight, None, None


# 在GlaSdLoraParameter中使用
def forward(self, x):
    if not self.training:
        # 推理模式: 直接使用原始权重 (已应用mask)
        return torch.matmul(x, self.base_layer.weight.t()) + self.base_layer.bias

    # 训练模式: 使用稀疏反向传播
    return ChannelSparseLinearFunction.apply(
        x,
        self.modified_weight,  # 已应用Zero mask
        self.base_layer.bias,
        self.train_mask
    )
```

### 3.2 Warmup迭代次数优化

**SMT发现**: 不同数据集需要不同的warmup次数

```python
# mamba_peft/lat_adapter.py

def _auto_tune_warmup_iterations(self, dataset_size, model_size):
    """
    自动调整warmup迭代次数

    SMT实验发现:
    - Commonsense (170k samples): 100 iterations
    - Math10K (较小): 25 iterations
    """
    # 基于数据集大小
    if dataset_size > 100000:
        base_iters = 100
    elif dataset_size > 50000:
        base_iters = 75
    elif dataset_size > 10000:
        base_iters = 50
    else:
        base_iters = 25

    # 基于模型大小调整 (大模型需要更多warmup)
    model_size_multiplier = {
        'gla-1.3b': 0.8,
        'gla-3b': 1.0,
        'gla-7b': 1.2,
    }

    model_name = self.config.get('model_name', 'gla-1.3b')
    for key, mult in model_size_multiplier.items():
        if key in model_name:
            base_iters = int(base_iters * mult)
            break

    return base_iters
```

### 3.3 梯度计算策略对比

**SMT实验**: `mean().abs()` 是最佳策略

```python
# 检查当前SD-LoRA使用的梯度计算策略

def _compute_channel_importance(self, grad, strategy='mean_abs'):
    """
    计算每个channel的重要性

    Args:
        grad: [out_features, in_features] 梯度张量
        strategy: 计算策略
    """
    if strategy == 'mean_abs':
        # SMT最佳策略
        return grad.mean(dim=(1,)).abs() if grad.dim() == 2 else grad.abs()

    elif strategy == 'abs_mean':
        return grad.abs().mean(dim=(1,)) if grad.dim() == 2 else grad.abs()

    elif strategy == 'l1_norm':
        return grad.abs().sum(dim=(1,)) if grad.dim() == 2 else grad.abs()

    elif strategy == 'l2_norm':
        return grad.norm(dim=1) if grad.dim() == 2 else grad.abs()

    elif strategy == 'max_abs':
        return grad.abs().max(dim=(1,))[0] if grad.dim() == 2 else grad.abs()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# 支持环境变量配置
# HP_IMPORTANCE_STRATEGY=mean_abs python train_lat.py ...
```

---

## 4. 实验验证方案

### 4.1 消光实验矩阵

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        消光实验设计                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  实验组                    │  变量                              │  对照  │
│  ──────────────────────────┼────────────────────────────────────┼────────│
│  baseline                 │  当前配置 (40/50/10, per-layer)    │   -    │
│  ──────────────────────────┼────────────────────────────────────┼────────│
│  global_selection         │  全局Top-K选择                      │ baseline│
│  ──────────────────────────┼────────────────────────────────────┼────────│
│  value_boost              │  V/gk_proj维度×1.5                   │ baseline│
│  ──────────────────────────┼────────────────────────────────────┼────────│
│  adaptive_ratios          │  基于CV动态调整比例                  │ baseline│
│  ──────────────────────────┼────────────────────────────────────┼────────│
│  partial_backward         │  部分反向传播                        │ baseline│
│  ──────────────────────────┼────────────────────────────────────┼────────│
│  combined                 │  global + value + adaptive          │ baseline│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 评估指标

```python
# 评估指标
metrics = {
    # 性能指标
    'accuracy': '任务准确率',
    'perplexity': '困惑度',

    # 效率指标
    'trainable_params': '可训练参数数量',
    'gpu_memory': '峰值显存使用',
    'training_time': '训练时间',

    # SD-LoRA特有
    'train_distribution': 'Train维度在各层分布',
    'zero_ratio_actual': '实际Zero比例',
    'gradient_sparsity': '梯度稀疏度',
}

# 可视化
plots = [
    'layer_train_distribution',  # 各层Train维度数量
    'gradient_importance_curve',  # 梯度重要性曲线
    'memory_over_time',           # 显存随训练变化
]
```

---

## 5. 实施路线图

### Phase 1: 快速验证 (Week 1-2)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: 快速验证                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1.1 全局Top-K选择                                                         │
│      实现全局channel排序                                                   │
│      在小数据集上验证效果                                                  │
│                                                                             │
│  1.2 V向量优先级                                                           │
│      为gk_proj分配更多Train维度                                            │
│      与baseline对比                                                         │
│                                                                             │
│  1.3 梯度计算策略                                                          │
│      对比mean_abs vs abs_mean vs l2_norm                                   │
│      验证SMT的结论是否适用于GLA                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: 优化实现 (Week 3-4)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 2: 优化实现                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  2.1 部分反向传播                                                           │
│      实现ChannelSparseLinearFunction                                       │
│      测量显存和速度提升                                                    │
│                                                                             │
│  2.2 动态比例调整                                                           │
│      实现基于CV的自适应比例                                                │
│      消光实验验证最优阈值                                                  │
│                                                                             │
│  2.3 Warmup优化                                                             │
│      自适应warmup迭代次数                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 3: 集成测试 (Week 5-6)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 3: 集成测试                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  3.1 组合实验                                                              │
│      global + value + partial_backward                                    │
│                                                                             │
│  3.2 多任务验证                                                             │
│      在多个数据集上验证                                                    │
│                                                                             │
│  3.3 与SMT对比                                                             │
│      相同参数量下的性能对比                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 代码修改清单

### 6.1 核心文件修改

```
需要修改的文件:
├── mamba_peft/mamba_ssm_peft/peft/gla_sd_lora.py
│   ├── _select_dimensions(): 添加全局选择模式
│   ├── _compute_channel_importance(): 添加多种策略
│   └── _apply_value_boost(): V向量优先级
│
├── mamba_peft/lat_adapter.py
│   ├── _apply_sdlora_env_overrides(): 添加新环境变量
│   └── _auto_tune_warmup(): 自适应warmup
│
├── mamba_peft/train_lat.py
│   └── 添加实验配置参数
│
└── configs/gla_sdlora/
    └── experimental/*.json: 新配置文件
```

### 6.2 新增环境变量

```bash
# 全局选择模式
export HP_GLOBAL_SELECT=true    # 启用全局Top-K选择

# V向量优先级
export HP_VALUE_BOOST=true       # 启用V维度优先
export HP_VALUE_BOOST_FACTOR=1.5 # 优先级倍数

# 自适应比例
export HP_ADAPTIVE_RATIO=true    # 启用自适应比例
export HP_CV_THRESHOLD=1.0       # CV阈值

# 梯度策略
export HP_IMPORTANCE_STRATEGY=mean_abs  # mean_abs|abs_mean|l2_norm

# Warmup优化
export HP_AUTO_WARMUP=true        # 自动调整warmup次数
export HP_WARMUP_MODE=dataset     # dataset|fixed
```

---

## 7. 风险与缓解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  风险                          │  影响        │  缓解措施                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  全局选择可能让某些层完全冻结  │  性能下降    │  保留最小Train维度       │
│  部分反向传播实现复杂          │  Bug风险     │  充分单元测试             │
│  动态比例可能不稳定            │  训练波动    │  添加限制范围             │
│  V向量优先可能不适用于GLA      │  效果相反    │  先小规模验证             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 预期效果

基于SMT的结果，保守预期：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  改进项                        │  预期提升                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  全局Top-K选择                  │  +1-2% accuracy               │
│  V向量优先级                    │  +0.5-1% accuracy             │
│  部分反向传播                    │  -20% 显存, +10% 速度        │
│  自适应比例                      │  ±0.5% accuracy (自适应)     │
│  组合效果                        │  +2-4% accuracy (累计)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 总结

SD-LoRA可以从SMT中借鉴的核心思想：

1. **全局优先**: 让重要参数自动获得更多训练资源
2. **V向量优先**: 识别并优先更新类似V的关键模块
3. **部分计算**: 只计算必要参数的梯度
4. **自适应调整**: 根据梯度分布动态调整策略

这些改进在保持SD-LoRA对GLA特化优势的同时，可以进一步提升性能和效率。
