# SMT代码实现深度分析

> **分析日期**: 2025-12-31
> **目标**: 理解SMT实现细节，为GLA SD-LoRA改进提供技术参考

---

## 1. 代码结构总览

```
Sparse_Matrix_Tuning/
├── peft/
│   ├── tuner/smt.py          # 主训练逻辑
│   ├── smt_config.py         # SMT配置类
│   ├── train_smt.py          # 训练入口
│   └── utils.py              # 工具函数
├── deepspeed/
│   └── smt/
│       ├── smt.py            # DeepSpeed版本的SMT实现
│       └── smt_helper.py     # 辅助函数
└── README.md
```

---

## 2. 核心配置类 (SMTConfig)

### 2.1 配置参数

```python
@dataclass
class SMTConfig(PeftConfig):
    # 子矩阵数量配置
    num_submatrix_mlp: int = 0      # MLP层的子矩阵数量
    num_submatrix_attn: int = 0    # Attention层的子矩阵数量
    smt_dropout: float = 0.0       # Dropout率

    # 模型相关
    model_name: str = None          # 模型名称（用于确定维度）
    full_ft_steps: int = 100        # Warmup迭代次数

    # 选择策略
    selection_strategy: str = "no_restriction"  # 参数分配策略
    calculation_strategy: str = "mean_abs"      # 梯度计算策略

    target_modules: Optional[Union[List[str], str]] = None
    merge_weights: bool = False
```

**关键设计**:
- `num_submatrix_mlp/attn` 直接指定子矩阵数量，而非比例
- `full_ft_steps` 控制warmup阶段长度
- `selection_strategy` 决定子矩阵在层间分配方式

### 2.2 支持的模型维度

```python
# 不同模型的矩阵维度配置
if model in ["yahma/llama-13b-hf", "NousResearch/Llama-2-13b-hf"]:
    Block_dimension = 256
    large_d = 54   # gate_proj, up_proj: 54×256 × 16×256
    small_d = 20   # down_proj: 20×256 × 54×256

elif model in [...llama-7b variants...]:
    Block_dimension = 256
    large_d = 43
    small_d = 16

elif model in [...llama-3-8b...]:
    Block_dimension = 256
    large_d = 56
    small_d = 16
    # k_proj和v_proj使用small_d=4
```

**设计理由**: 256是LLaMA系列所有Linear层维度（hidden_size和intermediate_size）的最大公约数

---

## 3. 子矩阵选择算法

### 3.1 核心函数: `select_submatrix_based_on_grads`

```python
def select_submatrix_based_on_grads(
    grads,              # {(module_name, layer_num): grad_tensor}
    n,                  # 要选择的子矩阵数量
    selection_strategy="no_restriction",
    calculate_strategy="mean_abs",
    model="yahma/llama-7b-hf"
):
```

### 3.2 梯度重塑与分块

```python
# gate_proj和up_proj: (large_d × 256) × (small_d × 256)
# → 重塑为 (large_d, 256, small_d, 256)
if key[0] in ['gate_proj', 'up_proj']:
    reshaped_grad = grad.reshape(large_d, Block_dimension, small_d, Block_dimension)

# down_proj: (small_d × 256) × (large_d × 256)
# → 重塑为 (small_d, 256, large_d, 256)
elif key[0] == 'down_proj':
    reshaped_grad = grad.reshape(small_d, Block_dimension, large_d, Block_dimension)

# q_proj, k_proj, v_proj: (small_d × 256) × (small_d × 256)
# → 重塑为 (small_d, 256, small_d, 256)
elif key[0] in ['q_proj', 'k_proj', 'v_proj']:
    reshaped_grad = grad.reshape(small_d, Block_dimension, small_d, Block_dimension)
```

### 3.3 梯度计算策略

```python
# 四种策略对比
def mean_abs(grad_tensor):
    # SMT最佳策略
    return grad_tensor.mean(dim=(1, 3)).abs()

def abs_mean_(grad_tensor):
    return grad_tensor.abs().mean(dim=(1, 3))

def L1_norm(grad_tensor):
    return grad_tensor.abs().sum(dim=(1, 3))

def L2_norm(grad_tensor):
    return torch.sqrt(torch.sum(grad_tensor.abs() ** 2, dim=(1, 3)))
```

**维度理解**:
- 输入: `(large_d, 256, small_d, 256)`
- `mean(dim=(1, 3))`: 对两个256维取平均 → `(large_d, small_d)`
- `abs()`: 取绝对值
- 输出: 每个子矩阵的平均梯度幅度的2D矩阵

### 3.4 子矩阵选择策略

#### Strategy 1: `no_restriction` (默认，推荐)

```python
# 全局Top-K选择，使用min-heap高效维护
top_blocks = []
for key, block_mean in block_means.items():
    for i in range(block_mean.shape[0]):
        for j in range(block_mean.shape[1]):
            abs_mean = block_mean[i, j].item()
            if len(top_blocks) < n:
                heapq.heappush(top_blocks, (abs_mean, (key, i, j)))
            else:
                heapq.heappushpop(top_blocks, (abs_mean, (key, i, j)))

# 按模块分组
ranked_blocks = defaultdict(list)
for mean, (info, row, col) in top_blocks:
    ranked_blocks[info].append((row, col))
```

**特点**:
- 跨所有层全局排序
- 自动分配子矩阵到最重要的层
- 某些层可能完全没有子矩阵被选中

#### Strategy 2: `norm_dist` (每层固定比例)

```python
# 每个层内部独立选择Top-K
ranked_blocks = defaultdict(list)
for key, block_mean in block_means.items():
    indices = torch.argsort(block_mean.view(-1), descending=True)
    top_indices = indices[:n]  # 每层选n个
    for idx in top_indices:
        row = idx // block_mean.shape[1]
        col = idx % block_mean.shape[1]
        ranked_blocks[key].append((row.item(), col.item()))
```

**特点**:
- 每层都有固定数量的子矩阵
- 可能浪费参数在不重要的层上
- 论文实验表现较差

---

## 4. 自定义稀疏Linear层

### 4.1 LinearLayer_MatrixSparsity

```python
class LinearLayer_MatrixSparsity(torch.nn.Module):
    def __init__(self, weight, bias=None, index_list=[]):
        super().__init__()
        self.weight = weight
        self.weight.requires_grad = False  # 原权重冻结
        self.index_list = index_list         # [(block_row, block_col), ...]

        # 提取选中的子矩阵，拼接成连续内存
        self.selected_weight = torch.empty(
            len(index_list) * Block_dimension,  # 总行数
            Block_dimension,                    # 列数
            dtype=self.weight.data.dtype,
            device=self.weight.data.device
        )

        # 从原权重复制初始值
        for i in range(len(index_list)):
            index = index_list[i]
            self.selected_weight[i*256 : (i+1)*256, :] = \
                self.weight.data[
                    index[0]*256 : (index[0]+1)*256,
                    index[1]*256 : (index[1]+1)*256
                ]

        self.selected_weight.requires_grad = True
        self.selected_weight = nn.Parameter(self.selected_weight)
        self.fn = linearZ.apply
```

**关键设计**:
1. **原权重冻结**: `self.weight.requires_grad = False`
2. **子矩阵拼接**: 选中的子矩阵存储在连续的`selected_weight`中
3. **参数化**: `selected_weight`是可训练参数

### 4.2 Forward: 稀疏矩阵乘法

```python
class linearZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, selected_weight, matrix_index_list, weight):
        # 仅保存选中列的激活值（用于backward）
        input_list = []
        for index in matrix_index_list:
            # input: [batch, seq, hidden]
            # 仅保存对应的256维
            input_list.append(
                input[:, :, index[1]*256 : index[1]*256+256]
            )

        ctx.list1 = input_list
        ctx.list2 = matrix_index_list
        ctx.save_for_backward(weight)

        # 使用完整的weight做矩阵乘法
        output = torch.matmul(input, weight.t())
        return output
```

**内存优化**:
- 仅保存选中列的激活值（z%比例）
- 减少activation memory

### 4.3 Backward: 部分梯度计算

```python
@staticmethod
def backward(ctx, grad_output):
    weight, = ctx.saved_tensors
    input_list = ctx.list1
    matrix_index_list = ctx.list2

    # 仅计算选中子矩阵的梯度
    grad_weight = torch.empty(
        len(input_list) * Block_dimension,
        Block_dimension,
        dtype=grad_output.dtype,
        device=grad_output.device
    )

    for i in range(len(input_list)):
        index = matrix_index_list[i]

        # 核心公式: ∂L/∂W = (∂L/∂Z)^T @ X
        # 仅计算选中行的梯度
        grad_weight[i*256:(i+1)*256, :] = torch.sum(
            torch.matmul(
                grad_output.permute(0, 2, 1)[:, index[0]*256:(index[0]+1)*256, :],
                input_list[i]
            ),
            dim=0
        )

    # 输入梯度需要完整计算
    grad_input = torch.matmul(grad_output, weight)
    return grad_input, grad_weight, None, None
```

**关键理解**:
```
完整反向传播:  ∂L/∂W = (∂L/∂Z)^T @ X
                [batch, out, hidden] @ [batch, hidden, in]

SMT部分反向传播: 仅计算选中行的(∂L/∂Z)^T和对应列的X
```

---

## 5. 训练流程

### 5.1 Warmup阶段（梯度收集）

```python
def gradient_collection(model, mlp_warmup_grads, attention_warmup_grads,
                        num_submatrix, num_submatrix_attention):
    from deepspeed.utils import safe_get_full_grad
    pattern = re.compile(r'model\.layers\.(\d+)\.')

    for name, param in model.module.named_parameters():
        match = pattern.search(name)
        layer_number = int(match.group(1)) if match else None

        # 收集MLP梯度
        if 'mlp' in name and num_submatrix > 0:
            grad = safe_get_full_grad(param)
            module_name = 'gate_proj' if 'gate_proj' in name else \
                          'up_proj' if 'up_proj' in name else 'down_proj'

            if (module_name, layer_number) not in mlp_warmup_grads:
                mlp_warmup_grads[(module_name, layer_number)] = \
                    grad.detach().cpu().to(torch.float32)
            else:
                mlp_warmup_grads[(module_name, layer_number)] += \
                    grad.detach().cpu().to(torch.float32)

        # 收集Attention梯度
        if 'self_attn' in name and num_submatrix_attention > 0:
            grad = safe_get_full_grad(param)
            module_name = 'q_proj' if 'q_proj' in name else \
                          'k_proj' if 'k_proj' in name else \
                          'v_proj' if 'v_proj' in name else None

            if module_name is not None:
                attention_warmup_grads[(module_name, layer_number)] += \
                    grad.detach().cpu().to(torch.float32)

    return mlp_warmup_grads, attention_warmup_grads
```

**特点**:
1. 使用`safe_get_full_grad`确保获取完整梯度（DS Zero兼容）
2. 梯度累积到CPU（节省GPU内存）
3. 按模块和层号组织

### 5.2 模型转换（SMT初始化）

```python
def mark_only_smt_as_trainable(model, select_parameters,
                                select_attention_parameters):
    """冻结未选中的参数"""
    pattern = re.compile(r'model\.layers\.(\d+)\.')

    for name, param in model.named_parameters():
        if "mlp" in name:
            module_name = ...
            layer_number = ...
            if (module_name, layer_number) in select_parameters.keys():
                param.requires_grad = True
            else:
                param.requires_grad = False

        elif "self_attn" in name:
            module_name = 'q_proj' if 'q_proj' in name else \
                          'k_proj' if 'k_proj' in name else \
                          'v_proj' if 'v_proj' in name else None
            layer_number = ...
            if (module_name, layer_number) in select_attention_parameters.keys():
                param.requires_grad = True
            else:
                param.requires_grad = False

        else:
            param.requires_grad = False  # 其他参数全部冻结

    return model
```

### 5.3 Linear层替换

```python
def convert_linear_layer_to_matrix_sparsity(model, selected_submatrix,
                                            selected_submatrix_attention):
    """将选中的Linear层替换为稀疏版本"""
    pattern = re.compile(r'model\.layers\.(\d+)\.')

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and '.layers.' in name:
            if "mlp" in name and module.weight.requires_grad:
                module_name = ...
                layer_number = ...
                index_list = selected_submatrix[(module_name, layer_number)]

                # 替换为稀疏Linear
                tmp = LinearLayer_MatrixSparsity(
                    module.weight,
                    bias=None,
                    index_list=index_list
                ).to(module.weight.device).to(module.weight.dtype)
                recursive_setattr(model, name, tmp)

            # similar for self_attn...

    return model
```

---

## 6. 优化器配置

### 6.1 分组优化器

```python
def get_optimizer_sparse_grouped_parameters(model, weight_decay, smt_lr):
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                      if (not any(nd in n.lower() for nd in no_decay_name_list)
                          and p.requires_grad
                          and not any(nd in n.lower() for nd in lora_name_list))],
            "weight_decay": weight_decay,
            "lr": smt_lr
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if (not any(nd in n.lower() for nd in no_decay_name_list)
                          and p.requires_grad
                          and any(nd in n.lower() for nd in lora_name_list))],
            "weight_decay": weight_decay,
            "lr": lora_lr
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if (any(nd in n.lower() for nd in no_decay_name_list)
                          and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]
    return [g for g in optimizer_grouped_parameters if g["params"]]
```

---

## 7. 关键实现细节总结

### 7.1 内存优化技术

| 技术 | 实现 | 效果 |
|------|------|------|
| 梯度存储到CPU | `grad.detach().cpu()` | 节省GPU内存 |
| 部分激活存储 | 仅保存选中列 | activation memory × z% |
| 无掩码存储 | 直接投影到dense tensor | 0额外内存 |
| 冻结层跳过 | `requires_grad=False` | 节省反向传播 |

### 7.2 计算优化技术

| 技术 | 实现 | 效果 |
|------|------|------|
| 部分反向传播 | 自定义autograd.Function | backward计算 × z% |
| Dense矩阵乘法 | 拼接子矩阵为连续块 | 避免SPMM开销 |
| FusedAdam | 使用DeepSpeed实现 | 优化器速度 |

### 7.3 与现有框架的兼容

```python
# DeepSpeed兼容
from deepspeed.utils import safe_get_full_grad

# 支持Zero-3的梯度获取
grad = safe_get_full_grad(param)

# 支持CPU offload (warmup阶段)
# 主训练阶段移到GPU
```

---

## 8. 代码质量观察

### 8.1 优点

1. **清晰的模块分离**: 选择、冻结、替换各自独立
2. **灵活的配置**: 支持多种选择和计算策略
3. **DeepSpeed集成**: 充分利用现有优化

### 8.2 可改进之处

1. **硬编码维度**: 每个新模型需要手动添加维度配置
2. **固定子矩阵大小**: 256×256作为常量，不可配置
3. **缺乏子矩阵位置可视化**: 难以调试选择结果
4. **无动态更新**: 选中子矩阵后不再调整

---

## 9. 对SD-LoRA实现的启发

### 9.1 可直接借鉴的代码模式

```python
# 1. 梯度累积模式
grad_accumulator[(module_name, layer_num)] += grad.detach().cpu()

# 2. Min-heap高效Top-K
heapq.heappushpop(top_blocks, (value, (key, i, j)))

# 3. 自定义autograd.Function
class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, selected_indices, ...):
        # 仅保存需要的信息
        ctx.save_for_backward(...)

    @staticmethod
    def backward(ctx, grad_output):
        # 仅计算选中参数的梯度
        ...
```

### 9.2 GLA适配需要注意

| SMT设计 | GLA差异 | SD-LoRA适配 |
|--------|---------|------------|
| Attention QKV | GLA的gk_proj | 需要确认gk_proj的维度 |
| 固定256块大小 | Channel数量可变 | 使用channel级别而非块 |
| softmax饱和 | logsigmoid | 梯度行为可能不同 |
| Linear层 | Gate层 | 结构不同 |

---

## 10. 关键代码片段速查

### 子矩阵选择核心逻辑

```python
# 步骤1: 梯度重塑为块
reshaped_grad = grad.reshape(d1, 256, d2, 256)

# 步骤2: 计算每个块的平均梯度
block_means = reshaped_grad.mean(dim=(1, 3)).abs()

# 步骤3: 全局Top-K选择
top_blocks = heapq.nlargest(n, [
    (block_means[i, j], (key, i, j))
    for key, block_means in all_blocks.items()
    for i in range(block_means.shape[0])
    for j in range(block_means.shape[1])
])

# 步骤4: 按模块分组
selected = defaultdict(list)
for mean, (key, i, j) in top_blocks:
    selected[key].append((i, j))
```

### 稀疏Forward/Backward核心

```python
# Forward: 保存选中列的激活
for (row, col) in selected_blocks:
    saved_activations.append(input[:, :, col*256:(col+1)*256])

# Backward: 仅计算选中块的梯度
for idx, (row, col) in enumerate(selected_blocks):
    grad_w[idx*256:(idx+1)*256, :] = \
        grad_output[:, row*256:(row+1)*256, :].T @ saved_activations[idx]
```
