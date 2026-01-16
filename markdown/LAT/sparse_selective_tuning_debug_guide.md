# Sparse Selective Tuning 调试和验证指南

## 概述

Sparse Selective Tuning 是一种在大语言模型上进行参数高效微调的技术，通过在选定的权重上添加稀疏增量来实现高效的模型适配。本文档详细介绍该技术的调试、验证过程，以及常见的易错点和解决方案。

## 核心概念

### Sparse Selective Tuning 的工作原理

1. **静态选择**：训练前基于梯度重要性选择需要微调的权重子集
2. **稀疏重参数化**：将选定的权重替换为 `SparseDeltaLinear` 模块
3. **优化器状态控制**：只为选定的 K 个参数维护梯度状态，实现 O(K) 内存复杂度

### 支持的微调模式

- `lora_only`: 只对 LoRA 参数进行稀疏选择
- `base_only`: 只对骨干网络权重进行稀疏选择
- `hybrid`: 同时对 LoRA 和骨干权重进行稀疏选择
- `lora_dense_base_sparse`: LoRA 保持稠密，骨干权重进行稀疏选择

## 易错点分析

### 1. 候选池构建错误

#### 问题描述
PEFT LoRA 注入后，模型结构变得复杂，容易出现权重重复选择或遗漏。

#### 具体表现
- **重复 sparsify**: 同一权重被多次替换，导致运行时错误
- **影子 base_layer 问题**: PEFT wrapper 的内部子模块被误当作独立候选
- **LoRA 内部污染**: LoRA A/B 参数被错误地包含在 base 候选池中

### 2. PEFT 注入后的复杂模型结构

#### 三类容易混淆的 Linear 层

从模型模块列表可以明确看到每个被 LoRA 注入的投影同时存在三类 Linear：

1. **PEFT wrapper 层**
   - 示例: `base_model.model.model.layers.0.attn.o_proj` → `peft.tuners.lora.layer.Linear`
   - 这是 PEFT 注入后的顶层包装器，负责 LoRA 计算

2. **wrapper 的子模块 base_layer**
   - 示例: `base_model.model.model.layers.0.attn.o_proj.base_layer` → `torch.nn.Linear`
   - 这是原始骨干网络的权重，LoRA 基于它进行计算

3. **LoRA 内部线性层**
   - 示例: `base_model.model.model.layers.0.attn.o_proj.lora_A.default` → `torch.nn.Linear`
   - 示例: `base_model.model.model.layers.0.attn.o_proj.lora_B.default` → `torch.nn.Linear`
   - 这些是 LoRA 适配器的内部权重矩阵

#### 关键风险
如果处理不当，会出现：
- 同时选择 `...o_proj.weight` 和 `...o_proj.base_layer.weight`（重复视图）
- 将 `...lora_A.default.weight` 误认为 backbone 权重
- 稀疏替换时尝试对已替换的模块再次操作

### 3. 预算计算错误

#### 问题类型
- **match_reference 模式**: 预算计算依赖参考配置的 LoRA 参数量
- **混合模式**: 需要正确区分 LoRA 和 base 贡献

#### 验证要点
- `dense_LoRA_trainable(current)` + `sparse_base_k` == `K_ref`
- 最终 `trainable_params` 应等于预算 K

## 验证工具

### inspect_all_linear_pool.py

#### 功能特性

1. **多池模式检查**
   - `all_linear`: 验证全骨干网络候选池
   - `from_peft_json`: 验证基于 JSON 配置的候选池
   - `from_current_peft`: 验证基于当前 YAML target 的候选池

2. **严格验证**
   - **重复检测**: 基于 Parameter 对象身份验证无重复引用
   - **影子 key 检查**: 检测 `*.base_layer.weight` / `*.linear.weight`
   - **覆盖验证**: 确保池完全匹配预期目标模块

3. **PEFT 兼容性**
   - 支持 PEFT 注入后的模型结构
   - 正确区分 LoRA 内部和骨干权重

#### 使用方法

```bash
# 检查全骨干候选池（PEFT 注入后）
python mamba-peft/tools/inspect_all_linear_pool.py \
  --model /path/to/model \
  --model-type retnet \
  --prec bf16 \
  --peft-json /path/to/peft.json \
  --base-pool all_linear \
  --out-dir /tmp/check_all_linear

# 检查 QKVOMLP 子集候选池
python mamba-peft/tools/inspect_all_linear_pool.py \
  --model /path/to/model \
  --model-type retnet \
  --prec bf16 \
  --peft-json /path/to/vo_peft.json \
  --base-pool from_peft_json \
  --base-pool-peft-json /path/to/qkvo_mlp_peft.json \
  --out-dir /tmp/check_qkvomlp
```

#### 输出解读

##### 成功标准
```
=== SUMMARY ===
total_modules=800
pool_entries(dict_keys)=193
pool_unique_parameter_objects=193
duplicate_groups=0

=== COVERAGE CHECK ===
[from_peft_json] expected_matches=168 pool_matches=168 missing=0 extra=0

=== DUPLICATES ===
no_duplicates_by_identity

=== 2D-WEIGHT NON-LINEAR MODULES ===
count=0
```

##### 错误模式
- `duplicate_groups > 0`: 存在重复 Parameter 引用
- `missing > 0`: 候选池不完整
- `SHADOW_KEY` 错误: 检测到影子 base_layer 键
- `missing_expected > 0`: 预期模块未被覆盖

## 修复过程

### 1. 候选池构建修复

#### 核心修改 (sparse_selective_engine.py)

```python
def _iter_all_backbone_linear_weight_params(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    # 排除 LoRA 内部模块
    if "lora_" in module_name:
        continue
    
    # 排除 PEFT wrapper 的影子子模块
    if isinstance(module, torch.nn.Linear):
        if _is_peft_shadow_child_linear(module_name, name_to_module):
            continue  # 跳过 ...base_layer / ...linear 子模块
        yield f"{module_name}.weight", module.weight
        continue
    
    # 处理 PEFT wrapper: 选择 base_layer 权重
    if _is_peft_lora_linear(module):
        base = _get_base_linear_from_peft_linear(module)
        w = getattr(base, "weight", None)
        if isinstance(w, torch.nn.Parameter) and w.dim() == 2:
            yield f"{module_name}.weight", w
        continue
```

### 2. 严格验证集成

#### 训练前验证 (maybe_run_sparse_selective_tuning)

```python
# 构建候选池后立即验证
_validate_base_pool_strict(
    model=model,
    base_pool=effective_base_pool,
    base_params=base_params,
    model_type=model_type,
)

# 检查项目:
# - 无重复 Parameter 引用
# - 无影子 base_layer 键
# - 对于 all_linear: 完全覆盖所有 eligible 模块
```

### 3. 错误处理增强

#### 替换时的保护 (_replace_linear_weight_with_sparse_delta)

```python
if isinstance(module, SparseDeltaLinear):
    raise TypeError(
        f"Cannot sparsify already-sparsified module at '{module_name}' "
        "(type=SparseDeltaLinear). This indicates duplicate selection keys."
    )
```

## 验证结果分析

### RetNet-1.3B-100B 验证结果

#### all_linear 模式 (全骨干)
```
pool_entries=193 (24层 × 8模块/层 + lm_head)
duplicate_groups=0
missing_expected=0 (修正脚本后)
expected_linear_modules=193
```

#### from_peft_json 模式 (QKVOMLP)
```
pool_entries=168 (24层 × 7模块/层)
duplicate_groups=0
missing=0 extra=0
expected_matches=168 pool_matches=168
```

### 关键发现

1. **无重复问题**: Parameter 对象身份验证确保无重复引用
2. **无影子问题**: 严格过滤 PEFT wrapper 子模块
3. **精确覆盖**: 候选池与目标模块集合完全匹配
4. **LoRA 隔离**: base 池不包含 LoRA 内部权重

## 最佳实践

### 1. 训练前验证
```bash
# 总是先用检查脚本验证候选池
python mamba-peft/tools/inspect_all_linear_pool.py \
  --model $MODEL --model-type $TYPE --prec bf16 \
  --peft-json $PEFT_JSON --base-pool $POOL_MODE \
  --out-dir /tmp/verify_pool
```

### 2. 错误排查流程
1. 检查 `SHADOW_KEY` 错误 → 更新引擎过滤逻辑
2. 检查 `missing_expected > 0` → 验证目标模块匹配
3. 检查 `duplicate_groups > 0` → 修复 Parameter 重复引用

### 3. 配置建议
- `all_linear`: 用于全模型稀疏微调
- `from_peft_json`: 用于子集权重稀疏
- 避免混合使用不同池模式的配置

### 4. 调试输出
训练后检查:
- `parameter_counts.json`: 验证最终 trainable 参数量
- `sparse_selective_meta.json`: 检查预算分配
- `sparse_delta.pt`: 确认稀疏增量保存

## 结论

通过严格的候选池验证和错误处理，Sparse Selective Tuning 能够可靠地实现：
- **精确选择**: 无重复、无遗漏的权重选择
- **安全替换**: 避免运行时重复 sparsify 错误
- **语义正确**: 正确区分 LoRA 和骨干权重
- **可重现**: 静态选择确保实验一致性

这种方法为大语言模型的高效微调提供了可靠的技术基础。