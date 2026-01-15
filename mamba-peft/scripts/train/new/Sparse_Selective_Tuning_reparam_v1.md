# Sparse Selective Tuning（reparam_v1）环境变量文档

> 基于原 dense LoRA bash 的**新增 / 修改 export** 汇总  
> 默认模式：**Gradient + Static + Global top‑K**  
> Sparse 开启后使用**稀疏重参数化**，optimizer state 严格 **O(K)**  
> 产物：
> - `sparse_selective_selection.pt`
> - `sparse_selective_meta.json`
> - 每个 checkpoint 额外写 `sparse_delta.pt`（O(K)）

---

## 0) 最小必配（所有 Sparse 模式都需要）

```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=...            # lora_only | base_only | hybrid
export HP_SPARSE_BUDGET_MODE=...      # fixed_ratio | fixed_count | match_reference
export HP_SPARSE_SCORE_SAMPLES=1024   # 梯度打分样本数（默认 1024）
```

---

## 1) Scope（三种功能 F1 / F2 / F3）

### F1 Sparse‑LoRA  
只在 **LoRA A/B 线性层**里进行稀疏训练：

```bash
export HP_SPARSE_SCOPE=lora_only
```

### F2 Sparse‑Base  
不稀疏 LoRA，仅稀疏 **base 模型**中 `target_modules` 对应的 Linear 权重：

```bash
export HP_SPARSE_SCOPE=base_only
```

### F3 Sparse‑Hybrid  
LoRA + base 的并集一起做 **全局 top‑K**：

```bash
export HP_SPARSE_SCOPE=hybrid
```

> **注意**  
> - base 的候选池来自 *当前 `cfg.yaml` 的 peft json* 中的 `target_modules`  
> - 这些 target **必须对应 `nn.Linear`**，否则直接报错

---

## 2) Budget Mode（预算策略）

### A) 固定比例（fixed_ratio）

```bash
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
```

### B) 固定数量（fixed_count）

```bash
export HP_SPARSE_BUDGET_MODE=fixed_count
export HP_SPARSE_K=1234567
```

### C) 对齐参考预算（match_reference）

```bash
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_REFERENCE_CFG=/abs/path/to/E7_KVONLY_r8_alpha16.yaml
```

> **参考预算 K_ref 的计算方式**  
> 1. 读取 reference YAML  
> 2. 提取 peft json 中的 `(target_modules, r)`  
> 3. 在**当前已实例化模型**的匹配 Linear 上累计  
>    ```
>    r * (in_features + out_features)
>    ```
>    （LoRA A+B 的 dense 参数量估计）

---

## 3) 训练 / 保存策略（推荐统一入口）

### A) 不保存任何权重（最省磁盘）

```bash
export HP_SAVE_MODE=none
```

### B) 只保存 last（1 份 checkpoint）

```bash
export HP_SAVE_MODE=last
```

### C) 保存 best + last（2 份 checkpoint，默认推荐）

```bash
export HP_SAVE_MODE=best_last
```

> **兼容性说明**  
> - 若不设置 `HP_SAVE_MODE`，则沿用原有逻辑：  
>   `HP_NO_SAVE / HP_SAVE_TOTAL_LIMIT / HP_LOAD_BEST_MODEL_AT_END`

---

## 4) Sparse 专用的轻量保存 / 恢复（自动）

当允许保存 checkpoint（`HP_SAVE_MODE=last|best_last` 或旧逻辑）时，系统会自动：

- 在每个 `checkpoint-*` 目录中写入  
  ```
  sparse_delta.pt   # O(K)
  ```
- resume 时：
  - 若 **不是 full‑model 保存**
  - 必须从 `resume_from_checkpoint/sparse_delta.pt` 恢复
  - 缺失或不匹配 **直接报错**

> **目的**：  
> 在磁盘受限情况下，实现 **可 resume**，无需保存 full model

---

## 5) 可选参数（通常无需修改）

### SparseDeltaLinear 缩放 / Dropout

```bash
export HP_SPARSE_ALPHA=1.0
export HP_SPARSE_DROPOUT=0.0
```

### 是否保存 full model（体积很大，不建议）

```bash
export HP_SAVE_FULL_MODEL=1
```

---

## 6) 六种典型模式（仅新增 / 修改的 export）

### F1‑A：Sparse‑LoRA + fixed_ratio

```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=lora_only
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F1‑B：Sparse‑LoRA + match_reference

```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=lora_only
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_REFERENCE_CFG=/abs/path/to/E7_KVONLY_r8_alpha16.yaml
export HP_SPARSE_SCORE_SAMPLES=1024
```

---

### F2‑A：Sparse‑Base + fixed_ratio

```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=base_only
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F2‑B：Sparse‑Base + match_reference

```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=base_only
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_REFERENCE_CFG=/abs/path/to/E7_KVONLY_r8_alpha16.yaml
export HP_SPARSE_SCORE_SAMPLES=1024
```

---

### F3‑A：Sparse‑Hybrid + fixed_ratio

```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=hybrid
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F3‑B：Sparse‑Hybrid + match_reference

```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=hybrid
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_REFERENCE_CFG=/abs/path/to/E7_KVONLY_r8_alpha16.yaml
export HP_SPARSE_SCORE_SAMPLES=1024
```
# HP_SAVE_FULL_MODEL 与 HP_SAVE_MODE 的关系（独立控制）

它们是**完全独立的两个维度**，组合起来决定“**保存什么**”和“**存多少份**”。  
你可以使用任意组合，**不会发生冲突**。

---

## 1) HP_SAVE_MODE 的作用（存多少份）

`HP_SAVE_MODE` 控制：

- `save_total_limit`
- `load_best_model_at_end`
- 是否在 `output_dir` 根目录写 **final snapshot**

**不控制** checkpoint 的内容格式（是否 full model）。

---

### HP_SAVE_MODE=none

```bash
export HP_SAVE_MODE=none
```

行为：

- `save_total_limit = None`（不存任何 checkpoint）
- `load_best_model_at_end = False`
- **跳过**写 final snapshot 到 `output_dir` 根目录（最省磁盘）

结果：

- 磁盘上**完全没有** checkpoint 或 `model.pt`

---

### HP_SAVE_MODE=last

```bash
export HP_SAVE_MODE=last
```

行为：

- `save_total_limit = 1`
- `load_best_model_at_end = False`

结果：

- 只存 **1 个 checkpoint**（最后一个 step）

---

### HP_SAVE_MODE=best_last

```bash
export HP_SAVE_MODE=best_last
```

行为：

- `save_total_limit = 2`
- `load_best_model_at_end = True`

结果：

- 存 **2 个 checkpoint**（best + last）

---

## 2) HP_SAVE_FULL_MODEL 的作用（每份存什么）

`HP_SAVE_FULL_MODEL` 控制 **每个 checkpoint 的内容格式**：

- 是写 **完整 `model.pt`**（巨大）
- 还是 **adapter-only（PEFT 格式）**（很小）

**不控制**存多少份（那是 `HP_SAVE_MODE` 的职责）。

---

### 默认行为（HP_SAVE_FULL_MODEL=0 或不设）

```bash
export HP_SAVE_FULL_MODEL=0
# 或者完全不设置
```

行为：

- 每个 checkpoint 写 **adapter-only**
  - `adapter_model.bin`
  - `adapter_config.json`
  - 等 PEFT 文件
- 如果启用 Sparse：
  - 额外写 `sparse_delta.pt`（O(K)，小文件）

---

### HP_SAVE_FULL_MODEL=1

```bash
export HP_SAVE_FULL_MODEL=1
```

行为：

- 每个 checkpoint 写 **full `model.pt`**
  - 完整 PyTorch 模型对象（包含所有权重，体积巨大）
- Sparse 的 `sparse_delta.pt` **仍然会写**
  - 但通常不需要，因为 full model 已包含一切

---

## 3) 典型组合示例

### 最省磁盘（不保存任何权重）

```bash
export HP_SAVE_MODE=none
export HP_SAVE_FULL_MODEL=0   # 不设也行
```

结果：

- 磁盘上只有：
  - `sparse_selective_selection.pt`
  - `sparse_selective_meta.json`
  - 等元信息文件
- **没有任何 checkpoint**

---

### 只存 1 份 last（adapter-only）

```bash
export HP_SAVE_MODE=last
export HP_SAVE_FULL_MODEL=0   # 或不设
```

结果：

- 1 个 `checkpoint-*` 目录
- 内容：
  - adapter-only（PEFT）
  - `sparse_delta.pt`（如果启用 Sparse）

---

### 存 2 份 best + last（full model，巨大）

```bash
export HP_SAVE_MODE=best_last
export HP_SAVE_FULL_MODEL=1
```

结果：

- 2 个 checkpoint
- 每个都包含：
  - `model.pt`（全模型，体积巨大）

---

### 存 2 份 best + last（adapter-only，最常用平衡方案）

```bash
export HP_SAVE_MODE=best_last
export HP_SAVE_FULL_MODEL=0   # 或不设
```

结果：

- 2 个 checkpoint
- 每个都包含：
  - adapter-only（PEFT）
  - `sparse_delta.pt`（Sparse 可恢复）

---

## 4) Sparse 场景下的额外行为

无论 `HP_SAVE_FULL_MODEL` 取值如何，只要启用 Sparse：

- **总是**在每个 checkpoint 中写：

  ```text
  sparse_delta.pt   # O(K)，小文件
  ```

### Resume 逻辑

- **adapter-only 保存（HP_SAVE_FULL_MODEL=0）**
  - resume 时 **强制** 从 `sparse_delta.pt` 恢复 Sparse 参数
  - 文件缺失或不匹配 → **直接报错**

- **full model 保存（HP_SAVE_FULL_MODEL=1）**
  - Sparse 参数已包含在 `model.pt` 中
  - `sparse_delta.pt` 作为备用存在

---

## 5) 推荐结论

最均衡、最实用的组合：

```bash
export HP_SAVE_MODE=best_last
export HP_SAVE_FULL_MODEL=0
```

优势：

- 只存 **2 份**
- **磁盘占用极小**
- Sparse **完全可 resume**
- 不需要保存巨大 `model.pt`