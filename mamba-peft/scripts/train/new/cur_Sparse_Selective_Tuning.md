#  Sparse Selective Tuning（Gradient + Static + Global top-K）使用说明

这套能力在你的原训练链路 lat_batch_tmux.sh → lat_round.sh → train_lat.py → prepare_lat_model_and_tokenizer() → GenericLMTrainer 上无侵入接入：
- 默认关闭（不设置 HP_SPARSE_ENABLE 就不会触发任何稀疏逻辑）
- 不修改 GenericLMTrainer
- mask 仅影响训练反传（grad hook），不改变推理结构

---

## 1) 向后兼容：原模式是否无损

### 结论
是的。只要你不设置 `HP_SPARSE_ENABLE=1`，训练行为保持原样（dense LoRA 还是 dense LoRA）。

### 你怎么验证（建议）
- **日志**：原模式下不会出现 `"[xxx][sparse] enabled ..."` 的打印。
- **输出目录**：原模式下不会生成：
  - `sparse_selective_mask.pt`
  - `sparse_selective_meta.json`
- **可训练参数统计**：仍由原来的 LoRA/PEFT config 决定（`parameter_counts.json`、trainable names 等保持原逻辑）。

---

## 2) 新增能力概览：三个功能（Scope）+ 两种预算模式（Budget Mode）

### 2.1 三个功能（Scope）
通过环境变量 `HP_SPARSE_SCOPE` 控制候选池：

**F1 Sparse-LoRA（lora_only）**
- 候选池：LoRA 参数（A/B 等，按参数名包含 `lora_` 识别）
- 用途：在不破坏原 LoRA 结构前提下，对 LoRA 内部做稀疏选择性更新（类似 FISH-Tuning 的训练阶段 mask）

**F2 Sparse-Base（base_only）**
- 候选池：base 模型参数（当前 YAML 的 peft JSON 里的 `target_modules` 所命中的模块的 `.weight`）
- 用途：不注入/不训练 LoRA（或即使注入也不更新 LoRA），只在 base 权重上做稀疏更新

**F3 Sparse-Hybrid（hybrid）**
- 候选池：base 候选 + LoRA 候选的并集
- 用途：用同一份全局 top-K 预算在 base 与 LoRA 之间“自动分配”（不引入 split 超参）

> 注意：F2/F3 的 base 候选池依赖“当前 YAML 有 peft: 并且其 JSON 里有 target_modules”，否则 base 候选可能为空并报错（这是按你“候选范围按 YAML target 限定”的设计）。

---

### 2.2 两种预算模式（Budget Mode）
通过环境变量 `HP_SPARSE_BUDGET_MODE` 控制：

**Fixed Budget（固定预算）：`fixed_ratio`**
- 用 `HP_SPARSE_RHO` 指定候选池内保留比例（例如 `0.3`）

**Match Reference Budget（对齐参考预算）：`match_reference`**
- 用 `HP_SPARSE_REFERENCE_CFG` 指定一个 reference YAML（dense LoRA）
- 系统会读取 reference 的 LoRA `r` 和 `target_modules`，并在当前模型结构上估算 dense LoRA 的参数量 `K_ref`，再在当前候选池内选 top-`K_ref`

---

## 3) 关键环境变量（只要记这些就够了）

### 总开关（默认不设 = 关闭）
- `HP_SPARSE_ENABLE=1`：开启稀疏选择性微调
- 不设 / 设为 `0`：完全不触发稀疏逻辑（向后兼容）

### 选择“功能 / Scope”
- `HP_SPARSE_SCOPE`：`lora_only | base_only | hybrid`

### 选择“预算模式”
- `HP_SPARSE_BUDGET_MODE`：`fixed_ratio | match_reference`

### fixed_ratio 所需
- `HP_SPARSE_RHO`：例如 `0.3`

### match_reference 所需
- `HP_SPARSE_REFERENCE_CFG`：reference YAML 的路径（建议写绝对路径）
  - 例：`/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/yaml/E7_KVONLY_r8_alpha16.yaml`

### salience 样本数（默认 1024）
- `HP_SPARSE_SCORE_SAMPLES`：例如 `1024`

---

## 4) 六个模式怎么设置 
 
> 说明：这套实现默认就是 Gradient + Static + Global，不需要你再配置指标 / 动态 / 局部等```bash

export HP_INIT=pissa   
export HP_SAVE_MODE=best_last
export HP_SAVE_FULL_MODEL=0

```
### F1-A：Sparse-LoRA + 固定预算（fixed_ratio）
```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=lora_only
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F1-B：Sparse-LoRA + 对齐参考预算（match_reference）
```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=lora_only
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_REFERENCE_CFG=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/yaml/E7_KVONLY_r8_alpha16.yaml
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F2-A：Sparse-Base + 固定预算（fixed_ratio）
```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=base_only
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F2-B：Sparse-Base + 对齐参考预算（match_reference）
```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=base_only
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_REFERENCE_CFG=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/yaml/E7_KVONLY_r8_alpha16.yaml
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F3-A：Sparse-Hybrid（LoRA + Base）+ 固定预算（fixed_ratio）
```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=hybrid
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F3-B：Sparse-Hybrid（LoRA + Base）+ 对齐参考预算（match_reference）
```bash
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=hybrid
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_REFERENCE_CFG=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/yaml/E7_KVONLY_r8_alpha16.yaml
export HP_SPARSE_SCORE_SAMPLES=1024
```

---

## 5) 完整 bash 示例（在 delta_net + sst2 原脚本上仅加 Sparse-LoRA fixed_ratio）

你原来的 bash 不用动，只需要在一堆 `export` 中间加上下面这段（建议放在 `HP_*` 超参附近）：

```bash
# === Sparse Selective Tuning (Gradient + Static + Global) ===
export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=lora_only
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
export HP_SPARSE_SCORE_SAMPLES=1024
```

其余部分（conda activate、MODEL_TYPE / LAT_MODEL / LAT_PREC、HP_*、SwanLab、最后 `./lat_batch_tmux.sh ...`）保持不变即可。

---

## 6) 运行后你应当看到 / 拿到什么（验收与复现）

每个 run 的 `output_dir` 下会多出两份文件：

### `sparse_selective_mask.pt`
- 训练 mask（每个参数一个布尔 mask tensor）+ 元数据
- resume 时会自动复用，避免重复算 salience

### `sparse_selective_meta.json`
你验收最关心的信息都在这：
- `candidate_elems`：候选池元素总数
- `budget_k`：理论预算 K
- `realized_k`：实际 `mask=1` 数量（应 ≈ K）
- `scope / budget_mode / score_samples / reference_cfg`
- `targets_from_current_peft`：当前 YAML 的 `target_modules`（用于解释 `base_only / hybrid`）

另外日志里会打印类似：
```text
[delta_net][sparse] candidate_elems=... budget_k=... realized_k=...
saved: sparse_selective_mask.pt, sparse_selective_meta.json
```
