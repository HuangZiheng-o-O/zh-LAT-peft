# 稀疏选择性微调（Sparse Selective Tuning）总览

- 新增 `SparsityConfig` + `apply_sparse_training`，在进入 `GenericLMTrainer` 前对候选参数做一次 **Gradient Salience + Static + Global top‑ρ** 的筛选。
- 默认 `HP_SPARSE_ENABLED=0`，保持 LoRA Dense 原行为；开启后，贯穿 **LoRA / Base / Hybrid** 三个 Scope 及 **Fixed / Match reference** 两类预算。
- mask 只算一次，写入 `output_dir/sparse_mask_meta.json` & `sparse_mask.pt`，resume 会自动复用。
- 所有例子均可叠加原本的 `MODEL_TYPE / LAT_MODEL / ...`，下面以你给的 `delta_net:ss t2` 命令为基准展示差异。

---

## 公共环境变量（开启稀疏时）

```bash
export HP_SPARSE_ENABLED=1                 # 启动稀疏训练
export HP_SPARSE_SCOPE=<lora|base|hybrid>  # 作用范围：LoRA / Base / Hybrid
export HP_SPARSE_BUDGET_MODE=<fixed_ratio|fixed_count|match_reference>

# 其他常用：
export HP_SPARSE_FIXED_RATIO=0.3           # 固定比例模式
export HP_SPARSE_FIXED_COUNT=500000        # 固定数量模式
export HP_SPARSE_MATCH_COUNT=1200000       # 对齐参考预算（目前需显式给出 trainable count）
export HP_SPARSE_SCORE_SAMPLES=1024        # salience 估计样本数
export HP_SPARSE_SAMPLE_BATCH_SIZE=1       # salience 估计 batch size
export HP_SPARSE_BASE_INCLUDE="q_proj k_proj"   # base-only 模式想精确控制某些模块
export HP_SPARSE_BASE_EXCLUDE="embed lm_head"   # 默认已排除 embedding / lm_head，可按需调整
```

---

## 原始 Dense LoRA 命令（无稀疏）

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16
...
# （保持你提供的 HF 缓存、数据、SwanLab 等所有 env 不动）
...
./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:glue-tvt_sst2" \
  --gpus "0 1 2 3 4 5 6 7" \
  --gpu-plan "2,2,2,2,2,2,2,2" \
  --model-type delta_net
```

此时 `HP_SPARSE_ENABLED` 未设或为 0，训练与旧版本完全一致。

---

## 三大功能 × 两种预算的增量配置

下文都默认“在上述 Dense LoRA 命令前增加若干 `export` 行”。若只列“新增的 env”，即在原 Dense 命令基础上追加这些设置即可。

### 1. F1‑A Sparse‑LoRA + Fixed Ratio

```bash
export HP_SPARSE_ENABLED=1
export HP_SPARSE_SCOPE=lora
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_FIXED_RATIO=0.3    # 30% 的 LoRA 参数留可训练
```

### 2. F1‑B Sparse‑LoRA + Match Reference

```bash
export HP_SPARSE_ENABLED=1
export HP_SPARSE_SCOPE=lora
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_MATCH_COUNT=250000  # 先统计参考 dense LoRA 参数数再填写
```

> 当前版本尚未自动解析 reference YAML，因此需要手动把参考配置的 trainable 参数数量填入 `HP_SPARSE_MATCH_COUNT`。  
> 若未来实现自动统计，只需额外提供  
> `HP_SPARSE_MATCH_REFERENCE=E7_KVONLY_r8_alpha16.yaml` 即可。

### 3. F2‑A Sparse‑Base + Fixed Ratio

```bash
export HP_SPARSE_ENABLED=1
export HP_SPARSE_SCOPE=base
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_FIXED_RATIO=0.01        # 只解冻 base 参数的 1%
export HP_SPARSE_BASE_INCLUDE="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
export HP_SPARSE_BASE_EXCLUDE="embed lm_head"   # 可省略（为默认值）
```

> 若 `BASE_INCLUDE` 为空，则默认只排除 `embed / lm_head`，其余层都可进候选。

### 4. F2‑B Sparse‑Base + Match Reference

```bash
export HP_SPARSE_ENABLED=1
export HP_SPARSE_SCOPE=base
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_MATCH_COUNT=250000     # 与某个 LoRA 配置的 trainable count 对齐
export HP_SPARSE_BASE_INCLUDE="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
```

### 5. F3‑A Sparse‑Hybrid + Fixed Ratio

```bash
export HP_SPARSE_ENABLED=1
export HP_SPARSE_SCOPE=hybrid
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_FIXED_RATIO=0.25   # LoRA + Base 合并候选里选 25%
export HP_SPARSE_BASE_INCLUDE="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
```

### 6. F3‑B Sparse‑Hybrid + Match Reference

```bash
export HP_SPARSE_ENABLED=1
export HP_SPARSE_SCOPE=hybrid
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_MATCH_COUNT=250000
export HP_SPARSE_BASE_INCLUDE="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
```

> 以上 6 个例子都沿用了同一条 `lat_batch_tmux.sh ...` 命令，差异仅在新增的 `HP_SPARSE_*` 变量。  
> `HP_SPARSE_SCORE_SAMPLES` 等可按论文默认改成 `1024 / batch=1`（不填即走默认）。

---

## 执行结果与验证

- 训练日志开头会输出 `Sparse selection summary: {...}`，包含候选数量、budget、实际 mask 数、`score_samples` 等；若没有这行，代表未启用或某步骤失败。
- `output/<run_name>/sparse_mask_meta.json` 记录 meta 信息，`sparse_mask.pt` 保存逐 tensor mask。resume 时若检测到这两文件即跳过 salience 计算并提示 **“Loaded existing mask”**。
- 关闭 `HP_SPARSE_ENABLED` 即完全恢复为 dense LoRA；任何 Scope / Budget 设置错误都会抛出异常（不会静默 fallback）。

通过以上环境变量组合，你即可针对 **LoRA / Base / Hybrid** 灵活试验 **Gradient + Static + Global** 的稀疏 selective tuning，同时保证旧工作流无缝延续。
