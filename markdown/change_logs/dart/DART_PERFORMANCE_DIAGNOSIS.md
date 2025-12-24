# DART 模型性能诊断报告

## 🔴 核心问题总结

你的 DART 训练结果（BLEU ≈ 10, chrF ≈ 31, METEOR ≈ 0.42）明显偏低，**主要原因是以下几个关键配置问题**：

---

## 🚨 问题 #1：**没有使用 Beam Search！**

### 当前配置
```bash
unset EVAL_GEN_NUM_BEAMS  # ← 这意味着 num_beams 未设置！
```

### 代码分析
在 `train_gla_only.py` 中，`eval_gen` 配置只包含 `max_length` 和 `min_length`：

```python
# train_gla_only.py:673-679
if (cfg.get("eval_gen") is None) and (is_gen_task or force_eval_gen):
    max_len = _maybe(env.get("EVAL_GEN_MAX_LENGTH"), int) or 1024
    min_len = _maybe(env.get("EVAL_GEN_MIN_LENGTH"), int) or 5
    cfg["eval_gen"] = {
        "max_length": int(max_len),
        "min_length": int(min_len),
        # ❌ 没有 num_beams！
    }
```

而在 `gla_hf_decoder.py` 中：

```python
# gla_hf_decoder.py:12-14
num_beams: Optional[int] = None  # ← 默认是 None！
do_sample: bool = False

# gla_hf_decoder.py:61-63
if self.num_beams is not None and self.num_beams > 1:
    gen_kwargs["num_beams"] = int(self.num_beams)
    gen_kwargs["do_sample"] = False
```

### 后果
- **你现在用的是 Greedy Decoding（贪心解码）！**
- 对于 DART 这种 data-to-text 任务，Greedy 解码会导致：
  - 生成质量明显下降
  - 重复词汇
  - 无法探索多样化的输出
- **业界标准是使用 beam_size=4~5**

### 修复方案
```bash
# 添加 beam search
export EVAL_GEN_NUM_BEAMS=5
```

并且需要修改 `train_gla_only.py` 来读取这个环境变量。

---

## 🚨 问题 #2：**学习率可能过高**

### 当前配置
```bash
export HP_LR=0.002  # 2e-3
```

### 分析
- 对于 1.8B 参数的 GLA 模型 + LoRA (r=8)，2e-3 的学习率**非常高**
- 典型的 LoRA fine-tuning 学习率范围是 **1e-4 ~ 5e-4**
- 过高的学习率可能导致：
  - 训练初期 loss 下降快，但后期震荡
  - 无法收敛到更好的局部最优
  - 生成质量不稳定

### 修复方案
```bash
export HP_LR=0.0002  # 或 0.0005
```

---

## 🚨 问题 #3：**训练数据可能没有正确展开**

### 代码分析
在 `dart_data.py` 中，`mode="lm"` 时会展开多参考：

```python
# dart_data.py:435-449
if self.mode == "lm":
    # 手动展开：把每个样本的多参考拆成多行
    rows = []
    for idx, row in df.iterrows():
        tripleset = row["tripleset"]
        sources = row["source"] if isinstance(row["source"], list) else [row["source"]]
        texts = row["text"] if isinstance(row["text"], list) else [row["text"]]
        ...
```

### 潜在问题
- 训练时用 `mode="lm"`，评估时用 `mode="gen"`
- 如果数据展开逻辑有问题，可能导致训练样本不完整
- 多参考的处理可能不一致

---

## 🚨 问题 #4：**评估指标计算可能有问题**

### 代码分析
```python
# dart_data.py:540-543
# Basic normalization: trim whitespace; split labels into multi-refs by sep_token
predictions = [p.strip() if isinstance(p, str) else "" for p in predictions]
references = [
    [r.strip() for r in (rs.split(self.sep_token) if isinstance(rs, str) else []) if r.strip()]
    for rs in references
]
```

### 潜在问题
- `sep_token` 的选择可能影响多参考的分割
- 如果 `sep_token` 在参考文本中出现，会导致错误分割
- 空参考的处理可能导致评估样本减少

---

## 🔧 完整修复方案

### 1. 修改 `train_gla_only.py` 支持 num_beams

```python
# 在 main() 函数中，修改 eval_gen 配置部分
if (cfg.get("eval_gen") is None) and (is_gen_task or force_eval_gen):
    max_len = _maybe(env.get("EVAL_GEN_MAX_LENGTH"), int) or 1024
    min_len = _maybe(env.get("EVAL_GEN_MIN_LENGTH"), int) or 5
    num_beams = _maybe(env.get("EVAL_GEN_NUM_BEAMS"), int) or 5  # ← 添加这行
    cfg["eval_gen"] = {
        "max_length": int(max_len),
        "min_length": int(min_len),
        "num_beams": int(num_beams),  # ← 添加这行
    }
```

### 2. 修改 `build_and_run_trainer_gla_only()` 传递 num_beams

```python
# 在创建 eval_generator 时
if eval_gen is not None:
    _eval = dict(eval_gen)
    max_length = int(_eval.get("max_length", 1024))
    min_length = int(_eval.get("min_length", 5))
    num_beams = int(_eval.get("num_beams", 5))  # ← 添加这行
    eval_generator = create_gla_decoder(
        tokenizer,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,  # ← 添加这行
        do_sample=False,
    )
```

### 3. 调整训练超参数

```bash
# 推荐的 DART 训练配置
export HP_LR=0.0002            # 降低学习率
export HP_EPOCHS=10            # 适当增加 epochs
export HP_BATCH_SIZE=8
export EVAL_GEN_NUM_BEAMS=5    # 使用 beam search
export EVAL_GEN_MAX_LENGTH=128 # DART 输出通常不需要 1024
```

---

## 📊 预期改进

| 修复项 | 预期 BLEU 提升 | 预期 chrF 提升 |
|--------|---------------|---------------|
| 添加 Beam Search (num_beams=5) | +10~15 | +10~15 |
| 降低学习率 (2e-4) | +5~10 | +5~8 |
| 增加训练 epochs | +3~5 | +3~5 |
| **总计** | **+18~30** | **+18~28** |

修复后预期：
- BLEU: 10 → 28~40
- chrF: 31 → 49~59
- METEOR: 0.42 → 0.45~0.50

---

## 🔍 额外诊断：添加 Debug 代码

在 `dart_data.py` 的 `compute_metrics()` 中添加：

```python
def compute_metrics(self, eval_preds):
    if self.mode == "gen":
        predictions = getattr(eval_preds, "preds", [])
        references = getattr(eval_preds, "labels", [])
        
        # ===== DEBUG: 检查生成质量 =====
        print("\n[DEBUG][DART] ----- compute_metrics -----")
        print(f"[DEBUG][DART] Total samples: {len(predictions)}")
        print(f"[DEBUG][DART] sep_token used: '{self.sep_token}'")
        
        # 打印前 5 个样本
        for i in range(min(5, len(predictions))):
            print(f"\n[DEBUG][DART] Sample {i}:")
            print(f"  Prediction: {predictions[i][:200]}...")
            refs = references[i] if i < len(references) else "N/A"
            if isinstance(refs, list):
                for j, r in enumerate(refs[:3]):
                    print(f"  Reference {j}: {r[:200]}...")
            else:
                print(f"  Reference: {refs[:200] if isinstance(refs, str) else refs}...")
        
        # 统计空预测
        empty_preds = sum(1 for p in predictions if not p or not p.strip())
        print(f"[DEBUG][DART] Empty predictions: {empty_preds}/{len(predictions)}")
        print("[DEBUG][DART] -----------------------------------------\n")
        # ===== END DEBUG =====
```

---

## 📋 执行清单

1. [ ] 修改 `train_gla_only.py` 支持 `EVAL_GEN_NUM_BEAMS`
2. [ ] 在启动脚本中添加 `export EVAL_GEN_NUM_BEAMS=5`
3. [ ] 降低学习率到 `2e-4`
4. [ ] 添加 debug 代码检查生成质量
5. [ ] 重新运行训练并观察指标变化


