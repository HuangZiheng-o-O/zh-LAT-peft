# GLUE/RTE 评估与训练参数传递修复记录

更新时间：2025-11-30

本文档记录近期在运行 GLUE（特别是 RTE）任务时遇到的评估偏低、环境变量未生效等问题，给出根因分析、代码改动、使用方式与验证要点。适用于 `zh-LAT-peft/mamba-peft` 的 clean GLA 路线。

---

## 1. 现象与影响

- 在 RTE（小样本二分类）上，验证集精度经常徘徊在 0.52–0.57，`eval/out_of_cls` 指标接近样本总数，`eval/loss` 却很低（约 0.23），与低精度不一致。
- 部分 HP\_\* 环境变量（如 `HP_VAL_SPLIT`、`HP_LR/HP_EPOCHS/HP_BATCH_SIZE/HP_EVAL_BATCH_SIZE` 等）在多卡/批量脚本链路中未完全传入 Python，导致运行时步数、学习率峰值与预期不一致。
- 由于默认验证集拆分为 `train(80%)/val(20%)`，评估并非严格使用官方 dev；学术合规性与外部可比性受到影响。

影响：RTE 验证精度被系统性低估；参数调度与日志可审计性变差；最终报告不便与标准 GLUE 结果对齐。

---

## 2. 根因分析

### 2.1 评估逻辑默认“全词表 argmax”导致 `out_of_cls` 爆炸

GLUE 的默认评估路径是 `eval_all_logits=True`：在“全词表”维度做 `argmax`，再只把落在类 token（`'0'/'1'`，RTE 为 2 类）的位置记为有效样本，其他 token 统统记为 `out_of_cls` 丢弃。对于 RTE/MRPC/CoLA 等小样本任务，如果模型尚未牢固学习在标签位置输出 `'0'/'1'`，大量预测会落到非类 token，导致 `out_of_cls` 极高、精度被明显低估。

> 标准的分类评估应在“类 token 子集”上计算 argmax（闭集评估），而不是在全词表上拼极值。

### 2.2 类别 token 解析与对齐的脆弱性

- 旧实现依赖 `tokenizer.vocab['0'/'1']`，未显式验证 `'0'/'1'` 是否为“单 token”。个别分词器可能将数字拆成多 token，从而导致类别映射与标签位置不一致。
- 评估使用的 logits 必须与 label\_ids 对齐到同一位置（仅对不为 `-100` 的位置计算）；若标签位置或 token 列表与评估取值不一致，会出现“`eval/loss` 很低但 `accuracy` 很低”的假象。

### 2.3 环境变量传递链路不完整

部分关键环境变量未通过 `gla_batch_tmux_clean.sh → gla_round_clean.sh → train_gla_only.py` 全链路透传，造成：

- `HP_VAL_SPLIT` 未被打印/难以审计；
- `HP_LR/HP_EPOCHS/HP_BATCH_SIZE/HP_EVAL_BATCH_SIZE` 等未必覆盖 YAML 导致运行时参数与期望不一致。

---

## 3. 修复与改动

### 3.1 GLUE 评估默认改为“闭集二分类”

- `dataset/glue.py`：`GlueDataset(eval_all_logits=None)`，默认从环境变量 `GLUE_EVAL_ALL_LOGITS` 解析；若未设置，则默认关闭“全词表 argmax”，统一改为：只在类 token 集（`'0'/'1'/(’2’)`）上取 `argmax` 进行分类评估。
- 添加 `in_class_rate` 诊断指标：表示“全词表 argmax 落在类 token 集的比例”，用于判断模型是否已学会把类 token logit 顶到最高；闭集评估仍会输出该指标以便对照。
- 若显式需要旧行为（不推荐，仅做对照），可手动开启：`export GLUE_EVAL_ALL_LOGITS=1`。

### 3.2 强制单 token 类别映射，避免“静默回退”

- 构建类别 token 时，统一使用 `tokenizer.encode("0/1/2", add_special_tokens=False)` 并**强制单 token**。如果任一类编码为多 token，**直接抛错**，提示需更换分词器或自行实现“多 token 类别打分”。
- 不再在评估中“静默回退”到全词表模式，避免把对齐问题掩盖成结果下降。

### 3.3 环境变量透明传递与日志可审计

- `scripts/train/new/gla_batch_tmux_clean.sh` 补充注入：
  - `HP_VAL_SPLIT`、`HP_LR/HP_EPOCHS/HP_BATCH_SIZE/HP_EVAL_BATCH_SIZE/HP_NO_SAVE`、`GLUE_EVAL_ALL_LOGITS`、`GLUE_DATASET_ID`、`GLUE_METRIC_DIR`、`HF_EVALUATE_LOCAL_GLUE_DIR`。
- `scripts/train/new/gla_round_clean.sh` 的 `ENV_OVERRIDES` 打印中加入：
  - `HP_VAL_SPLIT`、`GLUE_EVAL_ALL_LOGITS`（以及所有 HP\_\* 训练超参）。
- `train_gla_only.py`：继续读取 `HP_VAL_SPLIT` 覆盖 `cfg["val_data_split"]`；评估集 `split="test"` 映射为官方 `validation`（dev），用于严格的模型选择。

---

## 4. 推荐运行方式（RTE 示例）

```bash
export HP_VAL_SPLIT=test            # 用官方 dev 选 checkpoint（学术合规）
export GLUE_EVAL_ALL_LOGITS=0       # 显式关闭全词表 argmax，走闭集二分类评估

conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export EVAL_GEN=0
export HP_EPOCHS=3
export HP_BATCH_SIZE=8
export HP_LR=0.00005
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=100
export HP_SAVE_STEPS=400
export HP_LOGGING_STEPS=50
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.10
export NUM_DATA_WORKERS=4
export DATALOADER_PREFETCH_FACTOR=2
export DATALOADER_PIN_MEMORY=1
export DATALOADER_PERSISTENT_WORKERS=0
export GRADIENT_CHECKPOINTING=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

./gla_batch_tmux_clean.sh \
  --suite E15 \
  --round all \
  --pairs "87:glue-tvt_rte" \
  --gpus "0" \
  --gpu-plan "1"
```

> 说明：`glue-tvt_*` 数据名约定中，`-tvt` 表示“具有官方 test/validation 切分能力”；当 `HP_VAL_SPLIT=test` 时，内部会将 `split="test"` 映射到官方 `validation`，实现“用 dev 选最优 checkpoint”。

---

## 5. 验证与排错清单

1) 环境变量是否生效（日志）  
查看 `ENV_OVERRIDES`：应包含
`HP_VAL_SPLIT=test`、`GLUE_EVAL_ALL_LOGITS=0`、所有 `HP_*` 超参。

2) 输出配置是否一致（输出目录）  
`<output_dir>/cfg.yaml` 中应有：`val_data_split: test`、`learning_rate / num_epochs / batch_size` 等最终值。

3) 指标是否合理  
- 闭集评估下，`eval/out_of_cls` 应为 0；`eval/in_class_rate` 应较高（>0.7 更健康）。
- 若 `accuracy` 仍接近随机，重点检查：
  - `tokenizer.encode('0'/'1')` 是否为**单 token**（现已强制，否则抛错）；
  - `label_ids` 是否正好仅在类别位置非 `-100`；
  - 训练超参是否过大/过小（小数据建议 `LR∈[2e-5,5e-5]`，`epoch∈[2,4]`，`warmup_ratio≈0.06–0.10`）。

4) 学术合规性  
- 使用 `HP_VAL_SPLIT=test` → dev 选最优 checkpoint；严禁“看 test 选最优”。
- 若需要 test 指标，仅在选定的单个 checkpoint 上“test-once”。

---

## 6. 变更清单（摘要）

- `dataset/glue.py`
  - 默认从 `GLUE_EVAL_ALL_LOGITS` 解析评估模式，未设置时默认关闭“全词表 argmax”；改为闭集评估。
  - 类 token 构建：使用 `tokenizer.encode("0/1/2", add_special_tokens=False)` 并**强制单 token**；如不满足直接报错。
  - 增加 `in_class_rate` 诊断指标；闭集评估下 `out_of_cls=0`。
- `scripts/train/new/gla_batch_tmux_clean.sh`
  - 增加透传：`GLUE_EVAL_ALL_LOGITS / GLUE_DATASET_ID / GLUE_METRIC_DIR / HF_EVALUATE_LOCAL_GLUE_DIR`；补齐若干 `HP_*`。
- `scripts/train/new/gla_round_clean.sh`
  - 在 `ENV_OVERRIDES` 中新增：`HP_VAL_SPLIT / GLUE_EVAL_ALL_LOGITS / HP_EVAL_BATCH_SIZE` 等，便于审计。
- `train_gla_only.py`
  - 保持：读取 `HP_VAL_SPLIT` 并映射 `split="test"→validation(dev)` 的行为，用于 dev 选 checkpoint。

---

## 7. FAQ

- Q：这会影响 MRPC/CoLA 等任务吗？  
  A：不会。它们的类别 token 通常是单 token；闭集评估更符合标准分类头语义。若分词器极端导致标签非单 token，会在启动时显式报错，促使修正分词器或实现“多 token 类别打分”。

- Q：为什么不在评估时“自动回退到全词表模式”？  
  A：已知全词表评估在小样本 + 文本生成式标签时会严重低估分类性能，且会掩盖 token 对齐/分词问题。为保证结果可解释与可比性，明确禁止静默回退。

- Q：还能强制旧的全词表评估吗？  
  A：可以手动 `export GLUE_EVAL_ALL_LOGITS=1`，但仅建议用于对照与定位问题，不建议用于对外汇报。

---

如需在 RTE 上加入“多 token 类别打分”（label span 的 log-prob 求和）以兼容更特殊的分词器，请告知，我可以在 `compute_metrics` 中扩展该路径，并保证与闭集评估一致。***
















