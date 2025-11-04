## GLA/Mamba 批量实验运行手册（2025 版）

本文档介绍如何在本仓库中批量、稳定地运行 GLA/Mamba 的 LoRA 家族实验。涵盖：启动方式、关键环境变量、并发与 GPU 计划、不同 GLUE 数据集的推荐命令、输出路径、断点恢复与聚合报告、常见问题排查。

---

### 1) 快速开始（模板）

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

# 可选的稳定性/性能环境变量（见下文详解）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
NUM_DATA_WORKERS=2 GRADIENT_CHECKPOINTING=true LOGITS_TO_KEEP=1 \
./gla_batch_tmux.sh --suite E4 --round all \
  --pairs "87:glue-tvt_cola" \
  --gpus "2 3 5 6" \
  --gpu-plan "3,3,1,3"
```

说明：
- `--pairs "种子:数据集 ..."` 会在一个 tmux 会话里顺序跑多个作业，每个作业再由核心脚本按 GPU 并发计划切片并行。
- `--gpu-plan` 控制每张 GPU 同时跑几份作业（并发槽位），详见下文“并发与 GPU 计划”。
- 数据集名统一用 `glue-tvt_*`（如 `glue-tvt_cola`、`glue-tvt_qqp`）。

---

### 2) 关键环境变量详解（按用途分组）

— 训练稳定性/内存（推荐）：
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：启用可扩展段，缓解显存碎片。
- `TOKENIZERS_PARALLELISM=false`：避免分词多线程占用过多 CPU/内存。
- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`：限制 BLAS 线程数，避免 CPU 抢占。
- `MALLOC_ARENA_MAX=2`：减少 glibc arena，降低堆内存膨胀。

— DataLoader（CPU 内存与吞吐的平衡）：
- `NUM_DATA_WORKERS`：DataLoader 工作进程数。小数据集建议 2–4；大数据集（QQP/MNLI）建议 1。
- `DATALOADER_PREFETCH_FACTOR`：预取批次数。小集 2；大集 1。
- `DATALOADER_PIN_MEMORY`：是否锁页内存（0/1）。小集 1；大集 0（可显著降内存峰值）。
- `DATALOADER_PERSISTENT_WORKERS`：worker 常驻（0/1）。一般建议 0。
- `EVAL_ACCUMULATION_STEPS`：评估时梯度累积（节省显存/内存）。小集 64–128；大集 32。

— 训练行为与日志（最高优先级覆盖 YAML）：
- `GRADIENT_CHECKPOINTING=true`：启用激活检查点，降显存峰值；仓库内部已统一使用 `use_reentrant=False` 以兼容 LoRA 冻结基座。
- `LOGITS_TO_KEEP`：评估/生成时保留的 logits 数（降低评估内存）。
- `HP_EVAL_STEPS`、`HP_SAVE_STEPS`、`HP_LOGGING_STEPS`：分别控制评估/保存/日志步频（覆盖配置/默认）。
- `HP_BATCH_SIZE`、`HP_LR`、`HP_EPOCHS`、`HP_PREC`（bf16|fp16|fp32）、`HP_SEED`：训练核心超参覆盖。
- `HP_DATA`：训练时强制覆盖数据集（可传 `rte` 或 `glue-tvt_rte`）。
- `SAVE_OPTIMIZER_STATE=1`：默认不保存 `optimizer.pt`/`scheduler.pt`/`rng_state.pth`；设为 1 时恢复保存。

— 追踪（SwanLab，可选）：
- `SWANLAB_ENABLE=1`：启用 SwanLab 回调。
- `SWANLAB_MODE=cloud|local`：运行模式。
- `SWANLAB_PROJECT=<name>`：项目名。
- （可选）`SWANLAB_EXPERIMENT_PREFIX=<prefix>`：给实验名加前缀。

— 启动脚本相关：
- `DATA`：核心脚本会把 `data: <DATA>` 注入到每个 YAML 的临时副本（不改原 YAML）。
- `GPU_IDS`/`--gpus`：参与调度的 GPU 列表（空格或逗号分隔）。
- `GPU_PLAN`/`--gpu-plan`：与 `GPU_IDS` 对齐的并发槽位（单值广播或等长列表）。

提示：训练端内置了“按数据规模自适应”的内存调优（样本数 ≥ 200000 时自动更保守）。当你显式设置了 `DATALOADER_*`/`EVAL_ACCUMULATION_STEPS`/`NUM_DATA_WORKERS` 时，会以你的值为准，不会被自动策略覆盖。

---

### 3) 并发与 GPU 计划

- `--gpus "0 1 2 3"` 表示参与调度的设备是 0/1/2/3。
- `--gpu-plan "3,2,2,1"` 表示每张卡的并发槽位（同时跑几份作业）。
- 也可以只给单个数字（广播给所有 GPU），例如 `--gpu-plan 2`。
- 设置某卡并发为 0 表示“可见但不调度”（通常建议直接从 `--gpus` 移除该卡，以更干净）。

---

### 4) 不同数据集的推荐命令

说明：以下是“稳妥且高效”的经验组合。你可以按机器与任务需求微调 batch/concurrency。小数据集允许更高并发；大数据集优先稳内存。

— 小型数据集（CoLA / MRPC / RTE / SST-2）

推荐：
- `NUM_DATA_WORKERS=2`、`DATALOADER_PREFETCH_FACTOR=2`、`DATALOADER_PIN_MEMORY=1`、`DATALOADER_PERSISTENT_WORKERS=0`、`EVAL_ACCUMULATION_STEPS=64`。
- 可开较高并发（每卡 2–3）。

示例（CoLA）：
```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
NUM_DATA_WORKERS=2 DATALOADER_PREFETCH_FACTOR=2 DATALOADER_PIN_MEMORY=1 DATALOADER_PERSISTENT_WORKERS=0 \
GRADIENT_CHECKPOINTING=true LOGITS_TO_KEEP=1 EVAL_ACCUMULATION_STEPS=64 \
./gla_batch_tmux.sh --suite E4 --round all \
  --pairs "87:glue-tvt_cola" \
  --gpus "2 3 5 6" \
  --gpu-plan "3,3,1,3"
```

— 中等数据集（QNLI）

推荐：
- `NUM_DATA_WORKERS=2`、`DATALOADER_PREFETCH_FACTOR=2`、`DATALOADER_PIN_MEMORY=1`、`DATALOADER_PERSISTENT_WORKERS=0`、`EVAL_ACCUMULATION_STEPS=64`。
- 并发建议每卡 2（根据显存可调）。

示例：
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
NUM_DATA_WORKERS=2 DATALOADER_PREFETCH_FACTOR=2 DATALOADER_PIN_MEMORY=1 DATALOADER_PERSISTENT_WORKERS=0 \
GRADIENT_CHECKPOINTING=true LOGITS_TO_KEEP=1 EVAL_ACCUMULATION_STEPS=64 \
./gla_batch_tmux.sh --suite E4 --round all \
  --pairs "87:glue-tvt_qnli" \
  --gpus "0 1 2 3" \
  --gpu-plan "2,2,2,2"
```

— 大型数据集（QQP / MNLI）

推荐（内存友好优先）：
- `NUM_DATA_WORKERS=1`、`DATALOADER_PREFETCH_FACTOR=1`、`DATALOADER_PIN_MEMORY=0`、`DATALOADER_PERSISTENT_WORKERS=0`、`EVAL_ACCUMULATION_STEPS=32`。
- `GRADIENT_CHECKPOINTING=true`，`LOGITS_TO_KEEP=1`。
- 并发建议每卡 1，保证稳定性；如显存/内存充足可逐步提高。

示例（QQP）：
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
NUM_DATA_WORKERS=1 GRADIENT_CHECKPOINTING=true LOGITS_TO_KEEP=1 \
DATALOADER_PREFETCH_FACTOR=1 DATALOADER_PIN_MEMORY=0 DATALOADER_PERSISTENT_WORKERS=0 EVAL_ACCUMULATION_STEPS=32 \
HP_EVAL_STEPS=5000 HP_SAVE_STEPS=5000 HP_LOGGING_STEPS=1000 \
SWANLAB_ENABLE=1 SWANLAB_MODE=cloud SWANLAB_PROJECT="gla-mamba-qqp" \
./gla_batch_tmux.sh --suite E8 --round 1 \
  --pairs "87:glue-tvt_qqp" \
  --gpus "0 1 2 3 4 5" \
  --gpu-plan "1,1,1,1,1,1"
```

示例（MNLI）：
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
NUM_DATA_WORKERS=1 GRADIENT_CHECKPOINTING=true LOGITS_TO_KEEP=1 \
DATALOADER_PREFETCH_FACTOR=1 DATALOADER_PIN_MEMORY=0 DATALOADER_PERSISTENT_WORKERS=0 EVAL_ACCUMULATION_STEPS=32 \
HP_EVAL_STEPS=5000 HP_SAVE_STEPS=5000 HP_LOGGING_STEPS=1000 \
./gla_batch_tmux.sh --suite E8 --round 1 \
  --pairs "87:glue-tvt_mnli" \
  --gpus "0 1 2 3" \
  --gpu-plan "1,1,1,1"
```

---

### 5) 输出路径与结构

- 默认输出：`/home/user/mzs_h/output/benchmark/glue/<data>_seed<seed>/<yaml_stem>/`。
- 例如：`/home/user/mzs_h/output/benchmark/glue/glue-tvt_sst2_seed87/E1_QKVO_r8_alpha16/`。
- 每个 checkpoint 目录包含 `trainer_state.json` 等；默认不保存 `optimizer.pt`/`scheduler.pt`/`rng_state.pth`（除非 `SAVE_OPTIMIZER_STATE=1`）。

---

### 6) 断点恢复与覆盖

- 批量脚本默认走 `--overwrite`（从头开始）。
- 若需从 checkpoint 恢复，请直接单独运行：
  ```bash
  CUDA_VISIBLE_DEVICES=0 python mamba-peft/train.py --cfg <path-to-cfg-in-output>/cfg.yaml --resume
  ```

---

### 7) 结果聚合与汇总

整合所有实验（按数据集）并产出 CSV/JSON/Markdown/可选图表：

```bash
cd /Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft
python -m aggregate_result.main \
  --base_dir /Users/huangziheng/Documents/zotero附件/transformer改造/research_trackers/glue \
  --output   /Users/huangziheng/Documents/zotero附件/transformer改造/research_trackers/all_agg_results/glue_all \
  --workers 5 --small   # --small 可选：省略 checkpoint_path/cfg_path 列
```

聚合工具会：
- 读取每个 experiment 的 eval 日志/CSV，按 GLUE 官方主指标（含 MNLI m/mm、MRPC/QQP F1+Acc、STS-B Pearson+Spearman、CoLA MCC 等）选取最优 checkpoint；
- 生成 dataset 级汇总表与（可选）图表。

---

### 8) 常见问题排查

- 日志出现 `Killed` 且无 Python Traceback：多为系统级 OOM（主机内存不足）。
  - 立刻止血：`NUM_DATA_WORKERS=1`、`DATALOADER_PREFETCH_FACTOR=1`、`DATALOADER_PIN_MEMORY=0`、`DATALOADER_PERSISTENT_WORKERS=0`、`EVAL_ACCUMULATION_STEPS=32`、并发每卡 1。
  - 评估间隔加大：`HP_EVAL_STEPS/HP_SAVE_STEPS` 取更大值，减少频繁评估/保存开销。
- GPU OOM：减小 `batch_size` 或提高 `gradient_accumulation_steps`（通过 YAML 或 `HP_*` 覆盖）。
- 重复 YAML 同轮冲突：核心脚本已为临时 YAML 自动加唯一后缀（`__repK`），无需手动处理。
- 倒计时与计划：`gla_batch_tmux.sh` 支持 `--sleep-mins`；倒计时结束后打印计划并启动任务。

---

### 9) 建议的稳定组合（速查）

- 小集（CoLA/MRPC/RTE/SST-2）：`workers=2`、`prefetch=2`、`pin=1`、`eval_acc=64`、并发每卡 2–3。
- 中集（QNLI）：`workers=2`、`prefetch=2`、`pin=1`、`eval_acc=64`、并发每卡 2。
- 大集（QQP/MNLI）：`workers=1`、`prefetch=1`、`pin=0`、`eval_acc=32`、并发每卡 1；必要时加大 `HP_*_STEPS`。

---

如需更多定制或排障，请在脚本所在目录运行并查看 `logs/` 下的批量/分步日志，或将 `SWANLAB_ENABLE=1` 打开云端可视化以对比实验细节。



