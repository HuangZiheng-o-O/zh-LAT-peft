## zh-LAT-peft：Commonsense170k 混合训练 + lm_eval 分别评测接入 
 

### 0. 背景与目标


 希望实现 
- **训练**：用一个混合训练集 `commonsense_170k.json`（多任务混合）训练一次（每个 LoRA 配置一次训练）。  
- **评测**：训练结束后，用 `lm-evaluation-harness`（简称 **lm_eval**）对 8 个 task **逐个 task 独立算分**，但可 **一条命令串行跑完一串 task**（`--tasks boolq,piqa,...`）。  
  参考：[`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)

同时你的工程诉求是：

- **每张 GPU 跑一个 job slot**（或每张 GPU 跑 N 个 slot），每个 slot 对应 **一个 LoRA 配置**；训练完立刻在该 GPU 上跑评测，形成 “train→eval” 的闭环。
- **不使用 dataset-pairs 并行模式**（例如 `LAT_BATCH_PAIR_CONCURRENCY=auto` 这种用于“多 dataset 冒烟测试”的模式），而是固定一个数据集：`commonsense_170k`。

---

### 1. 本轮改动总览（你关心的点一网打尽）

本轮改动分成 5 类：

1) **commonsense_170k 混合训练接入**
- 新增 dataset：`mamba-peft/dataset/commonsense_170k.py`
- `HP_DATA=commonsense_170k` 即可走你原有训练链路（不需要新训练入口）

2) **lm_eval 评测接入（融入现有 train→eval 流程）**
- 仍然由 `lat_round.sh` 调 `eval_lat.py`，但 `eval_lat.py` 新增 **backend** 选择：
  - `EVAL_BACKEND=lat`：走你内部 dataset/* 的评测
  - `EVAL_BACKEND=lm_eval`：调用 `scripts/eval/lat_lm_harness_eval.py` 走 lm_eval（标准 benchmark 风格）
- `lat_batch_tmux.sh` / `lat_round.sh` 支持透传 `--eval-backend`（或 env）

3) **“1 is not in list” 评测崩溃：根因定位 + 严谨修复**
- 根因：label tokenization 混入了 tokenizer special tokens（常见 BOS=1），以及旧缓存仍被加载导致 label_ids 异常
- 修复：统一 `add_special_tokens=False` + cache 版本化 + 修复 `get_cache_name()` 覆盖导致的版本后缀丢失

4) **Checkpoint 策略：避免磁盘爆炸 + 论文风格（不作弊）**
- 默认策略：**只保留 last+best 两个 checkpoint**，且 best 只用 **val** 指标（默认 `eval_loss`）选择
- 训练结束写入稳定的 “final(best)” adapter 到 output_dir 根目录，便于 eval 自动定位

5) **断点续训（resume）优雅接入**
- `lat_batch_tmux.sh` 新增 `--resume/--overwrite`，通过 env 传到 `lat_round.sh`，再转成 `train_lat.py --resume`  
- 默认仍是 `overwrite`（保持你历史行为），但你可一键切换为 resume

---

### 2. 关键稳定性问题：`ValueError: 1 is not in list` 的严谨分析与修复

#### 2.1 现象

你跑 BoolQ/HellaSwag/WinoGrande/ARC/OpenBookQA 等时，在 `compute_metrics()` 中经常看到：

- `references_ind = [self.choice_ids.index(r) for r in references]`
- 崩溃：`ValueError: 1 is not in list`

#### 2.2 根因（两层）

**根因 A：label tokenization 被注入 special tokens（最常见 BOS=1）**

你原来（旧逻辑）使用 `tokenizer.encode(text)` 默认 `add_special_tokens=True`，会把 BOS/EOS 等注入 token 序列。  
对于“单 token 选项分类”任务，label 本应是单 token（如 `A/B/C/D` 或 `0/1`），一旦变成 `[BOS, label]`，则 label_ids 的第一个 token 就可能是 1，导致 `.index(1)` 不存在。

**根因 B：旧 cache 仍在被 FAST-PATH 加载**

即使你修了 encode 逻辑，如果数据预处理结果被写进 `.pkl` cache，后续训练/评测走：

- `FAST-PATH: Loading from cache...`

那么你实际用的是“旧格式 label_ids”，仍然会崩。

此外还有一个隐藏坑：

**根因 C：部分 dataset 重写了 `get_cache_name()`，导致 cache 版本后缀继承链断裂**

如果 dataset 重写 `get_cache_name()` 而不调用 `super()`，即使 base 层加了 cache format version，也会失效。

#### 2.3 修复点（你可以直接定位到代码）

**(1) 全局禁止 encode 注入 special tokens**

- 文件：`mamba-peft/dataset/base.py`
- 逻辑：`DatasetBase.encode()` 固定使用 `add_special_tokens=False`

**(2) cache format 版本化（并可通过 env 强制重建）**

- 文件：`mamba-peft/dataset/base.py`
- 关键变量：
  - `_BASE_CACHE_FORMAT_VERSION = "fmt3_nospecial"`
  - `_CACHE_FORMAT_VERSION = fmt3_nospecial[-$LAT_CACHE_FORMAT_VERSION]`
- cache 文件名会带后缀：`...__fmt3_nospecial-fmt4.pkl`

你可以通过：

```bash
export LAT_CACHE_FORMAT_VERSION=fmt4
```

强制让所有 cache 重新构建（无需手动删文件）。

**(3) 修复所有新 dataset 的 `get_cache_name()` 继承链**

例如 `winogrande.py/openbookqa.py/arc.py/social_iqa.py/hellaswag.py` 等重写 `get_cache_name()` 的地方，都改为优先调用 `super().get_cache_name()`，避免丢失 base 层的 cache 版本后缀。

---

### 3. commonsense_170k 混合训练接入（路线 A：复用你已存在的 JSON）

#### 3.1 数据格式确认

你本地已有文件：

- `commonsense_170k_data/commonsense_170k.json`

它是一个 JSON array，每条样本包含：

- `instruction`: 任务指令（含 BoolQ 之类的自然语言提示）
- `input`: 可为空字符串
- `output`: 训练目标（例如 `"the correct answer is true"`）
- `answer`: 结构化答案（例如 `"true"`；参考项目会保留，但训练用 `output`）

#### 3.2 训练数据集实现

- 新增文件：`mamba-peft/dataset/commonsense_170k.py`
- 设计目标：
  - prompt 模板尽量对齐参考项目 `reference/MambaPEFT/.../finetune.py::generate_prompt()`
  - 训练时让模型学习生成 `output`
  - 内置 train/val 切分逻辑（默认 `val_set_size=2000`，对齐参考实现）

关键环境变量：

- `LAT_COMMONSENSE_170K_PATH`：显式指定 `commonsense_170k.json` 路径（默认会读仓库内那个）
- `LAT_COMMONSENSE_170K_VAL_SET_SIZE`：验证集大小（默认 2000）

#### 3.3 接入你的 dispatcher

文件：`mamba-peft/dataset/__init__.py`

新增分支：

- `HP_DATA=commonsense_170k` 或 `--pairs "87:commonsense_170k"` 即可走 `Commonsense170kDataModule`

---

### 4. lm_eval 评测：优雅融入你现有 “train→eval” 流程

#### 4.1 为什么仍需要 `eval_lat.py`（而不是直接让脚本调 lm_eval）

你强调 “不要另起评测加载机制”，所以：

- 训练后评测必须仍由 `lat_round.sh` 控制（同 GPU、同 cfg 注入、同 output_dir 规则）
- `eval_lat.py` 是你训练链路旁的统一评测入口，最适合做 “backend switch”

#### 4.2 两条评测路径

**Path A：backend=lat（内部 dataset/* 评测）**

- 优点：可以更严格地走你自己的 `LAT_DATA_DIR` 本地 repo 优先策略
- 缺点：任务 prompt/metric 可能与社区标准实现存在差异（做 benchmark 时可比性弱）

**Path B：backend=lm_eval（标准 benchmark 评测）**

- 参考：[`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
- 特点：`--tasks boolq,piqa,...` 会逐个 task 跑、逐个 task 出指标（最后可能有汇总）
- 数据集走 HF datasets cache（你说这也属于“本地优先”，不要求严格从 mamba-peft/data 读）

#### 4.3 实现细节

**(1) lm_eval wrapper：`scripts/eval/lat_lm_harness_eval.py`**

- 注册模型：`@register_model("LAT")`
- 内部加载模型必须走你的：
  - `prepare_lat_model_and_tokenizer(model_type=..., model_id=..., prec=...)`
  - `attach_peft_weights(model, peft_dir)`
- 兼容性设置：
  - `tokenizer.padding_side="left"`
  - `pad_token_id` 为空则设为 `eos_token_id`
  - `model.to(device)` + `model.eval()`
  - `model.config.use_cache=False`

**(2) eval_lat backend switch：`mamba-peft/eval_lat.py`**

新增参数：

- `--backend lat|lm_eval`
- 或 env：`EVAL_BACKEND=lat|lm_eval`

当 backend=lm_eval 时：

- `eval_lat.py` 会调用 wrapper 脚本（subprocess）去跑 harness
- 输出目录仍然统一落在 `EVAL_OUTPUT_ROOT` 下

**(3) 融入批量脚本**

文件：`mamba-peft/scripts/train/new/lat_round.sh`

- eval 命令自动追加：`--backend $EVAL_BACKEND`

文件：`mamba-peft/scripts/train/new/lat_batch_tmux.sh`

- 支持 CLI：`--eval-backend lm_eval`
- 并把 `EVAL_BACKEND` 透传到 tmux runner 环境

---

### 5. checkpoint 策略（避免磁盘爆炸 + 不作弊）

你提出的两个方案，本质是：

- “只保留最新”：省空间，但无法回溯最优点
- “只保留 best + last”：这是更标准/更科学的工程折中

这里的关键在于 **best 的选择必须只使用验证集（val）**，不能用 benchmark task 的 test 分数反向挑 checkpoint，否则会被认为“用测试集调参”（不规范）。

#### 5.1 我实现的默认策略（推荐）

文件：`mamba-peft/train_lat.py`

- `save_total_limit=2`：默认只保留 2 个 checkpoint（last + best）
- `load_best_model_at_end=True`：训练结束加载 best checkpoint
- `metric_for_best_model=eval_loss`：默认用验证集 loss 选 best（最稳、最不争议）
- 训练结束会把 **best adapter** 保存到 `output_dir` 根目录，方便评测自动定位

你可通过 env 覆盖：

- `HP_SAVE_TOTAL_LIMIT`：保留多少个 checkpoint
- `HP_LOAD_BEST_MODEL_AT_END=1/0`
- `HP_METRIC_FOR_BEST_MODEL`（默认 `eval_loss`）
- `HP_GREATER_IS_BETTER`（eval_loss 用 0；accuracy 用 1）

#### 5.2 为什么还改了 `GenericLMTrainer.save_model`

文件：`mamba-peft/trainer/generic_lm_trainer.py`

之前 `save_model()` 只在 `save_full_model=True` 时保存 `model.pt`，这对 PEFT 不友好：  

1) **checkpoint 目录里缺少 `adapter_config.json / adapter_model.*`**，导致：
   - `eval_lat.py` 很难“自动定位可用 adapter”
   - `--resume` 时也无法可靠恢复

2) 你不想保存 full base model（太大），只想保存 adapter（小很多）

所以我们把 `GenericLMTrainer.save_model()` 改成：

- 若 `save_full_model=True`：保持旧行为，保存 full 模型快照（`model.pt`）
- 否则：优先调用 `transformers.Trainer.save_model()`（底层会走 `model.save_pretrained()`）
  - 对 **PeftModel** 来说，`save_pretrained()` 会保存 **adapter 文件**（这是我们需要的）
  - 如果上层失败，再 fallback 到 `state_dict` 保存（兜底）

#### 5.3 “每次保存就删上一次” vs “保留 best+last” 的取舍

你提的“每次存储完删上一次”≈ `save_total_limit=1`，优点是极省空间，但缺点是：

- 你无法回溯最佳 checkpoint（特别是 loss 曲线波动时）
- resume 的可恢复性也更差（只有一个点）

更标准/更科学的折中是：

- `save_total_limit=2`：保留 **last + best**
- `load_best_model_at_end=True`：训练结束自动切到 best，然后写入 output_dir 根目录（最终用于评测）

这套策略是很多论文/工业训练脚本的常见默认（不作弊，因为只用 val 选 best）。

---

### 6. 断点续训（resume）优雅接入

你之前关掉 resume 的主要原因是：

- 脚本层默认一直 `--overwrite`（训练输出目录存在就会被当成“必须重跑”）
- checkpoint 保存策略不稳定（要么爆磁盘，要么缺 adapter 文件）

本轮做了两处“融入式”的接入：

#### 6.1 `lat_batch_tmux.sh` 支持 `--resume/--overwrite`

文件：`mamba-peft/scripts/train/new/lat_batch_tmux.sh`

新增 CLI：

- `--resume`：启用续训（并自动关闭 overwrite）
- `--overwrite`：强制覆盖（默认行为）

它会透传两个 env 到 runner：

- `LAT_TRAIN_RESUME=1/0`
- `LAT_TRAIN_OVERWRITE=1/0`

#### 6.2 `lat_round.sh` 按 env 决定是否给 `train_lat.py` 传 `--resume`

文件：`mamba-peft/scripts/train/new/lat_round.sh`

- `LAT_TRAIN_RESUME=1` → `_cmd += --resume`
- 否则默认 `_cmd += --overwrite`（保持你历史行为不变）

#### 6.3 “resume 是否科学”的关键：是否保存 optimizer 状态

如果你希望严格恢复（学习率调度、动量、随机数等都一致），建议：

```bash
export SAVE_OPTIMIZER_STATE=1
```

说明：

- 我们的 Trainer 支持“是否保存 optimizer/scheduler/rng_state”可控
- 不保存 optimizer 状态可以省盘，但 resume 的严格性会下降（工程上仍可用，学术上略逊）

---

### 7. 如何在你的框架里“丝滑”跑：Commonsense170k 训练一次 + lm_eval 评测 8 tasks

下面给你一份 **全变量齐全版**（可直接复制跑）。  
注意：这是你强调的模式——**固定单数据集 `commonsense_170k`**，不使用 dataset-pairs 并行模式（不设置 `LAT_BATCH_PAIR_CONCURRENCY`）。

#### 7.1 依赖安装（一次性）

```bash
pip install lm-eval
# 或（如果你需要跟随仓库主线）
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
```

#### 7.2 训练 + 训练后评测（推荐：best+last + 不作弊）

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

# ===== model =====
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

# ===== output layout（可选但推荐：把 commonsense 任务单独分组）=====
export LAT_OUTPUT_ROOT=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/output/benchmark
export LAT_DATASET_ROOT_NAME=commonsense

# ===== HF cache + offline（推荐）=====
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ===== commonsense_170k.json（仓库内默认可读；显式指定也可以）=====
# export LAT_COMMONSENSE_170K_PATH=/home/user/mzs_h/code/zh-LAT-peft/commonsense_170k_data/commonsense_170k.json
export LAT_COMMONSENSE_170K_VAL_SET_SIZE=2000

# ===== dataset cache（建议放到大盘，避免写爆系统盘）=====
export LAT_DATA_CACHE_DIR=/home/user/mzs_h/data/lat_cache

# ===== train hparams =====
export EVAL_GEN=0
export HP_VAL_SPLIT=val

export HP_EPOCHS=3
export HP_BATCH_SIZE=8
export HP_LR=0.0003

# 训练中间 eval/save 频率（建议对齐）
export HP_EVAL_BATCH_SIZE=8
export HP_EVAL_STEPS=1000
export HP_SAVE_STEPS=1000
export HP_LOGGING_STEPS=100

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

# ===== checkpoint policy（关键：防爆磁盘 + best+last）=====
export HP_SAVE_TOTAL_LIMIT=2
export HP_LOAD_BEST_MODEL_AT_END=1
export HP_METRIC_FOR_BEST_MODEL=eval_loss
export HP_GREATER_IS_BETTER=0

# ===== resume 严格性（可选）=====
export SAVE_OPTIMIZER_STATE=0   # 盘紧张就 0；想严格 resume 就 1

# ===== dataloader / perf =====
export NUM_DATA_WORKERS=4
export DATALOADER_PREFETCH_FACTOR=2
export DATALOADER_PIN_MEMORY=1
export DATALOADER_PERSISTENT_WORKERS=0
export GRADIENT_CHECKPOINTING=true

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

# ===== SwanLab（可选）=====
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-commonsense170k-lora-batch"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

# ===== eval：训练后用 lm_eval 跑 8 tasks（逐 task 单独算分，一条命令跑完）=====
export EVAL_AFTER_TRAIN=1
export EVAL_BACKEND=lm_eval
export EVAL_TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'
export EVAL_BATCH_SIZE=64
export EVAL_OUTPUT_ROOT=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/outputs/lm_eval

# ===== cache 格式版本（遇到旧 cache 崩溃就 bump）=====
export LAT_CACHE_FORMAT_VERSION=fmt4

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:commonsense_170k" \
  --gpus "0 1 2 3 4 5 6 7" \
  --gpu-plan "1,1,1,1,1,1,1,1" \
  --model-type delta_net \
  --eval-after-train \
  --eval-backend lm_eval \
  --eval-tasks "$EVAL_TASKS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-output-root "$EVAL_OUTPUT_ROOT"
```

#### 7.3 断点续训（resume）

当某个输出目录已经存在且包含 `checkpoint-*` 时，你可以直接：

```bash
./lat_batch_tmux.sh ... --resume
```

建议：

- 如果你要严格续训：`SAVE_OPTIMIZER_STATE=1`
- 如果你只是工程上想接着跑完：`SAVE_OPTIMIZER_STATE=0` 也可以

---

### 8. 常见问题与排错清单（非常实用）

#### 8.1 仍然出现 “1 is not in list”

90% 是旧 cache 仍被加载：

- 看日志是否出现 `FAST-PATH: Loading from cache...`
- 解决：`export LAT_CACHE_FORMAT_VERSION=fmt5`（随便 bump 一个新值）

#### 8.2 checkpoint 目录没有 adapter 文件

确认你在跑的是 PEFT（LoRA）配置，并且：

- `GenericLMTrainer.save_model()` 已经按 HF/PEFT 语义保存（本轮已修）
- `train_lat.py` 训练结束会把 best adapter 写到 output_dir 根目录（本轮已修）

#### 8.3 “磁盘爆炸”

优先检查：

- `HP_SAVE_TOTAL_LIMIT`（推荐 2）
- `HP_SAVE_STEPS` 是否过于频繁
- `SAVE_OPTIMIZER_STATE` 是否不必要地开启

---

### 9. 参考与对齐说明

- lm_eval 官方仓库：[`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
- 参考项目思路：`reference/MambaPEFT/language/commonsense_reasoning`
  - 训练：`commonsense_170k.json` 混合训练
  - 评测：`TASKS=...` 一条命令跑完多个 task，逐 task 独立打分

---

### 10. 文件级变更清单（便于你 code review）

#### 10.1 数据集与缓存
- `mamba-peft/dataset/base.py`：encode 禁用 special tokens；cache format 版本化；debug 打印
- `mamba-peft/dataset/*`（boolq/piqa/social_iqa/hellaswag/winogrande/openbookqa/arc）：choice_ids 单 token 严格化；cache name 继承链修复
- `mamba-peft/dataset/commonsense_170k.py`：混合训练集接入（json 读入 + train/val split）
- `mamba-peft/dataset/__init__.py`：支持 `commonsense_170k`

#### 10.2 评测
- `mamba-peft/scripts/eval/lat_lm_harness_eval.py`：lm_eval wrapper（复用 lat_adapter）
- `mamba-peft/eval_lat.py`：新增 `--backend` / `EVAL_BACKEND`，并支持 lm_eval 后端

#### 10.3 训练与保存
- `mamba-peft/train_lat.py`：save_total_limit、load_best_model_at_end、metric_for_best_model；训练结束保存 best adapter 到 output_dir
- `mamba-peft/trainer/generic_lm_trainer.py`：save_model 改成默认走 HF/PEFT save_pretrained 语义

#### 10.4 批量脚本
- `mamba-peft/scripts/train/new/lat_batch_tmux.sh`：新增 `--eval-backend`、`--resume/--overwrite` 并透传 env
- `mamba-peft/scripts/train/new/lat_round.sh`：训练命令按 resume/overwrite 选择；评测命令透传 backend

