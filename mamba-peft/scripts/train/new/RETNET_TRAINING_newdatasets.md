## Run Guide (LAT Framework) — RetNet 版（Commonsense / MambaPEFT eval 任务）

本文件用于你新加入的 commonsense 评测任务：

- BoolQ
- PIQA
- SocialIQA (`social_iqa`)
- HellaSwag (`hellaswag`)
- WinoGrande (`winogrande`, 默认用 `winogrande_xl`)
- ARC-Easy / ARC-Challenge (`arc_easy` / `arc_challenge`)
- OpenBookQA (`openbookqa`, 默认用 `main`)

> 这些任务在本仓库里按 **“单 token 选项分类”**做训练/评测（`EVAL_GEN=0`）。  
> 评测指标是 `accuracy`（`dataset/*` 内的 `compute_metrics()`）。

---

### 1) 数据集规模（HF 默认配置，常用统计）

| task | train | val(dev) | test | 备注 |
| --- | ---:| ---:| ---:| --- |
| `boolq` | 9,427 | 3,270 | — | SuperGLUE BoolQ；建议 `HP_VAL_SPLIT=val` |
| `piqa` | 16,113 | 1,838 | 3,084 | PIQA；本仓库建议用 `val` 做带 label 评测 |
| `social_iqa` | 33,410 | 1,954 | 2,224 | SocialIQA；建议 `val` |
| `hellaswag` | 39,905 | 10,042 | 10,003 | HellaSwag；lm_eval 常用 `validation` |
| `winogrande` (xl) | 40,398 | 1,267 | 1,767 | WinoGrande；lm_eval 常用 `winogrande_xl` |
| `arc_easy` | 2,251 | 570 | 2,376 | ARC-Easy |
| `arc_challenge` | 1,119 | 299 | 1,172 | ARC-Challenge |
| `openbookqa` (main) | 4,957 | 500 | 500 | OpenBookQA |

> 说明：不同 HF repo / config 可能会有轻微差异；上表是 lm_eval 常用/默认拆分规模。

---

### 2) 超参数怎么改（你最关心的几项）

- **`EVAL_GEN=0`**  
  - 这些任务不是“生成长文本”，而是 **在 `Answer:` 后预测一个选项 token**（如 `A/B/C/D` 或 `0/1`），所以用 `0`。

- **`HP_VAL_SPLIT=val`**  
  - 你在训练脚本里设置它，等价于告诉 Trainer：用哪个 split 做 `trainer.evaluate()`。
  - commonsense 任务一般建议用 `val`（更稳：大多数 benchmark 的 test 不是给你调参用的）。

- **`HP_EPOCHS / HP_BATCH_SIZE / HP_LR`（最核心）**  
  - 你的框架里 `total_steps = epochs * ceil(len(train)/batch_size)`  
  - 一套非常稳的起点（4090/1.3B/LoRA）：
    - `HP_BATCH_SIZE=8`
    - `HP_LR=3e-4 ~ 4e-4`
    - `HP_EPOCHS=3~5`（大数据集取 3，小数据集可以取 5~10）

- **`HP_EVAL_STEPS / HP_SAVE_STEPS / HP_LOGGING_STEPS`**  
  - 你可以继续沿用 GLUE 那套（`200/800/50`），但更推荐按 “每 epoch 评测 1~2 次” 来设：
    - 先估算：`steps_per_epoch = ceil(train_size / HP_BATCH_SIZE)`
    - 经验：`HP_EVAL_STEPS ≈ steps_per_epoch`（每 epoch 一次），`HP_SAVE_STEPS ≈ 2*steps_per_epoch` 或等于 eval_steps
  - 如果你想保持和 GLUE 一样频率，也可以直接固定：
    - 小数据（<10k）：`HP_EVAL_STEPS=100~200`
    - 中数据（10k~30k）：`HP_EVAL_STEPS=300~600`
    - 大数据（>30k）：`HP_EVAL_STEPS=800~1500`

- **`LR_SCHEDULER_TYPE=cosine` + `LR_WARMUP_RATIO=0.1`**  
  - 沿用你现有经验即可，commonsense 任务同样适用。

- **`NUM_DATA_WORKERS=4`**  
  - 你现在的配置是合理起点；数据较小也可以降低到 2。

---


### 3) Deltnet 训练模板（correct）

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

# ===== model =====
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

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

# （可选）如果你把数据集用 hf download 放在这里
export LAT_DATA_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data

# ===== task / train hparams =====
export EVAL_GEN=0
export HP_VAL_SPLIT=val   # BoolQ 没有带 label 的 test，建议用 val

export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0004

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=200
export HP_SAVE_STEPS=800
export HP_LOGGING_STEPS=50

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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

# ===== SwanLab（可选，和你cola那套一致）=====
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-boolq-2-4090"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

# ===== launch =====
./lat_batch_tmux.sh \
  --suite E11 \
  --round all \
  --pairs "87:boolq" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type delta_net

```
### 3) RetNet 训练模板（通用）

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=retnet
export LAT_MODEL=/home/user/mzs_h/model/retnet-1.3B-100B/
export LAT_PREC=bf16

# （推荐）固定HF cache + 离线
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# （可选）如果你用 hf download --local-dir 放在这里
export LAT_DATA_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data

export EVAL_GEN=0
export HP_VAL_SPLIT=val

export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0004

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=500
export HP_SAVE_STEPS=1000
export HP_LOGGING_STEPS=100

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

export NUM_DATA_WORKERS=4
export GRADIENT_CHECKPOINTING=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
```

---

### 4) 每个数据集的推荐起点（直接可跑）

> 下面只列出关键差异；其余 env 复用上面模板即可。

#### 4.1 BoolQ（9.4k）

```bash
export HP_EVAL_STEPS=200
export HP_SAVE_STEPS=800
export HP_LOGGING_STEPS=50

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:boolq" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type retnet
```

#### 4.2 PIQA（16k）

```bash
export HP_EVAL_STEPS=400
export HP_SAVE_STEPS=1200
export HP_LOGGING_STEPS=100

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:piqa" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type retnet
```

#### 4.3 SocialIQA（33k）

```bash
export HP_LR=0.0003
export HP_EPOCHS=3
export HP_EVAL_STEPS=1000
export HP_SAVE_STEPS=2000
export HP_LOGGING_STEPS=100

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:social_iqa" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type retnet
```

#### 4.4 HellaSwag（40k）

```bash
export HP_LR=0.0003
export HP_EPOCHS=3
export HP_EVAL_STEPS=1200
export HP_SAVE_STEPS=2400
export HP_LOGGING_STEPS=100

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:hellaswag" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type retnet
```

#### 4.5 WinoGrande XL（40k）

```bash
export HP_LR=0.0003
export HP_EPOCHS=3
export HP_EVAL_STEPS=1200
export HP_SAVE_STEPS=2400
export HP_LOGGING_STEPS=100

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:winogrande" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type retnet
```

#### 4.6 ARC-Easy（2.2k，小数据，易过拟合）

```bash
export HP_LR=0.0002
export HP_EPOCHS=8
export HP_EVAL_STEPS=100
export HP_SAVE_STEPS=200
export HP_LOGGING_STEPS=50

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:arc_easy" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type retnet
```

#### 4.7 ARC-Challenge（1.1k，更小，更易过拟合）

```bash
export HP_LR=0.0002
export HP_EPOCHS=10
export HP_EVAL_STEPS=80
export HP_SAVE_STEPS=160
export HP_LOGGING_STEPS=40

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:arc_challenge" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type retnet
```

#### 4.8 OpenBookQA（5k）

```bash
export HP_LR=0.0003
export HP_EPOCHS=6
export HP_EVAL_STEPS=150
export HP_SAVE_STEPS=300
export HP_LOGGING_STEPS=50

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:openbookqa" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type retnet
```

---

### 5) 训练后“统一多任务评测”（可选但强烈推荐）

如果你想：**训练某个 cfg（某个 LoRA 配置）后，立刻在一组 tasks 上评测**，可以：

```bash
export EVAL_AFTER_TRAIN=1
export EVAL_TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'
export EVAL_BATCH_SIZE=64
export EVAL_OUTPUT_ROOT=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/outputs/lm_eval

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:boolq" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type retnet \
  --eval-after-train \
  --eval-tasks "$EVAL_TASKS" \
  --eval-batch-size 64 \
  --eval-output-root "$EVAL_OUTPUT_ROOT"
```

