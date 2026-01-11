# Commonsense_170K 超参数对齐说明（MambaPEFT / zh-LAT-peft）
H1
/home/user/mzs_h/code/zh-LAT-peft/output/benchmark/retnet/commonsense_170k_seed87/

H2
/home/user/mzs_h/code/zh-LAT-peft/output/benchmark/delta_net/commonsense_170k_seed87/

3090
/home/user/mzs_h/code/zh-LAT-peft/output/benchmark/delta_net/commonsense_170k_seed87/
第一次
rm -rf /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/cache/commonsense_170k/*

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

# ===== model =====
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

# ===== HF cache + offline =====
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ===== commonsense_170k =====
export LAT_COMMONSENSE_170K_VAL_SET_SIZE=2000
export LAT_DATA_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data

# cutoff_len 对齐（=256）
# export HP_MAX_SEQLEN=256

# ===== 预热加速：CPU 拉满（只影响“建 cache”，不影响训练 dataloader）=====
export LAT_DATA_PREPROC_WORKERS=35   # 或者 64/96（看你机器核数）

# （可选）cache 写到更快的盘（默认写 mamba-peft/data/cache）
# export LAT_DATA_CACHE_DIR=/path/to/fast_ssd/lat_data_cache

# ===== 预热阶段：只为建 cache，不跑 lm_eval，不跑长训练 =====
export EVAL_AFTER_TRAIN=0
export EVAL_BACKEND=lat
export HP_MAX_STEPS=1          # cache 建完后，训练只跑 1 step 就停
export HP_NO_SAVE=1            # 预热阶段不落 checkpoint（纯预热）

# ===== 其他（避免环境报错/更稳定）=====
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

./lat_batch_tmux.sh \
  --suite E14 \
  --round 1 \
  --pairs "87:commonsense_170k" \
  --gpus "2" \
  --gpu-plan "1" \
  --model-type delta_net
```

## H2 delt

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export HP_NO_SAVE=0  
# ===== model =====
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

# ===== HF cache + offline =====
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ===== commonsense_170k =====
export LAT_COMMONSENSE_170K_VAL_SET_SIZE=2000   # 对齐官方 finetune.py

export LAT_DATA_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data

# ===== train hparams（论文对齐）=====
export EVAL_GEN=0
export HP_VAL_SPLIT=val

export HP_EPOCHS=3
export HP_BATCH_SIZE=8              # 若显存允许，优先 16
export HP_LR=0.0003

# cutoff_len 对齐（=256）
# export HP_MAX_SEQLEN=256

# ===== eval / save 频率（样本数等价 25600）=====
# 25600 / 8 = 3200
export HP_EVAL_BATCH_SIZE=8
export HP_EVAL_STEPS=3200
export HP_SAVE_STEPS=3200
export HP_LOGGING_STEPS=100

# ===== scheduler（严格按论文）=====
export LR_SCHEDULER_TYPE=linear
export LR_WARMUP_STEPS=100
export LR_WARMUP_RATIO=0

# ===== checkpoint policy =====
export HP_SAVE_TOTAL_LIMIT=2
export HP_LOAD_BEST_MODEL_AT_END=1

# export HP_METRIC_FOR_BEST_MODEL=eval_loss
# export HP_GREATER_IS_BETTER=0
export HP_METRIC_FOR_BEST_MODEL=eval_token_accuracy
export HP_GREATER_IS_BETTER=1

# ===== resume =====
export SAVE_OPTIMIZER_STATE=1

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

# ===== SwanLab =====
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-commonsense170k-Jan7-7GPU-v8-2-4090"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

# ===== lm_eval（保守起步，避免 OOM）=====
export EVAL_AFTER_TRAIN=1
export EVAL_BACKEND=lm_eval
export EVAL_TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'
export EVAL_BATCH_SIZE=16
export EVAL_OUTPUT_ROOT=/home/user/mzs_h/code/zh-LAT-peft/output/lm_eval

# export LAT_CACHE_FORMAT_VERSION=fmt3

# ===== launch =====
./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:commonsense_170k" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "1,1,2,2,2,2,2" \
  --model-type delta_net \
  --eval-after-train \
  --eval-backend lm_eval \
  --eval-tasks "$EVAL_TASKS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-output-root "$EVAL_OUTPUT_ROOT"
```
## RetNet（MODEL_TYPE=retnet，用 --suite E13）

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export HP_NO_SAVE=0  
# ===== model =====
export MODEL_TYPE=retnet
export LAT_MODEL=/home/user/mzs_h/model/retnet-1.3B-100B/
export LAT_PREC=bf16

# ===== HF cache + offline =====
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ===== commonsense_170k =====
export LAT_COMMONSENSE_170K_VAL_SET_SIZE=2000
export LAT_DATA_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data

# ===== train hparams =====
export EVAL_GEN=0
export HP_VAL_SPLIT=val
export HP_EPOCHS=3
export HP_BATCH_SIZE=8
export HP_LR=0.0003
# export HP_MAX_SEQLEN=256

# ===== eval / save 频率（样本数等价 25600）=====
export HP_EVAL_BATCH_SIZE=8
export HP_EVAL_STEPS=3200
export HP_SAVE_STEPS=3200
export HP_LOGGING_STEPS=100

# ===== scheduler（严格按论文）=====
export LR_SCHEDULER_TYPE=linear
export LR_WARMUP_STEPS=100
export LR_WARMUP_RATIO=0

# ===== checkpoint policy =====
export HP_SAVE_TOTAL_LIMIT=2
export HP_LOAD_BEST_MODEL_AT_END=1

# export HP_METRIC_FOR_BEST_MODEL=eval_loss
# export HP_GREATER_IS_BETTER=0
export HP_METRIC_FOR_BEST_MODEL=eval_token_accuracy
export HP_GREATER_IS_BETTER=1

# ===== resume =====
export SAVE_OPTIMIZER_STATE=1

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

# ===== SwanLab =====
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="retnet-commonsense170k-Jan7-7GPU-1-4090-v8"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

# ===== lm_eval =====
export EVAL_AFTER_TRAIN=1
export EVAL_BACKEND=lm_eval
export EVAL_TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'
export EVAL_BATCH_SIZE=16
export EVAL_OUTPUT_ROOT=/home/user/mzs_h/code/zh-LAT-peft/output/lm_eval

./lat_batch_tmux.sh \
  --suite E13 \
  --round all \
  --pairs "87:commonsense_170k" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type retnet \
  --eval-after-train \
  --eval-backend lm_eval \
  --eval-tasks "$EVAL_TASKS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-output-root "$EVAL_OUTPUT_ROOT"

```

## GLA（MODEL_TYPE=gla，当前建议用 --suite E12）

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export HP_NO_SAVE=0  
# ===== model =====
export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/gla-1.3B-100B/
export LAT_PREC=bf16

# ===== HF cache + offline =====
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ===== commonsense_170k =====
export LAT_COMMONSENSE_170K_VAL_SET_SIZE=2000
export LAT_DATA_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data

# ===== train hparams（论文对齐）=====
export EVAL_GEN=0
export HP_VAL_SPLIT=val
export HP_EPOCHS=3
export HP_BATCH_SIZE=8
export HP_LR=0.0003
# export HP_MAX_SEQLEN=256

# ===== eval / save 频率（样本数等价 25600）=====
export HP_EVAL_BATCH_SIZE=8
export HP_EVAL_STEPS=3200
export HP_SAVE_STEPS=3200
export HP_LOGGING_STEPS=100

# ===== scheduler（严格按论文）=====
export LR_SCHEDULER_TYPE=linear
export LR_WARMUP_STEPS=100
export LR_WARMUP_RATIO=0

# ===== checkpoint policy =====
export HP_SAVE_TOTAL_LIMIT=2
export HP_LOAD_BEST_MODEL_AT_END=1

# export HP_METRIC_FOR_BEST_MODEL=eval_loss
# export HP_GREATER_IS_BETTER=0
export HP_METRIC_FOR_BEST_MODEL=eval_token_accuracy
export HP_GREATER_IS_BETTER=1

# ===== resume =====
export SAVE_OPTIMIZER_STATE=1

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

# ===== SwanLab =====
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-commonsense170k-Jan7-7GPU-3090-v8"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

# ===== lm_eval =====
export EVAL_AFTER_TRAIN=1
export EVAL_BACKEND=lm_eval
export EVAL_TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'
export EVAL_BATCH_SIZE=16
export EVAL_OUTPUT_ROOT=/home/user/mzs_h/code/zh-LAT-peft/output/lm_eval

./lat_batch_tmux.sh \
  --suite E12 \
  --round all \
  --pairs "87:commonsense_170k" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type gla \
  --eval-after-train \
  --eval-backend lm_eval \
  --eval-tasks "$EVAL_TASKS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-output-root "$EVAL_OUTPUT_ROOT"


```
---

## 1. 可靠来源（唯一依据）

### 1.1 MambaPEFT 论文（v3）

文件：`2411.03855v3.md`  
章节：**Appendix B.1 – Language Tasks**

原文（Lines 320–326）：

> We follow the fine-tuning setup of Liu et al. (2024a); Hu et al. (2023) for commonsense reasoning tasks.  
> Each model is fine-tuned with about **140,000 data for three epochs with a batch size of 16**.  
> A **linear learning rate scheduler** is used with a **warmup period of 100 iterations**.  
> As to the learning rate, we use suitable values for each method…

**可直接确定的硬约束：**

- Epochs = **3**
- Batch size = **16**
- LR scheduler = **Linear**
- Warmup steps = **100**
- 训练样本规模 ≈ 140k（commonsense_170k）

---

### 1.2 官方开源代码（finetune.py）

路径：`MambaPEFT/language/finetune.py`

#### 关键默认训练参数（Lines 66–76）

```python
batch_size = 128
micro_batch_size = 4
num_epochs = 3
learning_rate = 3e-4
weight_decay = 0.0
cutoff_len = 256
val_set_size = 2000
eval_step = 200
save_step = 200
```

#### Warmup 实现（Lines 563–565）

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=...
)
```

**可以直接继承的事实：**

- `cutoff_len = 256`
- `val_set_size = 2000`
- `eval_step = save_step = 200`
- `warmup_steps = 100`
- `learning_rate = 3e-4`（baseline / LoRA 类方法安全默认）

---

## 2. 与你当前框架的参数映射关系

| 论文 / 官方名 | zh-LAT-peft 中变量 |
|---|---|
| epochs | `HP_EPOCHS` |
| batch size | `HP_BATCH_SIZE` |
| learning rate | `HP_LR` |
| linear scheduler | `LR_SCHEDULER_TYPE=linear` |
| warmup=100 | `LR_WARMUP_STEPS=100` |
| cutoff_len=256 | `HP_MAX_SEQLEN=256` |
| val_set_size=2000 | `LAT_COMMONSENSE_170K_VAL_SET_SIZE=2000` |

你已在 `train_lat.py` 中补充：

```python
HP_MAX_SEQLEN / LAT_MAX_SEQLEN → dataset max_seqlen
```

该实现 **严格等价于 MambaPEFT 的 cutoff_len 行为**。

---

## 3. eval / save 频率的严格推导（不是拍脑袋）

官方代码：

- `batch_size = 128`
- `eval_step = 200`

等价于：

```
200 × 128 = 25600 samples / eval
```

因此你当前设置应满足：

```
HP_EVAL_STEPS = 25600 / HP_BATCH_SIZE
HP_SAVE_STEPS = 同上
```

### 推荐值

| HP_BATCH_SIZE | HP_EVAL_STEPS | HP_SAVE_STEPS |
|---|---|---|
| 16 | 1600 | 1600 |
| 8  | 3200 | 3200 |

---

## 4. 最终推荐“论文对齐版”训练超参

> 以下配置 **完全对齐论文 + 官方代码语义**，仅在显存受限处做等价缩放。

```bash
# ===== commonsense_170k（论文对齐）=====
export HP_EPOCHS=3
export HP_BATCH_SIZE=8            # 16 若不 OOM 优先用 16
export HP_LR=0.0003

# cutoff_len 对齐
# export HP_MAX_SEQLEN=256

# scheduler / warmup（严格对齐）
export LR_SCHEDULER_TYPE=linear
export LR_WARMUP_STEPS=100
export LR_WARMUP_RATIO=0

# eval / save（样本数等价）
export HP_EVAL_STEPS=3200
export HP_SAVE_STEPS=3200
export HP_EVAL_BATCH_SIZE=8
```

---

## 5. lm_eval 阶段建议

- 官方 wrapper 常用 `batch_size=64`
- **1.3B + RTX 4090 很容易 OOM**
- 推荐策略：

```bash
export EVAL_BATCH_SIZE=16   # 稳定优先
# 稳定后可尝试 32 / 64
```

---