## Run Guide (LAT Framework)


### spider

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 
    
# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

export NLTK_DATA=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/nltk_data
export SPIDER_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/spider_data

export GLA_FORCE_LEFT_PAD=1
export GLA_USE_MAX_NEW_TOKENS=1
export GLA_USE_FUSED_SWIGLU=0
export GLA_VERBOSE=1

export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=256
export EVAL_GEN_MIN_LENGTH=0
export EVAL_GEN_NUM_BEAMS=1

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

export HP_EVAL_STEPS=1500
export HP_SAVE_STEPS=1500
export HP_LOGGING_STEPS=100
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-spider-1-4090-E155-mail02-r4"
export SWANLAB_LOGDIR="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/my_swanlog/local_eval_logs"
export SWANLAB_EMAIL_YAML="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/dangerous/email_notify.yaml"
export SWANLAB_EMAIL_ON_START=1
export SWANLAB_EMAIL_ON_FINISH=1
export SWANLAB_EMAIL_ON_INTERRUPT=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUM_DATA_WORKERS=8
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1

export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0

./lat_batch_tmux.sh \
  --suite E155 \
  --round all \
  --pairs "87:spider-tvt" \
  --gpus "0 1 3 4 6" \
  --gpu-plan "2,2,2,2,2" \
  --model-type gla
```



### samsum

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 

# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

########################################
# ↓↓↓ 覆盖旧版本中已有的变量（只是数值不同） ↓↓↓
########################################

# ✔ 替换 EVAL_GEN 配置
export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=128
export EVAL_GEN_MIN_LENGTH=8

# ✔ 解码/GLA（同名变量更新）
export GLA_FORCE_LEFT_PAD=1
export GLA_USE_MAX_NEW_TOKENS=1
export GLA_VERBOSE=1
export GLA_USE_FUSED_SWIGLU=0

# ✔ 训练步骤（替换原来的）
export HP_EVAL_STEPS=1000      # ← 从 1500 → 1000
export HP_SAVE_STEPS=1000      # ← 从 1500 → 1000
export HP_LOGGING_STEPS=50     # ← 从 100 → 50

# ✔ LR 配置（保持一致）
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

# ✔ DataLoader / CPU / CUDA
export NUM_DATA_WORKERS=2               # ← 原 8 → 2
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ✔ SWANLAB（保持原有结构但更新 project）
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-samsum-E15-clean-decoder-r3-3090-t4"
export SWANLAB_EMAIL_ON_START=1
export SWANLAB_EMAIL_ON_FINISH=1

########################################
# ↓↓↓ 下面这些是 SamSum 版本新增的变量 ↓↓↓
########################################

# SamSum 数据路径
export SAMSUM_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/samsum

# LoRA 初始化策略（新增）
#export HP_INIT=pissa_niter_4
# 或：export HP_PISSA_FAST=1   # 可选

# DataLoader 新增项
#export DATALOADER_PREFETCH_FACTOR=2
#export DATALOADER_PIN_MEMORY=1
#export DATALOADER_PERSISTENT_WORKERS=0

./lat_batch_tmux.sh \
  --suite E15 \
  --round 2 \
  --pairs "87:samsum" \
  --gpus "3 4 5 6" \
  --gpu-plan "2,2,2,2" \
  --model-type gla

```
  --suite E15 \
  --round 3 \
  --pairs "87:samsum" \
  --gpus "0 1 2 3 4 5 6 7" \
  --gpu-plan "1,1,1,1,1,1,1,1"

  --suite E15 \
  --round 1 \
  --pairs "87:samsum" \
  --gpus "3 4 5 6" \
  --gpu-plan "2,2,2,2"

```bash
### All caches → /mnt/data4/user_cache ###

# Generic cache
export XDG_CACHE_HOME=/mnt/data4/user_cache

# Triton kernel cache
export TRITON_CACHE_DIR=/mnt/data4/user_cache/triton

# Torch Inductor
export TORCHINDUCTOR_CACHE_DIR=/mnt/data4/user_cache/torch

# Torch CUDA extensions
export TORCH_EXTENSIONS_DIR=/mnt/data4/user_cache/torch_extensions

# HuggingFace caches
export HF_HOME=/mnt/data4/user_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/data4/user_cache/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=/mnt/data4/user_cache/huggingface/hub

# pip cache
export PIP_CACHE_DIR=/mnt/data4/user_cache/pip

# wandb
export WANDB_DIR=/mnt/data4/user_cache/wandb
```

### dart

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 

# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

# 🔧 新增：强制离线模式，防止网络死锁
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export NLTK_DATA=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/nltk_data
export DART_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart

# Core training knobs
export HP_BATCH_SIZE=8
export HP_EPOCHS=8
export HP_LR=0.002
export HP_EVAL_BATCH_SIZE=16          # now honored by trainer
export HP_EVAL_STEPS=4000
export HP_SAVE_STEPS=8000
export HP_LOGGING_STEPS=1000
export HP_NO_SAVE=1                   # disable checkpointing entirely (drop this if you still want 8k-step saves)

# Generation / decoding
export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=1024
export EVAL_GEN_MIN_LENGTH=5
unset  EVAL_GEN_NUM_BEAMS
export GLA_FORCE_LEFT_PAD=1
export GLA_USE_MAX_NEW_TOKENS=1
export GLA_USE_FUSED_SWIGLU=0
export GLA_VERBOSE=0

# LR schedule
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

# DataLoader & runtime
export NUM_DATA_WORKERS=4
export DATALOADER_PREFETCH_FACTOR=2
export DATALOADER_PIN_MEMORY=1
export DATALOADER_PERSISTENT_WORKERS=0
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error    # keeps HF logger quiet

# SwanLab (本地模式，避免网络上传)
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud             # 🔧 修改：从 cloud 改为 local
export SWANLAB_PROJECT="gla-dart-E15-2-4090-r11"
export SWANLAB_EMAIL_YAML="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/dangerous/email_notify.yaml"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite E15 \
  --round all \
  --pairs "87:dart" \
  --gpus "1 2 3 4 5 6 7" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type gla
```

### GLUE glue_multidata_e15

```bash

conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 

# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

# chmod +x *.sh
# 离线/本地资源（可选）
#export GLUE_DATASET_ID=nyu-mll/glue
#export GLUE_METRIC_DIR=/home/user/mzs_h/data/hf_cache/eval_metrics/glue

# 禁止生成
export EVAL_GEN=0

# 训练超参（ENV 优先级最高）
export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0004

# 评测/保存/日志
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=200
export HP_SAVE_STEPS=800
export HP_LOGGING_STEPS=50
# 若完全不存
# export HP_NO_SAVE=1

# 调度
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

# DataLoader/Runtime（可沿用）
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

# SwanLab（改项目名区分，邮件按需开）
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-glue-all-test"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

# 注意: multidata 功能直接通过 --pairs 传多个数据集实现
./lat_batch_tmux.sh \
  --suite E15 \
  --round all \
  --pairs "87:glue-tvt_sst2 87:glue-tvt_qqp 87:glue-tvt_mnli" \
  --gpus "0 1 2 3 4 5 6" \
  --name glue_multidata_e15 \
  --model-type gla

#87:glue-tvt_cola 87:glue-tvt_rte  87:glue-tvt_mrpc  87:glue-tvt_qnli
# 87:
# 87:
```



### ok tvt_cola

https://swanlab.cn/@zh2701/cola-1-4090-Dec25-2/overview



```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 

# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

# 离线/本地资源（可选）
#export GLUE_DATASET_ID=nyu-mll/glue
#export GLUE_METRIC_DIR=/home/user/mzs_h/data/hf_cache/eval_metrics/glue

# 禁止生成
export EVAL_GEN=0
export HP_VAL_SPLIT=test

# 训练超参（ENV 优先级最高）
export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0004

# 评测/保存/日志
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=200
export HP_SAVE_STEPS=800
export HP_LOGGING_STEPS=50
# 若完全不存
# export HP_NO_SAVE=1

# 调度
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

# DataLoader/Runtime（可沿用）
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

# SwanLab（改项目名区分，邮件按需开）
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="cola-1-4090-Dec25-2"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

./lat_batch_tmux.sh \
  --suite E15 \
  --round all \
  --pairs "87:glue-tvt_cola" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type gla

```

### ok tvt_rte

https://swanlab.cn/@zh2701/rte-2-4090-Dec25/overview

```bash

conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 

# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

# 禁止生成
export EVAL_GEN=0
export HP_VAL_SPLIT=test
# ==== 训练超参 ====
# RTE 数据量很小，epoch 不用太多；主要降低学习率，防止过拟合+训练不稳定
export HP_EPOCHS=3          # 从 4 降到 3（必要的话第二轮再试 2 或 4）
export HP_BATCH_SIZE=8      # 先保持不动，稳定后再看是否增大
export HP_LR=0.00005        # ★从 4e-4 大幅降到 5e-5

# ==== 评测/保存/日志 ====
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=100    # 从 200 改成 100，更密集观察过拟合
export HP_SAVE_STEPS=400    # 从 800 改成 400，更早留一个"没过拟合"的 checkpoint
export HP_LOGGING_STEPS=50

# ==== 调度 ====
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1  # 先维持 0.1 就行

# DataLoader/Runtime
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

# SwanLab
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="rte-2-4090-Dec25"   
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite E15 \
  --round all \
  --pairs "87:glue-tvt_rte" \
  --gpus "4 5 6" \
  --gpu-plan "2,2,1" \
  --model-type gla

```

### ok QNLI  

https://swanlab.cn/@zh2701/tvt_qnli-2-4090-Dec25/overview

- 任务：二分类自然语言推理（Question, Sentence → entailment / not_entailment）
- 输入字段：question + sentence（代码里拼接用 `sep_token`）
- 标签数：2
- 主要指标：accuracy
- 典型规模（GLUE 官方）：train ≈ 104,743；dev ≈ 5,463
- 我们的 tvt 变体：
  - train/val 来自 train 的 80/20 切分
  - 将 split=test 映射到官方 dev（validation）

说明与校验要点
- EVAL_GEN=0：分类任务不走生成分支
- HP_VAL_SPLIT=test：对应官方 dev（validation），与代码逻辑一致
- GLUE_DATASET_ID/GLUE_METRIC_DIR：启用离线评测时建议设置
- 批大小/评测步数/保存步数已按 QNLI 的训练样本量上调；若单卡显存紧张，可降 `HP_EVAL_BATCH_SIZE=32`
- 其余 DataLoader 与调度 knobs 与仓库一致，适用于 QNLI

如需更快验证，可将：
- HP_EPOCHS=2
- HP_EVAL_STEPS=800
- HP_SAVE_STEPS=1600
用于小步快跑检查流程与指标。
- GLUE_DATASET_ID 是用来告诉 datasets.load_dataset 用哪个数据集 ID。代码里默认就是 nyu-mll/glue，不设也行。你本地已有缓存 datasets--nyu-mll--glue 和 nyu-mll___glue，默认可直接命中。
- GLUE_METRIC_DIR 是 evaluate.load("glue", name) 失败时的本地兜底目录。不设也行；代码会自动用内置 metrics（accuracy/MCC/F1）。只有当你本地放了 glue 的 metric 脚本时才需要显式指向那个目录。

给你一份更干净、严格离线的 QNLI 命令（不依赖这两个变量）：

```bash

conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 

# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

# 严格离线 + 指向你的本地 HF 缓存
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

# 分类任务：禁用生成；val=test → 官方 dev
export EVAL_GEN=0
export HP_VAL_SPLIT=test
export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0004

# 评测/保存/日志（按 QNLI 量级）
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=1500
export HP_SAVE_STEPS=3000
export HP_LOGGING_STEPS=100

# 调度
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

# DataLoader/Runtime
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

# SwanLab（可选）
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="tvt_qnli-2-4090-Dec25"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite E15 \
  --round all \
  --pairs "87:glue-tvt_qnli" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "1,1,2,2,2,2,2" \
  --model-type gla
```

- 想显式指定数据集 ID（但默认已是这个）:
  - export GLUE_DATASET_ID=nyu-mll/glue
- 你本地真的放了 glue 的 metric 脚本目录时:
  - export GLUE_METRIC_DIR=/path/to/your/local/glue/metric



### ok MNLI 在2-4090中断 直接在3090重新跑 打算

export SWANLAB_PROJECT="gla-glue-tvt_mnli-2-4090-Dec25"

https://swanlab.cn/@zh2701/gla-glue-tvt_mnli-2-4090-Dec25/overview



```bash
# =========================
# MNLI 训练命令（修正版）
# =========================
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 

# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

########################
# 严格离线 + 本地 HF 缓存
########################
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

########################################
# GLUE / MNLI：多领域三分类 NLI
########################################
# 告诉 train_lat.py：这是 GLUE 的 mnli 任务
# glue.py 会自动把 dev = matched + mismatched 合并
export HP_DATA=mnli

export EVAL_GEN=0
export HP_VAL_SPLIT=test      # 按 glue-tvt 约定：test=官方 dev

# 超参（结合 GLUE 论文 / RapidBERT / LoNAS-BERT）
export HP_EPOCHS=3            # 大数据 GLUE 上经典设置
export HP_BATCH_SIZE=8
export HP_LR=0.0004           # 4e-4：与 QNLI / LoNAS-BERT MNLI 设置保持同量级

########################################
# 评测 / 保存 / 日志
########################################
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=500
export HP_SAVE_STEPS=1000     # save 比 eval 稍稀

export HP_LOGGING_STEPS=100

#############
# 学习率调度
#############
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

#########################
# DataLoader / Runtime
#########################
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

############
# SwanLab（可选）
############
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-glue-tvt_mnli-3090-Dec27"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

##############################
# 启动 MNLI 训练
##############################
./lat_batch_tmux.sh \
  --suite E15 \
  --round all \
  --pairs "87:glue-tvt_mnli" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type gla
```





```java
# =========================
# MNLI 训练命令（修正版）
# =========================
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

########################
# 严格离线 + 本地 HF 缓存
########################
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

########################################
# GLUE / MNLI：多领域三分类 NLI
########################################
# 告诉 train_lat.py：这是 GLUE 的 mnli 任务
# glue.py 会自动把 dev = matched + mismatched 合并
export HP_DATA=mnli

export EVAL_GEN=0
export HP_VAL_SPLIT=test      # 按 glue-tvt 约定：test=官方 dev

# 超参（结合 GLUE 论文 / RapidBERT / LoNAS-BERT）
export HP_EPOCHS=3            # 大数据 GLUE 上经典设置
export HP_BATCH_SIZE=8
export HP_LR=0.0004           # 4e-4：与 QNLI / LoNAS-BERT MNLI 设置保持同量级

########################################
# 评测 / 保存 / 日志
########################################
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=500
export HP_SAVE_STEPS=1000     # save 比 eval 稍稀

export HP_LOGGING_STEPS=100

#############
# 学习率调度
#############
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

#########################
# DataLoader / Runtime
#########################
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

############
# SwanLab（可选）
############
export SWANLAB_ENABLE=0
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

##############################
# 启动 MNLI 训练
##############################
./lat_batch_tmux.sh \
  --suite E15 \
  --round all \
  --pairs "87:glue-tvt_mnli" \
  --gpus "1" \
  --gpu-plan "1" \
  --model-type gla
```

### OK SST-2

https://swanlab.cn/@zh2701/sst2-1-4090-Dec26/overview

```bash
# =========================
# SST-2 训练命令（GLUE）
# =========================
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 

# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

########################
# 严格离线 + 本地 HF 缓存
########################
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

########################################
# GLUE / SST-2：单句二分类
########################################
# 告诉 train_lat.py：这是 GLUE 的 sst2 任务
export HP_DATA=sst2

# 纯分类任务：禁用生成；val=test → 使用 tvt 中的 test 作为官方 dev
export EVAL_GEN=0
export HP_VAL_SPLIT=test

# 超参（参考 GLUE + LoNAS-BERT）
export HP_EPOCHS=4          # GLUE 常规 3–4 epoch，SST-2 用 4 轮
export HP_BATCH_SIZE=8      # 按你现有 QNLI 设置，兼顾显存
export HP_LR=0.0003         # 3e-4：与 LoNAS-BERT SST-2 一致的量级

########################################
# 评测 / 保存 / 日志（按 GLUE 超参表）
########################################
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=200    # 来自一份 GLUE 统一配置：SST-2 / QNLI eval 每 200 步
export HP_SAVE_STEPS=400    # save 比 eval 稍稀一点
export HP_LOGGING_STEPS=100

#############
# 学习率调度
#############
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

#########################
# DataLoader / Runtime
#########################
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

############
# SwanLab（可选）
############
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="sst2-1-4090-Dec26"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

##############################
# 启动 SST-2 训练
##############################
./lat_batch_tmux.sh \
  --suite E15 \
  --round all \
  --pairs "87:glue-tvt_sst2" \
  --gpus "1 2 3 4 5 6 7" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type gla
```

### QQP



```bash
# =========================
# QQP 训练命令（GLUE）
# =========================
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 


# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

########################
# 严格离线 + 本地 HF 缓存
########################
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

########################################
# GLUE / QQP：句对复述检测（二分类）
########################################
# 纯分类任务：禁用生成；val=test → 按你的 glue-tvt 约定，
# 使用 tvt 中的 test 作为官方 dev（本地验证）
export EVAL_GEN=0
export HP_VAL_SPLIT=test

# === 核心超参 ===
# QQP 训练集 363k+，规模和 MNLI 接近，但业界通常给 QQP
# 稍多 epoch（RapidBERT: MNLI 3 epoch, QQP 5 epoch）
export HP_EPOCHS=5          # QQP 稍多迭代，适配噪声 & 类别不平衡
export HP_BATCH_SIZE=8      # 沿用你在 QNLI/MNLI/SST-2 的设定，显存压力可控

# LoNAS-BERT 在 GLUE 上给的 QQP 学习率是 3e-4（MNLI 4e-4）
# 这里保持同样的相对关系：MNLI 4e-4，QQP 3e-4
export HP_LR=0.0003         # 3e-4：相比 MNLI 稍低一点，更稳

########################################
# 评测 / 保存 / 日志
########################################
# QQP 和 MNLI 同属"大样本句对任务"，一般 eval 周期会
# 比 SST-2 / QNLI 稍稀一点；这里对齐 MNLI：
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=500    # 大任务常用 ~500 step 评测一次
export HP_SAVE_STEPS=1000   # save 频率 ≈ eval 的 2 倍
export HP_LOGGING_STEPS=100

#############
# 学习率调度
#############
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

#########################
# DataLoader / Runtime
#########################
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

############
# SwanLab（可选）
############
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-glue-tvt_qqp-2-4090-Dec27"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

# 支持 LAT_* 或 GLA_* 前缀
export GLA_LAUNCH_STAGGER_MINUTES=15

##############################
# 启动 QQP 训练
##############################
./lat_batch_tmux.sh \
  --suite E15 \
  --round all \
  --pairs "87:glue-tvt_qqp" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "1,1,1,1,1,1,1" \
  --model-type gla
```

### ok MRPC

```bash
############################
# 1. 环境 & 路径
############################
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16 

# === LAT框架: 指定模型类型 ===
export MODEL_TYPE=gla

############################
# 2. 任务类型 & 运行模式
############################
# MRPC 是句对二分类（paraphrase / non-paraphrase）
# 只做分类，不做生成
export EVAL_GEN=0

############################
# 3. 训练超参（MRPC 专用）
############################
# MRPC 训练集：3668 对；总共只有 ~5800 对，属于 *小数据* GLUE 任务
# 典型做法是：epoch 比 MNLI/QQP 更多一点（因为步数少）

export HP_EPOCHS=10        # 解释见下面：对齐 LoNAS-BERT 的"总步数级别"
export HP_BATCH_SIZE=8     # 沿用你当前所有 GLUE 任务的 batch 设置

# LoNAS-BERT GLUE 表里：
#   MRPC:   lr = 5e-4, Epoch = 35, Batch = 32
#   MNLI:   lr = 4e-4
#   QQP:    lr = 3e-4
#   SST-2:  lr = 3e-4
# 所以 MRPC 在它们那套里是 **学习率最高的一个小任务**。
# 你这套 GLA 里：MNLI=4e-4，QQP/SST-2=3e-4 已经对齐了这个相对关系，
# 那 MRPC 用 5e-4 是合理延续。
export HP_LR=0.0005        # 5e-4：与 LoNAS-BERT MRPC 超参同量级

############################
# 4. 评测 / 保存 / 日志
############################
# MRPC 一个 epoch 的 step：3668 / 8 ≈ 459 steps
# 10 个 epoch ≈ 4590 个优化步，和 LoNAS-BERT (35 epoch, bs 32) 的总步数同级：
#   35 * 3668 / 32 ≈ 4012 步
# 所以 eval 每 100 步，大约：
#   每个 epoch 评测 ~4–5 次，10 个 epoch 一共 ~45 次评测，粒度够细。
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=100    # 小数据任务，评测稍频繁，方便早停/挑 ckpt
export HP_SAVE_STEPS=400    # 每 ~0.8 个 epoch 存一次，既不太密也不太稀
export HP_LOGGING_STEPS=20  # log 稍微密一点，方便看 loss 曲线

############################
# 5. 学习率调度
############################
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1   # 和你其它 GLUE 任务保持一致

############################
# 6. DataLoader / Runtime
############################
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

############################
# 7. SwanLab（按需）
############################
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="pad-tvt_mrpc-1-4090-Dec25"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

############################
# 8. 启动 MRPC 训练
############################
./lat_batch_tmux.sh \
  --suite E15 \
  --round all \
  --pairs "87:glue-tvt_mrpc" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type gla
```
