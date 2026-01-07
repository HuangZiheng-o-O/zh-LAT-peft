DeltaNet + BoolQ
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
export SWANLAB_PROJECT="delta_net-many-1-4090-v2"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

# ===== 关键新增：pairs 并行（auto=每块GPU一个dataset）=====
export LAT_BATCH_PAIR_CONCURRENCY=auto
# 强制 cache 格式版本升级
export LAT_CACHE_FORMAT_VERSION=fmt4

# ===== launch =====
./lat_batch_tmux.sh \
  --suite E11 \
  --round all \
  --pairs "87:boolq,87:piqa,87:social_iqa,87:hellaswag,87:winogrande,87:openbookqa,87:arc_easy,87:arc_challenge" \
  --gpus "0 1 2 3 4 5 6 7" \
  --gpu-plan "1,1,1,1,1,1,1,1" \
  --model-type delta_net

```



```angular2html
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
export SWANLAB_PROJECT="delta_net-many-2-4090"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

# ===== 关键新增：pairs 并行（auto=每块GPU一个dataset）=====
export LAT_BATCH_PAIR_CONCURRENCY=auto

# ===== launch =====
./lat_batch_tmux.sh \
  --suite E11 \
  --round all \
  --pairs "87:piqa,87:social_iqa,87:hellaswag" \
  --gpus "1 2 3" \
  --gpu-plan "1,1,1" \
  --model-type delta_net

```