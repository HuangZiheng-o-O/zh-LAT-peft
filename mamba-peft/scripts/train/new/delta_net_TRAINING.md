## Run Guide (LAT Framework) — RetNet 版（完整 bash）

> 你只需要把 `LAT_MODEL=...` 改成你本机 RetNet checkpoint 的真实目录 

### Global (可选：放在每次会话最开始)

```bash

export HP_USE_RSLoRA=0
export HP_USE_DORA=1
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16
```


### caches（原样保留）

```bash
export XDG_CACHE_HOME=/mnt/data4/user_cache
export TRITON_CACHE_DIR=/mnt/data4/user_cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/data4/user_cache/torch
export TORCH_EXTENSIONS_DIR=/mnt/data4/user_cache/torch_extensions
export HF_HOME=/mnt/data4/user_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/data4/user_cache/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=/mnt/data4/user_cache/huggingface/hub
export PIP_CACHE_DIR=/mnt/data4/user_cache/pip
export WANDB_DIR=/mnt/data4/user_cache/wandb
```
---

### spider

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export NLTK_DATA=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/nltk_data
export SPIDER_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/spider_data

export LAT_FORCE_LEFT_PAD=1
export LAT_USE_MAX_NEW_TOKENS=1
export LAT_USE_FUSED_SWIGLU=0
export LAT_VERBOSE=1
 
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
export SWANLAB_PROJECT="delta_net-spider-1-4090-ROUND_E12_DELTANET5-mail02-r4"
export SWANLAB_LOGDIR="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/my_swanlog/local_eval_logs"
export SWANLAB_EMAIL_YAML="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/dangerous/email_notify.yaml"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUM_DATA_WORKERS=8
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET5 \
  --round all \
  --pairs "87:spider-tvt" \
  --gpus "0 1 3 4 6" \
  --gpu-plan "2,2,2,2,2" \
  --model-type delta_net
```



### samsum

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=128
export EVAL_GEN_MIN_LENGTH=8

export LAT_FORCE_LEFT_PAD=1
export LAT_USE_MAX_NEW_TOKENS=1
export LAT_VERBOSE=1
export LAT_USE_FUSED_SWIGLU=0

export HP_EVAL_STEPS=1000
export HP_SAVE_STEPS=1000
export HP_LOGGING_STEPS=50

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

export NUM_DATA_WORKERS=2
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-samsum-ROUND_E12_DELTANET-clean-decoder-r3-3090-t4"
export SWANLAB_EMAIL_ON_START=1
export SWANLAB_EMAIL_ON_FINISH=1

export SAMSUM_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/samsum

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET \
  --round 1 \
  --pairs "87:samsum" \
  --gpus "3 4 5 6" \
  --gpu-plan "2,2,2,2" \
  --model-type delta_net
```

---


---

### dart

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export NLTK_DATA=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/nltk_data
export DART_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart

export HP_BATCH_SIZE=8
export HP_EPOCHS=8
export HP_LR=0.002
export HP_EVAL_BATCH_SIZE=16
export HP_EVAL_STEPS=4000
export HP_SAVE_STEPS=8000
export HP_LOGGING_STEPS=1000
export HP_NO_SAVE=1

export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=1024
export EVAL_GEN_MIN_LENGTH=5
unset EVAL_GEN_NUM_BEAMS

export LAT_FORCE_LEFT_PAD=1
export LAT_USE_MAX_NEW_TOKENS=1
export LAT_USE_FUSED_SWIGLU=0
export LAT_VERBOSE=0

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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
export TRANSFORMERS_VERBOSITY=error

export SWANLAB_ENABLE=1
export SWANLAB_MODE=local
export SWANLAB_PROJECT="delta_net-dart-ROUND_E12_DELTANET-2-4090-r11"
export SWANLAB_EMAIL_YAML="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/dangerous/email_notify.yaml"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET \
  --round all \
  --pairs "87:dart" \
  --gpus "1 2 3 4 5 6 7" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type delta_net
```

---

### GLUE glue_multidata_e15

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export EVAL_GEN=0
export HP_VAL_SPLIT=test

export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0004

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=200
export HP_SAVE_STEPS=800
export HP_LOGGING_STEPS=50

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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

export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-glue-all-test"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET \
  --round all \
  --pairs "87:glue-tvt_sst2 87:glue-tvt_qqp 87:glue-tvt_mnli" \
  --gpus "0 1 2 3 4 5 6" \
  --name glue_multidata_e15 \
  --model-type delta_net
```

---

### tvt_cola

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export EVAL_GEN=0
export HP_VAL_SPLIT=test

export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0004

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=200
export HP_SAVE_STEPS=800
export HP_LOGGING_STEPS=50

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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

export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-cola-1-4090-Dec25-2"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET \
  --round all \
  --pairs "87:glue-tvt_cola" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type delta_net
```

---

### tvt_rte

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export EVAL_GEN=0
export HP_VAL_SPLIT=test

export HP_EPOCHS=3
export HP_BATCH_SIZE=8
export HP_LR=0.00005

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=100
export HP_SAVE_STEPS=400
export HP_LOGGING_STEPS=50

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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

export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-rte-2-4090-Dec25"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET \
  --round all \
  --pairs "87:glue-tvt_rte" \
  --gpus "4 5 6" \
  --gpu-plan "2,2,1" \
  --model-type delta_net
```

---

### tvt_qnli（严格离线版）

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

export EVAL_GEN=0
export HP_VAL_SPLIT=test

export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0004

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=1500
export HP_SAVE_STEPS=3000
export HP_LOGGING_STEPS=100

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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

export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-tvt_qnli-2-4090-Dec25"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET \
  --round all \
  --pairs "87:glue-tvt_qnli" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "1,1,2,2,2,2,2" \
  --model-type delta_net
```

---

### tvt_mnli（3090 方案）

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

export HP_DATA=mnli
export EVAL_GEN=0
export HP_VAL_SPLIT=test

export HP_EPOCHS=3
export HP_BATCH_SIZE=8
export HP_LR=0.0004

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=500
export HP_SAVE_STEPS=1000
export HP_LOGGING_STEPS=100

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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

export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-glue-tvt_mnli-3090-Dec27"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET \
  --round all \
  --pairs "87:glue-tvt_mnli" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type delta_net
```

### tvt_mnli（单卡 debug 方案）

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export HF_HUB_OFFLINE=0
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

export HP_DATA=mnli
export EVAL_GEN=0
export HP_VAL_SPLIT=test

export HP_EPOCHS=3
export HP_BATCH_SIZE=8
export HP_LR=0.0004

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=500
export HP_SAVE_STEPS=1000
export HP_LOGGING_STEPS=100

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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

export SWANLAB_ENABLE=0
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET \
  --round all \
  --pairs "87:glue-tvt_mnli" \
  --gpus "1" \
  --gpu-plan "1" \
  --model-type delta_net
```

---

### tvt_sst2

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

export HP_DATA=sst2
export EVAL_GEN=0
export HP_VAL_SPLIT=test

export HP_EPOCHS=4
export HP_BATCH_SIZE=8
export HP_LR=0.0003

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=200
export HP_SAVE_STEPS=400
export HP_LOGGING_STEPS=100

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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

export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-sst2-1-4090-Dec26"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET \
  --round all \
  --pairs "87:glue-tvt_sst2" \
  --gpus "1 2 3 4 5 6 7" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type delta_net
```

---

### tvt_qqp

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

export EVAL_GEN=0
export HP_VAL_SPLIT=test

export HP_EPOCHS=5
export HP_BATCH_SIZE=8
export HP_LR=0.0003

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=500
export HP_SAVE_STEPS=1000
export HP_LOGGING_STEPS=100

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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

export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-glue-tvt_qqp-1-4090-Jan1-v2"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

export LAT_LAUNCH_STAGGER_MINUTES=15

./lat_batch_tmux.sh \
  --suite E251 \
  --round all \
  --pairs "87:glue-tvt_qqp" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "1,1,1,1,1,1,1" \
  --model-type delta_net
```

---

### tvt_mrpc

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export EVAL_GEN=0

export HP_EPOCHS=10
export HP_BATCH_SIZE=8
export HP_LR=0.0005

export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=100
export HP_SAVE_STEPS=400
export HP_LOGGING_STEPS=20

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

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

export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="delta_net-tvt_mrpc-1-4090-Dec25"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite ROUND_E12_DELTANET \
  --round all \
  --pairs "87:glue-tvt_mrpc" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "2,2,2,2,2,2,2" \
  --model-type delta_net
```

---
 
 