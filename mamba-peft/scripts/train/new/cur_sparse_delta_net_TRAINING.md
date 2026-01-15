
export HP_SAVE_MODE=none
• 只 last：
export HP_SAVE_MODE=last
• best + last:
export HP_SAVE_MODE=best_last

  - F1‑A → _SPARSE_LoraOnly_R30
  - F1‑B → _SPARSE_LoraOnly_REF_E7_KVONLY_r8_alpha16
  - F2‑A → _SPARSE_BaseOnly_R30
  - F2‑B → _SPARSE_BaseOnly_REF_E7_KVONLY_r8_alpha16
  - F3‑A → _SPARSE_Hybrid_R30
  - F3‑B → _SPARSE_Hybrid_REF_E7_KVONLY_r8_alpha16



### F1‑A：Sparse‑LoRA + fixed_ratio

```bash

export HP_SAVE_MODE=best_last
export HP_SAVE_FULL_MODEL=0

export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=lora_only
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F1‑B：Sparse‑LoRA + match_reference

```bash
export HP_SAVE_MODE=best_last
export HP_SAVE_FULL_MODEL=0

export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=lora_only
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_REFERENCE_CFG=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/yaml/E7_KVONLY_r8_alpha16.yaml
export HP_SPARSE_SCORE_SAMPLES=1024
```

---

### F2‑A：Sparse‑Base + fixed_ratio

```bash
export HP_SAVE_MODE=best_last
export HP_SAVE_FULL_MODEL=0

export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=base_only
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F2‑B：Sparse‑Base + match_reference

```bash
export HP_SAVE_MODE=best_last
export HP_SAVE_FULL_MODEL=0

export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=base_only
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_REFERENCE_CFG=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/yaml/E7_KVONLY_r8_alpha16.yaml
export HP_SPARSE_SCORE_SAMPLES=1024
```

---

### F3‑A：Sparse‑Hybrid + fixed_ratio

```bash
export HP_SAVE_MODE=best_last
export HP_SAVE_FULL_MODEL=0

export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=hybrid
export HP_SPARSE_BUDGET_MODE=fixed_ratio
export HP_SPARSE_RHO=0.3
export HP_SPARSE_SCORE_SAMPLES=1024
```

### F3‑B：Sparse‑Hybrid + match_reference

```bash
export HP_SAVE_MODE=best_last
export HP_SAVE_FULL_MODEL=0

export HP_SPARSE_ENABLE=1
export HP_SPARSE_SCOPE=hybrid
export HP_SPARSE_BUDGET_MODE=match_reference
export HP_SPARSE_REFERENCE_CFG=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/yaml/E7_KVONLY_r8_alpha16.yaml
export HP_SPARSE_SCORE_SAMPLES=1024
```



##

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
export SWANLAB_PROJECT="delta_net-rte-1-4090-Jan11"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

./lat_batch_tmux.sh \
  --suite E1414 \
  --round all \
  --pairs "87:glue-tvt_rte" \
  --gpus "0" \
  --gpu-plan "1" \
  --model-type delta_net
```
 