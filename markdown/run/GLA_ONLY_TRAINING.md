## GLA‑only Training & Evaluation (HF generate) – Run Guide

This document describes the clean GLA‑only training/eval pipeline that avoids any Mamba/SD‑LoRA decoder or resume code paths, and follows the official `flash-linear-attention` + Hugging Face `.generate()` best practices.

It also lists all environment toggles and shows how the launcher scripts propagate them through `tmux` and into Python.

---

### 0) Prerequisites

- Conda env prepared (CUDA, PyTorch, transformers ≥ 4.36 recommended).
- Local `fla` available under:
  - `zh-LAT-peft/3rdparty/flash-linear-attention/fla` (preferred) or
  - `zh-LAT-peft/fla` (symlink/folder).
- GPU visible via `nvidia-smi`, or set `CUDA_VISIBLE_DEVICES`.

The GLA model and tokenizer are loaded from `fla-hub/gla-...` using:

```python
from fla.models.gla import GLAForCausalLM, GLAConfig
from transformers import AutoTokenizer
model = GLAForCausalLM.from_pretrained("fla-hub/gla-1.3B-100B", torch_dtype="bfloat16").cuda()
tok = AutoTokenizer.from_pretrained("fla-hub/gla-1.3B-100B", trust_remote_code=True)
```

Our `train_gla_only.py` enforces the same behavior.

---

### 1) Clean GLA‑only Launchers

We use two shell launchers living under `mamba-peft/scripts/train/new/`:

- `gla_round_clean.sh` – runs a full “suite/round” of YAML configs on the current host with GPU slot slicing and env injection (data, workers, gradient checkpointing, etc.). It calls `train_gla_only.py` directly (no `train.py`).
- `gla_batch_tmux_clean.sh` – spawns a single tmux session and executes multiple `gla_round_clean.sh` jobs sequentially, each with its own log file and env overrides (e.g., different `HP_SEED`, `DATA` pairs).

Both scripts are GLA‑only and never call the old Mamba path.

You can verify they use the clean path:

```1:5:../mamba-peft/scripts/train/new/gla_round_clean.sh
#!/bin/bash
set -euo pipefail

LAUNCHER_PY="train_gla_only.py"
```

```16:21:../mamba-peft/scripts/train/new/gla_batch_tmux_clean.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/gla_round_clean.sh"
if [[ ! -x "$LAUNCHER" ]]; then
  echo "ERROR: gla_round_clean.sh not found or not executable at: $LAUNCHER" >&2
  exit 1
fi
```

---

### 2) Environment Toggles (GLA‑only)

These toggles are honored by the Python entry (`train_gla_only.py`) and passed through by the launchers:

- `GLA_USE_FUSED_SWIGLU` (default `0`)
  - `0`: disables `config.fuse_swiglu` and globally patches `fla.modules` to pure PyTorch SwiGLU (stable).
  - `1`: do not patch, use fused SwiGLU from `fla` (faster but depends on triton/toolchain).
- `GLA_FORCE_LEFT_PAD` (default `1`)
  - `1`: force `tokenizer.padding_side='left'`; if no `pad_token_id` but `eos_token` exists, set `pad_token=eos`. Avoids decoder‑only right‑padding warnings.
  - `0`: respect tokenizer’s original padding policy.
- `GLA_USE_MAX_NEW_TOKENS` (default `1`)
  - `1`: interpret `EVAL_GEN_MAX_LENGTH`/`EVAL_GEN_MIN_LENGTH` as `max_new_tokens`/`min_new_tokens`. Uses HF `.generate(max_new_tokens=...)`. If `min_new_tokens` not supported by your transformers version, we fail fast with a clear error (set this to `0` or upgrade).
  - `0`: legacy semantics: total length = `prompt_len + EVAL_GEN_MAX_LENGTH` (and `min_length = prompt_len + EVAL_GEN_MIN_LENGTH`).
- `GLA_VERBOSE` (default `0`):
  - `1`: extra logging for generate semantics & fused SwiGLU patch decisions.

Both launchers propagate these toggles:

```166:177:../mamba-peft/scripts/train/new/gla_round_clean.sh
echo "ENV_OVERRIDES:"
for _k in \
  ... \
  GLA_USE_FUSED_SWIGLU GLA_FORCE_LEFT_PAD GLA_USE_MAX_NEW_TOKENS GLA_VERBOSE \
  EVAL_GEN EVAL_GEN_MAX_LENGTH EVAL_GEN_MIN_LENGTH EVAL_GEN_NUM_BEAMS \
  ...
```

```117:121:../mamba-peft/scripts/train/new/gla_batch_tmux_clean.sh
# GLA-specific toggles
printf 'export GLA_USE_FUSED_SWIGLU=%q\n' "${GLA_USE_FUSED_SWIGLU:-}"
printf 'export GLA_FORCE_LEFT_PAD=%q\n' "${GLA_FORCE_LEFT_PAD:-}"
printf 'export GLA_USE_MAX_NEW_TOKENS=%q\n' "${GLA_USE_MAX_NEW_TOKENS:-}"
printf 'export GLA_VERBOSE=%q\n' "${GLA_VERBOSE:-}"
```

---

### 3) Correct Command Lines (Replaces legacy Mamba‑polluted path)

Old (incorrect) batch command used `gla_batch_tmux.sh` (Mamba‑decoder path). Replace with `gla_batch_tmux_clean.sh`:

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export NLTK_DATA=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/nltk_data
export SPIDER_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/spider_data

# GLA-only toggles (recommended defaults)
export GLA_FORCE_LEFT_PAD=1
export GLA_USE_MAX_NEW_TOKENS=1
export GLA_USE_FUSED_SWIGLU=0
export GLA_VERBOSE=1

# Generation (for Spider / DART / SAMSum, etc.)
export EVAL_GEN=1
export EандVAL_GEN_MAX_LENGTH=256
export EVAL_GEN_MIN_LENGTH=0
export EVAL_GEN_NUM=1

# Trainer cadence
export HP_EVAL_STEPS=1500
export HP_SAVE_STEPS=1500
export HP_LOGGING_STEPS=100

# Runtime
export SWANLAB_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUM_DATA_WORKERS=8
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1

# Batch launcher (tmux). Runs E5 suite across all YAMLs, for given seed/dataset.
./gla_batch_tmux_clean.sh \
  --suite E5 \
  --round all \
  --pairs "87:spider-tvt" \
  --gpus "7" \
  --gpu-plan "1"
```

Notes:
- If you see a `TypeError` complaining about `min_new_tokens`, either:
  - `export GLA_USE_MAX_NEW_TOKENS=0` (fallback to legacy `max_length = prompt_len + N`), or
  - upgrade `transformers` to a version supporting `min_new_tokens`.
- Set `--gpu-plan` to the desired per‑GPU concurrency. With `--gpus "7"` and `--gpu-plan "1"`, one job at a time on GPU 7.

Single-shot (no tmux) using `gla_round_clean.sh` directly:

```bash
export DATA=spider-tvt
export HP_SEED=87
bash gla_round_clean.sh E5 all
# This will iterate over ROUND_E5 YAMLs, inject DATA/GRADIENT_CHECKPOINTING/NUM_DATA_WORKERS, and call train_gla_only.py
```

---

### 4) What to Look for in Logs

- The launchers print the full `ENV_OVERRIDES`, including `GLA_*` and `EVAL_GEN_*`. Example snippet:

```
ENV_OVERRIDES:
  GLA_FORCE_LEFT_PAD=1
  GLA_USE_MAX_NEW_TOKENS=1
  GLA_USE_FUSED_SWIGLU=0
  GLA_VERBOSE=1
  EVAL_GEN=1
  EVAL_GEN_MAX_LENGTH=256
  EVAL_GEN_MIN_LENGTH=0
  EVAL_GEN_NUM_BEAMS=1
  ...
```

- In Python logs:
  - Loader: `"[GLA] fuse_swiglu disabled; globally patching fla.modules..."` (unless `GLA_USE_FUSED_SWIGLU=1`).
  - Tokenizer policy: `"[GLA] Using left padding..."` (if `GLA_FORCE_LEFT_PAD=1`).
  - Generate semantics: printed when `GLA_VERBOSE=1` (max_new_tokens vs legacy).
  - Resume: `"[GLA] Resuming from checkpoint: ..."` only when `--resume` and a valid `checkpoint-*` exists; otherwise a clear error is raised.

---

### 5) Resume Behavior (Strict)

If you pass `--resume`, the trainer will refuse to start unless a `checkpoint-*` folder exists in the `output_dir`. The last checkpoint is auto‑discovered and passed to HF Trainer. This prevents ambiguous “resume from nothing” behaviors.

---

### 6) Known Pitfalls & Remedies

- Decoder‑only right padding warning:
  - Ensure `GLA_FORCE_LEFT_PAD=1` and that the tokenizer has either `pad_token_id` or an `eos_token` for fallback.
  - Our entry will construct and pass `attention_mask` to `.generate()`.
- `min_new_tokens` not supported:
  - Set `GLA_USE_MAX_NEW_TOKENS=0` or upgrade `transformers`.
- Fused SwiGLU instability on some triton/driver stacks:
  - Leave `GLA_USE_FUSED_SWIGLU=0` (default). Enable with `1` only if your environment is verified.

---

### 7) Verifying the Clean Path (no Mamba decoder)

This pipeline never imports or uses `mamba_ssm_peft/utils/decoder.py` or `mamba_beam_search`. You can inspect the launchers and the entry:

```460:466:../mamba-peft/scripts/train/new/gla_round_clean.sh
HP_SEED=${FORCE_SEED} CUDA_VISIBLE_DEVICES="$GPU" \
  python "train_gla_only.py" --cfg "$CFG_INJ" --overwrite &
```

`train_gla_only.py` uses `GenericLMTrainer` and a HF `.generate()` decoder; no Mamba dependency.

---

### 8) Optional: Switch on Weights & Biases

By default, we keep `report_to="none"` in the trainer args to avoid accidental logging. To enable W&B:

```bash
export USE_WANDB=1
export WANDB_PROJECT=your_project
export WANDB_NAME=your_experiment_name
```

Then in `train_gla_only.py`, set `report_to = "wandb"` when `USE_WANDB=1` (if you decide to enable it).

---

### 9) SwanLab Integration (Recommended Alternative to W&B)

For cloud-based experiment tracking with automatic email notifications:

```bash
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="your_project"
export SWANLAB_EMAIL_YAML="/path/to/email_config.yaml"
```

**Note**: Common tokenizer padding warnings are automatically filtered out to prevent excessive logging and connection issues.

---

### 9) FAQ / Debugging

- “Why is my output repeating (e.g. `199999...`)?”
  - Ensure you are on the GLA‑only path (no Mamba decoder). Use this guide’s launchers.
  - Use `GLA_USE_MAX_NEW_TOKENS=1` and `GLA_FORCE_LEFT_PAD=1`.
  - Try `EVAL_GEN_NUM_BEAMS=4` to rule out degenerate greedy outputs on your prompt.
- “How do I sanity‑check the model outside training?”
  - Use `mamba-peft/debug/spider_debug.py` (already patched to HF `.generate()` and left padding).

---

### 10) One‑shot Cheat Sheet

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

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

export GLA_FORCE_LEFT_PAD=1 

export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

export HP_EVAL_STEPS=1500
export HP_SAVE_STEPS=1500
export HP_LOGGING_STEPS=100
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-spider-1-4090-E155-mail02-r3"
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

export SWANLAB_EMAIL_ON_START=1
export SWANLAB_EMAIL_ON_FINISH=1

./gla_batch_tmux_clean.sh \
  --suite E155 \
  --round all \
  --pairs "87:spider-tvt" \
  --gpus "0 1 3 4 6" \
  --gpu-plan "2,2,2,2,2"
```

This is the GLA‑only, HF‑native `.generate()` path with strict error reporting and no Mamba decoder/resume contamination.

---

### Modern LR Scheduler Configuration (Recommended)

The pipeline now supports advanced learning rate scheduling with warmup for better convergence. This is controlled via environment variables and applies to all experiments without modifying individual YAML files.

#### Using Modern LR Scheduler (Recommended)
```bash
# Enable cosine annealing with warmup (most advanced approach)
export LR_SCHEDULER_TYPE=cosine     # Options: constant|linear|cosine|polynomial
export LR_WARMUP_RATIO=0.1          # Warmup steps as ratio of total training steps (recommended: 0.05-0.15)

# Alternative: set fixed warmup steps
# export LR_WARMUP_STEPS=500
```

**Benefits:**
- **Warmup phase**: Prevents early training instability
- **Cosine annealing**: Smooth LR decay following cosine curve, often better than linear decay
- **Better convergence**: Typically achieves 1-3% higher final performance

#### Keeping Traditional Fixed LR (Backward Compatible)
```bash
# No LR scheduler variables set = constant LR throughout training
# This matches the behavior of your existing YAML files
# export LR_SCHEDULER_TYPE=constant  # Optional explicit setting
```

**When to use each:**
- **Modern LR**: For new experiments where you want optimal convergence (recommended)
- **Fixed LR**: When reproducing existing results or if you have YAML files with carefully tuned LR schedules


```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

# 本地 SamSum CSV 目录（需包含 train.csv / validation.csv / test.csv 或 dev.csv）
export SAMSUM_LOCAL_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/samsum

# 生成评测（ROUGE）
export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=128
export EVAL_GEN_MIN_LENGTH=8
export EVAL_GEN_NUM_BEAMS=5

# 解码/生成策略（GLA）
export GLA_FORCE_LEFT_PAD=1
export GLA_USE_MAX_NEW_TOKENS=1
export GLA_VERBOSE=1
export GLA_USE_FUSED_SWIGLU=0

# 训练超参（覆盖 YAML）
export HP_EVAL_STEPS=1000     # ← 从 500 调整为 1000
export HP_SAVE_STEPS=1000     # ← 从 500 调整为 1000
export HP_LOGGING_STEPS=50

# Modern LR scheduler (recommended for better convergence)
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

# LoRA 初始化（两种二选一；推荐第一种直接生效）
export HP_INIT=pissa_niter_4
# 或：export HP_PISSA_FAST=1   #（当 JSON 里是 "pissa" 时自动切到 fast 变体）

# DataLoader 与内存
export NUM_DATA_WORKERS=2      # ← 从 8 调整为 2（SamSum 小数据 + 多并发）
export DATALOADER_PREFETCH_FACTOR=2
export DATALOADER_PIN_MEMORY=1
export DATALOADER_PERSISTENT_WORKERS=0
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SWANLAB_EMAIL_ON_START=1
export SWANLAB_EMAIL_ON_FINISH=1
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-samsum-E15-clean-decoder"

./gla_batch_tmux_clean.sh \
  --suite E15 \
  --round all \
  --pairs "87:samsum" \
  --gpus "7" \
  --gpu-plan "1"
```


```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft
python -m scripts.utils.email_notify --event STARTED --group smoketest --yaml "$SWANLAB_EMAIL_YAML"
```

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

########################################
# ↓↓↓ 覆盖旧版本中已有的变量（只是数值不同） ↓↓↓
########################################

# ✔ 替换 EVAL_GEN 配置
export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=128     
export EVAL_GEN_MIN_LENGTH=8        
export EVAL_GEN_NUM_BEAMS=5        

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
export SWANLAB_PROJECT="gla-samsum-E15-clean-decoder-r1-2-4090"
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

./gla_batch_tmux_clean.sh \
  --suite E15 \
  --round 1 \
  --pairs "87:samsum" \
  --gpus "3 4 5 6" \
  --gpu-plan "2,2,2,2"


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

```



```bash

```