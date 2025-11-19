### GLA-only training and evaluation guide (clean pipeline)

This is the canonical, Mamba-free path for training and evaluating GLA models in this repo. It uses:
- `mamba-peft/train_gla_only.py` as the Python entry
- `scripts/train/new/gla_round_clean.sh` for one-round orchestration
- `scripts/train/new/gla_batch_tmux_clean.sh` for multi-job tmux batches

Key properties:
- Strict HF-native generation via `model.generate()` (no custom mamba decoders)
- Left padding enforced by default for decoder-only generation
- Optional SwanLab logging via environment variables
- Clear/explicit erroring on misconfigurations or missing files

---

### 0) Environment prerequisites

```bash
conda activate mzsz
cd /Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft
```

Optional HF mirrors/caches (adjust to your environment as needed):

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$HOME/.cache/hf"
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
```

Spider evaluation data and NLTK path:

```bash
export SPIDER_LOCAL_DIR="/Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft/data/spider_data"
export NLTK_DATA="/Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft/data/nltk_data"
```

Recommended general settings:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

### 1) Core GLA toggles (env)

- Left padding policy
  - `GLA_FORCE_LEFT_PAD=1` (default; force left padding for decoder-only generation)
- Generation length semantics
  - `GLA_USE_MAX_NEW_TOKENS=1` (default; use `max_new_tokens`/`min_new_tokens`)
- Verbose logs
  - `GLA_VERBOSE=1` (optional; prints helpful runtime info)

Example:

```bash
export GLA_FORCE_LEFT_PAD=1
export GLA_USE_MAX_NEW_TOKENS=1
export GLA_VERBOSE=1
```

---

### 2) Generation settings (Spider text-to-SQL)

Use these for eval-time generation runs:

```bash
export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=256     # or use max_new_tokens semantics if enabled
export EVAL_GEN_MIN_LENGTH=0
export EVAL_GEN_NUM_BEAMS=1        # set >1 for beam search
```

Notes:
- With `GLA_USE_MAX_NEW_TOKENS=1` (default), `EVAL_GEN_MAX_LENGTH`/`EVAL_GEN_MIN_LENGTH` are interpreted as `max_new_tokens`/`min_new_tokens`.
- If your transformers version does not support `min_new_tokens`, you will see an explicit error instructing you to set `GLA_USE_MAX_NEW_TOKENS=0` or upgrade.

---

### 3) SwanLab logging (optional)

```bash
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud                 # or 'local'
export SWANLAB_PROJECT="gla-spider"       # your project
export SWANLAB_EXPERIMENT_PREFIX="E5"     # optional prefix
```

SwanLab is wired as a Trainer callback only when `SWANLAB_ENABLE` is truthy. HF’s `report_to` remains `"none"` by default, so WandB will not be used even if its env is set.

---

### 4) Single-round runner (recommended for manual runs)

Use `gla_round_clean.sh` to run a set of YAML configs in a round-robin fashion across specified GPUs.

Basic form:

```bash
cd /Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft/scripts/train/new

# GPU selection: one slot per GPU by default; customize via GPU_PLAN
export GPU_IDS="7"          # which GPUs to use
export GPU_PLAN="3"         # per-GPU concurrency (here: 3 concurrent jobs on GPU 7)

# Choose data and generation options
export DATA="spider-tvt"
export EVAL_GEN=1 EVAL_GEN_MAX_LENGTH=256 EVAL_GEN_MIN_LENGTH=0 EVAL_GEN_NUM_BEAMS=1

# Optional: training cadence
export HP_EVAL_STEPS=1500
export HP_SAVE_STEPS=1500
export HP_LOGGING_STEPS=100
export NUM_DATA_WORKERS=8
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1

# Run suite E5 (all rounds)
bash gla_round_clean.sh E5 all
```

What it does:
- Expands the E5 config list and injects `data: ${DATA}` and `num_data_workers` into temp YAMLs.
- Schedules jobs onto GPU slots based on `GPU_PLAN`.
- Ensures key env vars are echoed at the start for reproducibility.
- Calls `train_gla_only.py` for each job.

Locking behavior:
- If you pass `--lock` to `train_gla_only.py` (e.g., by editing a YAML Cmd to include it), the training function creates a filesystem lock at `share/lock/<output_dir>` and now behaves as:
  - If the lock already exists, the run is skipped.
  - If the lock does not exist, it is created and removed after the run finishes.

---

### 5) Batch tmux runner (run multiple rounds/pairs sequentially)

`gla_batch_tmux_clean.sh` runs multiple `gla_round_clean.sh` jobs inside a single tmux session, each with its own log. Example:

```bash
cd /Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft/scripts/train/new

export GPU_IDS="7"
export GPU_PLAN="3"
export DATA="spider-tvt"

# Evaluate/generation knobs (as above)
export EVAL_GEN=1 EVAL_GEN_MAX_LENGTH=256 EVAL_GEN_MIN_LENGTH=0 EVAL_GEN_NUM_BEAMS=1

# Run two seeds sequentially in one session
./gla_batch_tmux_clean.sh \
  --suite E5 \
  --round all \
  --pairs "87:spider-tvt,127:spider-tvt" \
  --name "batch_glaclean"
```

Notes:
- Detach from tmux with Ctrl-b d, reattach with `tmux attach -t <session>`.
- Master log is in `scripts/train/new/logs/<session>.log`. Each sub-job also has its own log file.
- All relevant env variables (including `SWANLAB_*`, `GLA_*`, `EVAL_GEN*`, etc.) are exported into the tmux environment.

---

### 6) Direct Python entry (advanced)

You can call the Python entry directly if needed:

```bash
cd /Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft
python train_gla_only.py \
  --cfg cfg/my_lora_exp/yaml/E1_QKVO_r8_alpha16.yaml \
  --overwrite \
  --resume
```

For `--resume`, a valid `checkpoint-*` folder must exist under the target `output_dir`, otherwise a clear error is raised.

---

### 7) Verifying correct behavior

- Left padding: on start you should see
  - `[GLA] Using left padding for decoder-only generation (GLA_FORCE_LEFT_PAD=1).`
- Generation semantics:
  - `[GLA] Using HF generate(max_new_tokens/min_new_tokens) semantics (GLA_USE_MAX_NEW_TOKENS=1).`
- If `min_new_tokens` is not supported by your transformers version, you will see an explicit error asking to set `GLA_USE_MAX_NEW_TOKENS=0` or upgrade transformers.
- Missing model files:
  - You will see `FileNotFoundError` pointing at the missing file (e.g., `pytorch_model.bin`), not a deep `torch.load(None, ...)` error.

---

### 8) Common pitfalls

- Repetition during generation:
  - Ensure `GLA_FORCE_LEFT_PAD=1`, `tokenizer.pad_token_id` is set (falls back to `eos_token_id`).
  - Avoid custom decoders; we use HF-native `model.generate()` exclusively here.
- SwiGLU kernels:
  - We always use PyTorch SwiGLU (fused kernels disabled).
- Locks:
  - If a previous run crashed and left a lock under `share/lock/<output_dir>`, remove it to allow a new run.

---

### 9) Minimal reproduction command (Spider, single GPU)

```bash
conda activate mzsz
cd /Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft/scripts/train/new

export SPIDER_LOCAL_DIR="/Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft/data/spider_data"
export NLTK_DATA="/Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft/data/nltk_data"

export GPU_IDS="7"
export GPU_PLAN="3"
export DATA="spider-tvt"

export GLA_FORCE_LEFT_PAD=1
export GLA_USE_MAX_NEW_TOKENS=1
export GLA_VERBOSE=1

export EVAL_GEN=1 EVAL_GEN_MAX_LENGTH=256 EVAL_GEN_MIN_LENGTH=0 EVAL_GEN_NUM_BEAMS=1
export HP_EVAL_STEPS=1500 HP_SAVE_STEPS=1500 HP_LOGGING_STEPS=100
export NUM_DATA_WORKERS=8 GRADIENT_CHECKPOINTING=true LOGITS_TO_KEEP=1

export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-spider-1-4090-E5-try8"

bash gla_round_clean.sh E5 all
```

This is the supported, clean GLA-only path. If something looks off, enable `GLA_VERBOSE=1` and inspect the script logs under `scripts/train/new/logs/`.


