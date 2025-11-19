### Environment variables reference (GLA-only, Spider/HF aligned)

This page documents every env var respected by the clean GLA-only pipeline. It lists purpose, accepted values, defaults, who reads it, and Spider‑friendly recommendations.

—

#### Dataset and resources

- SPIDER_LOCAL_DIR
  - Purpose: Local Spider dataset root (for offline/fast I/O).
  - Values: Absolute path to the Spider data root (must exist).
  - Default: Unset → loaders fall back to conventional paths/HF.
  - Used by: dataset/spider_data.py, metrics/spider/spider.py, launchers (echo).
  - Spider recommendation: Set to your local copy.

- NLTK_DATA
  - Purpose: NLTK data directory used by Spider metrics (e.g., punkt).
  - Values: Absolute path to directory with NLTK datasets.
  - Default: Unset → NLTK resolves per its own search paths.
  - Used by: metrics/spider/spider.py, launchers (echo).
  - Spider recommendation: Set to your local NLTK dir.

—

#### GLA model toggles (HF-compatible)

- GLA_FORCE_LEFT_PAD
  - Purpose: Force left padding for decoder-only generation.
  - Values: 1/true/on or 0/false/off.
  - Default: 1.
  - Used by: train_gla_only.py (tokenizer policy).
  - Spider/HF: Use 1.

- GLA_USE_MAX_NEW_TOKENS
  - Purpose: Use HF generate(max_new_tokens/min_new_tokens) semantics.
  - Values: 1/true/on or 0/false/off (fallback legacy max_length).
  - Default: 1.
  - Used by: GLAHFDecoder (generation semantics).
  - Spider/HF: Use 1. If transformers doesn’t support min_new_tokens, set 0.

- SwiGLU kernels are always non‑fused (PyTorch ops). No env knob.

- GLA_VERBOSE
  - Purpose: Extra prints for generate semantics and SwiGLU patch.
  - Values: 1/true/on or 0/false/off.
  - Default: 0.
  - Used by: GLAHFDecoder, hf.load_gla.
  - Spider/HF: Optional; 1 during debugging.

—

#### Generation/evaluation (Spider task)

- EVAL_GEN
  - Purpose: Force-enable generation evaluation (auto for Spider).
  - Values: 1/true/on or 0/false/off.
  - Default: Auto-enabled if data startswith("spider").
  - Used by: train_gla_only.py (injects eval_gen cfg when true).
  - Spider/HF: Optional; you can keep 1 explicitly.

- EVAL_GEN_MAX_LENGTH
  - Purpose: Max new tokens (when GLA_USE_MAX_NEW_TOKENS=1).
  - Values: Integer > 0.
  - Default: 1024 (when unset and eval_gen is injected).
  - Used by: train_gla_only.py → eval_gen.max_length.
  - Spider/HF: 256 is a reasonable default.

- EVAL_GEN_MIN_LENGTH
  - Purpose: Min new tokens (when GLA_USE_MAX_NEW_TOKENS=1).
  - Values: Integer ≥ 0 (0 not set to min_new_tokens).
  - Default: 5 (when unset and eval_gen is injected).
  - Used by: train_gla_only.py → eval_gen.min_length.
  - Spider/HF: 0 to avoid forced minimum length.

- EVAL_GEN_NUM_BEAMS
  - Purpose: Beam size; if 1 → greedy (fast).
  - Values: Integer ≥ 1.
  - Default: 5 (when unset and eval_gen is injected).
  - Used by: train_gla_only.py → eval_gen.num_beams; decoder only sets beam search when >1.
  - Spider/HF: 1 for speed; increase (e.g., 4–5) for quality.

- SPIDER_EVAL_EXEC
  - Purpose: Whether to include execution accuracy (exec) in Spider evaluation.
  - Values: 1/true/on (enable exec+match) | 0/false/off (match only).
  - Default: 0 (match only; resilient against SQLite corner cases).
  - Used by: metrics/spider/spider.py (etype = 'all' if enabled, else 'match').

- SPIDER_EVAL_EXEC_SKIP_BAD_GOLD
  - Purpose: When exec is enabled, exclude examples whose gold SQL fails to execute under SQLite from the exec denominator.
  - Values: 1/true/on (skip) | 0/false/off (count as 0).
  - Default: 1 to align more closely with paper metrics ; set to 0 (count as 0 to keep totals stable).
  - Used by: metrics/spider/evaluation.py (adjusts exec denominator).

—

#### Trainer cadence and splits

- HP_LOGGING_STEPS
  - Purpose: Trainer logging interval (steps).
  - Values: Integer > 0.
  - Default: min(50, iters_per_epoch).
  - Used by: train_gla_only.py.

- HP_EVAL_STEPS
  - Purpose: Eval interval (steps).
  - Values: Integer > 0.
  - Default: iters_per_epoch × eval_epochs.
  - Used by: train_gla_only.py.

- HP_SAVE_STEPS
  - Purpose: Checkpoint save interval (steps).
  - Values: Integer > 0.
  - Default: iters_per_epoch × eval_epochs.
  - Used by: train_gla_only.py.

- HP_VAL_SPLIT
  - Purpose: Override validation split.
  - Values: train | val | test.
  - Default: cfg value (usually val).
  - Used by: train_gla_only.py.

—

#### Performance knobs

- NUM_DATA_WORKERS
  - Purpose: DataLoader workers; injected into YAML.
  - Values: Integer ≥ 0.
  - Default: 8 (in launcher if unset).
  - Used by: gla_round_clean.sh → YAML → train_gla_only.py.

- GRADIENT_CHECKPOINTING
  - Purpose: Enable gradient checkpointing (non‑reentrant by default).
  - Values: 1/true/on to enable.
  - Default: Off unless explicitly set.
  - Used by: gla_round_clean.sh → YAML → train_gla_only.py.

- LOGITS_TO_KEEP
  - Purpose: Optional diagnostic knob stored in args.info.
  - Values: Integer ≥ 1.
  - Default: Unset (ignored).
  - Used by: gla_round_clean.sh → YAML → trainer info.

—

#### System/runtime

- PYTORCH_CUDA_ALLOC_CONF
  - Purpose: CUDA memory allocation behavior.
  - Values: e.g., expandable_segments:True.
  - Used by: Torch runtime (env only).

- TOKENIZERS_PARALLELISM
  - Purpose: Disable tokenizer parallel threads/warnings.
  - Values: true | false.
  - Used by: HF tokenizers (env only).

- OMP_NUM_THREADS, MKL_NUM_THREADS
  - Purpose: CPU threading controls.
  - Values: Integers ≥ 1.
  - Used by: Torch/NumPy/MKL (env only).

—

#### Logging/experiment tracking

- SWANLAB_ENABLE
  - Purpose: Enable SwanLab logging via Trainer callback.
  - Values: 1/true/on (or “cloud”/“local”) to enable; 0/off to disable.
  - Default: Off.
  - Used by: train_gla_only.py (adds SwanLabCallback if enabled).

- SWANLAB_MODE
  - Purpose: SwanLab mode.
  - Values: cloud | local.
  - Default: Unset (SwanLab default).
  - Used by: train_gla_only.py (callback init).

- SWANLAB_PROJECT
  - Purpose: SwanLab project name.
  - Values: String.
  - Default: “gla-peft”.
  - Used by: train_gla_only.py (callback init).

- SWANLAB_EXPERIMENT_PREFIX
  - Purpose: Prefix for experiment name (= prefix + output_dir name).
  - Values: String.
  - Default: Unset.
  - Used by: train_gla_only.py.

—

#### GPU scheduling (launchers)

- --gpus / GPU_IDS
  - Purpose: Choose which GPU IDs to use.
  - Values: e.g., "7" or "0 1".
  - Used by: gla_batch_tmux_clean.sh → env → gla_round_clean.sh.

- --gpu-plan / GPU_PLAN
  - Purpose: Per‑GPU concurrency (slots per GPU).
  - Values: Integer or list (e.g., "1" or "3,3").
  - Used by: gla_batch_tmux_clean.sh → env → gla_round_clean.sh.

- --pairs
  - Purpose: List of seed:data pairs (e.g., "87:spider-tvt").
  - Values: Comma or space separated.
  - Used by: gla_batch_tmux_clean.sh; injected into each job (HP_SEED + DATA).

—

### Spider‑ready example (minimal)

```bash
export SPIDER_LOCAL_DIR=/path/to/spider_data
export NLTK_DATA=/path/to/nltk_data

export GLA_FORCE_LEFT_PAD=1
export GLA_USE_MAX_NEW_TOKENS=1
export GLA_USE_FUSED_SWIGLU=0
export GLA_VERBOSE=1

export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=256
export EVAL_GEN_MIN_LENGTH=0
export EVAL_GEN_NUM_BEAMS=1

export HP_EVAL_STEPS=1500
export HP_SAVE_STEPS=1500
export HP_LOGGING_STEPS=100

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUM_DATA_WORKERS=8
export GRADIENT_CHECKPOINTING=true
export LOGITS_TO_KEEP=1
```

This aligns with the HF GLA generation API: `.generate()` with left padding and `max_new_tokens` semantics. For more end‑to‑end instructions, see `GLA_ONLY_RUN_GUIDE.md`.

