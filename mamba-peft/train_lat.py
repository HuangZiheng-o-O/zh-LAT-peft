"""
Unified Linear Attention Training Entry Point.

This script provides a unified interface for training various Linear Attention models
from the FLA library, including GLA, RetNet, DeltaNet, and Mamba2.

Design Principles:
==================
1. **Backward Compatibility**: When model_type="gla", behavior is identical to train_gla_only.py
2. **Unified Interface**: All models use the same training flow and configuration
3. **Auto-Detection**: Model type can be automatically detected from config.json

Supported Models:
================
- gla: Gated Linear Attention (https://arxiv.org/abs/2312.06635)
- retnet: Retentive Network (https://arxiv.org/abs/2307.08621)
- delta_net: DeltaNet (https://arxiv.org/abs/2406.06484)
- mamba2: Mamba2 State Space Model (https://arxiv.org/abs/2405.21060)

Environment Variables:
=====================
- MODEL_TYPE: Model type override (gla, retnet, delta_net, mamba2, auto)
- LAT_* / GLA_*: Various configuration options (see documentation)

Usage:
======
    # With explicit model type
    python train_lat.py --cfg configs/gla.yaml --model-type gla

    # With auto-detection
    python train_lat.py --cfg configs/model.yaml --model-type auto

    # GLA backward compatibility (same as train_gla_only.py)
    python train_lat.py --cfg configs/gla.yaml
"""

import sys
from pathlib import Path

# --- ensure local 'fla' submodule is importable when running from mamba-peft/ ---
try:
    import fla  # noqa: F401
except Exception:
    try:
        repo_root = Path(__file__).resolve().parents[1]  # .../zh-LAT-peft
        fla_symlink = repo_root / "fla"
        if fla_symlink.exists():
            sys.path.insert(0, str(repo_root))
            import fla  # noqa: F401
    except Exception:
        pass

import json
import os
import shutil
from typing import Optional, Dict

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader  # noqa: F401  # kept for compatibility

import yaml

os.environ["WANDB_PROJECT"] = "mamba-peft"

from dataset import load_dataset
from trainer.generic_lm_trainer import GenericLMTrainer, GenericLMTrainingArguments
from mamba_ssm_peft import get_trainable_parameters_ratio, print_trainable_parameter_names
from utils.runtime_stats import gpu_memory_tracker

# Unified imports
from lat_adapter import prepare_lat_model_and_tokenizer
from mamba_ssm_peft.utils.lat_decoder import create_lat_decoder
from mamba_ssm_peft.utils.lat_model_loader import get_lat_env, get_lat_env_bool
from utils.sparse_selective_engine import maybe_run_sparse_selective_tuning

REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_sparse_run_enabled() -> bool:
    flag = os.environ.get("HP_SPARSE_ENABLE") or os.environ.get("LAT_SPARSE_ENABLE")
    if flag is None:
        return False
    return str(flag).strip().lower() in ("1", "true", "yes", "on")


def _default_output_root(*suffix: str) -> Path:
    root = REPO_ROOT / "output" / ("sparse" if _is_sparse_run_enabled() else "benchmark")
    for part in suffix:
        root /= part
    return root


def _sparse_run_suffix() -> str:
    """
    Stable suffix to append to output folders / experiment names when sparse tuning is enabled.
    """
    try:
        from utils.sparse_selective_engine import SparseSelectiveConfig  # local import to avoid cycles

        cfg = SparseSelectiveConfig.from_env()
    except Exception:
        return ""

    if not getattr(cfg, "enabled", False):
        return ""

    scope_map = {
        "lora_only": "LoraOnly",
        "base_only": "BaseOnly",
        "hybrid": "Hybrid",
        "lora_dense_base_sparse": "LoraDenseBaseSparse",
    }
    scope_raw = str(getattr(cfg, "scope", ""))
    scope = scope_map.get(scope_raw.lower().strip(), scope_raw)

    mode = str(getattr(cfg, "budget_mode", "")).lower().strip()
    budget: str
    if mode == "fixed_ratio":
        try:
            pct = int(round(float(getattr(cfg, "rho", 0.0)) * 100))
            budget = f"R{pct}"
        except Exception:
            budget = "R?"
    elif mode == "fixed_count":
        k = getattr(cfg, "k", None)
        budget = f"K{int(k)}" if k is not None else "KNA"
    elif mode == "match_reference":
        ref = getattr(cfg, "reference_cfg", "") or "REF"
        ref_stem = Path(ref).stem if ref else "REF"
        budget = f"REF_{ref_stem}"
    else:
        budget = mode.upper() if mode else "BUDGET"

    suffix = f"_SPARSE_{scope}_{budget}"
    return suffix.replace("/", "_").replace(" ", "_")


DEFAULT_NOW_OUTPUT_ROOT = _default_output_root()
NOW_OUTPUT_ROOT = Path(os.environ.get("LAT_OUTPUT_ROOT", DEFAULT_NOW_OUTPUT_ROOT)).expanduser()
DATASET_ROOT_NAME = os.environ.get("LAT_DATASET_ROOT_NAME", "glue")


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")



def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError as e:
        raise ValueError(
            f"Environment variable '{name}' must be a float, got '{v}'"
        )


def _apply_sparse_env_defaults_from_cfg(cfg: Dict) -> None:
    """
    YAML-driven defaults for sparse selective tuning.

    Contract (per your requirement):
    - ENV has higher priority than YAML.
    - Therefore we only set HP_SPARSE_* if that env var is currently unset/empty.
    - Pure LoRA YAMLs do not contain this block, so behavior is unchanged.

    Supported YAML shape:
      sparse_selective:
        enable: true|false
        scope: lora_only|base_only|hybrid|lora_dense_base_sparse
        budget_mode: fixed_ratio|fixed_count|match_reference
        rho: 0.3
        k: 7569408
        score_samples: 1024
        reference_cfg: /abs/or/relative/path.yaml   # optional; env still overrides
    """
    def _set_default_env(k: str, v) -> None:
        if v is None:
            return
        cur = os.environ.get(k)
        if cur is None or str(cur).strip() == "":
            os.environ[k] = str(v)

    # Optional generic env defaults block (useful for batch YAMLs):
    #   env_defaults:
    #     HP_INIT: pissa
    #     HP_SAVE_MODE: best_last
    #     HP_SAVE_FULL_MODEL: 0
    env_defaults = cfg.get("env_defaults")
    if isinstance(env_defaults, dict):
        for k, v in env_defaults.items():
            if not isinstance(k, str):
                continue
            _set_default_env(k, v)

    node = cfg.get("sparse_selective")
    if not isinstance(node, dict):
        return

    # Only set defaults (env remains authoritative).
    if "enable" in node:
        _set_default_env("HP_SPARSE_ENABLE", "1" if bool(node.get("enable")) else "0")
    if "scope" in node:
        _set_default_env("HP_SPARSE_SCOPE", str(node.get("scope")))
    if "budget_mode" in node:
        _set_default_env("HP_SPARSE_BUDGET_MODE", str(node.get("budget_mode")))
    if "rho" in node:
        _set_default_env("HP_SPARSE_RHO", node.get("rho"))
    if "k" in node:
        _set_default_env("HP_SPARSE_K", node.get("k"))
    if "score_samples" in node:
        _set_default_env("HP_SPARSE_SCORE_SAMPLES", node.get("score_samples"))
    if "reference_cfg" in node:
        _set_default_env("HP_SPARSE_REFERENCE_CFG", node.get("reference_cfg"))


def _lock_share(name: str, model_type: str = "LAT") -> bool:
    """
    Acquire a simple filesystem lock under share/lock/<name>.
    Returns:
      True  -> lock already exists (another process holds it) – caller SHOULD skip.
      False -> lock created successfully – caller MAY proceed and SHOULD remove it after completion.
    """
    path = Path("share/lock") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[{model_type}][lock] {path} exists; skipping this run to avoid duplicate training.")
        return True
    try:
        with open(path, "x"):
            pass
        print(f"[{model_type}][lock] acquired {path}")
        return False
    except OSError:
        print(f"[{model_type}][lock] {path} exists; skipping this run.")
        return True


def build_and_run_trainer_lat(
    *,
    model,
    tokenizer,
    model_type: str,
    output_dir: str,
    cfg: Dict,
    cfg_path: str,
    learning_rate: float,
    total_steps: int,
    logging_steps: int,
    gradient_accumulation_steps: int,
    num_data_workers: int,
    batch_size: int,
    eval_batch_size: int,
    eval_epochs: int,
    skip_eval: bool,
    no_save: bool,
    eval_steps_override: Optional[int],
    save_steps_override: Optional[int],
    eval_gen: Optional[Dict],
    resume_from_checkpoint: bool,
    min_eval_metric_after_epoch,
    seed: int,
    data: str,
    val_data: Optional[str],
    val_data_split: str,
    debug: bool,
    gradient_checkpointing: bool = False,
    logits_to_keep: int | None = None,
    train_data_module=None,
    val_data_module=None,
):
    """
    Unified Linear Attention training and evaluation entry point:
      - Uses GenericLMTrainer / GenericLMTrainingArguments
      - Generation evaluation uses HF-native model.generate() (create_lat_decoder)
      - Data loading reuses existing dataset modules (Spider/GLUE etc.)
    """
    # Log tag based on model type
    log_tag = model_type.upper() if model_type != "auto" else "LAT"

    # NOTE: We intentionally DO NOT write parameter_counts.json here.
    # For sparse selective tuning, trainable parameters are finalized only after
    # maybe_run_sparse_selective_tuning() finishes (post-sparse).
    # To make parameter_counts.json a trustworthy "post-sparse" artifact, we write it later.
    print("Loaded model")

    # Optional dataset-side max length filter (matches MambaPEFT cutoff_len behavior).
    # If set, samples with (prompt+label) token length > max_seqlen are dropped at preprocessing time.
    _max_seqlen_env = os.environ.get("HP_MAX_SEQLEN") or os.environ.get("LAT_MAX_SEQLEN")
    max_seqlen = int(_max_seqlen_env) if _max_seqlen_env not in (None, "") else None

    # Force left padding for decoder-only generation
    _force_left = get_lat_env_bool("FORCE_LEFT_PAD", "1")
    if _force_left and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if get_lat_env_bool("VERBOSE"):
            print(f"[{log_tag}] Using left padding for decoder-only generation.")

    # Build train data module (reuse pre-built module when provided)
    if train_data_module is None:
        train_data_module = load_dataset(data, tokenizer, "train", return_module=True, max_seqlen=max_seqlen)

    # Save cfg.yaml
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "cfg.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # ---------------------------------------------------------------------
    # Sparse Selective Tuning (Gradient + Static + Global Top-K)
    #
    # Requirement constraints:
    # - must not modify GenericLMTrainer training loop
    # - must run after model/tokenizer exist and LoRA (if any) is injected
    # - must run before optimizer is constructed (HF Trainer constructs optimizer later)
    #
    # We implement this by registering backward hooks: grad <- grad * mask
    # and persisting mask+metadata under output_dir for resume/reuse.
    # ---------------------------------------------------------------------
    try:
        sparse_meta = maybe_run_sparse_selective_tuning(
            model=model,
            train_dataset=train_data_module.dataset,
            data_collator=train_data_module.data_collator,
            batch_size=batch_size,
            output_dir=output_dir,
            cfg_path=cfg_path,
            model_type=model_type,
        )
        # If resuming and sparse is enabled, load sparse delta snapshot from checkpoint unless full model is saved.
        _sfm_env = str(os.environ.get("HP_SAVE_FULL_MODEL", "") or os.environ.get("LAT_SAVE_FULL_MODEL", "")).lower()
        _save_full_model = _sfm_env in ("1", "true", "yes", "on")
        if (sparse_meta is not None) and resume_from_checkpoint and (not _save_full_model):
            from utils.sparse_selective_engine import load_sparse_delta_snapshot_strict
            load_sparse_delta_snapshot_strict(model, resume_from_checkpoint)
    except Exception as _sparse_e:
        # Fail-fast when explicitly enabled; otherwise never triggered.
        raise

    # ------------------------------------------------------------
    # Post-sparse parameter accounting (authoritative)
    #
    # Hard requirement: parameter_counts.json must reflect the FINAL trainable
    # parameters used in training (post-sparse / post-resume restore).
    # ------------------------------------------------------------
    print_trainable_parameter_names(model, output_dir=output_dir, cfg_path=cfg_path)

    # Build generation decoder
    eval_generator = None
    if eval_gen is not None:
        _eval = dict(eval_gen)
        max_length = int(_eval.get("max_length", 1024))
        min_length = int(_eval.get("min_length", 5))
        eval_generator = create_lat_decoder(
            tokenizer,
            model_type=model_type,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )

    # Build validation data module
    if val_data_module is None:
        val_data_module = load_dataset(
            val_data if val_data is not None else data,
            tokenizer,
            val_data_split,
            mode="lm" if eval_generator is None else "gen",
            return_module=True,
            max_seqlen=max_seqlen,
        )
    compute_metrics = val_data_module.dataset.compute_metrics

    # Debug mode: truncate dataset size
    if debug:
        train_data_module.dataset = torch.utils.data.Subset(
            train_data_module.dataset, range(8)
        )
        val_data_module.dataset = torch.utils.data.Subset(
            val_data_module.dataset, range(2)
        )

    # Gradient checkpointing kwargs
    _gc_kwargs = {"use_reentrant": False} if gradient_checkpointing else None

    # Save optimizer state control
    _sos_env = str(os.environ.get("SAVE_OPTIMIZER_STATE", "")).lower()
    _save_optimizer_state = _sos_env in ("1", "true", "yes", "on")

    # Save full model weights (model.pt) instead of PEFT adapter-only save_pretrained.
    # This is REQUIRED for sparse base-only / hybrid modes, because they modify base weights.
    _sfm_env = str(os.environ.get("HP_SAVE_FULL_MODEL", "") or os.environ.get("LAT_SAVE_FULL_MODEL", "")).lower()
    _save_full_model = _sfm_env in ("1", "true", "yes", "on")

    # DataLoader configuration from env
    def _env_int(name: str, default: int) -> int:
        v = os.environ.get(name)
        if v is None or str(v).strip() == "":
            return default
        return int(v)

    _prefetch = _env_int("DATALOADER_PREFETCH_FACTOR", 2)
    _pin_memory = _env_bool("DATALOADER_PIN_MEMORY", True)
    _persist_workers = _env_bool("DATALOADER_PERSISTENT_WORKERS", False)
    _eval_acc_steps = _env_int("EVAL_ACCUMULATION_STEPS", 128)

    # LR scheduler configuration
    _lr_scheduler_type = os.environ.get("LR_SCHEDULER_TYPE", "constant")
    _warmup_steps = _env_int("LR_WARMUP_STEPS", None)
    _warmup_ratio = _env_float("LR_WARMUP_RATIO", 0.1)
    if _warmup_steps is None and _warmup_ratio > 0:
        _warmup_steps = int(_warmup_ratio * total_steps)

    # Optional SwanLab integration
    callbacks = []
    _sl_enable = str(os.environ.get("SWANLAB_ENABLE", "")).lower() in ("1", "true", "yes", "on", "cloud", "local")
    if _sl_enable:
        try:
            import warnings
            warnings.filterwarnings("ignore", message=".*For correct generation results, please set.*padding_side.*left.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*decoder-only architecture is being used, but right-padding was detected.*", category=UserWarning)

            from swanlab.integration.transformers import SwanLabCallback
            sl_project = os.environ.get("SWANLAB_PROJECT", f"{model_type}-peft")
            exp_prefix = os.environ.get("SWANLAB_EXPERIMENT_PREFIX", "")
            exp_name = Path(output_dir).name
            _sparse_suffix = _sparse_run_suffix()
            if _sparse_suffix and not exp_name.endswith(_sparse_suffix):
                exp_name = f"{exp_name}{_sparse_suffix}"
            if exp_prefix:
                exp_name = f"{exp_prefix}_{exp_name}"
            sl_mode = os.environ.get("SWANLAB_MODE", "")
            if sl_mode:
                callbacks.append(SwanLabCallback(project=sl_project, experiment_name=exp_name, mode=sl_mode))
            else:
                callbacks.append(SwanLabCallback(project=sl_project, experiment_name=exp_name))
            # Email callback setup (same as GLA)
            try:
                import swanlab
                from swanlab.plugin.notification import EmailCallback
                email_yaml = os.environ.get("SWANLAB_EMAIL_YAML", "dangerous/email_notify.yaml")
                if Path(email_yaml).is_file():
                    with open(email_yaml, "r") as _ef:
                        _ecfg = yaml.safe_load(_ef) or {}
                    if all(k in _ecfg for k in ("sender_email", "receiver_email", "password", "smtp_server", "port")):
                        _email_cb = EmailCallback(
                            sender_email=str(_ecfg["sender_email"]),
                            receiver_email=str(_ecfg["receiver_email"]),
                            password=str(_ecfg["password"]),
                            smtp_server=str(_ecfg["smtp_server"]),
                            port=int(_ecfg.get("port", 587)),
                            language=str(_ecfg.get("language", "zh")),
                        )
                        swanlab.register_callbacks([_email_cb])
                        _start_env = str(os.environ.get("SWANLAB_EMAIL_ON_START", "1")).lower()
                        if _start_env in ("1", "true", "yes", "on"):
                            try:
                                _msg = f"Output: {output_dir}\nData: {cfg.get('data')}\nSeed: {cfg.get('seed')}\nCfg: {cfg_path}"
                                _email_cb.send_email(subject=f"SwanLab | STARTED | {exp_name}", content=_msg)
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception as e:
            print(f"[{log_tag}][swanlab][warn] Failed to initialize SwanLabCallback: {e}")

    _eval_batch_size = int(cfg.get("eval_batch_size", 1) or 1)

    # ---------------------------------------------------------------------
    # Checkpoint policy (disk-safe, paper-style, non-cheating)
    #
    # Recommended default:
    # - save_total_limit=2  (keep last + best)
    # - load_best_model_at_end=True (select best on *validation*, NOT test)
    # - metric_for_best_model=eval_loss (always available; lower is better)
    #
    # You can override via env:
    #   HP_SAVE_TOTAL_LIMIT / LAT_SAVE_TOTAL_LIMIT / SAVE_TOTAL_LIMIT
    #   HP_LOAD_BEST_MODEL_AT_END / LAT_LOAD_BEST_MODEL_AT_END
    #   HP_METRIC_FOR_BEST_MODEL / LAT_METRIC_FOR_BEST_MODEL
    #   HP_GREATER_IS_BETTER / LAT_GREATER_IS_BETTER
    env = os.environ
    def _env_int_opt(name: str) -> int | None:
        v = env.get(name)
        if v is None or str(v).strip() == "":
            return None
        return int(v)

    def _env_bool_opt(name: str) -> bool | None:
        v = env.get(name)
        if v is None or str(v).strip() == "":
            return None
        return str(v).lower() in ("1", "true", "yes", "on")

    # Unified save mode (optional, overrides defaults; keeps backward compatibility if unset):
    #   HP_SAVE_MODE=none|last|best_last
    # - none: no checkpoints, no final snapshot
    # - last: keep last checkpoint only
    # - best_last: keep best+last (default behavior today)
    save_mode = (env.get("HP_SAVE_MODE") or env.get("LAT_SAVE_MODE") or "").strip().lower()
    if save_mode:
        if save_mode in ("none", "no", "off", "0"):
            no_save = True
        elif save_mode in ("last",):
            no_save = False
        elif save_mode in ("best_last", "best+last", "bestlast"):
            no_save = False
        else:
            raise ValueError(f"[{log_tag}] Unknown HP_SAVE_MODE='{save_mode}' (use none|last|best_last)")

    save_total_limit = (
        _env_int_opt("HP_SAVE_TOTAL_LIMIT")
        or _env_int_opt("LAT_SAVE_TOTAL_LIMIT")
        or _env_int_opt("SAVE_TOTAL_LIMIT")
    )
    if save_total_limit is None:
        if no_save:
            save_total_limit = None
        elif save_mode == "last":
            save_total_limit = 1
        else:
            # default & best_last
            save_total_limit = 2

    load_best_model_at_end = (
        _env_bool_opt("HP_LOAD_BEST_MODEL_AT_END")
        if _env_bool_opt("HP_LOAD_BEST_MODEL_AT_END") is not None
        else _env_bool_opt("LAT_LOAD_BEST_MODEL_AT_END")
    )
    if load_best_model_at_end is None:
        if no_save:
            load_best_model_at_end = False
        elif save_mode == "last":
            load_best_model_at_end = False
        else:
            # default & best_last
            load_best_model_at_end = (not skip_eval)

    metric_for_best_model = env.get("HP_METRIC_FOR_BEST_MODEL") or env.get("LAT_METRIC_FOR_BEST_MODEL") or "eval_loss"
    greater_is_better_env = env.get("HP_GREATER_IS_BETTER") or env.get("LAT_GREATER_IS_BETTER")
    if greater_is_better_env is None or str(greater_is_better_env).strip() == "":
        # Infer direction to avoid accidental "pick worst checkpoint" bugs.
        m = str(metric_for_best_model).lower()
        greater_is_better = not any(k in m for k in ("loss", "ppl", "perplexity"))
    else:
        greater_is_better = str(greater_is_better_env).lower() in ("1", "true", "yes", "on")

    # Compute step intervals once (and validate best-model constraints)
    _steps_per_epoch = int(
        len(train_data_module.dataset) // batch_size
        + (len(train_data_module.dataset) % batch_size > 0)
    )
    if _steps_per_epoch <= 0:
        raise ValueError(f"[{log_tag}] steps_per_epoch computed as {_steps_per_epoch} (dataset too small?)")
    _default_interval = int(eval_epochs * _steps_per_epoch)
    _save_steps = int(save_steps_override) if save_steps_override is not None else int(_default_interval)
    _eval_steps = int(eval_steps_override) if eval_steps_override is not None else int(_default_interval)
    if _save_steps <= 0 or _eval_steps <= 0:
        raise ValueError(f"[{log_tag}] save_steps={_save_steps}, eval_steps={_eval_steps} must be > 0")
    # HF Trainer requirement: when load_best_model_at_end and step-based, save_steps must be multiple of eval_steps.
    if bool(load_best_model_at_end) and (not no_save) and (not skip_eval):
        if _save_steps % _eval_steps != 0:
            raise ValueError(
                f"[{log_tag}] Invalid steps for load_best_model_at_end: save_steps({_save_steps}) "
                f"must be a multiple of eval_steps({_eval_steps}). "
                "Recommend setting HP_SAVE_STEPS == HP_EVAL_STEPS."
            )

    trainer = GenericLMTrainer(
        model=model,
        train_dataset=train_data_module.dataset,
        tokenizer=tokenizer,
        args=GenericLMTrainingArguments(
            learning_rate=float(learning_rate),
            max_steps=total_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=_gc_kwargs,
            optim=cfg.get("optim", "adamw_torch"),
            lr_scheduler_type=_lr_scheduler_type,
            warmup_steps=_warmup_steps,
            output_dir=output_dir,
            logging_steps=logging_steps,
            dataloader_num_workers=num_data_workers,
            dataloader_prefetch_factor=_prefetch,
            dataloader_pin_memory=_pin_memory,
            dataloader_persistent_workers=_persist_workers,
            eval_accumulation_steps=_eval_acc_steps,
            info={
                "trainable_params": get_trainable_parameters_ratio(model),
                "cfg_path": cfg_path,
                "logits_to_keep": logits_to_keep,
                "model_type": model_type,
            },
            save_full_model=_save_full_model,
            save_optimizer_state=_save_optimizer_state,
            save_strategy="steps" if not no_save else "no",
            evaluation_strategy="steps" if not skip_eval else "no",
            save_total_limit=save_total_limit,
            load_best_model_at_end=bool(load_best_model_at_end) if not (no_save or skip_eval) else False,
            metric_for_best_model=str(metric_for_best_model),
            greater_is_better=bool(greater_is_better),
            save_steps=_save_steps,
            eval_steps=_eval_steps,
            dataloader_drop_last=True,
            report_to="none",
            seed=seed,
        ),
        compute_metrics=compute_metrics,
        data_collator=train_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        callbacks=callbacks or None,
        eval_generator=eval_generator,
        min_eval_metric_after_epoch=min_eval_metric_after_epoch,
    )

    # Train with best-effort email notifications
    try:
        with gpu_memory_tracker(output_dir):
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        # Write a stable "final" snapshot into output_dir root.
        # If load_best_model_at_end=True, this is the best checkpoint on validation.
        save_final_env = env.get("HP_SAVE_FINAL_SNAPSHOT") or env.get("LAT_SAVE_FINAL_SNAPSHOT")
        save_final = True
        if save_final_env is not None and str(save_final_env).strip() != "":
            save_final = str(save_final_env).lower() in ("1", "true", "yes", "on")
        # In save_mode=none, default to NOT writing final snapshot (disk-minimal).
        if save_mode in ("none", "no", "off", "0"):
            save_final = False

        if (not no_save) and save_final:
            try:
                # GenericLMTrainer.save_model uses HF/PEFT save_pretrained semantics.
                trainer.save_model(output_dir, _internal_call=True)
            except Exception as _save_e:
                print(f"[{log_tag}][warn] Failed to save final model snapshot to {output_dir}: {_save_e}")
        try:
            _fin_env = str(os.environ.get("SWANLAB_EMAIL_ON_FINISH", "1")).lower()
            if _sl_enable and _fin_env in ("1", "true", "yes", "on"):
                from scripts.utils.email_notify import send_event_email
                send_event_email("FINISHED", group=Path(output_dir).name, details=f"Finished OK: {output_dir}")
        except Exception:
            pass
    except Exception as _e:
        try:
            from scripts.utils.email_notify import send_event_email
            import traceback
            tb = "".join(traceback.format_exception_only(type(_e), _e))
            send_event_email("FAILED", group=Path(output_dir).name, details=f"Failed: {tb}")
        except Exception:
            pass
        raise


def run_train(
    output_dir,
    cfg_path,
    model,
    data,
    model_type: str = "auto",  # NEW: model type parameter
    val_data=None,
    val_data_split="val",
    tokenizer="EleutherAI/gpt-neox-20b",  # Kept for config compatibility, not actually used
    num_epochs=10,
    prec="bf16",
    peft=None,
    optim="adamw_torch",
    learning_rate=5e-4,
    gradient_accumulation_steps=1,
    num_data_workers=8,
    batch_size=4,
    eval_batch_size=1,
    eval_gen=None,
    backend="cuda",  # Kept for config compatibility
    debug=False,
    resume=False,
    overwrite=False,
    lock=False,
    no_save=False,
    skip_eval=False,
    eval_epochs=1,
    min_eval_metric_after_epoch=None,
    seed=42,
    is_sdlora=False,  # Kept for config compatibility
    gradient_checkpointing=False,
    logits_to_keep=None,
):
    """
    Unified Linear Attention run_train entry point.

    Supports GLA, RetNet, Mamba2 and other FLA models through the model_type parameter.
    """
    # Determine model type from env if not specified or "auto"
    if model_type == "auto":
        env_model_type = os.environ.get("MODEL_TYPE", "auto")
        if env_model_type != "auto":
            model_type = env_model_type

    # Log tag
    log_tag = model_type.upper() if model_type != "auto" else "LAT"

    # Legacy SD-LoRA check (kept for compatibility)
    if overwrite and is_sdlora:
        assert Path(output_dir).exists()

    # Snapshot cfg
    cfg = {**locals()}

    created_lock = False
    if not overwrite:
        if lock:
            if _lock_share(str(output_dir), log_tag):
                return
            created_lock = True

        if (Path(output_dir) / "cfg.yaml").exists():
            if resume:
                resume_from_checkpoint = True
            else:
                assert False, str(Path(output_dir) / "cfg.yaml") + " exists!"
        else:
            resume_from_checkpoint = False
    else:
        resume_from_checkpoint = False

    # Safety warning for multi-epoch without saving
    if not (
        data.startswith("glue_")
        or data in ("glue_rte", "glue_mrpc", "glue_cola", "spider_1000")
        or not (no_save and num_epochs > 1)
    ):
        print("Training for more than one epoch without saving ckpts!")

    # Load model using unified adapter
    print(f"Loading {log_tag} model: {model}")
    model_id = model

    # Sparse Selective Tuning: base_only should NOT inject LoRA adapters.
    # Reason: even if LoRA params are frozen, some inits (e.g., PiSSA) are non-zero and would
    # change the forward pass, violating the intended semantics of base-only sparse tuning.
    # We still keep `peft` in cfg/yaml so sparse engine can read target_modules as the candidate pool.
    peft_for_load = peft
    try:
        scope_env = os.environ.get("HP_SPARSE_SCOPE") or os.environ.get("LAT_SPARSE_SCOPE") or ""
        if _is_sparse_run_enabled() and scope_env.strip().lower() == "base_only":
            peft_for_load = None
            print(f"[{log_tag}][sparse] scope=base_only: skipping PEFT/LoRA injection (peft_json_path=None).")
    except Exception:
        pass

    model, tokenizer_obj, _ = prepare_lat_model_and_tokenizer(
        model_type=model_type,
        model_id=model_id,
        prec=prec,
        debug=debug,
        peft_json_path=peft_for_load,
    )

    # Force left padding
    force_left = get_lat_env_bool("FORCE_LEFT_PAD", "1")
    if force_left:
        tokenizer_obj.padding_side = "left"
        if getattr(tokenizer_obj, "pad_token_id", None) is None and getattr(tokenizer_obj, "eos_token", None) is not None:
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
        print(f"[{log_tag}] Using left padding for decoder-only generation.")
    else:
        print(f"[{log_tag}] Respecting tokenizer's original padding policy.")
    # Optional dataset-side max length filter (matches MambaPEFT cutoff_len behavior).
    _max_seqlen_env = os.environ.get("HP_MAX_SEQLEN") or os.environ.get("LAT_MAX_SEQLEN")
    max_seqlen = int(_max_seqlen_env) if _max_seqlen_env not in (None, "") else None
    # Build train data module once (reuse for both length calc and trainer)
    train_data_module = load_dataset(
        data, tokenizer_obj, "train", return_module=True, max_seqlen=max_seqlen
    )

    its_per_epoch = int(
        np.ceil(len(train_data_module.dataset) / batch_size)
    )

    # Logging and steps configuration with env overrides
    env = os.environ
    logging_steps = min(50, its_per_epoch)
    if env.get("HP_LOGGING_STEPS"):
        logging_steps = int(env.get("HP_LOGGING_STEPS"))

    total_steps = int(num_epochs * its_per_epoch)
    if env.get("HP_MAX_STEPS"):
        total_steps = int(env.get("HP_MAX_STEPS"))

    eval_steps_override = None
    save_steps_override = None
    if env.get("HP_EVAL_STEPS"):
        eval_steps_override = int(env.get("HP_EVAL_STEPS"))
    if env.get("HP_SAVE_STEPS"):
        save_steps_override = int(env.get("HP_SAVE_STEPS"))

    os.environ["WANDB_NAME"] = str(output_dir).replace("weights/", "")

    print("Dropping last batch")

    # Resume handling
    resume_arg = None
    if resume_from_checkpoint:
        last_ckpt = _find_last_checkpoint(Path(output_dir))
        if last_ckpt is None:
            raise RuntimeError(f"[{log_tag}] --resume was set but no checkpoint-* found under {output_dir}")
        resume_arg = str(last_ckpt)
        print(f"[{log_tag}] Resuming from checkpoint: {resume_arg}")

    try:
        build_and_run_trainer_lat(
            model=model,
            tokenizer=tokenizer_obj,
            model_type=model_type,
            output_dir=str(output_dir),
            cfg=cfg,
            cfg_path=cfg_path,
            learning_rate=learning_rate,
            total_steps=total_steps,
            logging_steps=logging_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_data_workers=num_data_workers,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_epochs=eval_epochs,
            skip_eval=skip_eval,
            no_save=no_save,
            eval_steps_override=eval_steps_override,
            save_steps_override=save_steps_override,
            eval_gen=eval_gen,
            resume_from_checkpoint=resume_arg,
            min_eval_metric_after_epoch=min_eval_metric_after_epoch,
            seed=seed,
            data=data,
            val_data=val_data,
            val_data_split=val_data_split,
            debug=debug,
            gradient_checkpointing=gradient_checkpointing,
            logits_to_keep=logits_to_keep,
            train_data_module=train_data_module,
        )
    finally:
        if created_lock:
            try:
                lock_path = Path("share/lock") / str(output_dir)
                lock_path.unlink(missing_ok=True)
                print(f"[{log_tag}][lock] released {lock_path}")
            except Exception as e:
                print(f"[{log_tag}][lock][warn] failed to remove lock: {e}")


def get_output_path_for_cfg(cfg_path, cfg, model_type: str = "gla"):
    """
    Target path:
      <NOW_OUTPUT_ROOT>/<model_type>/<data>_seed<seed>/<yaml_stem>
    Fallback (missing data/seed):
      <NOW_OUTPUT_ROOT>/<model_type>/<DATASET_ROOT_NAME>/cola_gla/<yaml_stem>

    Args:
        cfg_path: Path to config YAML file
        cfg: Parsed config dict
        model_type: Model type (gla, retnet, mamba2, deltanet, etc.)

    Returns:
        Output directory path with model_type to prevent cross-model overwrites
    """
    yaml_stem = Path(cfg_path).stem
    data = cfg.get("data")
    seed = cfg.get("seed")

    # Normalize model_type to lowercase for consistent path naming
    model_type = model_type.lower()

    if data and seed is not None:
        # Canonical layout (requested):
        #   sparse/<model_type>/<data>_seed<seed>/<yaml_stem>
        #   benchmark/<model_type>/<data>_seed<seed>/<yaml_stem>
        #
        # Root selection (sparse vs benchmark) is handled by NOW_OUTPUT_ROOT.
        folder = f"{data}_seed{seed}"
        return NOW_OUTPUT_ROOT / model_type / folder / yaml_stem

    # Fallback: use DATASET_ROOT_NAME (glue) for backward compatibility
    base_dir = NOW_OUTPUT_ROOT / model_type / DATASET_ROOT_NAME
    return base_dir / "Fallback" / yaml_stem


def _find_last_checkpoint(root: Path) -> Optional[Path]:
    """
    Scan output directory for checkpoint-* subdirs, return the one with highest step number.
    """
    if not root.exists():
        return None
    candidates = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    if not candidates:
        return None
    def step_of(p: Path) -> int:
        # Expected format: checkpoint-<int>. Non-matching folders sort to -1.
        parts = p.name.split("-")
        if len(parts) != 2:
            return -1
        try:
            return int(parts[1])
        except ValueError:
            return -1
    candidates.sort(key=step_of)
    best = candidates[-1] if candidates else None
    if best is None or step_of(best) < 0:
        return None
    return best


def main():
    parser = argparse.ArgumentParser(description="Unified Linear Attention Training")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--model-type", type=str, default="auto",
                        help="Model type: gla, retnet, mamba2, or auto (default: auto)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--prec")
    parser.add_argument("--device")
    args = parser.parse_args()

    # Allow --device to set VISIBLE_DEVICES
    if args.device is not None:
        os.environ["VISIBLE_DEVICES"] = args.device

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply environment overrides
    env = os.environ

    # ------------------------------------------------------------------
    # Sparse Selective Tuning: allow YAML to provide defaults (ENV wins)
    #
    # This enables "batch run via 8 YAMLs" without requiring per-job env exports,
    # while preserving strict backward compatibility for existing pure-LoRA YAMLs.
    # ------------------------------------------------------------------
    try:
        _apply_sparse_env_defaults_from_cfg(cfg)
    except Exception as _sparse_cfg_e:
        # Fail-fast: invalid sparse YAML defaults should be surfaced immediately.
        raise

    # Remove YAML-only metadata keys so they won't be passed into run_train(**kwargs).
    # These keys are used only for ENV default injection and should never affect training logic.
    if isinstance(cfg, dict):
        cfg.pop("env_defaults", None)
        cfg.pop("sparse_selective", None)

    model_env = env.get("LAT_MODEL") or env.get("GLA_MODEL")
    if model_env and not args.model:
        args.model = model_env

    prec_cli_env = env.get("LAT_PREC")
    if prec_cli_env and not args.prec:
        args.prec = prec_cli_env

    def _maybe(v, cast):
        return cast(v) if v is not None and v != "" else None

    # MODEL_TYPE env override (highest priority)
    model_type_env = env.get("MODEL_TYPE")
    if model_type_env:
        args.model_type = model_type_env

    # HP_DATA: GLUE/spider task alias mapping
    data_env = env.get("HP_DATA")
    if data_env:
        glue_tasks = {
            "rte", "mrpc", "cola", "sst2", "qnli", "qqp", "mnli", "wnli",
        }
        accepted_prefixes = (
            "glue", "samsum", "dart", "spider", "mnist", "cifar", "piqa", "boolq", "arc",
        )
        if data_env in glue_tasks:
            cfg["data"] = f"glue-tvt_{data_env}"
        elif data_env == "cifar":
            cfg["data"] = "cifar-tvt"
        elif data_env == "spider":
            cfg["data"] = "spider-tvt"
        else:
            cfg["data"] = (
                data_env
                if data_env.startswith(accepted_prefixes)
                else data_env
            )

    bs_env = _maybe(env.get("HP_BATCH_SIZE"), int)
    if bs_env is not None:
        cfg["batch_size"] = bs_env

    lr_env = _maybe(env.get("HP_LR"), float)
    if lr_env is not None:
        cfg["learning_rate"] = lr_env

    epochs_env = _maybe(env.get("HP_EPOCHS"), int)
    if epochs_env is not None:
        cfg["num_epochs"] = epochs_env

    prec_env = env.get("HP_PREC")
    if prec_env:
        cfg["prec"] = prec_env

    seed_env = _maybe(env.get("HP_SEED"), int)
    if seed_env is not None:
        cfg["seed"] = seed_env

    no_save_env = env.get("HP_NO_SAVE")
    if no_save_env is not None:
        cfg["no_save"] = str(no_save_env).lower() in ("1", "true", "yes", "on")

    val_split_env = env.get("HP_VAL_SPLIT")
    if val_split_env in {"train", "val", "test"}:
        cfg["val_data_split"] = val_split_env

    eval_bs_env = _maybe(env.get("HP_EVAL_BATCH_SIZE"), int)
    if eval_bs_env is not None and eval_bs_env > 0:
        cfg["eval_batch_size"] = eval_bs_env

    # eval_gen auto-injection for generation tasks
    def _truthy(x: Optional[str]) -> bool:
        if x is None:
            return False
        return str(x).lower() in ("1", "true", "yes", "on")

    data_name = str(cfg.get("data", ""))
    is_gen_task = any([
        data_name.startswith("samsum"),
        data_name.startswith("dart"),
        data_name.startswith("spider"),
    ])
    force_eval_gen = _truthy(env.get("EVAL_GEN"))
    if (cfg.get("eval_gen") is None) and (is_gen_task or force_eval_gen):
        max_len = _maybe(env.get("EVAL_GEN_MAX_LENGTH"), int) or 1024
        min_len = _maybe(env.get("EVAL_GEN_MIN_LENGTH"), int) or 5
        cfg["eval_gen"] = {
            "max_length": int(max_len),
            "min_length": int(min_len),
        }

    # Output directory (include model_type to prevent cross-model overwrites)
    output_dir = get_output_path_for_cfg(args.cfg, cfg, model_type=args.model_type)

    # Merge cfg + CLI args into run_train parameters
    train_args = {
        **cfg,
        **{k: v for k, v in vars(args).items() if v is not None},
        "output_dir": str(output_dir),
        "model_type": args.model_type,  # Add model_type
    }
    train_args["cfg_path"] = train_args.pop("cfg")
    if "device" in train_args:
        del train_args["device"]

    run_train(**train_args)


if __name__ == "__main__":
    main()
