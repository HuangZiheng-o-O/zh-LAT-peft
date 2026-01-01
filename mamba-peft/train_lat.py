"""
Unified Linear Attention Training Entry Point.

This script provides a unified interface for training various Linear Attention models
from the FLA library, including GLA, RetNet, and Mamba2.

Design Principles:
==================
1. **Backward Compatibility**: When model_type="gla", behavior is identical to train_gla_only.py
2. **Unified Interface**: All models use the same training flow and configuration
3. **Auto-Detection**: Model type can be automatically detected from config.json

Supported Models:
================
- gla: Gated Linear Attention (https://arxiv.org/abs/2312.06635)
- retnet: Retentive Network (https://arxiv.org/abs/2307.08621)
- mamba2: Mamba2 State Space Model (https://arxiv.org/abs/2405.21060)

PEFT Methods:
=============
- lora: Standard LoRA (default)
- sdlora / gla_sd_lora: Sparse Dimension LoRA for GLA models
  - Two-phase training: Warmup (gradient accumulation) → Training (sparse tuning)
  - Controlled via HP_PEFT_TYPE environment variable or peft_type in config

Environment Variables:
=====================
- MODEL_TYPE: Model type override (gla, retnet, mamba2, auto)
- HP_PEFT_TYPE: PEFT type override ("lora", "sdlora", "gla_sd_lora")
- LAT_* / GLA_*: Various configuration options (see documentation)

SD-LoRA Specific Environment Variables:
- HP_WARMUP_IT: Override warmup iterations (default: 100)
- HP_TRAIN_RATIO: Override train dimension ratio (default: 0.4)
  If set, HP_ZERO_RATIO is auto-computed as: 1 - train - freeze
- HP_FREEZE_RATIO: Override freeze dimension ratio (default: 0.5)
- HP_ZERO_RATIO: Override zero dimension ratio (default: 0.1)
  Note: Train + Freeze + Zero must equal 1.0

Usage:
======
    # Standard LoRA training
    python train_lat.py --cfg configs/gla.yaml --model-type gla

    # SD-LoRA training (via environment variable)
    HP_PEFT_TYPE=sdlora python train_lat.py --cfg configs/gla.yaml

    # SD-LoRA training (via config file with peft_type: GLA_SD_LORA)
    python train_lat.py --cfg configs/gla.yaml --peft configs/gla_sdlora/default.json

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
import math
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOW_OUTPUT_ROOT = REPO_ROOT / "runs"
NOW_OUTPUT_ROOT = Path(os.environ.get("LAT_OUTPUT_ROOT", DEFAULT_NOW_OUTPUT_ROOT)).expanduser()


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

    print_trainable_parameter_names(model, output_dir=output_dir, cfg_path=cfg_path)
    print("Loaded model")

    # Force left padding for decoder-only generation
    try:
        _force_left = get_lat_env_bool("FORCE_LEFT_PAD", "1")
        if _force_left and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
            if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            if get_lat_env_bool("VERBOSE"):
                print(f"[{log_tag}] Using left padding for decoder-only generation.")
    except Exception as _e:
        print(f"[{log_tag}][warn] Failed to enforce left padding policy early: {_e}")

    # Build train data module (reuse pre-built module when provided)
    if train_data_module is None:
        train_data_module = load_dataset(data, tokenizer, "train", return_module=True)

    # Save cfg.yaml
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "cfg.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

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

    # DataLoader configuration from env
    def _env_int(name: str, default: int) -> int:
        try:
            v = os.environ.get(name)
            return int(v) if v is not None else default
        except Exception:
            return default

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
            save_optimizer_state=_save_optimizer_state,
            save_strategy="steps" if not no_save else "no",
            evaluation_strategy="steps" if not skip_eval else "no",
            save_steps=(
                save_steps_override
                if save_steps_override is not None
                else int(
                    eval_epochs
                    * (
                        len(train_data_module.dataset) // batch_size
                        + (len(train_data_module.dataset) % batch_size > 0)
                    )
                )
            ),
            eval_steps=(
                eval_steps_override
                if eval_steps_override is not None
                else int(
                    eval_epochs
                    * (
                        len(train_data_module.dataset) // batch_size
                        + (len(train_data_module.dataset) % batch_size > 0)
                    )
                )
            ),
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
    model, tokenizer_obj, peft_cfg, is_sdlora = prepare_lat_model_and_tokenizer(
        model_type=model_type,
        model_id=model_id,
        prec=prec,
        debug=debug,
        peft_json_path=peft,
    )

    # SD-LoRA two-phase training
    if is_sdlora:
        print(f"[{log_tag}] SD-LoRA mode detected - using two-phase training")
        try:
            _run_sdlora_two_phase_training(
                model_type=model_type,
                model_id=model_id,
                prec=prec,
                debug=debug,
                peft_json_path=peft,
                output_dir=str(output_dir),
                cfg=cfg,
                cfg_path=cfg_path,
                learning_rate=learning_rate,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_data_workers=num_data_workers,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                eval_epochs=eval_epochs,
                skip_eval=skip_eval,
                no_save=no_save,
                eval_gen=eval_gen,
                seed=seed,
                data=data,
                val_data=val_data,
                val_data_split=val_data_split,
                num_epochs=num_epochs,
                gradient_checkpointing=gradient_checkpointing,
                logits_to_keep=logits_to_keep,
                log_tag=log_tag,
                base_total_steps=total_steps,
                base_logging_steps=logging_steps,
                base_eval_steps_override=eval_steps_override,
                base_save_steps_override=save_steps_override,
            )
        finally:
            if created_lock:
                try:
                    lock_path = Path("share/lock") / str(output_dir)
                    lock_path.unlink(missing_ok=True)
                    print(f"[{log_tag}][lock] released {lock_path}")
                except Exception as e:
                    print(f"[{log_tag}][lock][warn] failed to remove lock: {e}")
        return  # SD-LoRA training complete, exit early

    # Force left padding
    force_left = get_lat_env_bool("FORCE_LEFT_PAD", "1")
    if force_left:
        try:
            tokenizer_obj.padding_side = "left"
            if getattr(tokenizer_obj, "pad_token_id", None) is None and getattr(tokenizer_obj, "eos_token", None) is not None:
                tokenizer_obj.pad_token = tokenizer_obj.eos_token
            print(f"[{log_tag}] Using left padding for decoder-only generation.")
        except Exception as e:
            print(f"[{log_tag}][warn] Failed to enforce left padding policy: {e}")
    else:
        print(f"[{log_tag}] Respecting tokenizer's original padding policy.")
    # Build train data module once (reuse for both length calc and trainer)
    train_data_module = load_dataset(
        data, tokenizer_obj, "train", return_module=True
    )

    its_per_epoch = int(
        np.ceil(len(train_data_module.dataset) / batch_size)
    )
    optimizer_steps_per_epoch = max(
        1, math.ceil(its_per_epoch / max(1, gradient_accumulation_steps))
    )

    # Logging and steps configuration with env overrides
    env = os.environ
    logging_steps = min(50, its_per_epoch)
    try:
        if env.get("HP_LOGGING_STEPS"):
            logging_steps = int(env.get("HP_LOGGING_STEPS"))
    except Exception:
        pass

    total_steps = int(num_epochs * optimizer_steps_per_epoch)
    try:
        if env.get("HP_MAX_STEPS"):
            total_steps = int(env.get("HP_MAX_STEPS"))
    except Exception:
        pass

    eval_steps_override = None
    save_steps_override = None
    try:
        if env.get("HP_EVAL_STEPS"):
            eval_steps_override = int(env.get("HP_EVAL_STEPS"))
    except Exception:
        pass
    try:
        if env.get("HP_SAVE_STEPS"):
            save_steps_override = int(env.get("HP_SAVE_STEPS"))
    except Exception:
        pass

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


def _sanitize_dir_name(name: str) -> str:
    allowed = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        else:
            allowed.append("-")
    sanitized = "".join(allowed).strip("-")
    return sanitized or "run"


def get_output_path_for_cfg(cfg_path, cfg, peft_json_path=None):
    """
    Target path (shallow but unique):
      <NOW_OUTPUT_ROOT>/<data>__seed<seed>__<yaml-or-peft-stem>
    Fallback (missing data/seed):
      <NOW_OUTPUT_ROOT>/cola_gla__<yaml-or-peft-stem>
    """
    yaml_stem = _sanitize_dir_name(Path(cfg_path).stem)
    peft_stem = None
    if peft_json_path:
        peft_stem = _sanitize_dir_name(Path(peft_json_path).stem)

    run_suffix = peft_stem or yaml_stem
    data = _sanitize_dir_name(cfg.get("data") or "cola_gla")
    seed = cfg.get("seed")
    if seed is not None:
        seed_part = f"seed{seed}"
    else:
        seed_part = "seedNA"

    folder = f"{data}__{seed_part}__{run_suffix}"
    return NOW_OUTPUT_ROOT / folder


def _find_last_checkpoint(root: Path) -> Optional[Path]:
    """
    Scan output directory for checkpoint-* subdirs, return the one with highest step number.
    """
    if not root.exists():
        return None
    try:
        candidates = [p for p in root.glob("checkpoint-*") if p.is_dir()]
        if not candidates:
            return None
        def step_of(p: Path) -> int:
            try:
                return int(p.name.split("-")[-1])
            except Exception:
                return -1
        candidates.sort(key=step_of)
        return candidates[-1] if candidates else None
    except Exception:
        return None


def _run_sdlora_two_phase_training(
    *,
    model_type: str,
    model_id: str,
    prec: str,
    debug: bool,
    peft_json_path: str,
    output_dir: str,
    cfg: Dict,
    cfg_path: str,
    learning_rate: float,
    gradient_accumulation_steps: int,
    num_data_workers: int,
    batch_size: int,
    eval_batch_size: int,
    eval_epochs: int,
    skip_eval: bool,
    no_save: bool,
    eval_gen: Optional[Dict],
    seed: int,
    data: str,
    val_data: Optional[str],
    val_data_split: str,
    num_epochs: int,
    gradient_checkpointing: bool,
    logits_to_keep: Optional[int],
    log_tag: str,
    base_total_steps: Optional[int],
    base_logging_steps: Optional[int],
    base_eval_steps_override: Optional[int],
    base_save_steps_override: Optional[int],
):
    """
    SD-LoRA two-phase training: Warmup → Training.

    Phase 1 (Warmup):
        - Train with full gradients to collect importance information
        - Stops when model.should_training_stop is triggered
        - Saves dimension importance data via model.save_config()

    Phase 2 (Training):
        - Loads saved dimension masks
        - Trains only selected dimensions (zero/freeze/train)
    """
    from mamba_ssm_peft.peft.gla_sd_lora import GlaSdLoraConfig

    print(f"\n{'=' * 60}")
    print(f"[{log_tag}] SD-LoRA Phase 1: Warmup (gradient accumulation)")
    print(f"{'=' * 60}\n")

    run_root = Path(output_dir)
    sdlora_cfg_dir = run_root / "sdlora_cfg"
    warmup_marker = sdlora_cfg_dir / "warmup_complete.marker"

    def _attach_config_dir(peft_model):
        if hasattr(peft_model, "set_sdlora_config_dir"):
            peft_model.set_sdlora_config_dir(sdlora_cfg_dir)

    def _truthy_env(name: str) -> bool:
        return str(os.environ.get(name, "0")).lower() in ("1", "true", "yes", "on")

    reset_requested = _truthy_env("SDLORA_RESET")
    stale_cache = sdlora_cfg_dir.exists() and not warmup_marker.exists()
    if reset_requested or stale_cache:
        if sdlora_cfg_dir.exists():
            shutil.rmtree(sdlora_cfg_dir, ignore_errors=True)
        if reset_requested:
            print(f"[{log_tag}] SDLORA_RESET=1 → clearing previous SD-LoRA cache in {sdlora_cfg_dir}")
        elif stale_cache:
            print(f"[{log_tag}] Detected incomplete SD-LoRA cache. Re-initializing {sdlora_cfg_dir}")
    sdlora_cfg_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Load model and run warmup
    model, tokenizer_obj, peft_cfg, _ = prepare_lat_model_and_tokenizer(
        model_type=model_type,
        model_id=model_id,
        prec=prec,
        debug=debug,
        peft_json_path=peft_json_path,
    )
    _attach_config_dir(model)

    # Force left padding
    force_left = get_lat_env_bool("FORCE_LEFT_PAD", "1")
    if force_left:
        tokenizer_obj.padding_side = "left"
        if getattr(tokenizer_obj, "pad_token_id", None) is None:
            tokenizer_obj.pad_token = tokenizer_obj.eos_token

    # Load train data
    train_data_module = load_dataset(data, tokenizer_obj, "train", return_module=True)
    its_per_epoch = int(np.ceil(len(train_data_module.dataset) / batch_size))
    warmup_logging_steps = min(50, its_per_epoch)
    if base_logging_steps is not None:
        warmup_logging_steps = base_logging_steps

    # [FIX Issue #5] Warmup phase: account for gradient_accumulation_steps
    # num_warmup_it counts forward passes, but total_steps counts training steps
    # With gradient_accumulation_steps=N, each training step has N forward passes
    warmup_it = getattr(peft_cfg, "num_warmup_it", 100)

    # Calculate training steps needed to reach warmup_it forward passes
    # ceil(warmup_it / gradient_accumulation_steps) + buffer for safety
    warmup_training_steps = int(np.ceil(warmup_it / gradient_accumulation_steps)) + 10
    warmup_steps = warmup_training_steps

    print(f"[{log_tag}] Warmup config: {warmup_it} forward passes")
    print(f"[{log_tag}] With gradient_accumulation_steps={gradient_accumulation_steps}, "
          f"running {warmup_steps} training steps")

    # Check if warmup already completed
    warmup_complete = warmup_marker.exists()
    if warmup_complete:
        print(f"[{log_tag}] Warmup already completed, skipping to Phase 2")
    else:
        # Run warmup phase
        build_and_run_trainer_lat(
            model=model,
            tokenizer=tokenizer_obj,
            model_type=model_type,
            output_dir=str(output_dir),
            cfg=cfg,
            cfg_path=cfg_path,
            learning_rate=learning_rate,
            total_steps=warmup_steps,
            logging_steps=warmup_logging_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_data_workers=num_data_workers,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_epochs=eval_epochs,
            skip_eval=True,  # Skip eval during warmup
            no_save=True,  # Don't save checkpoints during warmup
            eval_steps_override=None,
            save_steps_override=None,
            eval_gen=None,
            resume_from_checkpoint=None,
            min_eval_metric_after_epoch=None,
            seed=seed,
            data=data,
            val_data=val_data,
            val_data_split=val_data_split,
            debug=debug,
            gradient_checkpointing=gradient_checkpointing,
            logits_to_keep=logits_to_keep,
            train_data_module=train_data_module,
        )

        # [FIX BUG #1] Save warmup gradient information BEFORE marking done
        # This is CRITICAL - without this, Phase 2 has no dimension importance data
        if hasattr(model, "save_config"):
            model.save_config(sdlora_cfg_dir)
            print(f"[{log_tag}] Saved warmup gradient information to {sdlora_cfg_dir}")
        else:
            raise RuntimeError(
                f"[{log_tag}] FATAL: Model does not have save_config method. "
                "SD-LoRA requires saving warmup gradients for Phase 2."
            )

        # Verify the config was actually saved
        saved_files = list(sdlora_cfg_dir.glob("*.pkl"))
        if not saved_files:
            raise RuntimeError(
                f"[{log_tag}] FATAL: No .pkl config files found in {sdlora_cfg_dir} after save_config(). "
                "Warmup gradient information was not saved correctly."
            )
        print(f"[{log_tag}] Verified {len(saved_files)} config file(s) saved: {[f.name for f in saved_files]}")

        # Mark warmup as done
        warmup_marker.touch()
        print(f"[{log_tag}] Warmup phase completed successfully")

    # Release warmup model resources before loading the train-phase model.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 2: Training with sparse dimensions
    print(f"\n{'=' * 60}")
    print(f"[{log_tag}] SD-LoRA Phase 2: Training (sparse dimension tuning)")
    print(f"{'=' * 60}\n")

    # Reload model for training phase
    model2, tokenizer_obj2, peft_cfg2, _ = prepare_lat_model_and_tokenizer(
        model_type=model_type,
        model_id=model_id,
        prec=prec,
        debug=debug,
        peft_json_path=peft_json_path,
    )
    _attach_config_dir(model2)

    # [FIX BUG #2] Load saved warmup config (dimension masks) with required=True
    # This ensures Phase 2 fails fast if warmup data is missing
    if not hasattr(model2, "load_config"):
        raise RuntimeError(
            f"[{log_tag}] FATAL: Model does not have load_config method. "
            "Cannot load warmup gradient data for Phase 2."
        )

    model2.load_config(sdlora_cfg_dir, required=True)

    # Verify model is now in train mode (not warmup)
    if hasattr(model2, "verify_train_mode"):
        model2.verify_train_mode()
        print(f"[{log_tag}] Verified: All SD-LoRA modules are in train mode")
    else:
        # Fallback verification
        if hasattr(model2, "get_sdlora_mode"):
            current_mode = model2.get_sdlora_mode()
            if current_mode != "train":
                raise RuntimeError(
                    f"[{log_tag}] FATAL: Model is in '{current_mode}' mode, expected 'train'. "
                    "load_config may have failed to transition the model."
                )

    # Force left padding
    if force_left:
        tokenizer_obj2.padding_side = "left"
        if getattr(tokenizer_obj2, "pad_token_id", None) is None:
            tokenizer_obj2.pad_token = tokenizer_obj2.eos_token

    # Reload train data for training phase
    train_data_module2 = load_dataset(data, tokenizer_obj2, "train", return_module=True)
    its_per_epoch2 = int(np.ceil(len(train_data_module2.dataset) / batch_size))
    optimizer_steps_per_epoch2 = max(
        1, math.ceil(its_per_epoch2 / max(1, gradient_accumulation_steps))
    )
    if base_total_steps is not None:
        total_steps = base_total_steps
    else:
        total_steps = int(num_epochs * optimizer_steps_per_epoch2)
    if base_logging_steps is not None:
        logging_steps2 = base_logging_steps
    else:
        logging_steps2 = min(50, its_per_epoch2)

    print(f"[{log_tag}] Training steps: {total_steps}")

    # Run training phase
    build_and_run_trainer_lat(
        model=model2,
        tokenizer=tokenizer_obj2,
        model_type=model_type,
        output_dir=str(output_dir),
        cfg=cfg,
        cfg_path=cfg_path,
        learning_rate=learning_rate,
        total_steps=total_steps,
        logging_steps=logging_steps2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_data_workers=num_data_workers,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        eval_epochs=eval_epochs,
        skip_eval=skip_eval,
        no_save=no_save,
        eval_steps_override=base_eval_steps_override,
        save_steps_override=base_save_steps_override,
        eval_gen=eval_gen,
        resume_from_checkpoint=None,
        min_eval_metric_after_epoch=None,
        seed=seed,
        data=data,
        val_data=val_data,
        val_data_split=val_data_split,
        debug=debug,
        gradient_checkpointing=gradient_checkpointing,
        logits_to_keep=logits_to_keep,
        train_data_module=train_data_module2,
    )

    print(f"\n[{log_tag}] SD-LoRA two-phase training completed!")


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
    parser.add_argument("--peft", type=str, help="Path to PEFT config JSON file (for SD-LoRA)")
    parser.add_argument("--device")
    args = parser.parse_args()

    # Allow --device to set VISIBLE_DEVICES
    if args.device is not None:
        os.environ["VISIBLE_DEVICES"] = args.device

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply environment overrides
    env = os.environ

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

    # Output directory
    output_dir = get_output_path_for_cfg(args.cfg, cfg, peft_json_path=args.peft)

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
