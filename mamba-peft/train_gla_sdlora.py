"""
GLA SD-LoRA Training Entry Point.

This script provides training infrastructure for GLA models using SD-LoRA
(Sparse Dimension LoRA) - a PEFT method that combines sparse dimension tuning
with LoRA for efficient fine-tuning.

SD-LoRA Training Flow:
=====================
1. **Warmup Phase**: Train with full gradient to identify important dimensions
2. **Dimension Selection**: Select train/freeze/zero dimensions based on gradients
3. **Training Phase**: Fine-tune only selected dimensions

Key Differences from Standard LoRA:
==================================
- Two-phase training (warmup → train)
- Sparse dimension selection on gate projections (gk_proj)
- Zero/freeze/train categorization based on gradient importance

Usage:
======
    # With default config
    python train_gla_sdlora.py --cfg configs/gla.yaml --peft configs/gla_sdlora/default.json

    # With custom warmup iterations
    HP_WARMUP_IT=200 python train_gla_sdlora.py --cfg configs/gla.yaml

Environment Variables:
=====================
- HP_WARMUP_IT: Override warmup iterations
- HP_ZERO_RATIO: Override zero dimension ratio
- HP_FREEZE_RATIO: Override freeze dimension ratio
"""

import sys
from pathlib import Path

# Ensure local 'fla' submodule is importable
try:
    import fla  # noqa: F401
except Exception:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        fla_symlink = repo_root / "fla"
        if fla_symlink.exists():
            sys.path.insert(0, str(repo_root))
            import fla  # noqa: F401
    except Exception:
        pass

import json
import os
import shutil
from typing import Optional, Dict, Tuple, Any

import torch
import argparse
import numpy as np

import yaml

os.environ["WANDB_PROJECT"] = "mamba-peft"

from dataset import load_dataset
from trainer.generic_lm_trainer import GenericLMTrainer, GenericLMTrainingArguments
from mamba_ssm_peft import get_trainable_parameters_ratio, print_trainable_parameter_names
from mamba_ssm_peft.utils.lat_model_loader import load_lat_model, get_lat_env_bool
from mamba_ssm_peft.utils.lat_decoder import create_lat_decoder

# Import peft BEFORE gla_sd_lora to ensure registration works
from peft import get_peft_model, PeftConfig

# Import GLA SD-LoRA (this triggers registration via decorators)
# Must import AFTER peft to register into PEFT_TYPE_TO_MODEL_MAPPING
import mamba_ssm_peft.peft  # Trigger _init() which imports gla_sd_lora
from mamba_ssm_peft.peft.gla_sd_lora import GlaSdLoraConfig, GlaSdLoraModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "benchmark" / "gla_sdlora"
OUTPUT_ROOT = Path(os.environ.get("GLA_SDLORA_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).expanduser()


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load_sdlora_config(peft_json_path: str) -> Dict[str, Any]:
    """Load and apply environment overrides to SD-LoRA configuration."""
    with open(peft_json_path, "r") as f:
        cfg = json.load(f)

    # Apply environment overrides
    warmup_it = _env_int("HP_WARMUP_IT", cfg.get("num_warmup_it", 100))
    cfg["num_warmup_it"] = warmup_it

    zero_ratio = _env_float("HP_ZERO_RATIO", cfg.get("num_zero", {}).get("channel", 0.3))
    freeze_ratio = _env_float("HP_FREEZE_RATIO", cfg.get("num_freeze", {}).get("channel", 0.3))

    cfg["num_zero"] = {"channel": zero_ratio}
    cfg["num_freeze"] = {"channel": freeze_ratio}

    return cfg


def prepare_gla_sdlora_model(
    model_id: str,
    prec: str,
    debug: bool,
    peft_json_path: str,
) -> Tuple[Any, Any, GlaSdLoraConfig]:
    """
    Prepare GLA model with SD-LoRA configuration.

    Args:
        model_id: HuggingFace model ID or local path
        prec: Precision string
        debug: If True, use CPU
        peft_json_path: Path to SD-LoRA config JSON

    Returns:
        Tuple of (model, tokenizer, sdlora_config)
    """
    # Determine device and dtype
    device = "cpu" if debug else "cuda"
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(prec, torch.bfloat16)

    # Load model and tokenizer
    loaded = load_lat_model(
        model_type="gla",
        model_id=model_id,
        trust_remote_code=True,
        device=device,
        dtype=dtype,
    )
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]

    # Load SD-LoRA configuration
    sdlora_cfg_dict = load_sdlora_config(peft_json_path)

    # Create SD-LoRA config
    sdlora_config = GlaSdLoraConfig(**sdlora_cfg_dict)

    # Apply SD-LoRA to model
    model = get_peft_model(model, sdlora_config)

    return model, tokenizer, sdlora_config


def run_sdlora_train(
    output_dir: str,
    cfg_path: str,
    model: str,
    data: str,
    peft: str,
    val_data: Optional[str] = None,
    val_data_split: str = "val",
    num_epochs: int = 10,
    prec: str = "bf16",
    learning_rate: float = 5e-4,
    gradient_accumulation_steps: int = 1,
    num_data_workers: int = 8,
    batch_size: int = 4,
    eval_gen: Optional[Dict] = None,
    debug: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    no_save: bool = False,
    skip_eval: bool = False,
    eval_epochs: int = 1,
    seed: int = 42,
    is_warmup_phase: bool = True,
):
    """
    Run a single phase of SD-LoRA training.

    Args:
        is_warmup_phase: If True, run warmup phase; if False, run training phase
    """
    cfg = {**locals()}

    # Check if already completed
    if not overwrite and (Path(output_dir) / "cfg.yaml").exists():
        if resume:
            resume_from_checkpoint = True
        else:
            if is_warmup_phase:
                print(f"Warmup phase already completed. Skipping to training phase.")
                return
            else:
                raise RuntimeError(f"{output_dir}/cfg.yaml exists!")
    else:
        resume_from_checkpoint = False

    # Load model
    print(f"[GLA-SDLoRA] Loading model: {model}")
    model_obj, tokenizer, sdlora_config = prepare_gla_sdlora_model(
        model_id=model,
        prec=prec,
        debug=debug,
        peft_json_path=peft,
    )

    # Load from saved config if available (for training phase)
    if hasattr(model_obj, "load_config"):
        model_obj.load_config(output_dir)

    print_trainable_parameter_names(model_obj, output_dir=output_dir, cfg_path=cfg_path)

    # Force left padding
    if get_lat_env_bool("FORCE_LEFT_PAD", "1"):
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[GLA-SDLoRA] Using left padding for decoder-only generation.")

    # Load datasets
    train_data_module = load_dataset(data, tokenizer, "train", return_module=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "cfg.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # Build generation decoder
    eval_generator = None
    if eval_gen is not None:
        max_length = int(eval_gen.get("max_length", 1024))
        min_length = int(eval_gen.get("min_length", 5))
        eval_generator = create_lat_decoder(
            tokenizer,
            model_type="gla",
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )

    # Validation data
    val_data_module = load_dataset(
        val_data if val_data is not None else data,
        tokenizer,
        val_data_split,
        mode="lm" if eval_generator is None else "gen",
        return_module=True,
    )
    compute_metrics = val_data_module.dataset.compute_metrics

    # Debug mode
    if debug:
        train_data_module.dataset = torch.utils.data.Subset(train_data_module.dataset, range(8))
        val_data_module.dataset = torch.utils.data.Subset(val_data_module.dataset, range(2))

    its_per_epoch = int(np.ceil(len(train_data_module.dataset) / batch_size))
    logging_steps = min(50, its_per_epoch)
    total_steps = int(num_epochs * its_per_epoch)

    # For warmup phase, limit steps
    if is_warmup_phase:
        warmup_it = sdlora_config.num_warmup_it or 100
        total_steps = min(total_steps, warmup_it + 10)  # A bit extra to ensure transition
        print(f"[GLA-SDLoRA] Warmup phase: max {total_steps} steps")

    # GenericLMTrainer already has SD-LoRA support:
    # - load_config() called at init
    # - should_training_stop checked in compute_loss()
    # - save_config() called when stopping
    trainer = GenericLMTrainer(
        model=model_obj,
        train_dataset=train_data_module.dataset,
        tokenizer=tokenizer,
        args=GenericLMTrainingArguments(
            learning_rate=learning_rate,
            max_steps=total_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim="adamw_torch",
            output_dir=output_dir,
            logging_steps=logging_steps,
            dataloader_num_workers=num_data_workers,
            dataloader_prefetch_factor=2,
            eval_accumulation_steps=128,
            info={
                "trainable_params": get_trainable_parameters_ratio(model_obj),
                "cfg_path": cfg_path,
                "phase": "warmup" if is_warmup_phase else "train",
            },
            save_strategy="steps" if not no_save else "no",
            evaluation_strategy="steps" if not skip_eval else "no",
            save_steps=int(eval_epochs * its_per_epoch),
            eval_steps=int(eval_epochs * its_per_epoch),
            dataloader_drop_last=True,
            report_to="none",
            seed=seed,
        ),
        compute_metrics=compute_metrics,
        data_collator=train_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        eval_generator=eval_generator,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def get_output_path_for_cfg(cfg_path: str, cfg: Dict) -> Path:
    """Generate output path from config."""
    yaml_stem = Path(cfg_path).stem
    data = cfg.get("data", "unknown")
    seed = cfg.get("seed", 42)
    folder = f"{data}_seed{seed}"
    return OUTPUT_ROOT / folder / yaml_stem


def main():
    parser = argparse.ArgumentParser(description="GLA SD-LoRA Training")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--peft", type=str, default="configs/gla_sdlora/default.json",
                        help="Path to SD-LoRA config JSON")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--prec")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.model:
        cfg["model"] = args.model
    if args.prec:
        cfg["prec"] = args.prec

    # Output directory
    output_dir = get_output_path_for_cfg(args.cfg, cfg)

    # Prepare train arguments
    train_args = {
        **cfg,
        "output_dir": str(output_dir),
        "cfg_path": args.cfg,
        "peft": args.peft,
        "debug": args.debug,
        "resume": args.resume,
    }

    # Two-phase training
    print("=" * 60)
    print("[GLA-SDLoRA] Phase 1: Warmup (gradient accumulation)")
    print("=" * 60)
    run_sdlora_train(**train_args, is_warmup_phase=True)

    print("")
    print("=" * 60)
    print("[GLA-SDLoRA] Phase 2: Training (sparse dimension tuning)")
    print("=" * 60)
    run_sdlora_train(**train_args, is_warmup_phase=False, overwrite=True)

    print("")
    print("[GLA-SDLoRA] Training complete!")


if __name__ == "__main__":
    main()
