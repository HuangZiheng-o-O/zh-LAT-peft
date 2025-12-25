"""
Unified Linear Attention Model Adapter.

This module provides a unified interface for preparing Linear Attention models
for fine-tuning, including PEFT/LoRA configuration and optional environment
variable overrides.

Design Principles:
==================
1. **Backward Compatibility**: `prepare_gla_model_and_tokenizer()` remains available
   and behaves identically to the original.
2. **Unified Interface**: `prepare_lat_model_and_tokenizer()` works with all supported
   Linear Attention models.
3. **PEFT Support**: Same PEFT/LoRA configuration logic applies to all model types.

Supported Models:
================
- gla: Gated Linear Attention
- retnet: Retentive Network
- mamba2: Mamba2 State Space Model

Environment Variables for PEFT overrides:
========================================
- HP_PEFT_R: Override LoRA rank
- HP_PEFT_ALPHA: Override LoRA alpha
- HP_PEFT_DROPOUT: Override LoRA dropout
- HP_INIT: Override init_lora_weights (e.g., "pissa", "pissa_niter_4")
- HP_PISSA_FAST: If set and init is "pissa", switch to "pissa_niter_4"

Usage:
======
    from lat_adapter import prepare_lat_model_and_tokenizer

    # For GLA (backward compatible)
    model, tokenizer, peft_cfg = prepare_lat_model_and_tokenizer(
        model_type="gla",
        model_id="fla-hub/gla-1.3B-100B",
        prec="bf16",
        debug=False,
        peft_json_path="configs/peft_lora.json",
    )

    # For RetNet
    model, tokenizer, peft_cfg = prepare_lat_model_and_tokenizer(
        model_type="retnet",
        model_id="fla-hub/retnet-1.3B",
        prec="bf16",
        debug=False,
        peft_json_path="configs/peft_lora.json",
    )

    # With auto-detection
    model, tokenizer, peft_cfg = prepare_lat_model_and_tokenizer(
        model_type="auto",
        model_id="fla-hub/gla-1.3B-100B",
        ...
    )
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

# Import unified model loader
from mamba_ssm_peft.utils.lat_model_loader import (
    load_lat_model,
    get_lat_env,
    get_lat_env_bool,
)


def _dtype_from_prec(prec: str) -> torch.dtype:
    """
    Convert precision string to torch dtype.

    Note: fp16 is mapped to bfloat16 for consistency with the original implementation.
    """
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.bfloat16,  # Legacy behavior: fp16 -> bfloat16
        "fp32": torch.float32,
    }
    if prec not in mapping:
        raise ValueError(f"Unknown precision '{prec}'. Supported: {list(mapping.keys())}")
    return mapping[prec]


def _apply_peft_env_overrides(peft_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to PEFT configuration.

    Supported overrides:
    - HP_PEFT_R: LoRA rank (int)
    - HP_PEFT_ALPHA: LoRA alpha (int)
    - HP_PEFT_DROPOUT: LoRA dropout (float)
    - HP_INIT: init_lora_weights (str, e.g., "pissa", "pissa_niter_4")
    - HP_PISSA_FAST: If true and init is "pissa", switch to "pissa_niter_4"

    Args:
        peft_json: Original PEFT configuration dict

    Returns:
        Modified PEFT configuration dict
    """
    env = os.environ

    # HP_PEFT_R: Override LoRA rank
    r_env = env.get("HP_PEFT_R")
    if r_env is not None:
        try:
            peft_json["r"] = int(r_env)
        except (ValueError, TypeError):
            pass

    # HP_PEFT_ALPHA: Override LoRA alpha
    alpha_env = env.get("HP_PEFT_ALPHA")
    if alpha_env is not None:
        try:
            peft_json["lora_alpha"] = int(alpha_env)
        except (ValueError, TypeError):
            pass

    # HP_PEFT_DROPOUT: Override LoRA dropout
    drop_env = env.get("HP_PEFT_DROPOUT")
    if drop_env is not None:
        try:
            peft_json["lora_dropout"] = float(drop_env)
        except (ValueError, TypeError):
            pass

    # HP_INIT: Override init_lora_weights (e.g., "pissa", "pissa_niter_4")
    init_env = env.get("HP_INIT")
    if init_env:
        peft_json["init_lora_weights"] = init_env
    else:
        # HP_PISSA_FAST: If set and init is "pissa", switch to fast SVD init
        fast_pissa_env = env.get("HP_PISSA_FAST")
        try:
            if fast_pissa_env and str(fast_pissa_env).lower() not in ("0", "false", "no", "off"):
                init_val = peft_json.get("init_lora_weights", None)
                if isinstance(init_val, str) and init_val.lower() == "pissa":
                    peft_json["init_lora_weights"] = "pissa_niter_4"
        except Exception:
            pass

    return peft_json


def _get_target_modules_for_model(model_type: str, model: Any) -> Optional[list]:
    """
    Get default LoRA target modules for a specific model type.

    Different Linear Attention models have different module naming conventions.
    This function provides sensible defaults when target_modules is not specified
    in the PEFT config.

    Args:
        model_type: Model type string
        model: The loaded model (for inspection if needed)

    Returns:
        List of target module names, or None to use PEFT's auto-detection
    """
    # Model-specific default target modules
    # These are the key projection layers in each architecture
    defaults = {
        "gla": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "retnet": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mamba2": ["in_proj", "out_proj"],  # Mamba2 has different layer structure
    }

    return defaults.get(model_type)


def prepare_lat_model_and_tokenizer(
    model_type: str,
    model_id: str,
    prec: str,
    debug: bool,
    peft_json_path: Optional[str],
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Prepare a Linear Attention model + tokenizer and (optionally) attach PEFT LoRA.

    This is the unified entry point for preparing any supported Linear Attention
    model for fine-tuning.

    Args:
        model_type: Model type ("gla", "retnet", "mamba2", or "auto")
        model_id: HuggingFace model ID or local path
        prec: Precision string ("bf16", "fp16", "fp32")
        debug: If True, use CPU instead of CUDA
        peft_json_path: Path to PEFT/LoRA config JSON (None to skip PEFT)

    Returns:
        Tuple of (model, tokenizer, peft_cfg)
        - model: The prepared model (with PEFT if configured)
        - tokenizer: The loaded tokenizer
        - peft_cfg: LoraConfig object (or None if no PEFT)

    Example:
        >>> model, tokenizer, peft_cfg = prepare_lat_model_and_tokenizer(
        ...     model_type="gla",
        ...     model_id="fla-hub/gla-1.3B-100B",
        ...     prec="bf16",
        ...     debug=False,
        ...     peft_json_path="configs/peft_lora.json",
        ... )
    """
    # Determine device and dtype
    device = "cpu" if debug else "cuda"
    dtype = _dtype_from_prec(prec)

    # Load model and tokenizer using unified loader
    loaded = load_lat_model(
        model_type=model_type,
        model_id=model_id,
        trust_remote_code=True,
        device=device,
        dtype=dtype,
    )
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    resolved_model_type = loaded["model_type"]

    # Apply PEFT if configured
    peft_cfg = None
    if peft_json_path is not None:
        # Lazy import to avoid hard dependency
        from peft import LoraConfig, get_peft_model

        with open(peft_json_path, "r") as f:
            peft_json = json.load(f)

        # Apply environment variable overrides
        peft_json = _apply_peft_env_overrides(peft_json)

        # Set default target_modules if not specified
        if "target_modules" not in peft_json or peft_json["target_modules"] is None:
            default_targets = _get_target_modules_for_model(resolved_model_type, model)
            if default_targets:
                peft_json["target_modules"] = default_targets

        peft_cfg = LoraConfig(**peft_json)
        model = get_peft_model(model, peft_cfg)

    return model, tokenizer, peft_cfg


# ============================================================================
# BACKWARD COMPATIBILITY: GLA-specific function
# ============================================================================
def prepare_gla_model_and_tokenizer(
    model_id: str,
    prec: str,
    debug: bool,
    peft_json_path: Optional[str],
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Prepare GLA model + tokenizer and (optionally) attach HF PEFT LoRA.

    This function provides exact backward compatibility with the original
    train_gla_adapter.py implementation.

    Behavior is intentionally identical to the inlined logic in train.py:
    - Uses load_gla(...) to get model & tokenizer
    - When peft_json_path is provided, loads JSON and applies env overrides:
      HP_PEFT_R, HP_PEFT_ALPHA, HP_PEFT_DROPOUT, HP_INIT, HP_PISSA_FAST
      Then builds peft.LoraConfig and wraps with peft.get_peft_model(...)

    Args:
        model_id: HuggingFace model ID or local path
        prec: Precision string ("bf16", "fp16", "fp32")
        debug: If True, use CPU instead of CUDA
        peft_json_path: Path to PEFT/LoRA config JSON (None to skip PEFT)

    Returns:
        Tuple of (model, tokenizer, peft_cfg) where peft_cfg is None when
        no PEFT config is provided.
    """
    return prepare_lat_model_and_tokenizer(
        model_type="gla",
        model_id=model_id,
        prec=prec,
        debug=debug,
        peft_json_path=peft_json_path,
    )
