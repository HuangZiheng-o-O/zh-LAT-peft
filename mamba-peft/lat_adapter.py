"""
Unified Linear Attention Model Adapter.

This module provides a unified interface for preparing Linear Attention models
for fine-tuning, supporting both standard LoRA and SD-LoRA (Sparse Dimension LoRA).

Design Principles:
==================
1. **Backward Compatibility**: Existing configs continue to work unchanged.
2. **Unified Interface**: Single entry point for all PEFT methods.
3. **Environment Control**: PEFT type can be switched via HP_PEFT_TYPE env.

Supported PEFT Methods:
======================
- lora: Standard LoRA (default)
- sdlora / gla_sd_lora: Sparse Dimension LoRA for GLA models

Environment Variables:
=====================
- HP_PEFT_TYPE: Override PEFT type ("lora", "sdlora")
- HP_PEFT_R: Override LoRA rank
- HP_PEFT_ALPHA: Override LoRA alpha
- HP_PEFT_DROPOUT: Override LoRA dropout
- HP_INIT: Override init_lora_weights
- HP_PISSA_FAST: Fast PiSSA init

SD-LoRA Specific (default: Train=40%, Freeze=50%, Zero=10%):
- HP_WARMUP_IT: Override warmup iterations (default: 100)
- HP_TRAIN_RATIO: Override train dimension ratio (default: 0.4)
  If set, HP_ZERO_RATIO is auto-computed as: 1 - train - freeze
- HP_FREEZE_RATIO: Override freeze dimension ratio (default: 0.5)
- HP_ZERO_RATIO: Override zero dimension ratio (default: 0.1)

Usage:
======
    # Standard LoRA (default)
    model, tokenizer, peft_cfg, is_sdlora = prepare_lat_model_and_tokenizer(...)

    # SD-LoRA via config file
    # (when peft_json contains "peft_type": "GLA_SD_LORA")
    model, tokenizer, peft_cfg, is_sdlora = prepare_lat_model_and_tokenizer(...)

    # SD-LoRA via environment variable
    HP_PEFT_TYPE=sdlora python train_lat.py ...
"""

import json
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch

from mamba_ssm_peft.utils.lat_model_loader import (
    load_lat_model,
    get_lat_env,
    get_lat_env_bool,
)


def _dtype_from_prec(prec: str) -> torch.dtype:
    """Convert precision string to torch dtype."""
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.bfloat16,  # Legacy behavior: fp16 -> bfloat16
        "fp32": torch.float32,
    }
    if prec not in mapping:
        raise ValueError(f"Unknown precision '{prec}'. Supported: {list(mapping.keys())}")
    return mapping[prec]


def _truthy(value: Optional[str]) -> Optional[bool]:
    """Parse truthy string value."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def _env_float(name: str, default: float) -> float:
    """Get float from environment variable."""
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Get int from environment variable."""
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _detect_peft_type(peft_json: Dict[str, Any]) -> str:
    """
    Detect PEFT type from config and environment.

    Priority:
    1. HP_PEFT_TYPE environment variable
    2. peft_type field in config
    3. Default to "LORA"

    Returns:
        Normalized PEFT type string ("LORA", "GLA_SD_LORA", etc.)
    """
    # Environment override has highest priority
    env_type = os.environ.get("HP_PEFT_TYPE", "").strip().lower()
    if env_type in ("sdlora", "sd_lora", "gla_sd_lora", "gla_sdlora"):
        return "GLA_SD_LORA"
    if env_type in ("lora",):
        return "LORA"

    # Check config file
    config_type = peft_json.get("peft_type", "LORA")
    if isinstance(config_type, str):
        config_type_upper = config_type.upper().replace("-", "_")
        if config_type_upper in ("GLA_SD_LORA", "SDLORA", "SD_LORA"):
            return "GLA_SD_LORA"

    return "LORA"


def _apply_lora_env_overrides(peft_json: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides for standard LoRA."""
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

    # HP_INIT: Override init_lora_weights
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

    # HP_USE_DORA / HP_USE_RSLoRA
    use_dora_env = _truthy(env.get("HP_USE_DORA"))
    if use_dora_env is not None:
        peft_json["use_dora"] = use_dora_env

    use_rslora_env = _truthy(env.get("HP_USE_RSLoRA"))
    if use_rslora_env is not None:
        peft_json["use_rslora"] = use_rslora_env

    return peft_json


def _apply_sdlora_env_overrides(peft_json: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides for SD-LoRA.

    Dimension ratio logic (Train + Freeze + Zero = 100%):
    - Default: Train=40%, Freeze=50%, Zero=10%
    - If HP_TRAIN_RATIO is set, Zero is auto-computed: Zero = 1 - Train - Freeze
    - All three ratios can be set explicitly via HP_TRAIN_RATIO, HP_FREEZE_RATIO, HP_ZERO_RATIO
    """
    # Apply standard LoRA overrides first (for proj_lora_r, etc.)
    peft_json = _apply_lora_env_overrides(peft_json)

    # SD-LoRA specific overrides
    warmup_it = _env_int("HP_WARMUP_IT", peft_json.get("num_warmup_it", 100))
    peft_json["num_warmup_it"] = warmup_it

    # Default ratios: Train=40%, Freeze=50%, Zero=10%
    # Train = 1 - Zero - Freeze = 1 - 0.1 - 0.5 = 0.4
    default_train = 0.4
    default_freeze = 0.5
    default_zero = 0.1

    # Get config values or use new defaults
    num_zero = peft_json.get("num_zero", {"channel": default_zero})
    num_freeze = peft_json.get("num_freeze", {"channel": default_freeze})

    # Check if HP_TRAIN_RATIO is explicitly set
    train_ratio_env = os.environ.get("HP_TRAIN_RATIO")
    freeze_ratio_env = os.environ.get("HP_FREEZE_RATIO")
    zero_ratio_env = os.environ.get("HP_ZERO_RATIO")

    if isinstance(num_freeze, dict):
        if freeze_ratio_env is not None:
            try:
                freeze_ratio = float(freeze_ratio_env)
            except (ValueError, TypeError):
                freeze_ratio = num_freeze.get("channel", default_freeze)
        else:
            freeze_ratio = num_freeze.get("channel", default_freeze)
        num_freeze["channel"] = freeze_ratio
    else:
        freeze_ratio = default_freeze

    if isinstance(num_zero, dict):
        if train_ratio_env is not None and zero_ratio_env is None:
            # HP_TRAIN_RATIO is set, auto-compute HP_ZERO_RATIO
            try:
                train_ratio = float(train_ratio_env)
                zero_ratio = max(0.0, 1.0 - train_ratio - freeze_ratio)
                print(f"[SD-LoRA] HP_TRAIN_RATIO={train_ratio:.2f} set, auto-computed "
                      f"zero_ratio={zero_ratio:.2f} (freeze={freeze_ratio:.2f})")
            except (ValueError, TypeError):
                zero_ratio = num_zero.get("channel", default_zero)
        elif zero_ratio_env is not None:
            try:
                zero_ratio = float(zero_ratio_env)
            except (ValueError, TypeError):
                zero_ratio = num_zero.get("channel", default_zero)
        else:
            zero_ratio = num_zero.get("channel", default_zero)
        num_zero["channel"] = zero_ratio

    peft_json["num_zero"] = num_zero
    peft_json["num_freeze"] = num_freeze

    # Print effective ratios
    train_ratio_effective = 1.0 - num_zero.get("channel", 0) - num_freeze.get("channel", 0)
    print(f"[SD-LoRA] Effective ratios: train={train_ratio_effective:.1%}, "
          f"freeze={num_freeze.get('channel', 0):.1%}, zero={num_zero.get('channel', 0):.1%}")

    # proj_lora_r from HP_PEFT_R
    r_env = os.environ.get("HP_PEFT_R")
    if r_env is not None:
        try:
            peft_json["proj_lora_r"] = int(r_env)
        except (ValueError, TypeError):
            pass

    return peft_json


def _get_target_modules_for_model(model_type: str, peft_type: str) -> Optional[list]:
    """Get default target modules for a specific model and PEFT type."""
    if peft_type == "GLA_SD_LORA":
        # SD-LoRA targets gate projection for SDT
        return ["gk_proj.1"]

    # Standard LoRA defaults
    defaults = {
        "gla": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "retnet": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mamba2": ["in_proj", "out_proj"],
    }
    return defaults.get(model_type)


def _get_lora_targets_for_sdlora(model_type: str) -> list:
    """Get default LoRA targets for SD-LoRA (in addition to SDT targets)."""
    defaults = {
        "gla": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "retnet": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    return defaults.get(model_type, ["q_proj", "k_proj", "v_proj", "o_proj"])


def prepare_lat_model_and_tokenizer(
    model_type: str,
    model_id: str,
    prec: str,
    debug: bool,
    peft_json_path: Optional[str],
) -> Tuple[Any, Any, Optional[Any], bool]:
    """
    Prepare a Linear Attention model + tokenizer with PEFT (LoRA or SD-LoRA).

    Args:
        model_type: Model type ("gla", "retnet", "mamba2", or "auto")
        model_id: HuggingFace model ID or local path
        prec: Precision string ("bf16", "fp16", "fp32")
        debug: If True, use CPU instead of CUDA
        peft_json_path: Path to PEFT config JSON (None to skip PEFT)

    Returns:
        Tuple of (model, tokenizer, peft_cfg, is_sdlora)
        - model: The prepared model (with PEFT if configured)
        - tokenizer: The loaded tokenizer
        - peft_cfg: PeftConfig object (or None if no PEFT)
        - is_sdlora: True if using SD-LoRA (requires two-phase training)
    """
    # Determine device and dtype
    device = "cpu" if debug else "cuda"
    dtype = _dtype_from_prec(prec)

    # Load model and tokenizer
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
    is_sdlora = False

    if peft_json_path is not None:
        with open(peft_json_path, "r") as f:
            peft_json = json.load(f)

        # Detect PEFT type
        peft_type = _detect_peft_type(peft_json)
        is_sdlora = (peft_type == "GLA_SD_LORA")

        if is_sdlora:
            # ============================================================
            # SD-LoRA: Sparse Dimension LoRA for GLA
            # ============================================================
            print(f"[LAT] Using SD-LoRA (Sparse Dimension LoRA) for {resolved_model_type}")

            # Ensure peft module is imported to trigger registration
            import mamba_ssm_peft.peft  # noqa: F401
            from peft import get_peft_model
            from mamba_ssm_peft.peft.gla_sd_lora import GlaSdLoraConfig

            # Apply env overrides
            peft_json = _apply_sdlora_env_overrides(peft_json)

            # Set defaults if not specified
            if "target_modules" not in peft_json or peft_json["target_modules"] is None:
                peft_json["target_modules"] = _get_target_modules_for_model(
                    resolved_model_type, peft_type
                )
            if "lora_targets" not in peft_json or peft_json["lora_targets"] is None:
                peft_json["lora_targets"] = _get_lora_targets_for_sdlora(resolved_model_type)

            # Ensure BaseTuner walks every LoRA target by adding them to target_modules
            target_modules = peft_json.get("target_modules") or []
            lora_targets = peft_json.get("lora_targets") or []
            for module_name in lora_targets:
                if module_name not in target_modules:
                    target_modules.append(module_name)
            peft_json["target_modules"] = target_modules

            # Remove non-config fields from dict
            peft_json.pop("peft_type", None)  # Not a valid GlaSdLoraConfig field
            peft_json.pop("_comment", None)   # Comment field for documentation

            # Create config
            peft_cfg = GlaSdLoraConfig(**peft_json)
            model = get_peft_model(model, peft_cfg)

            print(f"[LAT] SD-LoRA config: warmup_it={peft_cfg.num_warmup_it}, "
                  f"zero={peft_cfg.num_zero}, freeze={peft_cfg.num_freeze}")

        else:
            # ============================================================
            # Standard LoRA
            # ============================================================
            from peft import LoraConfig, get_peft_model

            # Apply env overrides
            peft_json = _apply_lora_env_overrides(peft_json)

            # Set defaults if not specified
            if "target_modules" not in peft_json or peft_json["target_modules"] is None:
                default_targets = _get_target_modules_for_model(resolved_model_type, peft_type)
                if default_targets:
                    peft_json["target_modules"] = default_targets

            # Remove non-LoRA fields
            peft_json.pop("peft_type", None)

            peft_cfg = LoraConfig(**peft_json)
            model = get_peft_model(model, peft_cfg)

    return model, tokenizer, peft_cfg, is_sdlora


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================
def prepare_gla_model_and_tokenizer(
    model_id: str,
    prec: str,
    debug: bool,
    peft_json_path: Optional[str],
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Backward compatible function for GLA model preparation.

    Returns 3-tuple (model, tokenizer, peft_cfg) for compatibility.
    Use prepare_lat_model_and_tokenizer() for SD-LoRA support.
    """
    model, tokenizer, peft_cfg, _ = prepare_lat_model_and_tokenizer(
        model_type="gla",
        model_id=model_id,
        prec=prec,
        debug=debug,
        peft_json_path=peft_json_path,
    )
    return model, tokenizer, peft_cfg
