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
- delta_net: DeltaNet (Linear Transformers with Delta Rule)
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

    # HP_PISSA_FAST: If set and init is "pissa", switch to fast SVD init.
    # IMPORTANT: apply this even when HP_INIT is explicitly set, so users can do:
    #   HP_INIT=pissa + HP_PISSA_FAST=1  -> pissa_niter_4
    fast_pissa_env = env.get("HP_PISSA_FAST")
    try:
        if fast_pissa_env and str(fast_pissa_env).lower() not in ("0", "false", "no", "off"):
            init_val = peft_json.get("init_lora_weights", None)
            if init_val is None:
                peft_json["init_lora_weights"] = "pissa_niter_4"
            elif isinstance(init_val, str) and init_val.lower() == "pissa":
                peft_json["init_lora_weights"] = "pissa_niter_4"
    except Exception:
        pass

    # If lora_alpha is missing, default to 2 * r (matches FISH-Tuning impl detail; opt-in via "missing key").
    # We DO NOT override if user already set lora_alpha in JSON or via HP_PEFT_ALPHA.
    if ("lora_alpha" not in peft_json) or (peft_json.get("lora_alpha") is None):
        try:
            r_val = int(peft_json.get("r"))
            if r_val > 0:
                peft_json["lora_alpha"] = 2 * r_val
        except Exception:
            pass

    # HP_USE_DORA / HP_USE_RSLoRA (truthy env toggles)
    def _truthy(value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        v = str(value).strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        return None

    use_dora_env = _truthy(env.get("HP_USE_DORA"))
    if use_dora_env is not None:
        peft_json["use_dora"] = use_dora_env

    use_rslora_env = _truthy(env.get("HP_USE_RSLoRA"))
    if use_rslora_env is not None:
        peft_json["use_rslora"] = use_rslora_env

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

    Model-specific projections:
    - GLA (GatedLinearAttention):
        Attention: q_proj, k_proj, v_proj, o_proj, g_proj, gk_proj
        MLP: gate_proj, up_proj, down_proj (SwiGLU)

    - RetNet (MultiScaleRetention):
        Attention: q_proj, k_proj, v_proj, o_proj, g_proj
        MLP: gate_proj, up_proj, down_proj (SwiGLU)
        Note: RetNet has NO gk_proj (uses RotaryEmbedding instead of learned gating)

    - DeltaNet (Delta Rule Linear Transformer):
        DeltaNet Layer: q_proj, k_proj, v_proj, o_proj, b_proj (optional: g_proj)
        MLP: gate_proj, up_proj, down_proj (SwiGLU)
        Note: b_proj outputs only num_heads scalars (beta/writing strength)
              Not included in default LoRA targets due to small output dimension.
        Reference: https://arxiv.org/abs/2406.06484

    - Mamba2:
        Mixer: in_proj, out_proj
    """
    # Model-specific default target modules
    # These are the key projection layers in each architecture
    defaults = {
        # GLA: includes gk_proj for low-rank gating mechanism
        "gla": [
            "q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "gk_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        # RetNet: includes g_proj but NO gk_proj (uses RotaryEmbedding, not learned gating)
        "retnet": [
            "q_proj", "k_proj", "v_proj", "o_proj", "g_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        # DeltaNet: uses delta rule for state updates
        # b_proj outputs only num_heads scalars - not included by default as it may
        # not benefit much from LoRA due to small output dimension
        # g_proj is optional (when use_gate=True)
        # Reference: https://arxiv.org/abs/2406.06484
        "delta_net": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        # Mamba2: different layer structure, no SwiGLU MLP
        "mamba2": ["in_proj", "out_proj"],
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

        # PEFT compatibility shim:
        # Some environments ship an older `peft` that doesn't recognize init_lora_weights="pissa"
        # (but may support "pissa_niter_4"), causing:
        #   ValueError: Unknown initialization init_lora_weights='pissa'
        #
        # We prefer the user's requested init, but if PEFT rejects it, we retry with a safe fallback.
        try:
            peft_cfg = LoraConfig(**peft_json)
        except ValueError as e:
            init_v = peft_json.get("init_lora_weights", None)
            msg = str(e).lower()
            if isinstance(init_v, str) and init_v.lower() == "pissa" and ("unknown initialization" in msg or "init_lora_weights" in msg):
                # Retry with fast PiSSA (more widely supported across PEFT versions)
                peft_json2 = dict(peft_json)
                peft_json2["init_lora_weights"] = "pissa_niter_4"
                try:
                    peft_cfg = LoraConfig(**peft_json2)
                    peft_json = peft_json2
                    print("[lat_adapter][warn] PEFT rejected init_lora_weights='pissa'; using 'pissa_niter_4' fallback.")
                except Exception:
                    # Final fallback: drop init_lora_weights so PEFT uses its default init.
                    peft_json3 = dict(peft_json)
                    peft_json3.pop("init_lora_weights", None)
                    peft_cfg = LoraConfig(**peft_json3)
                    peft_json = peft_json3
                    print("[lat_adapter][warn] PEFT rejected PiSSA init; falling back to default LoRA init (no init_lora_weights).")
            else:
                raise
        model = get_peft_model(model, peft_cfg)

    return model, tokenizer, peft_cfg


def attach_peft_weights(
    model: Any,
    peft_weights_path: str,
    torch_dtype: Optional[torch.dtype] = None,
) -> Any:
    """
    Attach an existing PEFT adapter (LoRA/DoRA/RSLoRA/...) onto a base model.

    This is used for evaluation workflows where the adapter is already trained and
    saved under a checkpoint directory (contains adapter_config.json + adapter weights).

    Args:
        model: Base model (NOT already wrapped by get_peft_model)
        peft_weights_path: Directory path containing PEFT adapter files
        torch_dtype: Optional dtype override for loading adapter weights

    Returns:
        PeftModel wrapping the base model
    """
    from peft import PeftModel

    return PeftModel.from_pretrained(
        model,
        peft_weights_path,
        torch_dtype=torch_dtype,
        is_trainable=False,
    )


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
