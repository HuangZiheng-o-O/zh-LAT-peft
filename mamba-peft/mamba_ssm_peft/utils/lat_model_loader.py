
"""
Unified Linear Attention Model Loader.

This module provides a unified interface for loading various Linear Attention models
from the FLA (Flash Linear Attention) library, including GLA, RetNet, Mamba2, and others.

Design Principles:
==================
1. **Backward Compatibility**: GLA loading remains identical to the original `load_gla()`.
2. **Unified Interface**: All models use the same `load_lat_model()` entry point.
3. **Auto-Detection**: Model type can be automatically detected from config.json.
4. **Extensibility**: Easy to add new model types by updating MODEL_REGISTRY.

Supported Models (First Batch):
==============================
- gla: Gated Linear Attention (https://arxiv.org/abs/2312.06635)
- retnet: Retentive Network (https://arxiv.org/abs/2307.08621)
- mamba2: Mamba2 State Space Model (https://arxiv.org/abs/2405.21060)

Environment Variables (LAT_* preferred, GLA_* fallback for compatibility):
=========================================================================
- LAT_FORCE_LEFT_PAD / GLA_FORCE_LEFT_PAD: Force left padding
- LAT_VERBOSE / GLA_VERBOSE: Enable verbose logging
- LAT_USE_FUSED_SWIGLU / GLA_USE_FUSED_SWIGLU: Enable fused SwiGLU (default: disabled)

Usage:
======
    from mamba_ssm_peft.utils.lat_model_loader import load_lat_model, detect_model_type

    # Auto-detect model type from config
    model_type = detect_model_type("fla-hub/gla-1.3B-100B")

    # Load model with explicit type
    result = load_lat_model("gla", "fla-hub/gla-1.3B-100B")
    model, tokenizer = result["model"], result["tokenizer"]

    # Or with auto-detection
    result = load_lat_model("auto", "fla-hub/gla-1.3B-100B")
"""

import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file


# ============================================================================
# MODEL REGISTRY
# ============================================================================
# Format: model_type -> (module_path, config_class_name, model_class_name, special_handling)
# special_handling: dict of flags for model-specific behavior
#   - has_fuse_swiglu: whether the model config has fuse_swiglu option
#   - cache_type: "past_key_values" (GLA/RetNet) or "cache_params" (Mamba2)
#   - inner_model_attr: attribute name for inner model ("model" or "backbone")

MODEL_REGISTRY: Dict[str, Tuple[str, str, str, Dict[str, Any]]] = {
    "gla": (
        "fla.models.gla",
        "GLAConfig",
        "GLAForCausalLM",
        {"has_fuse_swiglu": True, "cache_type": "past_key_values", "inner_model_attr": "model"},
    ),
    "retnet": (
        "fla.models.retnet",
        "RetNetConfig",
        "RetNetForCausalLM",
        {"has_fuse_swiglu": True, "cache_type": "past_key_values", "inner_model_attr": "model"},
    ),
    "mamba2": (
        "fla.models.mamba2",
        "Mamba2Config",
        "Mamba2ForCausalLM",
        {"has_fuse_swiglu": False, "cache_type": "cache_params", "inner_model_attr": "backbone"},
    ),
}

# Mapping from config.json model_type to our registry key
CONFIG_MODEL_TYPE_MAP: Dict[str, str] = {
    "gla": "gla",
    "retnet": "retnet",
    "mamba2": "mamba2",
}


# ============================================================================
# ENVIRONMENT VARIABLE HELPERS
# ============================================================================
def get_lat_env(key: str, default: str = "0") -> str:
    """
    Get environment variable with LAT_* prefix, falling back to GLA_* for compatibility.

    Priority: LAT_{key} > GLA_{key} > default

    Example:
        get_lat_env("VERBOSE") checks LAT_VERBOSE, then GLA_VERBOSE, then returns default.
    """
    lat_key = f"LAT_{key}"
    gla_key = f"GLA_{key}"
    return os.getenv(lat_key, os.getenv(gla_key, default))


def get_lat_env_bool(key: str, default: str = "0") -> bool:
    """Get environment variable as boolean."""
    return get_lat_env(key, default).lower() in ("1", "true", "yes", "on")


def _verbose_print(msg: str) -> None:
    """Print message if LAT_VERBOSE or GLA_VERBOSE is enabled."""
    if get_lat_env_bool("VERBOSE"):
        print(f"[LAT] {msg}")


# ============================================================================
# MODEL TYPE DETECTION
# ============================================================================
def detect_model_type(model_id: str, trust_remote_code: bool = True) -> str:
    """
    Auto-detect model type from config.json.

    Args:
        model_id: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code (for HF hub)

    Returns:
        Model type string (e.g., "gla", "retnet", "mamba2")

    Raises:
        ValueError: If model_type cannot be determined or is not supported
    """
    _verbose_print(f"Detecting model type from: {model_id}")

    # Try to load config.json
    try:
        resolved_config = cached_file(
            model_id, CONFIG_NAME, _raise_exceptions_for_missing_entries=True
        )
        with open(resolved_config, "r") as f:
            config_dict = json.load(f)
    except Exception as e:
        raise ValueError(
            f"[LAT] Failed to load config.json from '{model_id}': {e}. "
            f"Please specify model_type explicitly."
        ) from e

    # Extract model_type from config
    config_model_type = config_dict.get("model_type")
    if config_model_type is None:
        raise ValueError(
            f"[LAT] config.json for '{model_id}' does not contain 'model_type'. "
            f"Please specify model_type explicitly."
        )

    # Map to our registry key
    model_type = CONFIG_MODEL_TYPE_MAP.get(config_model_type)
    if model_type is None:
        supported = ", ".join(CONFIG_MODEL_TYPE_MAP.keys())
        raise ValueError(
            f"[LAT] Unsupported model_type '{config_model_type}' in config.json. "
            f"Supported types: {supported}"
        )

    _verbose_print(f"Detected model type: {model_type}")
    return model_type


# ============================================================================
# FUSED OPERATIONS PATCHING
# ============================================================================
def _apply_swiglu_patch() -> None:
    """
    Disable fused SwiGLU operations by replacing with PyTorch implementations.

    This ensures compatibility across different hardware and avoids potential
    issues with fused Triton kernels.
    """
    try:
        import torch.nn.functional as F
        from importlib import import_module

        _mlp = import_module("fla.modules.mlp")
        _act = import_module("fla.modules.activations")

        def _pt_swiglu(x, y):
            return F.silu(x) * y

        def _pt_swiglu_linear(x, y, weight, bias):
            return F.linear(F.silu(x) * y, weight, bias)

        _mlp.swiglu = _pt_swiglu
        _mlp.swiglu_linear = _pt_swiglu_linear
        _act.swiglu = _pt_swiglu
        _act.swiglu_linear = _pt_swiglu_linear
        _verbose_print("fuse_swiglu disabled; using PyTorch SwiGLU.")
    except Exception as patch_err:
        print(f"[LAT][warn] Failed to apply SwiGLU runtime patch: {patch_err}")


# ============================================================================
# MODEL LOADING
# ============================================================================
def _import_model_classes(model_type: str) -> Tuple[Any, Any]:
    """
    Dynamically import Config and ForCausalLM classes for a model type.

    Args:
        model_type: Model type key from MODEL_REGISTRY

    Returns:
        Tuple of (ConfigClass, ModelClass)
    """
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"[LAT] Unknown model_type '{model_type}'. Supported: {supported}")

    module_path, config_cls_name, model_cls_name, _ = MODEL_REGISTRY[model_type]

    try:
        from importlib import import_module

        module = import_module(module_path)
        config_cls = getattr(module, config_cls_name)
        model_cls = getattr(module, model_cls_name)
        return config_cls, model_cls
    except ImportError as e:
        raise ImportError(
            f"[LAT] Failed to import {module_path}. "
            f"Ensure flash-linear-attention is installed. Error: {e}"
        ) from e


def load_lat_model(
    model_type: str,
    model_id: str,
    trust_remote_code: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """
    Load a Linear Attention model and tokenizer.

    This is the unified entry point for loading GLA, RetNet, Mamba2, and other
    Linear Attention models from the FLA library.

    Args:
        model_type: Model type ("gla", "retnet", "mamba2", or "auto" for auto-detection)
        model_id: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code
        device: Target device ("cuda", "cpu", or "auto" for device_map="auto")
        dtype: Model dtype (default: torch.bfloat16)

    Returns:
        Dict with keys:
            - "model": The loaded model
            - "tokenizer": The loaded tokenizer
            - "model_type": The resolved model type string
            - "special_handling": Model-specific handling flags from registry

    Raises:
        ValueError: If model_type is invalid or cannot be auto-detected
        RuntimeError: If model loading fails

    Example:
        >>> result = load_lat_model("gla", "fla-hub/gla-1.3B-100B")
        >>> model = result["model"]
        >>> tokenizer = result["tokenizer"]
    """
    _verbose_print(f"Loading model: model_type={model_type}, model_id={model_id}")

    # Auto-detect model type if needed
    if model_type == "auto":
        model_type = detect_model_type(model_id, trust_remote_code)

    # Validate model type
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"[LAT] Unknown model_type '{model_type}'. Supported: {supported}")

    # Get model info from registry
    _, _, _, special_handling = MODEL_REGISTRY[model_type]

    # Import model classes
    ConfigClass, ModelClass = _import_model_classes(model_type)

    # Load config
    try:
        config = ConfigClass.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(
            f"[LAT] Failed to load {ConfigClass.__name__}.from_pretrained('{model_id}'). "
            f"Error: {e}"
        ) from e

    # Apply config patches
    if special_handling.get("has_fuse_swiglu", False):
        # Disable fused SwiGLU unless explicitly enabled
        if not get_lat_env_bool("USE_FUSED_SWIGLU"):
            if hasattr(config, "fuse_swiglu"):
                config.fuse_swiglu = False
            _apply_swiglu_patch()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    # Load model
    model = ModelClass.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map="auto" if device == "auto" else None,
    )

    # Move to device if not using device_map="auto"
    if device != "auto" and device is not None:
        model = model.to(device=device)

    _verbose_print(f"Model loaded successfully: {model_type}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_type": model_type,
        "special_handling": special_handling,
    }


def load_lat_tokenizer(
    model_id: str,
    trust_remote_code: bool = True,
) -> Any:
    """
    Load only the tokenizer for a Linear Attention model.

    Args:
        model_id: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code

    Returns:
        The loaded tokenizer
    """
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)


# ============================================================================
# BACKWARD COMPATIBILITY: GLA-specific functions
# ============================================================================
def load_gla(
    model_id: str,
    trust_remote_code: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """
    Load GLA model - backward compatible with original load_gla() function.

    This function provides the exact same interface as the original load_gla()
    in hf.py for backward compatibility.

    Args:
        model_id: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code
        device: Target device
        dtype: Model dtype

    Returns:
        Dict with "model" and "tokenizer" keys
    """
    result = load_lat_model("gla", model_id, trust_remote_code, device, dtype)
    # Return only model and tokenizer for backward compatibility
    return {"model": result["model"], "tokenizer": result["tokenizer"]}


def load_gla_tokenizer(
    model_id: str = "fla-hub/gla-1.3B-100B",
    trust_remote_code: bool = True,
) -> Any:
    """
    Load GLA tokenizer - backward compatible with original load_gla_tokenizer().
    """
    return load_lat_tokenizer(model_id, trust_remote_code)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get information about a model type from the registry.

    Args:
        model_type: Model type key

    Returns:
        Dict with model information including module_path, class names, and special_handling

    Raises:
        ValueError: If model_type is not in registry
    """
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: {supported}")

    module_path, config_cls, model_cls, special_handling = MODEL_REGISTRY[model_type]
    return {
        "model_type": model_type,
        "module_path": module_path,
        "config_class": config_cls,
        "model_class": model_cls,
        "special_handling": special_handling,
    }


def list_supported_models() -> list:
    """Return list of supported model types."""
    return list(MODEL_REGISTRY.keys())

