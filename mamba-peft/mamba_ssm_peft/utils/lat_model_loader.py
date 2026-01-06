"""
Unified Linear Attention Model Loader.

This module provides a unified interface for loading various Linear Attention models
from the FLA (Flash Linear Attention) library, including GLA, RetNet, Mamba2, and others.

Design Principles:
==================
1. **Backward Compatibility**: GLA loading remains identical to the original `load_gla()`.
2. **Unified Interface**: All models use the same `load_lat_model()` entry point.
3. **Auto-Detection**: Model type can be automatically detected from config.json.
4. **Extensibility**: Easy to add new model types via ModelRegistry.

Supported Models:
================
- gla: Gated Linear Attention (https://arxiv.org/abs/2312.06635)
- retnet: Retentive Network (https://arxiv.org/abs/2307.08621)
- delta_net: DeltaNet (https://arxiv.org/abs/2406.06484)
- mamba2: Mamba2 State Space Model (https://arxiv.org/abs/2405.21060)

Environment Variables (LAT_* preferred, GLA_* fallback for compatibility):
=========================================================================
- LAT_FORCE_LEFT_PAD: Force left padding
- LAT_VERBOSE: Enable verbose logging
- LAT_USE_FUSED_SWIGLU: Enable fused SwiGLU (default: disabled)

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
from typing import Any, Dict

import torch
from transformers import AutoTokenizer
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

# Import from new modular components
from .env_config import env_config, get_lat_env, get_lat_env_bool
from .lat_base import ModelRegistry, ModelSpec, ModelCapabilities, CONFIG_MODEL_TYPE_MAP
from .patches import apply_model_patches


def _verbose_print(msg: str) -> None:
    """Print message if LAT_VERBOSE is enabled."""
    if env_config.get_bool("VERBOSE"):
        print(f"[LAT] {msg}")


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
            - "capabilities": ModelCapabilities for this model type

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

    # Get model spec from registry
    spec = ModelRegistry.get(model_type)
    capabilities = spec.capabilities

    # Import model classes dynamically
    ConfigClass, ModelClass = spec.import_classes()

    # Load config
    try:
        config = ConfigClass.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(
            f"[LAT] Failed to load {spec.config_class_name}.from_pretrained('{model_id}'). "
            f"Error: {e}"
        ) from e

    # Apply patches (e.g., disable fused SwiGLU)
    if capabilities.has_fuse_swiglu:
        apply_model_patches(model_type, config)

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
        "capabilities": capabilities,
        # Backward compatibility alias
        "special_handling": {
            "has_fuse_swiglu": capabilities.has_fuse_swiglu,
            "cache_type": capabilities.cache_type,
            "inner_model_attr": capabilities.inner_model_attr,
        },
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
        Dict with model information

    Raises:
        ValueError: If model_type is not in registry
    """
    spec = ModelRegistry.get(model_type)
    return {
        "model_type": spec.model_type,
        "module_path": spec.module_path,
        "config_class": spec.config_class_name,
        "model_class": spec.model_class_name,
        "capabilities": spec.capabilities,
    }


def list_supported_models() -> list:
    """Return list of supported model types."""
    return ModelRegistry.list_models()
