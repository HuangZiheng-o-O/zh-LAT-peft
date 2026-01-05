"""
HuggingFace Model Loading Utilities for LAT Framework.

This module provides utilities for loading models and weights from the
HuggingFace Hub, with proper path setup for the FLA library.

Primary Functions:
=================
- load_gla: Load GLA (Gated Linear Attention) model
- load_retnet: Load RetNet (Retentive Network) model
- load_mamba2: Load Mamba2 (State Space Model)
- load_config_hf: Load config.json from HuggingFace Hub
- load_state_dict_hf: Load model weights from HuggingFace Hub

All model loading functions delegate to the unified lat_model_loader.py
for consistent behavior and reduced code duplication.
"""

import json
import sys
import os

import torch
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


# ============================================================================
# FLA LIBRARY PATH SETUP
# ============================================================================
# Add flash-linear-attention to path for FLA support
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))  # .../zh-LAT-peft

# Preferred: 3rdparty/flash-linear-attention (contains 'fla' package)
preferred_dir = os.path.join(repo_root, '3rdparty', 'flash-linear-attention')
legacy_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'flash-linear-attention'))

_fla_path_inserted = False
if os.path.isdir(preferred_dir):
    sys.path.insert(0, preferred_dir)
    _fla_path_inserted = True
elif os.path.isdir(os.path.join(repo_root, 'fla')):
    sys.path.insert(0, repo_root)
    _fla_path_inserted = True
elif os.path.isdir(legacy_dir):
    sys.path.insert(0, legacy_dir)
    _fla_path_inserted = True

if not _fla_path_inserted:
    print(f"[LAT][warn] flash-linear-attention not found under {preferred_dir} or {legacy_dir}; "
          "relying on environment")

# Backward-compat shim: provide a no-op decorator for deprecate_kwarg when missing
try:
    from transformers.utils.deprecation import deprecate_kwarg as _hf_deprecate_kwarg  # noqa: F401
except Exception:
    import types as _types
    _dep_mod = _types.ModuleType("transformers.utils.deprecation")

    def _noop_deprecate_kwarg(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator

    _dep_mod.deprecate_kwarg = _noop_deprecate_kwarg  # type: ignore[attr-defined]
    sys.modules["transformers.utils.deprecation"] = _dep_mod


# ============================================================================
# CONFIG AND WEIGHTS LOADING
# ============================================================================
def load_config_hf(model_name: str) -> dict:
    """
    Load config.json from HuggingFace Hub or local path.

    Args:
        model_name: HuggingFace model ID or local path

    Returns:
        Parsed config dictionary

    Raises:
        FileNotFoundError: If config.json is not found
    """
    resolved_archive_file = cached_file(
        model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
    )
    if resolved_archive_file is None:
        raise FileNotFoundError(f"[LAT] {CONFIG_NAME} not found for model '{model_name}'")
    with open(resolved_archive_file, "r") as f:
        return json.load(f)


def load_state_dict_hf(
    model_name: str,
    device: str = None,
    dtype: torch.dtype = None,
) -> dict:
    """
    Load model state_dict from HuggingFace Hub and optionally cast dtype and/or move to device.

    Args:
        model_name: HuggingFace model ID or local path
        device: Target device (optional)
        dtype: Target dtype (optional)

    Returns:
        Model state dictionary

    Raises:
        FileNotFoundError: If model weights are not found
    """
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(
        model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
    )
    if resolved_archive_file is None:
        raise FileNotFoundError(f"[LAT] {WEIGHTS_NAME} not found for model '{model_name}'")
    state_dict = torch.load(resolved_archive_file, map_location=mapped_device)
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    if device is not None and device != "cpu":
        state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict


# ============================================================================
# MODEL LOADING (delegate to lat_model_loader)
# ============================================================================
# Import from unified loader
from .lat_model_loader import (
    load_lat_model,
    load_lat_tokenizer,
    load_gla as _load_gla,
    load_gla_tokenizer as _load_gla_tokenizer,
)


def load_gla(
    model_id: str,
    trust_remote_code: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Load GLA (Gated Linear Attention) model and tokenizer.

    This function delegates to the unified lat_model_loader for consistent behavior.

    Args:
        model_id: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code
        device: Target device ("cuda", "cpu", or "auto")
        dtype: Model dtype (default: torch.bfloat16)

    Returns:
        Dict with "model" and "tokenizer" keys
    """
    return _load_gla(model_id, trust_remote_code, device, dtype)


def load_gla_tokenizer(
    model_id: str = "fla-hub/gla-1.3B-100B",
    trust_remote_code: bool = True,
):
    """Load GLA tokenizer."""
    return _load_gla_tokenizer(model_id, trust_remote_code)


def load_retnet(
    model_id: str,
    trust_remote_code: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Load RetNet (Retentive Network) model and tokenizer.

    Reference: "Retentive Network: A Successor to Transformer for Large Language Models"
               https://arxiv.org/abs/2307.08621

    Args:
        model_id: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code
        device: Target device ("cuda", "cpu", or "auto")
        dtype: Model dtype (default: torch.bfloat16)

    Returns:
        Dict with "model" and "tokenizer" keys
    """
    result = load_lat_model("retnet", model_id, trust_remote_code, device, dtype)
    return {"model": result["model"], "tokenizer": result["tokenizer"]}


def load_retnet_tokenizer(model_id: str, trust_remote_code: bool = True):
    """Load RetNet tokenizer."""
    return load_lat_tokenizer(model_id, trust_remote_code)


def load_mamba2(
    model_id: str,
    trust_remote_code: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Load Mamba2 (State Space Model) and tokenizer.

    Note: Mamba2 uses cache_params instead of past_key_values for caching.

    Reference: "Transformers are SSMs: Generalized Models and Efficient Algorithms
               Through Structured State Space Duality"
               https://arxiv.org/abs/2405.21060

    Args:
        model_id: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code
        device: Target device ("cuda", "cpu", or "auto")
        dtype: Model dtype (default: torch.bfloat16)

    Returns:
        Dict with "model" and "tokenizer" keys
    """
    result = load_lat_model("mamba2", model_id, trust_remote_code, device, dtype)
    return {"model": result["model"], "tokenizer": result["tokenizer"]}


def load_mamba2_tokenizer(model_id: str, trust_remote_code: bool = True):
    """Load Mamba2 tokenizer."""
    return load_lat_tokenizer(model_id, trust_remote_code)
