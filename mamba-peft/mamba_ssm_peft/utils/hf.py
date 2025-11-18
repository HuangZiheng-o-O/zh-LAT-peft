import json
import sys

import torch

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

# Add flash-linear-attention to path for GLA support
import sys
import os

#if os.path.exists('/home/user/mzs_h/code/flash-linear-attention'):
#   sys.path.insert(0, '/home/user/mzs_h/code/flash-linear-attention')
# --- Robustly locate flash-linear-attention (prefer new submodule path) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))  # .../zh-LAT-peft

# Preferred: 3rdparty/flash-linear-attention (contains 'fla' package)
preferred_dir = os.path.join(repo_root, '3rdparty', 'flash-linear-attention')
legacy_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'flash-linear-attention'))

inserted = False
if os.path.isdir(preferred_dir):
    sys.path.insert(0, preferred_dir)
    inserted = True
else:
    # Fallback: repo_root if it has a top-level 'fla' symlink/dir
    if os.path.isdir(os.path.join(repo_root, 'fla')):
        sys.path.insert(0, repo_root)
        inserted = True
    elif os.path.isdir(legacy_dir):
        sys.path.insert(0, legacy_dir)
        inserted = True

if not inserted:
    print(f"Warning: flash-linear-attention not found under {preferred_dir} or {legacy_dir}; relying on environment")

# Backward-compat shim: provide a no-op decorator for deprecate_kwarg when missing
try:
    from transformers.utils.deprecation import deprecate_kwarg as _hf_deprecate_kwarg  # type: ignore
except Exception:
    import types as _types
    _dep_mod = _types.ModuleType("transformers.utils.deprecation")
    def _noop_deprecate_kwarg(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator
    _dep_mod.deprecate_kwarg = _noop_deprecate_kwarg  # type: ignore[attr-defined]
    sys.modules["transformers.utils.deprecation"] = _dep_mod


def load_config_hf(model_name):
    resolved_archive_file = cached_file(
        model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
    )
    if resolved_archive_file is None:
        raise FileNotFoundError(f"[GLA] {CONFIG_NAME} not found for model '{model_name}'")
    with open(resolved_archive_file, "r") as f:
        return json.load(f)


def load_state_dict_hf(model_name, device=None, dtype=None):
    """
    Load a HF state dict and optionally cast dtype and/or move to device.
    """
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    if resolved_archive_file is None:
        raise FileNotFoundError(f"[GLA] {WEIGHTS_NAME} not found for model '{model_name}'")
    state_dict = torch.load(resolved_archive_file, map_location=mapped_device)
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    if device is not None and device != "cpu":
        state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict


def load_gla(model_id, trust_remote_code=True, device="cuda", dtype=torch.bfloat16):
    """
    Robust GLA loader (flash-linear-attention first; hard fail on mismatch).
    Current behavior:
      1) Use flash-linear-attention GLAForCausalLM/GLAConfig exclusively.
      2) If GLAConfig.from_pretrained fails, raise RuntimeError (no fallback to generic AutoModel).
    Returns: {"model": model, "tokenizer": tokenizer}
    """
    # Try flash-linear-attention implementation first (more reliable for GLA)
    from fla.models.gla import GLAForCausalLM, GLAConfig
    from transformers import AutoTokenizer

    # Strict config loading (no silent fallback)
    try:
        config = GLAConfig.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(
            f"[GLA] Failed to load GLAConfig.from_pretrained('{model_id}'). "
            f"Ensure flash-linear-attention and model weights are compatible. Underlying error: {e}"
        )

    # Optional fused SwiGLU control
    use_fused = str(os.environ.get("GLA_USE_FUSED_SWIGLU", "0")).lower() in ("1", "true", "yes", "on")
    if not use_fused:
        try:
            if hasattr(config, "fuse_swiglu"):
                config.fuse_swiglu = False
        except Exception:
            pass
        try:
            import torch.nn.functional as F
            from importlib import import_module
            _mlp = import_module('fla.modules.mlp')
            _act = import_module('fla.modules.activations')

            def _pt_swiglu(x, y):
                return F.silu(x) * y

            def _pt_swiglu_linear(x, y, weight, bias):
                return F.linear(F.silu(x) * y, weight, bias)

            _mlp.swiglu = _pt_swiglu
            _mlp.swiglu_linear = _pt_swiglu_linear
            _act.swiglu = _pt_swiglu
            _act.swiglu_linear = _pt_swiglu_linear
            print("[GLA] fuse_swiglu disabled; using PyTorch SwiGLU (set GLA_USE_FUSED_SWIGLU=1 to enable fused kernels).")
        except Exception as patch_err:
            print(f"[GLA][warn] Failed to apply SwiGLU runtime patch: {patch_err}")
    else:
        print("[GLA] Using fused SwiGLU kernels (GLA_USE_FUSED_SWIGLU=1).")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = GLAForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map="auto" if device == "auto" else None,
    )
    if device != "auto" and device is not None:
        model = model.to(device=device)
    return {"model": model, "tokenizer": tokenizer}


def load_gla_tokenizer(model_id="fla-hub/gla-1.3B-100B", trust_remote_code=True):
    """
    Load GLA tokenizer
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
