import json
import sys

import torch

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

# Add flash-linear-attention to path for GLA support
# Try server path first, then local path
import os
if os.path.exists('/home/user/mzs_h/code/flash-linear-attention'):
    sys.path.insert(0, '/home/user/mzs_h/code/flash-linear-attention')
elif os.path.exists('/Users/huangziheng/PycharmProjects/code/flash-linear-attention'):
    sys.path.insert(0, '/Users/huangziheng/PycharmProjects/code/flash-linear-attention')
else:
    print("Warning: flash-linear-attention path not found")

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
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def load_state_dict_hf(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, map_location=mapped_device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict


def load_gla(model_id, trust_remote_code=True, device="cuda", dtype=torch.bfloat16):
    """
    Robust GLA loader:
    1) Prefer flash-linear-attention GLA classes (GLAForCausalLM/GLAConfig)
    2) Fallback to HF AutoModel (only if registered)
    Returns: {"model": model, "tokenizer": tokenizer}
    """
    # Try flash-linear-attention implementation first (more reliable for GLA)
    e_fla_msg = None
    try:
        from fla.models.gla import GLAForCausalLM, GLAConfig
        from transformers import AutoTokenizer

        # Load config (with graceful fallback to raw config.json)
        try:
            config = GLAConfig.from_pretrained(model_id)
        except Exception:
            raw_cfg = load_config_hf(model_id)
            config = GLAConfig(
                vocab_size=raw_cfg.get("vocab_size", 32000),
                hidden_size=raw_cfg.get("hidden_size", 2048),
                intermediate_size=raw_cfg.get("intermediate_size", 5632),
                num_hidden_layers=raw_cfg.get("num_hidden_layers", 24),
                num_attention_heads=raw_cfg.get("num_attention_heads", 32),
                max_position_embeddings=raw_cfg.get("max_position_embeddings", 2048),
                rms_norm_eps=raw_cfg.get("rms_norm_eps", 1e-6),
                use_cache=True,
                pad_token_id=raw_cfg.get("pad_token_id", None),
                bos_token_id=raw_cfg.get("bos_token_id", 1),
                eos_token_id=raw_cfg.get("eos_token_id", 2),
            )
        # Disable fused SwiGLU to avoid Triton autotuner issues on certain torch/triton combos
        try:
            config.fuse_swiglu = False
        except Exception:
            pass

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Load model
        # Runtime monkey patch: disable Triton SwiGLU by overriding to PyTorch ops
        try:
            import torch.nn.functional as F
            from importlib import import_module
            _mlp = import_module('fla.modules.mlp')
            _act = import_module('fla.modules.activations')

            def _pt_swiglu(x, y):
                return F.silu(x) * y

            def _pt_swiglu_linear(x, y, weight, bias):
                return F.linear(F.silu(x) * y, weight, bias)

            # Patch both modules so global lookups resolve to PT versions
            _mlp.swiglu = _pt_swiglu
            _mlp.swiglu_linear = _pt_swiglu_linear
            _act.swiglu = _pt_swiglu
            _act.swiglu_linear = _pt_swiglu_linear
        except Exception:
            pass

        model = GLAForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            device_map="auto" if device == "auto" else None,
        )

        if device != "auto" and device is not None:
            model = model.to(device=device)

        return {"model": model, "tokenizer": tokenizer}

    except Exception as e_fla:
        e_fla_msg = str(e_fla)
        print(f"[load_gla] flash-linear-attention load failed: {e_fla_msg}")

    # Explicitly do NOT fallback to HF AutoModel for 'gla' (not registered in your transformers).
    # If we reach here, raise a clear error.
    raise RuntimeError(f"Failed to load GLA model via flash-linear-attention: {e_fla_msg}")


def load_gla_tokenizer(model_id="fla-hub/gla-1.3B-100B", trust_remote_code=True):
    """
    Load GLA tokenizer
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
