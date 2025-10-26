import json
import os
from typing import Tuple, Optional

import torch

# Reuse existing loader that already resolves flash-linear-attention pathing
from mamba_ssm_peft.utils.hf import load_gla


def _dtype_from_prec(prec: str):
    # Match original mapping exactly (fp16 mapped to bfloat16 in the legacy code)
    return {"bf16": torch.bfloat16, "fp16": torch.bfloat16, "fp32": torch.float32}[prec]


def prepare_gla_model_and_tokenizer(
    model_id: str,
    prec: str,
    debug: bool,
    peft_json_path: Optional[str],
) -> Tuple[object, object, Optional[object]]:
    """
    Prepare GLA model + tokenizer and (optionally) attach HF PEFT LoRA.

    Behavior is intentionally identical to the inlined logic in train.py:
    - Uses load_gla(...) to get model & tokenizer
    - When peft_json_path is provided, loads JSON and applies env overrides:
      HP_PEFT_R, HP_PEFT_ALPHA, HP_PEFT_DROPOUT, HP_INIT, HP_PISSA_FAST
      Then builds peft.LoraConfig and wraps with peft.get_peft_model(...)

    Returns (model, tokenizer, peft_cfg) where peft_cfg is None for GLA.
    """

    model_kwargs = dict(
        dtype=_dtype_from_prec(prec),
        device="cuda" if not debug else "cpu",
        trust_remote_code=True,
    )

    gla_loaded = load_gla(model_id, **model_kwargs)
    model = gla_loaded["model"]
    tokenizer = gla_loaded["tokenizer"]

    peft_cfg = None
    if peft_json_path is not None:
        # Lazy import to avoid a hard dependency unless needed
        from peft import LoraConfig, get_peft_model

        with open(peft_json_path, "r") as f:
            peft_json = json.load(f)

        # Optional env overrides from launcher (highest precedence)
        env = os.environ
        r_env = env.get("HP_PEFT_R")
        alpha_env = env.get("HP_PEFT_ALPHA")
        drop_env = env.get("HP_PEFT_DROPOUT")
        init_env = env.get("HP_INIT")
        fast_pissa_env = env.get("HP_PISSA_FAST")

        if r_env is not None:
            try:
                peft_json["r"] = int(r_env)
            except Exception:
                pass
        if alpha_env is not None:
            try:
                peft_json["lora_alpha"] = int(alpha_env)
            except Exception:
                pass
        if drop_env is not None:
            try:
                peft_json["lora_dropout"] = float(drop_env)
            except Exception:
                pass
        if init_env:
            # e.g., "pissa" or "pissa_niter_4"
            peft_json["init_lora_weights"] = init_env
        else:
            # If HP_PISSA_FAST is set, and config uses PiSSA, switch to fast SVD init
            try:
                if fast_pissa_env and str(fast_pissa_env).lower() not in ("0", "false", "no", "off"):
                    init_val = peft_json.get("init_lora_weights", None)
                    if isinstance(init_val, str) and init_val.lower() == "pissa":
                        peft_json["init_lora_weights"] = "pissa_niter_4"
            except Exception:
                pass

        peft_cfg = LoraConfig(**peft_json)
        model = get_peft_model(model, peft_cfg)

    return model, tokenizer, peft_cfg


