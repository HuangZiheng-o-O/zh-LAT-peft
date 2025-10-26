import json
from typing import Tuple, Optional

import torch

from mamba_ssm_peft import get_mamba_peft_model, load_mamba, load_tokenizer
from mamba_ssm_peft.peft.sd_lora import SdLoraModel


def _dtype_from_prec(prec: str):
    return {"bf16": torch.bfloat16, "fp16": torch.bfloat16, "fp32": torch.float32}[prec]


def prepare_mamba_model_and_tokenizer(
    model_id: str,
    tokenizer_id: str,
    prec: str,
    backend: str,
    is_custom_tokenizer: bool,
    peft_json_path: Optional[str],
    no_print: bool = True,
) -> Tuple[object, object, Optional[object], bool]:
    """
    Prepare Mamba model + tokenizer and (optionally) attach PEFT via project utilities.

    Mirrors the original logic in train.py, returning:
    (model, tokenizer, peft_cfg_or_None, is_sdlora)
    """

    tokenizer = load_tokenizer(tokenizer_id)

    model_kwargs = dict(
        dtype=_dtype_from_prec(prec),
        device="cuda",
        backend=backend,
    )
    model = load_mamba(model_id, **model_kwargs)["model"]

    peft_cfg = None
    is_sdlora = False
    if peft_json_path is not None:
        with open(peft_json_path, "r") as f:
            peft_cfg = json.load(f)

        model, peft_cfg = get_mamba_peft_model(
            model,
            peft_json_path,
            return_peft_cfg=True,
            train_embedding=is_custom_tokenizer,
            no_print=no_print,
        )

        is_sdlora = isinstance(model.base_model, SdLoraModel)

    return model, tokenizer, peft_cfg, is_sdlora


