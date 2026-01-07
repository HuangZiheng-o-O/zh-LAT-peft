"""
Optional: lm-evaluation-harness integration for zh-LAT-peft.

This mirrors reference/MambaPEFT/language/commonsense_reasoning/lm_harness_eval.py,
but MUST reuse zh-LAT-peft's unified loader/adapter stack:
  ModelRegistry + lat_model_loader + lat_adapter + env_config

Install (recommended, minimal):
  pip install 'lm-eval' || pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git

Example:
  TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'
  python scripts/eval/lat_lm_harness_eval.py \
    --model LAT \
    --model_args pretrained="fla-hub/gla-1.3B-100B,model_type=gla,prec=bf16,peft_weights=/path/to/adapter,trust_remote_code=True" \
    --tasks $TASKS \
    --output_path outputs/lm_eval/lm_harness

Notes:
- lm_eval_harness will load task datasets using HF datasets cache.
  If you require fully local/offline datasets from mamba-peft/data/, prefer eval_lat.py.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import transformers

try:
    from lm_eval.api.registry import register_model
    from lm_eval.models.huggingface import HFLM
    from lm_eval.__main__ import cli_evaluate
except Exception as e:  # pragma: no cover
    print(
        "[LAT][lm_eval] lm-evaluation-harness is not installed.\n"
        "Install one of:\n"
        "  - pip install lm-eval\n"
        "  - pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git\n"
        f"Original import error: {e}",
        file=sys.stderr,
    )
    raise

from lat_adapter import prepare_lat_model_and_tokenizer, attach_peft_weights


def _dtype_from_prec(prec: str) -> torch.dtype:
    prec = str(prec).lower()
    if prec in ("bf16", "bfloat16"):
        return torch.bfloat16
    if prec in ("fp16", "float16", "half"):
        # Match training behavior (fp16 -> bf16)
        return torch.bfloat16
    if prec in ("fp32", "float32"):
        return torch.float32
    return torch.bfloat16


@register_model("LAT")
class LATEvalWrapper(HFLM):
    """
    lm_eval_harness HF wrapper that loads zh-LAT-peft models via lat_adapter.
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
        self,
        pretrained: str,
        model_type: str = "auto",
        peft_weights: Optional[str] = None,
        prec: str = "bf16",
        max_length: int = 2048,
        batch_size: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        self.model_type = model_type
        self.peft_weights = peft_weights
        self.prec = prec
        self._device = torch.device(device)

        # HFLM will call our _create_model during init.
        super().__init__(
            pretrained=pretrained,
            tokenizer=pretrained,
            max_length=max_length,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        self._batch_size = int(batch_size) if batch_size is not None else 64

    @property
    def batch_size(self):
        return self._batch_size

    def _create_model(self, pretrained: str, dtype="float32", **kwargs) -> None:
        # Respect CUDA_VISIBLE_DEVICES; use debug=True for CPU
        debug = str(self._device) == "cpu"
        torch_dtype = _dtype_from_prec(self.prec)

        model, tokenizer, _ = prepare_lat_model_and_tokenizer(
            model_type=self.model_type,
            model_id=pretrained,
            prec=self.prec,
            debug=debug,
            peft_json_path=None,
        )

        if self.peft_weights:
            model = attach_peft_weights(model, self.peft_weights, torch_dtype=torch_dtype)

        try:
            if hasattr(model, "config"):
                model.config.use_cache = False
        except Exception:
            pass

        self._model = model
        self.tokenizer = tokenizer


if __name__ == "__main__":
    cli_evaluate()


