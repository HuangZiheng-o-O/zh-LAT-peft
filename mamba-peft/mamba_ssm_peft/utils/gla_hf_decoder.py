import torch
from dataclasses import dataclass
from typing import Optional, Any
import os


@dataclass
class GLAHFDecoder:
    tokenizer: Any
    max_length: int = 1024  # interpreted as max_new_tokens when GLA_USE_MAX_NEW_TOKENS=1 (default)
    min_length: int = 0     # interpreted as min_new_tokens when supported and GLA_USE_MAX_NEW_TOKENS=1
    num_beams: Optional[int] = None
    do_sample: bool = False

    def __call__(self, model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Use HF generate provided by fla.models.gla.FLAGenerationMixin
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if attention_mask is None:
            attention_mask = input_ids.ne(pad_id) if pad_id is not None else None

        use_max_new = str(os.getenv("GLA_USE_MAX_NEW_TOKENS", "1")).lower() in ("1", "true", "yes", "on")
        gen_kwargs = dict(
            input_ids=input_ids,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=bool(self.do_sample),
        )
        # Prefer official semantics: max_new_tokens/min_new_tokens
        if use_max_new:
            if os.getenv("GLA_VERBOSE", "0").lower() in ("1","true","yes","on"):
                print("[GLA] Using HF generate(max_new_tokens/min_new_tokens) semantics (GLA_USE_MAX_NEW_TOKENS=1).")
            gen_kwargs["max_new_tokens"] = int(self.max_length)
            if self.min_length and self.min_length > 0:
                gen_kwargs["min_new_tokens"] = int(self.min_length)
        else:
            # Legacy behavior: treat max_length/min_length as total length caps relative to prompt
            if os.getenv("GLA_VERBOSE", "0").lower() in ("1","true","yes","on"):
                print("[GLA] Using legacy generate(max_length=min_length+prompt) semantics (GLA_USE_MAX_NEW_TOKENS=0).")
            gen_kwargs["max_length"] = int(input_ids.shape[1] + self.max_length)
            if self.min_length and self.min_length > 0:
                gen_kwargs["min_length"] = int(input_ids.shape[1] + self.min_length)

        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
            # Optional runtime check to catch right-padding during generation
            if os.getenv("GLA_VERBOSE", "0").lower() in ("1","true","yes","on"):
                try:
                    if attention_mask.size(1) > 0 and (attention_mask[:, -1] == 0).any():
                        msg = ("[GLA][warn] Right-padding detected in attention_mask during generation. "
                               "Ensure tokenizer.padding_side='left' and collator applies left padding.")
                        if os.getenv("GLA_STRICT_LEFT_PAD", "0").lower() in ("1", "true", "yes", "on"):
                            raise RuntimeError(msg)
                        print(msg)

                except Exception:
                    pass
        if self.num_beams is not None and self.num_beams > 1:
            gen_kwargs["num_beams"] = int(self.num_beams)
            gen_kwargs["do_sample"] = False

        try:
            outputs = model.generate(**gen_kwargs)
        except TypeError as e:
            # Some transformers versions may not support min_new_tokens kwarg
            if use_max_new and "min_new_tokens" in str(e):
                raise RuntimeError(
                    "min_new_tokens is not supported by the current transformers version. "
                    "Set GLA_USE_MAX_NEW_TOKENS=0 to fall back to legacy max_length semantics, "
                    "or upgrade transformers."
                ) from e
            raise
        # Trim prompt for downstream metrics (fail fast if incompatible)
        if hasattr(outputs, "sequences"):
            seq = outputs.sequences
            if seq is not None and seq.dim() == 2 and input_ids is not None and input_ids.dim() == 2:
                # Always trim off the original prompt so metrics only see generated tokens
                outputs.sequences = seq[:, input_ids.shape[1]:]
        return outputs


def create_gla_decoder(tokenizer, **kwargs) -> GLAHFDecoder:
    return GLAHFDecoder(tokenizer=tokenizer, **kwargs)
