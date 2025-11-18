import torch
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class GLAHFDecoder:
    tokenizer: Any
    max_length: int = 1024
    min_length: int = 0
    num_beams: Optional[int] = None
    do_sample: bool = False

    def __call__(self, model, input_ids: torch.Tensor):
        # Use HF generate provided by fla.models.gla.FLAGenerationMixin
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        attention_mask = None
        try:
            if pad_id is not None:
                attention_mask = input_ids.ne(pad_id)
        except Exception:
            attention_mask = None
        gen_kwargs = dict(
            input_ids=input_ids,
            max_length=int(input_ids.shape[1] + self.max_length),
            min_length=int(self.min_length),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=bool(self.do_sample),
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if self.num_beams is not None and self.num_beams > 1:
            gen_kwargs["num_beams"] = int(self.num_beams)
            gen_kwargs["do_sample"] = False

        outputs = model.generate(**gen_kwargs)
        # Trim prompt for downstream metrics
        if hasattr(outputs, "sequences"):
            seq = outputs.sequences
            if seq is not None and seq.dim() == 2 and input_ids is not None and input_ids.dim() == 2:
                try:
                    outputs.sequences = seq[:, input_ids.shape[1]:]
                except Exception:
                    pass
        return outputs


def create_gla_decoder(tokenizer, **kwargs) -> GLAHFDecoder:
    return GLAHFDecoder(tokenizer=tokenizer, **kwargs)
