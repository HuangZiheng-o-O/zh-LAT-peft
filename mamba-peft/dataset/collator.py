import torch
import transformers

from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class DataCollator(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, label_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "label_ids"))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            pad_id = eos_id if eos_id is not None else 0
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        def _pad_sequences(seqs, pad_value, side: str):
            lengths = [t.size(0) for t in seqs]
            max_len = max(lengths) if lengths else 0
            out = seqs[0].new_full((len(seqs), max_len), pad_value) if max_len > 0 else torch.empty((len(seqs), 0), dtype=seqs[0].dtype, device=seqs[0].device)
            for i, t in enumerate(seqs):
                l = t.size(0)
                if l == 0:
                    continue
                if side == "left":
                    out[i, max_len - l:] = t
                else:
                    out[i, :l] = t
            return out

        input_ids = _pad_sequences(input_ids, pad_id, padding_side)
        label_ids = _pad_sequences(label_ids, -100, padding_side)

        return dict(
            input_ids=input_ids,
            label_ids=label_ids,
            attention_mask=input_ids.ne(pad_id),
        )