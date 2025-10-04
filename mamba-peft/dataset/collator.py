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
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            label_ids=label_ids,
            attention_mask=input_ids.ne(pad_id),
        )