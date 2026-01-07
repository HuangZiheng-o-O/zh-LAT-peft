
import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np

from dataset.hf_local import resolve_dataset_path


class BoolQDataset(NluDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "google/boolq"
        self.hf_dataset = None

        # Get vocab dict - compatible with both fast and slow tokenizers
        vocab_dict = tokenizer.vocab if (hasattr(tokenizer, 'vocab') and not callable(tokenizer.vocab)) else tokenizer.get_vocab()

        # IMPORTANT: Labels must be single-token for our next-token classification metric.
        # "true/false" may not be single-token under GPT-Neox BPE; use digits for robustness.
        self.choice_labels = ["0", "1"]  # 0 = false/no, 1 = true/yes
        self.choice_ids = [vocab_dict[c] for c in self.choice_labels]

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def get_cache_name(self):
        # Bump cache name to invalidate legacy caches that stored "true/false" labels.
        base = super().get_cache_name()
        return f"{base}_single_token_v2"

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            split_map = {"train": "train", "val": "validation", "test": "validation"}
            hf_split = split_map.get(self.split, self.split)
            ds_path = resolve_dataset_path(self.path)
            self.hf_dataset = load_dataset(ds_path, trust_remote_code=True)[hf_split]

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        question = self.hf_dataset["question"][idx]
        passage = self.hf_dataset["passage"][idx]
        label = self.hf_dataset["answer"][idx]

        # Map bool -> digit label (single token)
        label = {False: "0", True: "1"}[label]
        assert label in self.choice_labels
        
        input = (
            "Answer the question based on the passage. "
            "Respond with '0' for No/False and '1' for Yes/True.\n"
            f"Question: {question}\n"
            f"Passage: {passage}\n"
            "Answer: "
        )

        # print(input)

        return input, label
    
    def compute_metrics(self, eval_preds):
        references = np.concatenate(eval_preds.label_ids)
        predictions = np.concatenate(eval_preds.predictions)  # .argmax(-1)

        references_ind = [self.choice_ids.index(r) for r in references]
        predictions_ind = predictions[:, self.choice_ids].argmax(1)

        acc = float(np.mean(predictions_ind == references_ind))

        return {
            "accuracy": acc,
        }


class BoolQDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = BoolQDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
