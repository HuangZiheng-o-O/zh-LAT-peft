
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

        # IMPORTANT: Labels must be single-token for our next-token classification metric.
        # Use letter labels (A/B) instead of digits (0/1) because:
        # - GLA and RetNet tokenizers encode "0"/"1" as multi-token
        # - Letter labels are single-token across all model tokenizers (GLA, DeltaNet, RetNet)
        self.choice_labels = ["A", "B"]  # A = false/no, B = true/yes
        self.choice_ids = []
        for c in self.choice_labels:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"[BoolQ] label '{c}' is not single-token for this tokenizer: ids={ids}")
            self.choice_ids.append(ids[0])

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def get_cache_name(self):
        # Bump cache name to invalidate legacy caches.
        # v2: digit labels (0/1), v3: letter labels (A/B) for tokenizer compatibility
        base = super().get_cache_name()
        return f"{base}_single_token_v3"

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

        # Map bool -> letter label (single token across all tokenizers)
        label = {False: "A", True: "B"}[label]
        assert label in self.choice_labels

        input = (
            "Answer the question based on the passage. "
            "Respond with 'A' for No/False and 'B' for Yes/True.\n"
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
