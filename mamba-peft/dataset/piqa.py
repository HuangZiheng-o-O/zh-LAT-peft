
import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np
import os

from dataset.hf_local import resolve_dataset_path


class PiqaDataset(NluDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "piqa"
        self.hf_dataset = None

        # Use letter labels (A/B) instead of digits (0/1) because:
        # - GLA and RetNet tokenizers encode "0"/"1" as multi-token
        # - Letter labels are single-token across all model tokenizers (GLA, DeltaNet, RetNet)
        self.choice_labels = ["A", "B"]
        self.choice_ids = []
        for c in self.choice_labels:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"[PIQA] label '{c}' is not single-token for this tokenizer: ids={ids}")
            self.choice_ids.append(ids[0])

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            # Support split aliases
            split_map = {"train": "train", "val": "validation", "test": "validation"}
            hf_split = split_map.get(self.split, self.split)
            ds_path = resolve_dataset_path(self.path)
            self.hf_dataset = load_dataset(ds_path, trust_remote_code=True)[hf_split]

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        goal = self.hf_dataset["goal"][idx]
        sol1 = self.hf_dataset["sol1"][idx]
        sol2 = self.hf_dataset["sol2"][idx]
        label_raw = self.hf_dataset["label"][idx]  # 0 or 1

        # Map numeric label to letter label
        label = {0: "A", 1: "B", "0": "A", "1": "B"}[label_raw]
        assert label in self.choice_labels

        choices_txt = "\n".join([f"{l}. {c}" for l, c in zip(self.choice_labels, [sol1, sol2])])
        
        input = f"Question: {goal}\nChoices:\n{choices_txt}\nAnswer: "

        if str(os.environ.get("LAT_VERBOSE", os.environ.get("GLA_VERBOSE", "0"))).lower() in ("1", "true", "yes", "on"):
            print(input)

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


class PiqaDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = PiqaDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
