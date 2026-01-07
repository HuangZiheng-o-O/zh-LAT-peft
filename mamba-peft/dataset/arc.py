
import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np
import os

from dataset.hf_local import resolve_dataset_path


class ArcDataset(NluDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, name, split="train", use_cache=True, **kwargs):
        path = "allenai/ai2_arc"
        self.name = name
        self.hf_dataset = None

        self.choice_labels = ["A", "B", "C", "D", "E"]
        self.choice_ids = []
        for c in self.choice_labels:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"[ARC] label '{c}' is not single-token for this tokenizer: ids={ids}")
            self.choice_ids.append(ids[0])

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_cache_name(self):
        base = super().get_cache_name()
        return f"{base}_{self.name}"

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            ds_path = resolve_dataset_path(self.path)
            subset = {"arc-easy": "ARC-Easy", "arc-challenge": "ARC-Challenge"}[self.name]
            split_map = {"train": "train", "val": "test", "test": "test"}
            hf_split = split_map.get(self.split, self.split)
            self.hf_dataset = load_dataset(ds_path, subset, trust_remote_code=True)[hf_split]

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        question = self.hf_dataset["question"][idx]
        choices = self.hf_dataset["choices"][idx]["text"]
        choice_labels = self.hf_dataset["choices"][idx]["label"]
        answer = self.hf_dataset["answerKey"][idx]

        if any(choice_labels == ["1", "2", "3", "4", "5"][:i] for i in (3, 4, 5)):
            answer = self.choice_labels[choice_labels.index(answer)]
            choice_labels = self.choice_labels[:len(choice_labels)]

        assert any(choice_labels == self.choice_labels[:i] for i in (3, 4, 5))
        assert answer in choice_labels
        assert len(choices) == len(choice_labels)

        choices_txt = "\n".join([f"{l}. {c}" for l, c in zip(choice_labels, choices)])
        
        input = f"Question: {question}\nChoices:\n{choices_txt}\nAnswer: "
        label = answer

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


class ArcDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = ArcDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
