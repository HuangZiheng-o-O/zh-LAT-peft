import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np

from dataset.hf_local import resolve_dataset_path


class HellaSwagDataset(NluDatasetBase):
    """
    HellaSwag (Rowan/hellaswag)
    Multiple-choice next sentence completion with 4 endings.
    """

    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "Rowan/hellaswag"
        self.hf_dataset = None

        self.choice_labels = ["A", "B", "C", "D"]
        self.choice_ids = []
        for c in self.choice_labels:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"[HellaSwag] label '{c}' is not single-token for this tokenizer: ids={ids}")
            self.choice_ids.append(ids[0])

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_cache_name(self):
        return f"cache_hellaswag_{self.split}"

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            split_map = {"train": "train", "val": "validation", "test": "validation"}
            hf_split = split_map.get(self.split, self.split)
            ds_path = resolve_dataset_path(self.path)
            self.hf_dataset = load_dataset(ds_path, trust_remote_code=True)[hf_split]
        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        # HF columns differ across versions. Common ones include:
        # - ctx (string) OR (ctx_a, ctx_b) OR (activity_label + ctx_a/ctx_b)
        if "ctx" in getattr(self.hf_dataset, "column_names", []):
            ctx = self.hf_dataset["ctx"][idx]
        else:
            ctx_a = self.hf_dataset["ctx_a"][idx] if "ctx_a" in getattr(self.hf_dataset, "column_names", []) else ""
            ctx_b = self.hf_dataset["ctx_b"][idx] if "ctx_b" in getattr(self.hf_dataset, "column_names", []) else ""
            act = self.hf_dataset["activity_label"][idx] if "activity_label" in getattr(self.hf_dataset, "column_names", []) else ""
            ctx = " ".join([str(x).strip() for x in [act, ctx_a, ctx_b] if str(x).strip() != ""]).strip()
        endings = self.hf_dataset["endings"][idx]
        label_idx = int(self.hf_dataset["label"][idx])
        assert 0 <= label_idx <= 3
        label = self.choice_labels[label_idx]

        # Normalize endings type
        if not isinstance(endings, (list, tuple)) or len(endings) != 4:
            # Some variants might store endings as a dict; fail fast with a clear message
            raise ValueError(f"Unexpected 'endings' format for HellaSwag: {type(endings)}")

        choices_txt = "\n".join([f"{l}. {opt}" for l, opt in zip(self.choice_labels, endings)])
        input_txt = f"Context: {ctx}\nChoices:\n{choices_txt}\nAnswer: "
        return input_txt, label

    def compute_metrics(self, eval_preds):
        references = np.concatenate(eval_preds.label_ids)
        predictions = np.concatenate(eval_preds.predictions)

        references_ind = [self.choice_ids.index(r) for r in references]
        predictions_ind = predictions[:, self.choice_ids].argmax(1)
        acc = float(np.mean(predictions_ind == references_ind))
        return {"accuracy": acc}


class HellaSwagDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = HellaSwagDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)


