import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np

from dataset.hf_local import resolve_dataset_path


class OpenBookQADataset(NluDatasetBase):
    """
    OpenBookQA (allenai/openbookqa), config "main".
    Multiple-choice QA with 4 options labeled A-D.
    """

    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, config_name="main", **kwargs):
        path = "allenai/openbookqa"
        self.config_name = config_name
        self.hf_dataset = None

        self.choice_labels = ["A", "B", "C", "D"]
        self.choice_ids = []
        for c in self.choice_labels:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"[OpenBookQA] label '{c}' is not single-token for this tokenizer: ids={ids}")
            self.choice_ids.append(ids[0])

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_cache_name(self):
        return f"cache_openbookqa_{self.config_name}_{self.split}"

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            split_map = {"train": "train", "val": "validation", "test": "validation"}
            hf_split = split_map.get(self.split, self.split)
            ds_path = resolve_dataset_path(self.path)
            self.hf_dataset = load_dataset(ds_path, self.config_name, trust_remote_code=True)[hf_split]
        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        # HF format:
        # - question_stem: str
        # - choices: {"label": [...], "text": [...]}
        # - answerKey: "A"/"B"/"C"/"D"
        q = self.hf_dataset["question_stem"][idx]
        choices = self.hf_dataset["choices"][idx]["text"]
        choice_labels = self.hf_dataset["choices"][idx]["label"]
        answer = self.hf_dataset["answerKey"][idx]

        # Normalize labels to A-D order when possible
        if isinstance(choice_labels, (list, tuple)) and len(choice_labels) == len(choices):
            # Some versions might use "1","2","3","4" labels; map to A-D
            if choice_labels == ["1", "2", "3", "4"]:
                mapped = dict(zip(choice_labels, self.choice_labels))
                answer = mapped.get(answer, answer)
                choice_labels = self.choice_labels

        assert answer in self.choice_labels, f"Unexpected OpenBookQA answerKey={answer}"
        choices_txt = "\n".join([f"{l}. {c}" for l, c in zip(choice_labels, choices)])
        input_txt = f"Question: {q}\nChoices:\n{choices_txt}\nAnswer: "
        return input_txt, answer

    def compute_metrics(self, eval_preds):
        references = np.concatenate(eval_preds.label_ids)
        predictions = np.concatenate(eval_preds.predictions)

        references_ind = [self.choice_ids.index(r) for r in references]
        predictions_ind = predictions[:, self.choice_ids].argmax(1)
        acc = float(np.mean(predictions_ind == references_ind))
        return {"accuracy": acc}


class OpenBookQADataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = OpenBookQADataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)


