import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np

from dataset.hf_local import resolve_dataset_path


class SocialIQADataset(NluDatasetBase):
    """
    SocialIQA (allenai/social_i_qa)
    Format: context + question + 3 answers (A/B/C), label is correct option.
    """

    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "allenai/social_i_qa"
        self.hf_dataset = None

        vocab_dict = tokenizer.vocab if (hasattr(tokenizer, "vocab") and not callable(tokenizer.vocab)) else tokenizer.get_vocab()
        self.choice_labels = ["A", "B", "C"]
        self.choice_ids = [vocab_dict[c] for c in self.choice_labels]

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_cache_name(self):
        return f"cache_social_iqa_{self.split}"

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            split_map = {"train": "train", "val": "validation", "test": "validation"}
            hf_split = split_map.get(self.split, self.split)
            ds_path = resolve_dataset_path(self.path)
            self.hf_dataset = load_dataset(ds_path, trust_remote_code=True)[hf_split]
        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        # Standard SocialIQA columns on HF:
        # - context, question, answerA, answerB, answerC, label (0/1/2)
        context = self.hf_dataset["context"][idx]
        question = self.hf_dataset["question"][idx]
        a = self.hf_dataset["answerA"][idx]
        b = self.hf_dataset["answerB"][idx]
        c = self.hf_dataset["answerC"][idx]
        label_idx = int(self.hf_dataset["label"][idx])

        assert 0 <= label_idx <= 2
        label = self.choice_labels[label_idx]

        choices_txt = "\n".join([f"{l}. {opt}" for l, opt in zip(self.choice_labels, [a, b, c])])
        input_txt = f"Context: {context}\nQuestion: {question}\nChoices:\n{choices_txt}\nAnswer: "
        return input_txt, label

    def compute_metrics(self, eval_preds):
        references = np.concatenate(eval_preds.label_ids)
        predictions = np.concatenate(eval_preds.predictions)

        references_ind = [self.choice_ids.index(r) for r in references]
        predictions_ind = predictions[:, self.choice_ids].argmax(1)
        acc = float(np.mean(predictions_ind == references_ind))
        return {"accuracy": acc}


class SocialIQADataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = SocialIQADataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)


