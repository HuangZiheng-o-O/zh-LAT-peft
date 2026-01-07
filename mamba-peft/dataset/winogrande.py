import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset

from dataset.collator import DataCollator
from .base import NluDatasetBase
import numpy as np

from dataset.hf_local import resolve_dataset_path


class WinoGrandeDataset(NluDatasetBase):
    """
    WinoGrande (allenai/winogrande).
    We use the common "winogrande_xl" config (as lm_eval_harness does).
    """

    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, config_name="winogrande_xl", **kwargs):
        path = "allenai/winogrande"
        self.config_name = config_name
        self.hf_dataset = None

        self.choice_labels = ["1", "2"]
        self.choice_ids = []
        for c in self.choice_labels:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"[WinoGrande] label '{c}' is not single-token for this tokenizer: ids={ids}")
            self.choice_ids.append(ids[0])

        super().__init__(tokenizer, path, split, use_cache=use_cache, **kwargs)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_cache_name(self):
        return f"cache_winogrande_{self.config_name}_{self.split}"

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            split_map = {"train": "train", "val": "validation", "test": "validation"}
            hf_split = split_map.get(self.split, self.split)
            ds_path = resolve_dataset_path(self.path)
            self.hf_dataset = load_dataset(ds_path, self.config_name, trust_remote_code=True)[hf_split]
        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        sentence = self.hf_dataset["sentence"][idx]
        option1 = self.hf_dataset["option1"][idx]
        option2 = self.hf_dataset["option2"][idx]
        answer = str(self.hf_dataset["answer"][idx])  # "1" or "2"
        assert answer in self.choice_labels

        choices_txt = "\n".join([f"{l}. {opt}" for l, opt in zip(self.choice_labels, [option1, option2])])
        input_txt = f"Sentence: {sentence}\nChoices:\n{choices_txt}\nAnswer: "
        return input_txt, answer

    def compute_metrics(self, eval_preds):
        references = np.concatenate(eval_preds.label_ids)
        predictions = np.concatenate(eval_preds.predictions)

        references_ind = [self.choice_ids.index(r) for r in references]
        predictions_ind = predictions[:, self.choice_ids].argmax(1)
        acc = float(np.mean(predictions_ind == references_ind))
        return {"accuracy": acc}


class WinoGrandeDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = WinoGrandeDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)


