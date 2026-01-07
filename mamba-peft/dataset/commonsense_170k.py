import os
from pathlib import Path

import numpy as np
import transformers
from datasets import load_dataset
from transformers.models.auto import AutoTokenizer

from dataset.collator import DataCollator
from .base import DatasetBase


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMONSENSE_170K_PATH = REPO_ROOT / "commonsense_170k_data" / "commonsense_170k.json"


def _get_commonsense_170k_path() -> Path:
    # Prefer explicit env override (useful when the json is stored elsewhere)
    p = (
        os.environ.get("LAT_COMMONSENSE_170K_PATH")
        or os.environ.get("COMMONSENSE_170K_PATH")
        or os.environ.get("LAT_COMMONSENSE_170K_JSON")
        or os.environ.get("COMMONSENSE_170K_JSON")
    )
    if p:
        return Path(p).expanduser().resolve()
    return DEFAULT_COMMONSENSE_170K_PATH


def _build_prompt(instruction: str, inp: str) -> str:
    # Mirror reference/MambaPEFT/language/commonsense_reasoning/finetune.py::generate_prompt,
    # but we split prompt/output so our trainer learns to generate the response conditioned on prompt.
    instruction = "" if instruction is None else str(instruction)
    inp = "" if inp is None else str(inp)
    if inp.strip():
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{inp}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


class Commonsense170kDataset(DatasetBase):
    """
    Mixed commonsense instruction-tuning dataset (LLM-Adapters commonsense_170k.json).

    Data format (per row): {"instruction": str, "input": str, "output": str, "answer": str}
    We train causal LM to generate the "output" given the prompt built from instruction/input.

    Reference implementation:
      reference/MambaPEFT/language/commonsense_reasoning/finetune.py
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        split: str = "train",
        use_cache: bool = True,
        val_set_size: int | None = None,
        seed: int = 42,
        **kwargs,
    ):
        self.data_path = _get_commonsense_170k_path()
        if not self.data_path.is_file():
            raise FileNotFoundError(
                f"commonsense_170k.json not found at {self.data_path}. "
                "Set LAT_COMMONSENSE_170K_PATH=/path/to/commonsense_170k.json"
            )

        # Match reference default (val_set_size=2000) unless user overrides.
        if val_set_size is None:
            val_set_size = int(os.environ.get("LAT_COMMONSENSE_170K_VAL_SET_SIZE", "2000"))
        self.val_set_size = int(val_set_size)
        self.seed = int(seed)

        self.hf_dataset = None
        self._split_cache = {}

        # Use a stable cache namespace; actual file path is tracked separately.
        super().__init__(
            tokenizer=tokenizer,
            path="commonsense_170k",
            split=split,
            use_cache=use_cache,
            **kwargs,
        )

    def get_cache_name(self):
        base = super().get_cache_name()
        suffix = f"_val{self.val_set_size}"
        if getattr(self, "max_seqlen", None) is not None:
            suffix += f"_seqlen{self.max_seqlen}"
        return f"{base}{suffix}"

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def _load_base_dataset(self):
        # datasets can load JSON arrays and jsonl; commonsense_170k.json is a JSON array.
        return load_dataset("json", data_files=str(self.data_path), split="train")

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            base = self._load_base_dataset()

            # Build a deterministic train/val split like the reference code.
            if self.val_set_size and self.val_set_size > 0:
                tv = base.train_test_split(test_size=self.val_set_size, shuffle=True, seed=self.seed)
                self._split_cache["train"] = tv["train"]
                self._split_cache["val"] = tv["test"]
                self._split_cache["test"] = tv["test"]
            else:
                self._split_cache["train"] = base
                # If val_set_size==0, still provide a tiny val split to keep Trainer happy.
                n_val = min(2000, len(base))
                self._split_cache["val"] = base.select(range(n_val))
                self._split_cache["test"] = self._split_cache["val"]

            self.hf_dataset = self._split_cache.get(self.split, self._split_cache["train"])

        return self.hf_dataset

    def get_input_label(self, idx):
        ds = self.get_hf_dataset()
        instruction = ds["instruction"][idx]
        inp = ds["input"][idx] if "input" in ds.column_names else ""
        output = ds["output"][idx]

        prompt = _build_prompt(instruction, inp)
        label = "" if output is None else str(output)
        return prompt, label

    def preproc_input_label(self, input, label):
        # IMPORTANT: do NOT inject sep_token/eos_token *between* prompt and response.
        # We only append EOS at the end of the response to mark termination (matches common SFT practice).
        eos = getattr(self.tokenizer, "eos_token", None) or ""
        return input, label + eos

    def compute_metrics(self, eval_preds):
        # Token-level accuracy on response tokens only (prompt tokens are masked out by DatasetBase logic).
        refs = np.concatenate(eval_preds.label_ids) if len(eval_preds.label_ids) else np.array([], dtype=np.int64)
        preds = np.concatenate(eval_preds.predictions) if len(eval_preds.predictions) else np.array([], dtype=np.float32)
        if preds.size == 0 or refs.size == 0:
            return {"token_accuracy": 0.0, "n_label_tokens": 0}
        pred_ids = preds.argmax(-1)
        acc = float(np.mean(pred_ids == refs))
        return {"token_accuracy": acc, "n_label_tokens": int(refs.size)}


class Commonsense170kDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = Commonsense170kDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)


