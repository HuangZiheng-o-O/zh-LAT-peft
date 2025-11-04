import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import snapshot_download
from pathlib import Path
from huggingface_hub import hf_hub_download

from dataset.collator import DataCollator
from .base import NlgDatasetBase
import evaluate
import numpy as np
import pandas as pd


class SamSumDataset(NlgDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "samsum"
        self.hf_dataset = None
        self.input_formatter = None

        super().__init__(tokenizer, path, split,  
                         use_cache=use_cache, **kwargs)
    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            # file-based loading: snapshot samsum repo
            local_root = Path("data") / "samsum"
            local_root.mkdir(parents=True, exist_ok=True)
            snap = Path(snapshot_download(repo_id="samsum", repo_type="dataset", local_dir=str(local_root), local_dir_use_symlinks=False))

            def find_files(split_key: str):
                key = {"train": ["train", "training"], "val": ["validation", "valid", "dev", "val"], "test": ["test"]}[split_key]
                def _match(exts):
                    out = []
                    for hint in key:
                        for ext in exts:
                            out += list(snap.rglob(f"**/*{hint}*.{ext}"))
                    return out
                files_parquet = _match(["parquet"]) 
                files_jsonl  = _match(["jsonl"]) 
                files_json   = _match(["json"]) 
                if files_parquet: return "parquet", sorted(set(files_parquet))
                if files_jsonl:  return "json", sorted(set(files_jsonl))
                if files_json:   return "json", sorted(set(files_json))
                # fallback: try direct download of common names
                name_map = {
                    "train": ["train.json", "train.jsonl", "data/train.json"],
                    "val":   ["validation.json", "valid.json", "dev.json", "data/dev.json"],
                    "test":  ["test.json", "data/test.json"],
                }
                builder = None
                files = []
                dest = snap / "_files"
                dest.mkdir(parents=True, exist_ok=True)
                for fname in name_map[split_key]:
                    try:
                        local = hf_hub_download(repo_id="samsum", repo_type="dataset", filename=fname)
                        p = dest / Path(fname).name
                        if not p.exists():
                            Path(local).rename(p)
                        files.append(p)
                        if p.suffix == ".parquet": builder = builder or "parquet"
                        elif p.suffix in (".jsonl", ".json"): builder = builder or "json"
                    except Exception:
                        continue
                return builder, files

            want = {"train": "train", "val": "val", "test": "test"}[self.split]
            builder, files = find_files(want)
            assert files, f"samsum {want} files not found under {snap}"
            ds = load_dataset(builder, data_files={"train": [str(p) for p in files]})["train"]
            self.hf_dataset = ds

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        input = self.hf_dataset["dialogue"][idx]
        label = self.hf_dataset["summary"][idx]
        return input, label
    
    def compute_metrics(self, eval_preds, eval_mask=None):
        rouge = evaluate.load('rouge')

        if self.mode == "gen":
            if eval_mask is None:
                results = rouge.compute(predictions=eval_preds.preds, references=eval_preds.labels)
            else:
                results = rouge.compute(predictions=[eval_preds.preds[i] for i in eval_mask], references=[eval_preds.labels[i] for i in eval_mask])
        else:
            results = {}

        return results


class SamSumDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = SamSumDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)



