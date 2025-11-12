import os
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
            # 1) Prefer local CSV (offline-friendly)
            def _try_load_local_csv(root: Path):
                split_map = {"train": ["train.csv"], "val": ["validation.csv", "valid.csv", "dev.csv"], "test": ["test.csv"]}
                target_names = split_map[{"train": "train", "val": "val", "test": "test"}[self.split]]
                for name in target_names:
                    if (root / name).is_file():
                        files_dict = {}
                        for k, names in split_map.items():
                            for nm in names:
                                p = root / nm
                                if p.is_file():
                                    files_dict[k] = str(p)
                                    break
                        if files_dict:
                            ds_all = load_dataset("csv", data_files=files_dict)
                            key = {"train": "train", "val": "val", "test": "test"}[self.split]
                            # datasets.load_dataset("csv") will use provided keys
                            # Align 'val' key if not present
                            if key not in ds_all and key == "val" and "validation" in ds_all:
                                return ds_all["validation"]
                            return ds_all[key]
                return None

            # Try env SAMSUM_LOCAL_DIR first
            env_dir = os.environ.get("SAMSUM_LOCAL_DIR")
            if env_dir:
                maybe = _try_load_local_csv(Path(env_dir))
                if maybe is not None:
                    self.hf_dataset = maybe
                    return self.hf_dataset
            # Try default data/samsum
            default_dir = Path("data") / "samsum"
            maybe = _try_load_local_csv(default_dir)
            if maybe is not None:
                self.hf_dataset = maybe
                return self.hf_dataset

            # 2) Online snapshot (may require internet and valid HF endpoint)
            local_root = default_dir
            local_root.mkdir(parents=True, exist_ok=True)
            try:
                snap = Path(snapshot_download(repo_id="samsum", repo_type="dataset", local_dir=str(local_root), local_dir_use_symlinks=False))
            except Exception:
                snap = None

            def find_files(split_key: str):
                key = {"train": ["train", "training"], "val": ["validation", "valid", "dev", "val"], "test": ["test"]}[split_key]
                def _match(exts):
                    out = []
                    if snap is not None:
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
                dest = (snap if snap is not None else local_root) / "_files"
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
            try:
                builder, files = find_files(want)
                if not files:
                    raise FileNotFoundError(f"samsum {want} files not found under {local_root}")
                ds = load_dataset(builder, data_files={"train": [str(p) for p in files]})["train"]
                self.hf_dataset = ds
            except Exception:
                # 3) Last-resort: datasets hub (requires internet and correct HF endpoint)
                try:
                    ds_all = load_dataset("samsum")
                    self.hf_dataset = ds_all[{"train": "train", "val": "validation", "test": "test"}[self.split]]
                except Exception as e:
                    raise RuntimeError(
                        "Failed to load SAMSum dataset from HuggingFace Hub and local CSV.\n"
                        "Provide local CSVs (train.csv, validation.csv, test.csv) and set SAMSUM_LOCAL_DIR, "
                        "or ensure internet access and valid HuggingFace endpoint/token."
                    ) from e

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



