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


class DartDataset(NlgDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "GEM/dart"
        self.df = None
        self.input_formatter = None
        prompt_prefix = "Generate text for the following RDF triples:\n"
        self.sep_token = tokenizer.sep_token
        # prompt_prefix = None

        super().__init__(tokenizer, path, split, prompt_prefix=prompt_prefix,
                         use_cache=use_cache, **kwargs)
        
        assert not (self.mode == "lm" and split != "train")

    def get_cache_name(self):
        name = super().get_cache_name()
        if self.mode == "gen":
            name = name + "_gen"
        return name

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.load_df())

    def _snapshot_local_root(self) -> Path:
        """Download GEM/dart snapshot to a deterministic local directory and return the snapshot dir."""
        local_root = Path("data") / self.path.replace("/", "_")
        local_root.mkdir(parents=True, exist_ok=True)
        snap = snapshot_download(repo_id=self.path, repo_type="dataset", local_dir=str(local_root), local_dir_use_symlinks=False)
        return Path(snap)

    def _find_split_files(self, snap_dir: Path, split_key: str):
        # Map our split to filename hints
        key = {"train": ["train", "training"], "val": ["validation", "valid", "dev", "val"], "test": ["test"]}[split_key]
        # Prefer parquet > jsonl > json
        def _match(exts):
            out = []
            for hint in key:
                for ext in exts:
                    out += list(snap_dir.rglob(f"**/*{hint}*.{ext}"))
            return out
        files_parquet = _match(["parquet"]) 
        files_jsonl  = _match(["jsonl"]) 
        files_json   = _match(["json"]) 
        if files_parquet:
            return "parquet", sorted(set(files_parquet))
        if files_jsonl:
            return "json", sorted(set(files_jsonl))
        if files_json:
            return "json", sorted(set(files_json))
        return None, []

    def _download_candidates(self, split_key: str, dest_dir: Path):
        """Attempt to fetch known filename patterns directly from the repo when snapshot doesn't expose split files plainly.
        Returns (builder, files).
        """
        # Common DART file names observed in GEM releases
        name_map = {
            "train": [
                "train.json", "train.jsonl", "train.parquet",
                "train-v1.1.json", "train-v1.1.jsonl",
                "data/train.json", "data/train.jsonl",
            ],
            "val": [
                "validation.json", "validation.jsonl", "valid.json", "dev.json",
                "dev-v1.1.json", "validation-v1.1.json",
                "data/dev.json", "data/validation.json",
            ],
            "test": [
                "test.json", "test.jsonl", "test-v1.1.json",
                "data/test.json",
            ],
        }
        builder = None
        files = []
        dest_dir.mkdir(parents=True, exist_ok=True)
        for fname in name_map[split_key]:
            try:
                local = hf_hub_download(repo_id=self.path, repo_type="dataset", filename=fname)
                p = dest_dir / Path(fname).name
                # copy file to a flat location
                if not p.exists():
                    Path(local).rename(p)
                files.append(p)
                if p.suffix == ".parquet":
                    builder = builder or "parquet"
                elif p.suffix in (".jsonl", ".json"):
                    builder = builder or "json"
            except Exception:
                continue
        return builder, files

    def load_hf_dataset_split(self):
        snap_dir = self._snapshot_local_root()
        # decide which split to load from files. For train-*, still load full train then split
        if self.split.startswith("train-"):
            builder, files = self._find_split_files(snap_dir, "train")
            if not files:
                builder, files = self._download_candidates("train", snap_dir / "_files")
            assert files, f"GEM/dart train files not found under {snap_dir}"
            ds = load_dataset(builder, data_files={"train": [str(p) for p in files]})["train"]
            prefix, split, *seed_id = self.split.split("-")
            assert prefix == "train" and len(seed_id) == 0
            return ds.train_test_split(test_size=0.2, seed=self.shuffle_seeds[0])[{"train": "train", "val": "test"}[split]]
        else:
            want = {"train": "train", "val": "val", "test": "test"}[self.split]
            builder, files = self._find_split_files(snap_dir, want)
            if not files:
                # some repos may only provide validation/dev; map val->dev and test->validation as fallback
                fallback = "val" if want == "test" else want
                builder, files = self._find_split_files(snap_dir, fallback)
            if not files:
                builder, files = self._download_candidates(want, snap_dir / "_files")
                if not files and want != "val":
                    # try fallback explicitly
                    builder, files = self._download_candidates("val" if want == "test" else want, snap_dir / "_files")
            assert files, f"GEM/dart {want} files not found under {snap_dir}"
            ds = load_dataset(builder, data_files={"train": [str(p) for p in files]})["train"]
            return ds

    def load_df(self):
        if self.df is None:
            # load via file-based builder
            data = self.load_hf_dataset_split()
            df = data.to_pandas()
            df = pd.concat([df["tripleset"], df['annotations'].apply(pd.Series)], axis=1)

            if self.mode == "lm":
                # make separate entries for multiple annotations during training
                df = df.explode(["source", "text"])
            
            self.df = df

        return self.df

    def linearize_triples(self, triples):
        return " | ".join([" : ".join(t) for t in triples])

    # https://github.com/microsoft/AdaMix/blob/d361e9d6a24cb44d6d6169337128a0cf6feb6e1d/NLG/src/format_converting_webnlg.py
    def get_input_label(self, idx):
        self.load_df()

        triples = self.df.iloc[idx]["tripleset"]
        source = self.df.iloc[idx]["source"]
        text = self.df.iloc[idx]["text"]

        input = self.linearize_triples(triples)
        
        if self.mode == "lm":
            assert isinstance(source, str) and isinstance(text, str)
            label = text
        else:
            # need to handle multiple references
            assert isinstance(source, list) and isinstance(text, list)
            assert not any(self.sep_token in t for t in text)
            label = self.tokenizer.sep_token.join(text)

        return input, label
    
    def compute_metrics(self, eval_preds):
        if self.mode == "gen":
            predictions = eval_preds.preds
            references = eval_preds.labels
            references = [r.split(self.sep_token) for r in references]  # split to get all refs

            meteor = evaluate.load("meteor")
            bleu = evaluate.load("bleu")

            meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
            bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]

            results = {
                "meteor": meteor_score,
                "bleu": bleu_score,
            }
        else:
            results = {}

        return results


class DartDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = DartDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)



