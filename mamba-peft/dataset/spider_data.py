from pathlib import Path
import os
import requests
import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import snapshot_download

from dataset.collator import DataCollator
from metrics.spider.lib.evaluation import Evaluator, Schema, get_schema, get_sql
from metrics.spider.spider import SpiderMetric
from .base import NlgDatasetBase
import json
import numpy as np
import pandas as pd


class SpiderDataset(NlgDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", max_seqlen=1536, use_cache=True, hardness=None, has_test_split=False, **kwargs):
        path = "xlangai/spider"
        # self.path_table = "richardr1126/spider-schema"
        self.hf_dataset = None
        self.hardness = hardness
        self.has_test_split = has_test_split

        assert self.has_test_split
        assert max_seqlen is not None
        prompt_prefix = "Create a sql request for the following question and schema:\n"
        # For Text-to-SQL literature (e.g., PICARD/T5 baselines), inputs are separated by textual labels/newlines,
        # not by model-specific sep tokens. Use a plain newline-only prefix.
        super().__init__(tokenizer, path, split, prompt_prefix=prompt_prefix,  
                         use_cache=use_cache, max_seqlen=max_seqlen, **kwargs)
        self._snapshot_dir = None
        self._db_dir = None
    
    def _get_local_root(self) -> Path | None:
        env_dir = os.environ.get("SPIDER_LOCAL_DIR") or os.environ.get("HP_SPIDER_LOCAL_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists():
                return p
        return None
        
    def get_sql_hardness(self, sql, db):
        db_dir = self._db_dir or (Path("data") / self.path.replace("/", "_") / "spider" / "database")
        db = Path(db_dir) / db / (db + ".sqlite")
        schema = Schema(get_schema(str(db)))
        try:
            sql_proc = get_sql(schema, sql)
        except KeyError:
            print(f"Unknown hardness: {sql}")
            return "unknown"

        return Evaluator.eval_hardness(None, sql_proc)
    
    def get_cache_name(self):
        name = self.path.replace('/', '_')

        if self.has_test_split:
            name += "-tvt"

        name = f"cache_{name}_{self.split}_seqlen{self.max_seqlen}"

        if self.hardness is not None:
            name += "_" + "_".join(self.hardness)

        return name

    def __len__(self):
        l = len(self.data) if self.data is not None else len(self.get_hf_dataset()[0])
        return l

    def table_to_str(self, table_name, column_names, column_types, primary_keys, foreign_keys):
        return table_name + ": " + " ,  ".join([
            *[f"{n} {t}" for n, t in zip(column_names, column_types)],
            "foreign: " + " , ".join([f"{n} from {t}" for n, t in foreign_keys]),
            "primary: " + " , ".join(primary_keys)
        ]).lower()

    # https://github.com/AnMol12499/Text-to-SQL-using-T5-Model/blob/main/T5_finetuning%26inferencing%20.ipynb
    def get_schema(self, schema_data):
        tables_str = []

        for table_idx, table_name in enumerate(schema_data["table_names_original"]):
            column_names, column_types  = zip(*[(name, col_type) for (i, name), col_type in zip(
                schema_data["column_names_original"], schema_data["column_types"]) if i == table_idx])
            primary_keys = [schema_data["column_names_original"][k][1] for k in schema_data["primary_keys"] 
                            if schema_data["column_names_original"][k][0] == table_idx]
            foreign_keys = [(schema_data["column_names_original"][k2][1], schema_data["table_names_original"][schema_data["column_names_original"][k2][0]]) 
                            for k1, k2 in schema_data["foreign_keys"] if schema_data["column_names_original"][k1][0] == table_idx]

            tables_str.append(self.table_to_str(table_name, column_names, column_types, primary_keys, foreign_keys))

        out = " | ".join(tables_str)
        return out

    def load_table_dataset(self):
        # Prefer local official Spider if provided, else snapshot to data/
        root = self._get_local_root()
        if root is None:
            local_root = Path("data") / self.path.replace("/", "_")
            local_root.mkdir(parents=True, exist_ok=True)
            root = Path(snapshot_download(repo_id=self.path, repo_type="dataset", local_dir=str(local_root), local_dir_use_symlinks=False))
        self._snapshot_dir = root
        # Find tables.json anywhere
        cand = list(root.rglob("**/tables.json"))
        assert cand, f"tables.json not found under {root}"
        table_file = cand[0]
        # Infer db dir: prefer a sibling 'database' under the same tree
        db_dir = table_file.parent / "database"
        if not db_dir.exists():
            # search up to find a directory named 'database'
            db_cand = list(root.rglob("**/database"))
            if db_cand:
                db_dir = db_cand[0]
        self._db_dir = db_dir

        with open(table_file, "r") as f:
            dbs = json.load(f)

        out = {db["db_id"]: self.get_schema(db) for db in dbs}
        return out

    def load_hf_dataset_split(self):
        assert self.has_test_split
        # Prefer local official Spider if provided, else snapshot to data/
        root = self._get_local_root()
        if root is None:
            local_root = Path("data") / self.path.replace("/", "_")
            local_root.mkdir(parents=True, exist_ok=True)
            root = Path(snapshot_download(repo_id=self.path, repo_type="dataset", local_dir=str(local_root), local_dir_use_symlinks=False))

        def find_files(keywords, prefer_explicit=None):
            if prefer_explicit:
                cand_files = [root / name for name in prefer_explicit]
                cand_files = [p for p in cand_files if p.exists()]
                if cand_files:
                    builder = "json" if any(p.suffix in (".json", ".jsonl") for p in cand_files) else "parquet"
                    return builder, cand_files
            def _match(exts):
                out = []
                for hint in keywords:
                    for ext in exts:
                        out += list(root.rglob(f"**/*{hint}*.{ext}"))
                return out
            files_parquet = _match(["parquet"]) 
            files_jsonl  = _match(["jsonl"]) 
            files_json   = _match(["json"]) 
            if files_parquet: return "parquet", sorted(set(files_parquet))
            if files_jsonl:  return "json", sorted(set(files_jsonl))
            if files_json:   return "json", sorted(set(files_json))
            return None, []

        if self.split == "test":
            # Spider 没有公开 test 标签，沿用 validation/dev
            builder, files = find_files(["validation", "valid", "dev"], prefer_explicit=["dev.json", "dev_gold.sql"])
            assert files, f"spider dev/validation files not found under {root}"
            ds = load_dataset(builder, data_files={"train": [str(p) for p in files]})["train"]
            return ds
        else:
            # Prefer official JSONs if present (train_spider + train_others)
            builder, files = find_files(["train", "train_spider", "training"], prefer_explicit=["train_spider.json", "train_others.json"])
            assert files, f"spider train files not found under {root}"
            ds = load_dataset(builder, data_files={"train": [str(p) for p in files]})["train"]
            return ds.train_test_split(test_size=0.2, seed=self.shuffle_seeds[0])[{"train": "train", "val": "test"}[self.split]]

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            self.hf_dataset = [self.load_hf_dataset_split(), self.load_table_dataset()]

            if self.hardness is not None:
                self.hf_dataset[0] = self.hf_dataset[0].filter(
                    lambda example: self.get_sql_hardness(example["query"], example["db_id"]) in self.hardness)

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        question = self.hf_dataset[0]["question"][idx]
        db_id = self.hf_dataset[0]["db_id"][idx]
        query = self.hf_dataset[0]["query"][idx]

        table = self.hf_dataset[1][db_id]

        input = f"Question: {question}\\Schema: {table}\n"
        label = query.lower().strip()
        
        return input, label
    
    def preproc(self, idx):
        inputs_labels = super().preproc(idx)

        if inputs_labels is None:
            return None

        return inputs_labels, {"db_id": self.hf_dataset[0]["db_id"][idx]}
    
    def get_ids(self, idx):
        return self.data[idx][0]

    def get_db_id(self, idx):
        return self.data[idx][1]["db_id"]
    
    def compute_metrics(self, eval_preds, eval_mask=None):
        if self.mode == "gen":
            metric = SpiderMetric()

            db_ids = [self.get_db_id(i) for i in range(len(self))]
            assert len(db_ids) == len(eval_preds.preds)
            assert len(db_ids) == len(eval_preds.labels)

            predictions = list(zip(eval_preds.preds, db_ids))
            references = list(zip(eval_preds.labels, db_ids))

            if eval_mask is not None:
                predictions = [predictions[i] for i in eval_mask]
                references = [references[i] for i in eval_mask]

            metrics = metric.compute(predictions, references)

            # important metric first
            return {
                "all/exec": None,
                **metrics
            }
        else:
            return {}


class SpiderDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = SpiderDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)



