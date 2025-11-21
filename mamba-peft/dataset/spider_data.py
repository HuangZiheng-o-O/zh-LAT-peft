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
from typing import List, Dict


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
        # Priority:
        # 1) SPIDER_LOCAL_DIR / HP_SPIDER_LOCAL_DIR (user override)
        # 2) repo-local conventional paths under the project
        # 3) None (caller will use HF snapshot)
        candidates = []
        for key in ("SPIDER_LOCAL_DIR", "HP_SPIDER_LOCAL_DIR"):
            v = os.environ.get(key)
            if v:
                candidates.append(Path(v))
        # Project-local fallbacks
        try:
            repo_root = Path(__file__).resolve().parents[2]  # .../zh-LAT-peft
            candidates += [
                repo_root / "mamba-peft" / "data" / "spider_data",
                repo_root / "spider_data",
            ]
        except Exception:
            pass
        # CWD-relative fallbacks
        try:
            cwd = Path.cwd()
            candidates += [
                cwd / "mamba-peft" / "data" / "spider_data",
                cwd / "spider_data",
            ]
        except Exception:
            pass
        for p in candidates:
            try:
                if p.exists() and (p / "train_spider.json").exists():
                    return p
            except Exception:
                continue
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

    def _examples_from_json_files(self, root: Path, filenames: List[str]) -> List[Dict[str, str]]:
        """Read only necessary fields (question, db_id, query) to avoid Arrow schema issues in nested 'sql'."""
        records: List[Dict[str, str]] = []
        for name in filenames:
            p = root / name
            if not p.exists():
                continue
            with open(p, "r") as f:
                try:
                    arr = json.load(f)
                except Exception:
                    continue
            for ex in arr:
                records.append({
                    "question": ex.get("question", ""),
                    "db_id": ex.get("db_id", ""),
                    "query": ex.get("query", ""),
                })
        return records

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
        from datasets import Dataset
        if self.split == "test":
            # Use dev.json as evaluation split (Spider doesn't have public test)
            recs = self._examples_from_json_files(root, ["dev.json"])
            assert recs, f"spider dev.json not found or empty under {root}"
            return Dataset.from_list(recs)
        else:
            # Train/Val from official train JSONs
            recs = self._examples_from_json_files(root, ["train_spider.json", "train_others.json"])
            if not recs:
                # Fallback to any train*.json present
                cand = sorted(list(root.rglob("**/train*.json")))
                names = [str(p.relative_to(root)) for p in cand]
                recs = self._examples_from_json_files(root, names)
            assert recs, f"spider train files not found under {root}"
            ds = Dataset.from_list(recs)
            return ds.train_test_split(test_size=0.2, seed=self.shuffle_seeds[0])[{"train": "train", "val": "test"}[self.split]]

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            self.hf_dataset = [self.load_hf_dataset_split(), self.load_table_dataset()]

            if self.hardness is not None:
                self.hf_dataset[0] = self.hf_dataset[0].filter(
                    lambda example: self.get_sql_hardness(example["query"], example["db_id"]) in self.hardness)

        return self.hf_dataset

    def get_input_label(self, idx):
        hf_ds, _ = self.get_hf_dataset()

        question = hf_ds["question"][idx]
        db_id = hf_ds["db_id"][idx]
        query = hf_ds["query"][idx]

        table = self.hf_dataset[1][db_id]

        input = f"Question: {question}\nSchema: {table}\n"
        label = query.lower().strip()
        
        return input, label
    
    def preproc(self, idx):
        """
        Build one training/eval example.
        We:
        - Use the base class to create (input_ids, label_ids)
        - Attach metadata with db_id and canonical SQL query (lower+strip) so that
          generation metrics can safely use the ground-truth SQL without relying
          on decoded labels or fragile index assumptions.
        """
        inputs_labels = super().preproc(idx)

        if inputs_labels is None:
            return None

        hf_ds, _ = self.get_hf_dataset()
        meta = {
            "db_id": hf_ds["db_id"][idx],
            "query": str(hf_ds["query"][idx]).lower().strip(),
        }
        return inputs_labels, meta
    
    def get_ids(self, idx):
        return self.data[idx][0]

    def get_db_id(self, idx):
        return self.data[idx][1]["db_id"]
    
    def compute_metrics(self, eval_preds, eval_mask=None):
        if self.mode == "gen":
            # Ensure dataset and HF view are initialized (handles lazy reload cases)
            if self.data is None:
                # Import from base without circular import
                from .base import DatasetBase  # type: ignore
                DatasetBase._ensure_materialized(self)  # type: ignore[attr-defined]

            metric = SpiderMetric()

            size = len(self)
            # Guard against legacy caches that don't carry 'query' in metadata
            sample_meta = self.data[0][1] if (self.data and len(self.data) > 0) else {}
            if "query" not in sample_meta:
                raise RuntimeError(
                    "SpiderDataset.compute_metrics expected per-sample metadata with 'query', "
                    "but current cache is missing it. Please clear the Spider cache directory "
                    "(e.g., data/xlangai_spider_*/cache_*.pkl) or set DATA_CACHE_TAG to a new "
                    "value and rerun so the dataset can be rebuilt."
                )

            # For each in-memory example, use the attached db_id + canonical SQL query.
            db_ids = [self.data[i][1]["db_id"] for i in range(size)]
            gt_queries = [self.data[i][1]["query"] for i in range(size)]

            assert len(db_ids) == len(eval_preds.preds) == len(gt_queries)

            predictions = list(zip(eval_preds.preds, db_ids))
            references = list(zip(gt_queries, db_ids))

            if eval_mask is not None:
                predictions = [predictions[i] for i in eval_mask]
                references = [references[i] for i in eval_mask]

            metrics = metric.compute(predictions, references)

            # Save readable local log for debugging (only in cloud mode)
            self._save_local_eval_log(eval_preds, predictions, references, metrics)

            # important metric first
            return {
                "all/exec": None,
                **metrics
            }
        else:
            return {}

    def _save_local_eval_log(self, eval_preds, predictions, references, metrics):
        """Save readable local log with pred/gold comparisons for debugging."""
        import os
        import datetime

        # Only save log if SwanLab is in cloud mode
        if os.environ.get("SWANLAB_MODE", "").lower() != "cloud":
            return

        # Create output directory for local logs (use my_swanlog/ to avoid cleanup)
        base_log_dir = "my_swanlog"
        output_dir = os.path.join(base_log_dir, "local_eval_logs")
        os.makedirs(output_dir, exist_ok=True)

        # Get experiment group info from environment
        suite = os.environ.get("SUITE", "unknown")
        round_num = os.environ.get("ROUND", "unknown")
        seed = os.environ.get("HP_SEED", "unknown")
        data = os.environ.get("DATA", "unknown")

        # Create group identifier
        group_tag = f"{suite}_r{round_num}_s{seed}_{data.replace('-', '_')}"

        # Generate filename with group info, timestamp and step info
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        step = getattr(eval_preds, 'step', 'unknown')
        filename = f"{output_dir}/eval_log_{group_tag}_{timestamp}_step{step}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            # Write summary metrics
            f.write("=== EVALUATION SUMMARY ===\n")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")

            # Write detailed pred/gold comparisons
            f.write("=== PRED/GOLD COMPARISONS ===\n")
            eval_err_num = 0

            for i, ((pred_sql, db_id), (gold_sql, gold_db_id)) in enumerate(zip(predictions, references)):
                # Check if prediction is incorrect (exact match == 0)
                try:
                    from metrics.spider.evaluation import get_sql, Schema, get_schema
                    db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
                    schema = Schema(get_schema(db_path))
                    pred_parsed = get_sql(schema, pred_sql)
                    gold_parsed = get_sql(schema, gold_sql)

                    from metrics.spider.lib.evaluation import Evaluator
                    evaluator = Evaluator()
                    exact_score = evaluator.eval_exact_match(pred_parsed, gold_parsed)

                    if exact_score == 0:  # Incorrect prediction
                        eval_err_num += 1
                        # Determine hardness level
                        hardness = evaluator.eval_hardness(gold_parsed)
                        hardness = hardness if hardness in ['easy', 'medium', 'hard', 'extra'] else 'unknown'

                        f.write(f"{hardness} pred: {pred_sql}\n")
                        f.write(f"{hardness} gold: {gold_sql}\n")
                        f.write("\n")

                except Exception as e:
                    # If parsing fails, still log the raw strings
                    f.write(f"parse_error pred: {pred_sql}\n")
                    f.write(f"parse_error gold: {gold_sql}\n")
                    f.write(f"parse_error: {str(e)}\n")
                    f.write("\n")

            f.write(f"eval_err_num:{eval_err_num}\n")

        print(f"[SpiderEval] Saved detailed eval log to: {filename} (group: {group_tag})")


class SpiderDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = SpiderDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)



