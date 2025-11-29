import transformers
import os
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from pathlib import Path
from huggingface_hub import hf_hub_download
import json
import nltk

from dataset.collator import DataCollator
from .base import NlgDatasetBase
import evaluate
import numpy as np
import pandas as pd


def _ensure_local_nltk_data():
    data_root = os.environ.get("NLTK_DATA")
    if not data_root:
        return
    for segment in data_root.split(os.pathsep):
        if segment and segment not in nltk.data.path:
            nltk.data.path.append(segment)


_ensure_local_nltk_data()


class DartDataset(NlgDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, split="train", use_cache=True, **kwargs):
        path = "GEM/dart"
        self.df = None
        self.input_formatter = None
        prompt_prefix = "Generate text for the following RDF triples:\n"
        self.sep_token = tokenizer.sep_token or getattr(tokenizer, "eos_token", "</s>")
        # prompt_prefix = None

        # 🚨 关键修改：尽早验证数据存在性，避免中途失败
        self._validate_data_availability()

        super().__init__(tokenizer, path, split, prompt_prefix=prompt_prefix,
                         use_cache=use_cache, **kwargs)

        assert not (self.mode == "lm" and split != "train")

    def _validate_data_availability(self):
        """🚨 尽早验证DART本地数据存在性，避免训练中途失败。

        这个方法在__init__的早期就被调用，确保在任何数据加载开始前
        就发现问题并提供清晰的解决指引。
        """
        # 1) 检查环境变量指定的自定义路径
        env_dir = os.environ.get("DART_LOCAL_DIR") or os.environ.get("HP_DART_LOCAL_DIR")
        if env_dir:
            data_root = Path(env_dir)
            if not data_root.exists():
                self._raise_data_missing_error(data_root, f"环境变量 DART_LOCAL_DIR/HP_DART_LOCAL_DIR 指定的路径不存在: {env_dir}")
            if not self._has_required_files(data_root):
                self._raise_data_missing_error(data_root, f"环境变量指定的路径中缺少必需的文件: {env_dir}")
            return  # 数据验证通过

        # 2) 检查默认路径
        default_root = Path("data") / self.path.replace("/", "_")
        if not default_root.exists():
            self._raise_data_missing_error(default_root, f"默认数据路径不存在: {default_root}")
        if not self._has_required_files(default_root):
            self._raise_data_missing_error(default_root, f"默认数据路径中缺少必需的文件: {default_root}")

        # 数据验证通过
        print(f"[DART] ✓ 本地数据验证通过: {default_root}")

    def _has_required_files(self, root: Path) -> bool:
        """检查数据目录是否包含必需的文件。"""
        required_files = [
            "train.json", "validation.json", "test.json",  # 主要文件
            "train.jsonl", "validation.jsonl", "test.jsonl",  # 备用格式
        ]
        # 至少要有训练和验证数据
        has_train = any((root / f).exists() for f in ["train.json", "train.jsonl", "train.parquet"])
        has_val = any((root / f).exists() for f in ["validation.json", "valid.json", "dev.json", "validation.jsonl", "dev.jsonl"])
        has_test = any((root / f).exists() for f in ["test.json", "test.jsonl"])

        return has_train and has_val  # 训练和验证是必需的，测试可选

    def _raise_data_missing_error(self, expected_path: Path, reason: str):
        """抛出清晰的数据缺失错误，包含详细的解决指引。"""
        error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🚨 DART 数据缺失错误                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 原因: {reason}
║
║ 📁 期望路径: {expected_path}
║
║ 🔧 解决方法（选择其一）:
║
║   方法1 - 使用 HuggingFace CLI（推荐，稳定可靠）:
║   cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft
║   huggingface-cli download --repo-type dataset GEM/dart --local-dir {expected_path}
║
║   方法2 - 使用 Python 脚本:
║   python -c "
║   from huggingface_hub import snapshot_download
║   snapshot_download(repo_id='GEM/dart', repo_type='dataset',
║                     local_dir='{expected_path}', local_dir_use_symlinks=False)
║   "
║
║   方法3 - 手动下载（如果网络受限）:
║   1. 访问: https://huggingface.co/datasets/GEM/dart
║   2. 下载文件: train.json, validation.json, test.json
║   3. 放置到: {expected_path}/
║
║   方法4 - 使用自定义路径:
║   export DART_LOCAL_DIR=/your/custom/path/to/dart/data
║   # 然后重新运行训练脚本
║
║ 📋 数据集信息:
║   - 名称: GEM/dart (Data-to-Text Generation with RDF triples)
║   - 大小: ~50MB (压缩后)
║   - 包含: 训练/验证/测试分割
║
║ ⚠️  注意事项:
║   - 数据下载只需执行一次
║   - 请确保有足够的磁盘空间
║   - 网络下载可能需要几分钟
║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        raise FileNotFoundError(error_msg)

    def get_cache_name(self):
        name = super().get_cache_name()
        name = name.replace("/", "_").replace(" ", "_")  # 避免再次出现 "GEM dart" 这种文件名
        if self.mode == "gen":
            name += "_gen"
        return name

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.load_df())

    def _snapshot_local_root(self) -> Path:
        """OFFLINE-ONLY: 强制本地读取。不会触发任何网络下载。
        
        优先顺序：
        1) 环境变量指定目录：DART_LOCAL_DIR / HP_DART_LOCAL_DIR
        2) 项目内本地目录：data/GEM_dart（或已有缓存文件）
        
        如本地不存在，将抛出异常并给出下载指引（代码中保留了下载示例，但已注释，避免误触发网络）。
        """
        # 1) explicit override
        env_dir = os.environ.get("DART_LOCAL_DIR") or os.environ.get("HP_DART_LOCAL_DIR")
        if env_dir and Path(env_dir).exists():
            return Path(env_dir)

        local_root = Path("data") / self.path.replace("/", "_")
        local_root.mkdir(parents=True, exist_ok=True)

        # 2) 仅当已存在本地文件时使用；否则报错，不再尝试任何下载
        offline = str(os.environ.get("HF_HUB_OFFLINE", "")).lower() in ("1", "true", "yes", "on")
        has_local_files = any((local_root / name).exists() for name in [
            "train.json", "validation.json", "dev.json", "test.json",
            "train.parquet", "validation.parquet", "test.parquet",
        ]) or any(local_root.rglob("*.json")) or any(local_root.rglob("*.parquet"))
        if offline or has_local_files:
            return local_root

        # 3) 不再进行 snapshot_download。明确报错 + 下载指引（注释保留下载示例，方便他人了解如何获取数据）
        msg = (
            "[DART][offline] Local dataset not found.\n"
            f"  Expected under: {local_root}\n"
            "  Or set env DART_LOCAL_DIR=/path/to/dart\n\n"
            "Please prepare DART locally before running. For example:\n"
            "  - Place files like train.json/validation.json/test.json (or parquet/jsonl) under the directory above.\n"
            "  - Or set DART_LOCAL_DIR to an existing folder with these files.\n\n"
            "Optional reference (commented out to avoid network):\n"
            "  from huggingface_hub import snapshot_download\n"
            "  snapshot_download(repo_id='GEM/dart', repo_type='dataset', local_dir='data/GEM_dart', local_dir_use_symlinks=False)\n"
        )
        raise FileNotFoundError(msg)

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
        """已禁用的下载分支（保留注释说明）。
        
        出于稳定性与可控性考虑，代码不再发起任何网络请求。若需要从 Hub 拉取，请手动执行下面的示例命令，
        并将文件放入本地数据目录（或通过 DART_LOCAL_DIR 指向该目录）。

        示例（请在命令行中执行，非代码自动执行）：
          python -c "from huggingface_hub import snapshot_download; \
                     snapshot_download(repo_id='GEM/dart', repo_type='dataset', \
                     local_dir='data/GEM_dart', local_dir_use_symlinks=False)"

        返回 (None, []) 以显式表示不提供在线下载候选。
        """
        return None, []

    def load_hf_dataset_split(self):
        snap_dir = self._snapshot_local_root()
        # decide which split to load from files. For train-*, still load full train then split
        if self.split.startswith("train-"):
            builder, files = self._find_split_files(snap_dir, "train")
            # 不再尝试任何下载；若本地缺失，直接报错并给出指引
            assert files, (
                f"GEM/dart train files not found under {snap_dir}\n"
                "Please prepare local files (e.g., train.json/train.parquet) first.\n"
                "You can set DART_LOCAL_DIR to point to your local dataset folder."
            )
            assert builder is not None, f"Unable to determine dataset builder for files: {files}"
            ds = load_dataset(builder, data_files={"train": [str(p) for p in files]})["train"]
            prefix, split, *seed_id = self.split.split("-")
            assert prefix == "train" and len(seed_id) == 0
            ds = ds.train_test_split(test_size=0.2, seed=self.shuffle_seeds[0])[{"train": "train", "val": "test"}[split]]
            if len(ds) == 0:
                raise AssertionError(f"GEM/dart split '{self.split}' resolved to 0 samples. Please verify train files under {snap_dir}.")
            return ds
        else:
            want = {"train": "train", "val": "val", "test": "test"}[self.split]
            builder, files = self._find_split_files(snap_dir, want)
            if not files:
                # some repos may only provide validation/dev; map val->dev and test->validation as fallback
                fallback = "val" if want == "test" else want
                builder, files = self._find_split_files(snap_dir, fallback)
            # 不再尝试任何下载；若本地缺失，直接报错并给出指引
            assert files, (
                f"GEM/dart {want} files not found under {snap_dir}\n"
                "Please prepare local files (e.g., validation.json/test.json or parquet/jsonl).\n"
                "You can set DART_LOCAL_DIR to point to your local dataset folder."
            )
            assert builder is not None, f"Unable to determine dataset builder for files: {files}"
            ds = load_dataset(builder, data_files={"train": [str(p) for p in files]})["train"]
            if len(ds) == 0:
                # Try alternate splits present locally to avoid silent empty datasets
                alt_order = ("val", "train", "test") if want == "train" else ("train", "val", "test")
                for alt in alt_order:
                    b2, f2 = self._find_split_files(snap_dir, alt)
                    if f2:
                        assert b2 is not None, f"Unable to determine dataset builder for files: {f2}"
                        ds_alt = load_dataset(b2, data_files={"train": [str(p) for p in f2]})["train"]
                        if len(ds_alt) > 0:
                            print(f"[DART] Warning: split '{want}' empty. Falling back to '{alt}' ({len(ds_alt)} samples).")
                            ds = ds_alt
                            break
                if len(ds) == 0:
                    raise AssertionError(f"GEM/dart split '{want}' resolved to 0 samples. Please verify files under {snap_dir}.")
            return ds

    def load_df(self):
        if self.df is None:
            # load via file-based builder
            data = self.load_hf_dataset_split()
            df = data.to_pandas()

            # Build source/text lists robustly from various schemas
            def build_lists(row):
                # Prefer standard annotations
                if "annotations" in row and row["annotations"] is not None:
                    ann = row["annotations"]
                    # Handle both list and numpy.ndarray (pandas may convert lists to arrays)
                    if isinstance(ann, (list, np.ndarray)):
                        texts = []
                        sources = []
                        for a in ann:
                            if isinstance(a, dict):
                                t = a.get("text") or a.get("target") or a.get("reference")
                                s = a.get("source", "")
                                if isinstance(t, str) and t.strip():
                                    texts.append(t)
                                    sources.append(s)
                            elif isinstance(a, str):
                                texts.append(a)
                                sources.append("")
                        return sources, texts
                    if isinstance(ann, dict):
                        # dict-of-lists or dict-of-str
                        texts = ann.get("text") or ann.get("target") or ann.get("targets") or []
                        sources = ann.get("source") or [""] * (len(texts) if isinstance(texts, list) else 1)
                        if isinstance(texts, str):
                            texts = [texts]
                        if isinstance(sources, str):
                            sources = [sources]
                        if isinstance(texts, list) and not isinstance(sources, list):
                            sources = [""] * len(texts)
                        return sources, texts

                # Alternative fields when annotations missing
                texts = None
                if "references" in row and isinstance(row["references"], list):
                    cand = []
                    for r in row["references"]:
                        if isinstance(r, dict) and "text" in r:
                            cand.append(r["text"])
                        elif isinstance(r, str):
                            cand.append(r)
                    texts = cand
                # WebNLG-style keys occasionally present in merged corpora
                if texts is None and isinstance(row.get("verbalizations"), list):
                    cand = []
                    for r in row["verbalizations"]:
                        if isinstance(r, dict) and isinstance(r.get("text"), str):
                            cand.append(r["text"])
                        elif isinstance(r, str):
                            cand.append(r)
                    texts = cand
                if texts is None and isinstance(row.get("lexicalizations"), list):
                    cand = []
                    for r in row["lexicalizations"]:
                        if isinstance(r, dict) and isinstance(r.get("text"), str):
                            cand.append(r["text"])
                        elif isinstance(r, str):
                            cand.append(r)
                    texts = cand
                if texts is None and isinstance(row.get("targets"), list):
                    texts = [t for t in row["targets"] if isinstance(t, str)]
                if texts is None and isinstance(row.get("target"), str):
                    texts = [row["target"]]
                if texts is None and isinstance(row.get("text"), str):
                    texts = [row["text"]]
                if texts is None and isinstance(row.get("output"), str):
                    texts = [row["output"]]
                if texts is None and isinstance(row.get("outputs"), list):
                    texts = [t for t in row["outputs"] if isinstance(t, str)]
                if texts is None:
                    texts = []
                sources = [""] * len(texts)
                return sources, texts

            # Apply normalizer row-wise
            built = df.apply(build_lists, axis=1, result_type="reduce")
            # built is a Series of tuples (sources, texts)
            sources_col = built.apply(lambda x: x[0])
            texts_col = built.apply(lambda x: x[1])
            out = pd.DataFrame({
                "tripleset": df["tripleset"] if "tripleset" in df.columns else [[] for _ in range(len(df))],
                "source": sources_col,
                "text": texts_col,
            })

            # Ensure list[str] for both columns (hardened)
            def to_str_list(x):
                # Handle numpy arrays first
                if isinstance(x, np.ndarray):
                    x = x.tolist()
                
                if isinstance(x, list):
                    out = []
                    for e in x:
                        # Recursively handle nested structures
                        if isinstance(e, (list, np.ndarray)):
                            # Flatten one level
                            for sub_e in (e.tolist() if isinstance(e, np.ndarray) else e):
                                if isinstance(sub_e, (str, int, float)) or sub_e is None:
                                    s = "" if sub_e is None else str(sub_e)
                                    if s.strip() != "":
                                        out.append(s)
                        elif isinstance(e, (str, int, float)) or e is None:
                            s = "" if e is None else str(e)
                            if s.strip() != "":
                                out.append(s)
                    return out
                if isinstance(x, (str, int, float)) or x is None:
                    s = "" if x is None else str(x)
                    return [s] if s.strip() != "" else []
                return []
            out["source"] = out.get("source", pd.Series([[]] * len(out))).apply(to_str_list)
            out["text"]   = out.get("text",   pd.Series([[]] * len(out))).apply(to_str_list)

            # 强制保证两列都存在（即使上面全空也要有空列表）
            if "source" not in out.columns:
                out["source"] = [[]] * len(out)
            if "text" not in out.columns:
                out["text"] = [[]] * len(out)

            # Drop records without any reference text
            out = out[out["text"].apply(lambda lst: isinstance(lst, list) and len(lst) > 0)].reset_index(drop=True)
            if len(out) == 0:
                # Final fallback: if nothing matched, try to harvest any string-like field as text
                text_like_cols = [c for c in df.columns if c.lower() in ("reference", "references", "target", "targets", "text", "output", "outputs")]
                rows_fallback = []
                for _, r in df.iterrows():
                    texts_fb = []
                    for c in text_like_cols:
                        v = r.get(c)
                        if isinstance(v, str) and v.strip():
                            texts_fb.append(v)
                        elif isinstance(v, list):
                            texts_fb += [t for t in v if isinstance(t, str) and t.strip()]
                    if texts_fb:
                        rows_fallback.append({
                            "tripleset": r["tripleset"] if "tripleset" in df.columns else [],
                            "source": [""] * len(texts_fb),
                            "text": texts_fb,
                        })
                if rows_fallback:
                    out = pd.DataFrame(rows_fallback)
            df = out
            
            # 最终确保 source 和 text 列是正确的列表格式（不是 numpy 数组）
            # 这对 mode="gen" 很关键，因为后续会直接使用这些列表
            def ensure_list(x):
                if isinstance(x, np.ndarray):
                    return x.tolist()
                elif isinstance(x, list):
                    return x
                else:
                    return [x] if x else []
            
            df["source"] = df["source"].apply(ensure_list)
            df["text"] = df["text"].apply(ensure_list)

            if self.mode == "lm":
                # 手动展开：把每个样本的多参考拆成多行
                rows = []
                for idx, row in df.iterrows():
                    tripleset = row["tripleset"]
                    sources = row["source"] if isinstance(row["source"], list) else [row["source"]]
                    texts = row["text"] if isinstance(row["text"], list) else [row["text"]]
                    # 确保 sources 和 texts 长度一致
                    max_len = max(len(sources), len(texts))
                    sources = sources + [""] * (max_len - len(sources))
                    texts = texts + [""] * (max_len - len(texts))
                    for s, t in zip(sources, texts):
                        if isinstance(t, str) and t.strip():
                            rows.append({"tripleset": tripleset, "source": str(s) if s else "", "text": str(t)})
                df = pd.DataFrame(rows)
                # 最终确保列存在且为字符串
                if len(df) == 0:
                    df = pd.DataFrame(columns=["tripleset", "source", "text"])
                df["source"] = df["source"].astype(str)
                df["text"] = df["text"].astype(str)
            
            self.df = df

        return self.df

    def linearize_triples(self, triples):
        def as_str(x):
            s = "" if x is None else str(x)
            return s.replace("\n", " ").strip()

        # Handle numpy.ndarray (pandas may convert lists to arrays)
        if triples is None or (isinstance(triples, (list, np.ndarray)) and len(triples) == 0):
            triples = []
        return " | ".join([" : ".join(as_str(ti) for ti in t) for t in triples])

    # https://github.com/microsoft/AdaMix/blob/d361e9d6a24cb44d6d6169337128a0cf6feb6e1d/NLG/src/format_converting_webnlg.py
    def get_input_label(self, idx):
        self.load_df()

        triples = self.df.iloc[idx]["tripleset"]
        source = self.df.iloc[idx]["source"]
        text = self.df.iloc[idx]["text"]

        input = self.linearize_triples(triples)
        
        if self.mode == "lm":
            # Defensive fallback: guarantee scalar strings
            if isinstance(text, list):
                text = next((t for t in text if isinstance(t, str) and t.strip()), "")
            if isinstance(source, list):
                source = next((s for s in source if isinstance(s, str)), "")
            assert isinstance(source, str) and isinstance(text, str)
            label = text
        else:
            # need to handle multiple references (generation mode)
            # Ensure source and text are lists (not numpy arrays)
            if isinstance(source, np.ndarray):
                source = source.tolist()
            if isinstance(text, np.ndarray):
                text = text.tolist()
            
            # Ensure they are lists
            if not isinstance(source, list):
                source = [source] if source else []
            if not isinstance(text, list):
                text = [text] if text else []
            
            # Flatten nested lists (defensive)
            def flatten_once(lst):
                result = []
                for item in lst:
                    if isinstance(item, (list, np.ndarray)):
                        result.extend(item.tolist() if isinstance(item, np.ndarray) else item)
                    else:
                        result.append(item)
                return result
            
            text = flatten_once(text)
            source = flatten_once(source)
            
            # Filter out any non-string elements
            text = [str(t).strip() for t in text if t is not None and str(t).strip()]
            
            if len(text) == 0:
                # Don't raise, return None so preproc filters it out
                print(f"[DART] Warning: Sample {idx} has no valid text after filtering, skipping")
                return None, None
            
            # Check for sep_token collision
            if any(self.sep_token in t for t in text):
                print(f"[DART] Warning: Sample {idx} contains sep_token '{self.sep_token}', replacing with space")
                text = [t.replace(self.sep_token, " ") for t in text]
            
            label = self.sep_token.join(text)

        return input, label
    
    def compute_metrics(self, eval_preds):
        if self.mode == "gen":
            predictions = getattr(eval_preds, "preds", [])
            references = getattr(eval_preds, "labels", [])
            # Defensive: skip metric computation if no samples
            if not predictions or not references:
                return {}
            # Basic normalization: trim whitespace; split labels into multi-refs by sep_token
            predictions = [p.strip() if isinstance(p, str) else "" for p in predictions]
            references = [
                [r.strip() for r in (rs.split(self.sep_token) if isinstance(rs, str) else []) if r.strip()]
                for rs in references
            ]

            # Metrics commonly reported for GEM/DART: BLEU, METEOR, chrF
            bleu = evaluate.load("bleu")
            meteor = evaluate.load("meteor")
            chrf = evaluate.load("chrf")

            bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
            meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
            chrf_score = chrf.compute(predictions=predictions, references=references)["score"]

            results = {
                "bleu": bleu_score,
                "meteor": meteor_score,
                "chrf": chrf_score,
            }

            # Save concise local eval logs for debugging (cloud runs only), mirroring Spider
            self._save_local_eval_log(eval_preds, predictions, references, results)
        else:
            results = {}

        return results

    def _save_local_eval_log(self, eval_preds, predictions, references, metrics):
        """Save readable local log with pred/ref comparisons for debugging (cloud mode only).
        We log only 'low-overlap' samples to keep files small.
        """
        import os
        import datetime
        from difflib import SequenceMatcher

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
        group_tag = f"{suite}_r{round_num}_s{seed}_{str(data).replace('-', '_')}"

        # Generate filename with group info, timestamp and step info
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        step = getattr(eval_preds, 'step', 'unknown')
        filename = f"{output_dir}/eval_log_dart_{group_tag}_{timestamp}_step{step}.txt"

        # Heuristic: mark samples as 'low-overlap' if best Jaccard over refs is below threshold
        def jaccard(a: str, b: str) -> float:
            sa = set(a.split())
            sb = set(b.split())
            if not sa and not sb:
                return 1.0
            if not sa or not sb:
                return 0.0
            inter = len(sa & sb)
            union = len(sa | sb)
            return inter / union if union > 0 else 0.0

        low_overlap_threshold = 0.20  # conservative
        max_records = 200             # cap the number of logged examples
        records = []

        for i, (pred, refs) in enumerate(zip(predictions, references)):
            try:
                best_j = max((jaccard(pred, r) for r in refs), default=0.0)
                if best_j < low_overlap_threshold:
                    # Also compute a quick similarity ratio as auxiliary info
                    best_ratio = max((SequenceMatcher(None, pred, r).ratio() for r in refs), default=0.0)
                    records.append((i, best_j, best_ratio, pred, refs))
            except Exception:
                continue
            if len(records) >= max_records:
                break

        with open(filename, 'w', encoding='utf-8') as f:
            # Write summary metrics
            f.write("=== EVALUATION SUMMARY (DART) ===\n")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")

            # Write selected 'low-overlap' examples
            f.write(f"=== LOW-OVERLAP EXAMPLES (≤{low_overlap_threshold:.2f} Jaccard) ===\n")
            for i, best_j, best_ratio, pred, refs in records:
                f.write(f"[{i}] jaccard={best_j:.3f} ratio={best_ratio:.3f}\n")
                f.write(f"pred: {pred}\n")
                for ridx, r in enumerate(refs[:5]):  # limit references shown per example
                    f.write(f"ref{ridx+1}: {r}\n")
                f.write("\n")
            f.write(f"logged_examples: {len(records)}\n")

        print(f"[DartEval] Saved detailed eval log to: {filename} (group: {group_tag})")


class DartDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = DartDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)



