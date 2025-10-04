import transformers
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

from dataset.collator import DataCollator
from .base import NluDatasetBase
import evaluate
import numpy as np
import os


# https://github.com/microsoft/promptbench
prompts = {
    'rte': 
        "Determine if the given pair of sentences displays entailment or not_entailment. Respond with '0' if entailment or '1' if not_entailment: ",
    'cola': 
        "Review the sentence below and identify whether its grammar is Unacceptable or Acceptable. Respond with '0' if Unacceptable or '1' if Acceptable: ",
    'mrpc': 
        "Can the given sentences be considered semantically identical? Respond with '0' if not_equivalent or '1' if equivalent: ",
    'sst2': 
        "Read the provided excerpt and choose between negative and positive to describe its sentiment. Respond with '0' if negative or '1' if positive: ",
    'qnli': 
        "Consider the context and question, and indicate if the answer can be logically deduced from the context. Respond with '0' if entailment or '1' if not_entailment: ",
    'qqp': 
        "Can these two statements be considered equal in meaning? Respond with '0' if not_equivalent or '1' if equivalent: ",
    'mnli': 
        "Assess the connection between the following sentences and classify it as entailment, neutral, or contradiction. Respond with '0' if entailment, '1' if neutral or '2' if contradiction: ",
    'wnli': 
        "Identify whether the given pair of sentences demonstrates entailment or not_entailment. Respond with '0' if not_entailment or '1' if entailment: ",
}


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "wnli": ("sentence1", "sentence2"),
}

num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "wnli": 2,
}


class GlueDataset(NluDatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, name, split="train", use_cache=True, eval_all_logits=True, has_test_split=False, **kwargs):
        # Allow overriding dataset id via env; default mirrors HF hub id
        path = os.environ.get("GLUE_DATASET_ID", "nyu-mll/glue")
        self.name = name
        self.hf_dataset = None
        # Try to load GLUE metric from evaluate; fall back to local dir or built-in metrics
        self.metric = None
        try:
            self.metric = evaluate.load("glue", name)
        except Exception:
            # Optional local directory for offline metric scripts
            local_metric_dir = (
                os.environ.get("GLUE_METRIC_DIR")
                or os.environ.get("HF_EVALUATE_LOCAL_GLUE_DIR")
                or os.path.join(os.environ.get("HF_EVALUATE_CACHE", os.environ.get("HF_HOME", "")), "eval_metrics", "glue")
            )
            try:
                if local_metric_dir and os.path.isdir(local_metric_dir):
                    self.metric = evaluate.load(local_metric_dir, name)
            except Exception:
                self.metric = None
        self.eval_all_logits = eval_all_logits
        self.has_test_split = has_test_split

        assert self.has_test_split

        sep_tok = tokenizer.sep_token or getattr(tokenizer, "eos_token", None) or ""
        super().__init__(tokenizer, path, split, prompt_prefix=prompts[name] + sep_tok,  
                         use_cache=use_cache, **kwargs)

        emb_size = 50280
        self.id_to_int_label = np.full(emb_size, self.ignore_index, dtype=int)

        self.choice_ids = []
        for i in range(num_labels[name]):
            self.id_to_int_label[tokenizer.vocab[str(i)]] = i
            self.choice_ids.append(tokenizer.vocab[str(i)])

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.get_hf_dataset())
    
    def get_cache_name(self):
        name = self.name

        if self.has_test_split:
            name += "-tvt"
        
        return f"cache_{name}_{self.split}"

    def _load_dataset(self, split):
        if self.name == "mnli" and split == "val":
            hf_dataset = concatenate_datasets([
                load_dataset(self.path, "mnli_matched")["validation"],
                load_dataset(self.path, "mnli_mismatched")["validation"],
            ])
        else:
            target_split = {"train": "train", "val": "validation", "test": "test"}[split]
            try:
                hf_dataset = load_dataset(self.path, self.name)[target_split]
            except Exception:
                # Offline fallback: try local or alternative ids/paths
                fallback_candidates = [
                    os.environ.get("GLUE_DATASET_PATH"),                 # explicit local dir
                    "glue",                                              # official dataset id
                    "/home/user/mzs_h/data/nyu-mll_glue",               # user's provided local data root
                ]
                last_err = None
                for cand in fallback_candidates:
                    if not cand:
                        continue
                    try:
                        hf_dataset = load_dataset(cand, self.name)[target_split]
                        break
                    except Exception as e:
                        last_err = e
                        hf_dataset = None
                if hf_dataset is None:
                    raise last_err if last_err is not None else RuntimeError("Failed to load GLUE dataset (offline and no local fallback)")

        return hf_dataset

    def load_hf_dataset_split(self):
        assert self.has_test_split

        if self.split == "test":
            return self._load_dataset("val")
        else:
            # prefix, split, *seed_id = self.split.split("-")
            # assert prefix == "train"
            # assert len(seed_id) == 0
            data = self._load_dataset("train")
            return data.train_test_split(test_size=0.2, seed=self.shuffle_seeds[0])[{"train": "train", "val": "test"}[self.split]]

    def get_hf_dataset(self):
        if self.hf_dataset is None:
            self.hf_dataset = self.load_hf_dataset_split()

        return self.hf_dataset

    def get_input_label(self, idx):
        self.get_hf_dataset()

        key1, key2 = task_to_keys[self.name]

        input = self.hf_dataset[key1][idx]

        if key2 is not None:
            sep = getattr(self.tokenizer, "sep_token", None) or getattr(self.tokenizer, "eos_token", None) or ""
            input = input + sep + self.hf_dataset[key2][idx]

        label = self.hf_dataset["label"][idx]

        return input, label
    
    def compute_metrics(self, eval_preds):
        if self.eval_all_logits:
            references = np.concatenate(eval_preds.label_ids)
            predictions = np.concatenate(eval_preds.predictions).argmax(-1)

            references_int = self.id_to_int_label[references]
            predictions_int = self.id_to_int_label[predictions]

            valid = predictions_int != self.ignore_index
            references_int_valid = references_int[valid]
            predictions_int_valid = predictions_int[valid]
            if self.metric is not None:
                res = self.metric.compute(predictions=predictions_int_valid, references=references_int_valid)
            else:
                res = _compute_glue_metrics_local(self.name, predictions_int_valid, references_int_valid)

            res = {**res, "out_of_cls": np.sum(~valid)}
        else:
            references = np.concatenate(eval_preds.label_ids)
            predictions = np.concatenate(eval_preds.predictions)  # .argmax(-1)

            references_ind = np.array([self.choice_ids.index(r) for r in references])
            predictions_ind = predictions[:, self.choice_ids].argmax(1)
            if self.metric is not None:
                res = self.metric.compute(predictions=predictions_ind, references=references_ind)
            else:
                res = _compute_glue_metrics_local(self.name, predictions_ind, references_ind)

        return res


class GlueDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = GlueDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)


def _compute_glue_metrics_local(task_name: str, preds: np.ndarray, refs: np.ndarray):
    """Offline-safe minimal GLUE metrics.
    - cola: matthews_correlation
    - mrpc/qqp: accuracy and f1
    - others: accuracy
    """
    def _safe_div(a, b):
        return a / b if b != 0 else 0.0

    def _accuracy(p, r):
        return float(np.mean(p == r)) if len(p) else 0.0

    def _binary_stats(p, r):
        tp = int(np.sum((p == 1) & (r == 1)))
        tn = int(np.sum((p == 0) & (r == 0)))
        fp = int(np.sum((p == 1) & (r == 0)))
        fn = int(np.sum((p == 0) & (r == 1)))
        return tp, tn, fp, fn

    def _f1_binary(p, r):
        tp, tn, fp, fn = _binary_stats(p, r)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        return f1

    def _mcc_binary(p, r):
        tp, tn, fp, fn = _binary_stats(p, r)
        numerator = tp * tn - fp * fn
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if denom == 0:
            return 0.0
        return float(numerator / np.sqrt(denom))

    task = task_name.lower()
    if task == "cola":
        return {"matthews_correlation": _mcc_binary(preds, refs)}
    if task in {"mrpc", "qqp"}:
        return {"accuracy": _accuracy(preds, refs), "f1": _f1_binary(preds, refs)}
    # Default to accuracy for the rest (sst2, rte, qnli, wnli, mnli)
    return {"accuracy": _accuracy(preds, refs)}
