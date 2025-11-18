import numpy as np
import yaml
from dataclasses import dataclass
from typing import Any, List, Optional
from yaml import CSafeLoader


@dataclass
class EvalPredictionWithText:
    tokenizer: Any = None
    inputs: Optional[List[str]] = None
    preds: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    input_ids: Optional[List[np.ndarray]] = None
    pred_ids: Optional[List[np.ndarray]] = None
    label_ids: Optional[List[np.ndarray]] = None
    save_file: Optional[str] = None
    remove_eos: bool = False

    def __init__(self, tokenizer=None, input_ids=None, pred_ids=None, label_ids=None, save_file=None, remove_eos=False):
        self.tokenizer = tokenizer
        self.remove_eos = remove_eos
        if input_ids is not None:
            self.inputs = tokenizer.batch_decode(
                self._remove_pad_token_id(input_ids) if remove_eos else input_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        if pred_ids is not None:
            self.preds = tokenizer.batch_decode(
                self._remove_eos_token_id(pred_ids) if remove_eos else pred_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        if label_ids is not None:
            self.labels = tokenizer.batch_decode(
                self._remove_eos_token_id(label_ids) if remove_eos else label_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        self.input_ids = [t.cpu().numpy() for t in input_ids] if input_ids is not None else None
        self.pred_ids = [t.cpu().numpy() for t in pred_ids] if pred_ids is not None else None
        self.label_ids = [t.cpu().numpy() for t in label_ids] if label_ids is not None else None
        self.save_file = save_file

    def _remove_pad_token_id(self, ids):
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        return [(id if id[-1] != pad_id else id[:-1]) for id in ids]

    def _remove_eos_token_id(self, ids):
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        return [(id if id[-1] != eos_id else id[:-1]) for id in ids]

    @staticmethod
    def from_file(path: str) -> "EvalPredictionWithText":
        p = EvalPredictionWithText()
        p.load(path)
        return p

    def load(self, path: str):
        with open(path, "r") as f:
            state = yaml.load(f, Loader=CSafeLoader)
        self.inputs = state["inputs"]
        self.preds = state["preds"]
        self.labels = state["labels"]
        self.input_ids = [np.array(x) for x in state["input_ids"]]
        self.pred_ids = [np.array(x) for x in state["pred_ids"]]
        self.label_ids = [np.array(x) for x in state["label_ids"]]
        self.save_file = path

    def save(self, path: Optional[str] = None):
        if path is None:
            path = self.save_file
        out_dict = dict(
            inputs=self.inputs,
            preds=self.preds,
            labels=self.labels,
            input_ids=[t.astype(int).tolist() for t in self.input_ids],
            pred_ids=[t.astype(int).tolist() for t in self.pred_ids],
            label_ids=[t.astype(int).tolist() for t in self.label_ids],
        )
        with open(path, "w") as f:
            yaml.safe_dump(out_dict, f, sort_keys=False)


class TrainLossEarlyStop:
    def __init__(self) -> None:
        self.nan_limit = 10
        self.consec_nans = 0
        self.should_stop = False

    def __call__(self, control, train_loss) -> Any:
        train_loss = train_loss.item()
        if np.isnan(train_loss) or train_loss <= 1.e-6:
            self.consec_nans += 1
            if self.consec_nans >= self.nan_limit:
                print(f"Stopping after {self.consec_nans} 0/nan losses")
                self.should_stop = True
                control.should_training_stop = True
        else:
            self.consec_nans = 0


class BadEvalEarlyStop:
    def __init__(self, eval_after_epochs, metric=None):
        self.eval_after_epochs = eval_after_epochs
        self.metric = metric

    def __call__(self, control, metrics) -> Any:
        epoch = int(metrics["epoch"])
        if epoch in self.eval_after_epochs:
            metric = self.metric if self.metric is not None else next(iter(metrics.keys()))
            min_val = self.eval_after_epochs[epoch]
            val = metrics[metric]
            if val < min_val:
                control.should_training_stop = True

# Backward-compatible alias
MambaEvalPrediction = EvalPredictionWithText


