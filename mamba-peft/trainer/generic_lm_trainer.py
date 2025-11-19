from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import os
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import logger
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from peft import PeftModel

from trainer.loss import CrossEntropy, Accuracy
from trainer.eval_utils import (
    EvalPredictionWithText,
    TrainLossEarlyStop,
    BadEvalEarlyStop,
)


@dataclass
class GenericLMTrainingArguments(TrainingArguments):
    info: Dict[str, Any] = field(default=None)
    save_full_model: bool = False
    # Control whether to save optimizer state (optimizer.pt), scheduler state and rng_state during checkpointing
    # Default False to minimize disk usage unless explicitly enabled
    save_optimizer_state: bool = False


class GenericLMTrainer(Trainer):
    def __init__(self,
                 model: PreTrainedModel | nn.Module = None,
                 args: TrainingArguments = None,
                 data_collator: Any | None = None,
                 train_dataset: Dataset | None = None,
                 eval_dataset: Dataset | Dict[str, Dataset] | None = None,
                 tokenizer: PreTrainedTokenizerBase | None = None,
                 model_init: Callable[[], PreTrainedModel] | None = None,
                 compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
                 callbacks: List[TrainerCallback] | None = None,
                 optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                 eval_generator=None,
                 min_eval_metric_after_epoch=None,
                 **kwargs):
        if callbacks is None:
            callbacks = []
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks,
                         optimizers, preprocess_logits_for_metrics, **kwargs)

        self.train_crit = CrossEntropy()
        self.val_crits = [Accuracy()]
        self.train_loss_early_stop = TrainLossEarlyStop()
        self.eval_generator = eval_generator
        self.min_eval_metric_after_epoch_early_stop = BadEvalEarlyStop(min_eval_metric_after_epoch) if min_eval_metric_after_epoch is not None else None

        # Optional model-specific hook; safe no-op if absent
        if hasattr(model, "load_config"):
            try:
                model.load_config(self.args.output_dir)
            except Exception:
                pass

    def log_train_seq(self, input_ids, label_ids, lm_logits, idx=0):
        input_ids, label_ids, lm_logits = input_ids[idx], label_ids[idx], lm_logits[idx]
        output_ids = lm_logits.argmax(-1)
        valid_ids = label_ids != -100
        input_txt = self.tokenizer.decode(input_ids)
        input_txt_valid = self.tokenizer.decode(input_ids[valid_ids])
        label_txt_valid = self.tokenizer.decode(label_ids[valid_ids])
        output_txt_valid = self.tokenizer.decode(output_ids[valid_ids])
        print(input_txt)
        print(input_txt_valid, "->", label_txt_valid)
        print(output_txt_valid, "==", label_txt_valid)

    def _forward(self, model, inputs):
        input_ids = inputs["input_ids"]
        label_ids = inputs["label_ids"]
        add_inputs = {}
        if isinstance(model, PeftModel):
            base = model.base_model
            if "label_ids" in base.forward.__code__.co_varnames:
                add_inputs["label_ids"] = label_ids
        lm_logits = model(input_ids, **add_inputs).logits
        return input_ids, label_ids, lm_logits

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, label_ids, lm_logits = self._forward(model, inputs)
        lm_loss = self.train_crit(lm_logits, label_ids)
        if getattr(model, "should_training_stop", False):
            if hasattr(model, "save_config"):
                try:
                    model.save_config(self.args.output_dir)
                except Exception:
                    pass
                self.control.should_training_stop = True
        self.train_loss_early_stop(self.control, lm_loss)
        return lm_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        input_ids, label_ids, lm_logits = self._forward(model, inputs)
        lm_loss = self.train_crit(lm_logits, label_ids)

        logits_valid = []
        label_ids_valid = []
        for i, (logits_sample, label_ids_sample) in enumerate(zip(lm_logits, label_ids)):
            valid_pos = label_ids_sample != self.train_crit.ignore_index
            logits_sample_valid = logits_sample[valid_pos]
            label_ids_sample_valid = label_ids_sample[valid_pos]
            logits_valid.append(logits_sample_valid)
            label_ids_valid.append(label_ids_sample_valid)
        return (lm_loss, logits_valid, label_ids_valid)

    def generation_step(self, generator, model, inputs):
        if inputs is None:
            return ([], [])
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        label_ids = inputs.get("label_ids") if isinstance(inputs, dict) else None
        attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None
        if input_ids is None or label_ids is None:
            return ([], [])
        out_seq = generator(model, input_ids, attention_mask=attention_mask)
        if hasattr(out_seq, "sequences"):
            out_seq = out_seq.sequences
        if out_seq.dim() == 1:
            out_seq = out_seq.unsqueeze(0)
        if label_ids.dim() == 1:
            label_ids = label_ids.unsqueeze(0)
        pred_list = [row for row in out_seq]
        label_list = [row for row in label_ids]
        return (pred_list, label_list)

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not getattr(self.args, "save_full_model", False):
            return
        torch.save(self.model, f"{output_dir}/model.pt")

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.train_loss_early_stop.should_stop:
            self.control.should_evaluate = False
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

    def _save_optimizer_and_scheduler(self, output_dir: str):
        if not getattr(self.args, "save_optimizer_state", True):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                pass
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            pass
        return super()._save_optimizer_and_scheduler(output_dir)

    def _save_rng_state(self, output_dir: str):
        if not getattr(self.args, "save_optimizer_state", True):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                pass
            return
        return super()._save_rng_state(output_dir)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ):
        return data_collator

    def reset_optimizer(self):
        print("Resetting optimzer")
        self.optimizer = None
        self.lr_scheduler = None
        self.create_optimizer_and_scheduler(self.args.max_steps - self.state.global_step)

    def evaluate(self, eval_dataset: Dataset | Dict[str, Dataset] | None = None, ignore_keys: List[str] | None = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        if self.eval_generator is not None:
            metrics = self.evaluate_generation(self.eval_generator, metric_key_prefix=metric_key_prefix)
        else:
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if self.min_eval_metric_after_epoch_early_stop is not None:
            self.min_eval_metric_after_epoch_early_stop(self.control, metrics)
        return metrics

    @torch.no_grad()
    def evaluate_generation(self, generator, use_cache=True, skip_metrics=False, metric_key_prefix="eval"):
        eval_pred_file = Path(self.args.output_dir) / f"predictions-{self.state.global_step}.yaml"
        if not use_cache or not eval_pred_file.is_file():
            model = self.model
            model.eval()
            dataloader = self.get_eval_dataloader()
            input_ids_all = []
            pred_ids_all = []
            label_ids_all = []
            for step, inputs in enumerate(dataloader):
                if inputs is None:
                    continue
                pred_ids, label_ids = self.generation_step(generator, model, inputs)
                if not pred_ids or not label_ids:
                    continue
                batch_input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
                if batch_input_ids is None:
                    continue
                input_ids_all += [*batch_input_ids]
                pred_ids_all += [*pred_ids]
                label_ids_all += [*label_ids]
            eval_pred = EvalPredictionWithText(generator.tokenizer, input_ids_all, pred_ids_all, label_ids_all,
                                               save_file=eval_pred_file, remove_eos=True)
            eval_pred.save()
        else:
            if not skip_metrics:
                print(f"Loading prediction {eval_pred_file}")
        if not skip_metrics:
            eval_pred = EvalPredictionWithText.from_file(str(eval_pred_file))
            metrics = self.compute_metrics(eval_pred)
            if metric_key_prefix != "":
                metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics
        else:
            return None


