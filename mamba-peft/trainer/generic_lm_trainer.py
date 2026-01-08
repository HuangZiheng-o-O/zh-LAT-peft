"""
Generic Language Model Trainer for Linear Attention architectures.

This trainer is designed to work with the FLA (Flash Linear Attention) library's
models including GLA, RetNet, Mamba2, and other Linear Attention variants.
It properly handles attention_mask to ensure correct training behavior.

Key Design Considerations for Linear Attention:
===============================================

1. **attention_mask is REQUIRED**
   Unlike Transformer's softmax attention, Linear Attention uses state accumulation:
       S_t = Diag(α_t) · S_{t-1} + k_t^T ⊗ v_t

   Without attention_mask, padding tokens pollute the hidden state S_t.
   The FLA library implements an "unpadding" strategy that completely removes
   padding tokens from computation when attention_mask is provided.

2. **Left Padding for Generation**
   Generation tasks should use left padding (tokenizer.padding_side = "left")
   to ensure the last tokens are always valid (not padding).

3. **Loss Calculation**
   Loss is only computed at positions where label_ids != -100, so padding
   positions in labels are correctly ignored.

Environment Variables (LAT_* preferred, GLA_* fallback for compatibility):
=========================================================================
- LAT_LOG_PADDING_STATS=1: Log padding ratio every 500 steps for debugging
- LAT_FORCE_LEFT_PAD=1: Force left padding in tokenizer
- LAT_VERBOSE=1: Enable verbose logging

References:
- GLA Paper: https://arxiv.org/abs/2312.06635
- RetNet Paper: https://arxiv.org/abs/2307.08621
- FLA Library: https://github.com/sustcsonglin/flash-linear-attention
"""

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
from mamba_ssm_peft.utils.env_config import env_config


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
            model.load_config(self.args.output_dir)

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
        """
        Forward pass for training and evaluation.

        IMPORTANT: For Linear Attention models from the FLA library,
        attention_mask enables the unpadding strategy for correct behavior.
        References:
        - FLA Implementation: fla/layers/*.py (see get_unpad_data usage)
        """
        input_ids = inputs["input_ids"]
        label_ids = inputs["label_ids"]
        attention_mask = inputs.get("attention_mask")

        # Build model forward kwargs
        add_inputs = {}

        # Pass attention_mask for GLA/Linear Attention models
        # This enables the unpadding strategy in FLA library, which:
        # 1. Removes padding tokens before computation (via index_first_axis)
        # 2. Uses cu_seqlens to track sequence boundaries in Triton kernels
        # 3. Restores output shape after computation (via pad_input)
        if attention_mask is not None:
            add_inputs["attention_mask"] = attention_mask

        # Handle PEFT models that accept label_ids
        if isinstance(model, PeftModel):
            base = model.base_model
            if "label_ids" in base.forward.__code__.co_varnames:
                add_inputs["label_ids"] = label_ids

        # Optional: Log padding statistics for debugging
        # Enable via environment variable: LAT_LOG_PADDING_STATS=1 (or GLA_LOG_PADDING_STATS=1)
        if attention_mask is not None and env_config.get_bool("LOG_PADDING_STATS"):
            if hasattr(self, 'state') and self.state.global_step % 500 == 0:
                valid_tokens = attention_mask.sum().item()
                total_tokens = attention_mask.numel()
                valid_ratio = valid_tokens / total_tokens if total_tokens > 0 else 1.0
                logger.info(
                    f"[Step {self.state.global_step}] Padding stats: "
                    f"valid={valid_ratio:.1%}, pad={1-valid_ratio:.1%}, "
                    f"batch_shape={tuple(input_ids.shape)}"
                )

        # IMPORTANT: Explicitly disable caching for Linear Attention models during training/evaluation.
        # Many FLA models default to use_cache=True, but caching conflicts with the
        # unpadding/padding strategy used for variable-length sequences in batches.
        # This causes "CUDA driver error: invalid argument" during evaluation.
        add_inputs["use_cache"] = False

        lm_logits = model(input_ids, **add_inputs).logits
        return input_ids, label_ids, lm_logits

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, label_ids, lm_logits = self._forward(model, inputs)
        lm_loss = self.train_crit(lm_logits, label_ids)
        if getattr(model, "should_training_stop", False):
            if hasattr(model, "save_config"):
                # Fail-fast: if model exposes save_config(), saving errors should surface.
                model.save_config(self.args.output_dir)
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
        """
        Run generation for a single batch.

        IMPORTANT: Inputs from dataloader are on CPU by default.
        We must move them to the model's device before generation.
        """
        if inputs is None:
            return ([], [])
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        label_ids = inputs.get("label_ids") if isinstance(inputs, dict) else None
        attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None
        if input_ids is None or label_ids is None:
            return ([], [])

        # Move inputs to model device (critical for generation!)
        # DataLoader returns CPU tensors; model is typically on GPU
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        # label_ids stays on CPU - only used for comparison after generation

        out_seq = generator(model, input_ids, attention_mask=attention_mask)
        if hasattr(out_seq, "sequences"):
            out_seq = out_seq.sequences
        if out_seq.dim() == 1:
            out_seq = out_seq.unsqueeze(0)
        if label_ids.dim() == 1:
            label_ids = label_ids.unsqueeze(0)

        # Move generated sequences back to CPU for downstream processing
        # This ensures consistent device handling in EvalPredictionWithText
        out_seq = out_seq.cpu()

        pred_list = [row for row in out_seq]
        label_list = [row for row in label_ids]
        return (pred_list, label_list)

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if getattr(self.args, "save_full_model", False):
            torch.save(self.model, f"{output_dir}/model.pt")
            return

        # Fail-fast: rely on HF/PEFT save_pretrained() semantics.
        # For PEFT models, this saves adapter weights (adapter_config.json + adapter_model.*).
        return super().save_model(output_dir, _internal_call=_internal_call)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.train_loss_early_stop.should_stop:
            self.control.should_evaluate = False
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

    def _save_optimizer_and_scheduler(self, output_dir: str):
        if not getattr(self.args, "save_optimizer_state", True):
            os.makedirs(output_dir, exist_ok=True)
            return
        os.makedirs(output_dir, exist_ok=True)
        return super()._save_optimizer_and_scheduler(output_dir)

    def _save_rng_state(self, output_dir: str):
        if not getattr(self.args, "save_optimizer_state", True):
            os.makedirs(output_dir, exist_ok=True)
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


