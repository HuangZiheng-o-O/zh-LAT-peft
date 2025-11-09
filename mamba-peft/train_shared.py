from pathlib import Path
import os
from typing import Optional, Dict

import numpy as np
import torch
import yaml

from trainer.mamba_trainer import MambaTrainer, MambaTrainingArguments
from dataset import load_dataset
from mamba_ssm_peft import get_trainable_parameters_ratio, print_trainable_parameter_names
from mamba_ssm_peft.utils.decoder import create_decoder


def build_and_run_trainer(
    *,
    model,
    tokenizer,
    output_dir: str,
    cfg: Dict,
    cfg_path: str,
    learning_rate: float,
    total_steps: int,
    logging_steps: int,
    gradient_accumulation_steps: int,
    num_data_workers: int,
    batch_size: int,
    eval_epochs: int,
    skip_eval: bool,
    no_save: bool,
    eval_steps_override: Optional[int],
    save_steps_override: Optional[int],
    eval_gen: Optional[Dict],
    resume_from_checkpoint: bool,
    min_eval_metric_after_epoch,
    seed: int,
    data: str,
    val_data: Optional[str],
    val_data_split: str,
    debug: bool,
    gradient_checkpointing: bool = False,
    logits_to_keep: int | None = None,
):
    print_trainable_parameter_names(model)
    print("Loaded model")

    train_data_module = load_dataset(data, tokenizer, "train", return_module=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "cfg.yaml", "w") as f:
        yaml.dump(cfg, f)

    if eval_gen is not None:
        # Detect FLA/GLA models and default to parallel logits during beam search
        _eval_gen_cfg = dict(eval_gen)
        try:
            cls = model.__class__
            is_gla_model = ("GLA" in getattr(cls, "__name__", "")) or ("fla." in getattr(cls, "__module__", ""))
        except Exception:
            is_gla_model = False
        if is_gla_model and "mode" not in _eval_gen_cfg:
            _eval_gen_cfg["mode"] = "parallel"
        eval_generator = create_decoder(tokenizer, **_eval_gen_cfg)
    else:
        eval_generator = None

    val_data_module = load_dataset(
        val_data if val_data is not None else data,
        tokenizer,
        val_data_split,
        mode="lm" if eval_gen is None else "gen",
        return_module=True,
    )
    compute_metrics = val_data_module.dataset.compute_metrics

    if debug:
        train_data_module.dataset = torch.utils.data.Subset(train_data_module.dataset, range(8))
        val_data_module.dataset = torch.utils.data.Subset(val_data_module.dataset, range(2))

    # Prefer non-reentrant checkpointing to avoid 'inputs have no grad' issues with frozen LoRA bases
    _gc_kwargs = {"use_reentrant": False} if gradient_checkpointing else None

    # Env-only switch (no YAML needed): SAVE_OPTIMIZER_STATE=1|true|yes|on to enable saving optimizer/scheduler/rng
    _sos_env = str(os.environ.get("SAVE_OPTIMIZER_STATE", "")).lower()
    _save_optimizer_state = _sos_env in ("1", "true", "yes", "on")

    # Optional SwanLab logging via Transformers callback (enabled by env vars only)
    callbacks = []
    _sl_enable = str(os.environ.get("SWANLAB_ENABLE", "")).lower() in ("1", "true", "yes", "on", "cloud", "local")
    if _sl_enable:
        # try:
        from swanlab.integration.transformers import SwanLabCallback  # type: ignore
        sl_project = os.environ.get("SWANLAB_PROJECT", "mamba-peft")
        exp_prefix = os.environ.get("SWANLAB_EXPERIMENT_PREFIX", "")
        exp_name = Path(output_dir).name
        if exp_prefix:
            exp_name = f"{exp_prefix}_{exp_name}"
        callbacks.append(SwanLabCallback(project=sl_project, experiment_name=exp_name))
        # except Exception as _e:
        #     print(f"[SwanLab] disabled (import/init failed): {_e}")

    # Optional DataLoader/Eval memory tuning via env (defaults preserve old behavior)
    def _env_bool(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        if v is None:
            return default
        return str(v).lower() in ("1", "true", "yes", "on")

    _prefetch = int(os.environ.get("DATALOADER_PREFETCH_FACTOR", 2))
    _pin_memory = _env_bool("DATALOADER_PIN_MEMORY", True)
    _persist_workers = _env_bool("DATALOADER_PERSISTENT_WORKERS", False)
    _eval_acc_steps = int(os.environ.get("EVAL_ACCUMULATION_STEPS", 128))

    # Auto memory tuning for very large datasets (dataset-size aware; does not rely on task name)
    try:
        dataset_size = len(train_data_module.dataset)
    except Exception:
        dataset_size = 0
    _auto_on = _env_bool("MEMORY_TUNING_AUTO", True)
    _threshold = int(os.environ.get("MEMORY_TUNING_THRESHOLD_SAMPLES", 200000))
    if _auto_on and dataset_size >= _threshold:
        # Only adjust when user did not explicitly override via envs
        if os.environ.get("DATALOADER_PREFETCH_FACTOR") is None:
            _prefetch = 1
        if os.environ.get("DATALOADER_PIN_MEMORY") is None:
            _pin_memory = False
        if os.environ.get("DATALOADER_PERSISTENT_WORKERS") is None:
            _persist_workers = False
        if os.environ.get("EVAL_ACCUMULATION_STEPS") is None:
            _eval_acc_steps = 32
        # Be conservative with CPU workers
        if num_data_workers > 2:
            num_data_workers = 1

    trainer = MambaTrainer(
        model=model,
        train_dataset=train_data_module.dataset,
        tokenizer=tokenizer,
        args=MambaTrainingArguments(
            learning_rate=float(learning_rate),
            max_steps=total_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=_gc_kwargs,
            optim=cfg.get("optim", "adamw_torch"),
            output_dir=output_dir,
            logging_steps=logging_steps,
            dataloader_num_workers=num_data_workers,
            dataloader_prefetch_factor=_prefetch,
            dataloader_pin_memory=_pin_memory,
            dataloader_persistent_workers=_persist_workers,
            eval_accumulation_steps=_eval_acc_steps,
            info={
                "trainable_params": get_trainable_parameters_ratio(model),
                "cfg_path": cfg_path,
                "logits_to_keep": logits_to_keep,
            },
            save_optimizer_state=_save_optimizer_state,
            save_strategy="steps" if not no_save else "no",
            evaluation_strategy="steps" if not skip_eval else "no",
            save_steps=(save_steps_override if save_steps_override is not None else int(eval_epochs * np.ceil(len(train_data_module.dataset) / batch_size))),
            eval_steps=(eval_steps_override if eval_steps_override is not None else int(eval_epochs * np.ceil(len(train_data_module.dataset) / batch_size))),
            dataloader_drop_last=True,
            report_to="none",
            seed=seed,
        ),
        compute_metrics=compute_metrics,
        data_collator=train_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        eval_generator=eval_generator,
        min_eval_metric_after_epoch=min_eval_metric_after_epoch,
        callbacks=callbacks or None,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


