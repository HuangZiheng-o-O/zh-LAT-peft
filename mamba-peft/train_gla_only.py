import os
import json
from pathlib import Path
from typing import Optional, Dict

import torch
import yaml

from dataset import load_dataset
from trainer.generic_lm_trainer import GenericLMTrainer, GenericLMTrainingArguments
from mamba_ssm_peft import get_trainable_parameters_ratio, print_trainable_parameter_names
from mamba_ssm_peft.utils.gla_hf_decoder import create_gla_decoder
from train_gla_adapter import prepare_gla_model_and_tokenizer


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")


def build_and_run_trainer_gla_only(
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
        yaml.safe_dump(cfg, f)

    # Force HF-native generation for GLA
    eval_generator = None
    if eval_gen is not None:
        _eval = dict(eval_gen)
        # Normalize keys from our launcher
        max_length = int(_eval.get("max_length", 1024))
        min_length = int(_eval.get("min_length", 0))
        num_beams = _eval.get("num_beams", None)
        eval_generator = create_gla_decoder(
            tokenizer,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=False,
        )

    val_data_module = load_dataset(
        val_data if val_data is not None else data,
        tokenizer,
        val_data_split,
        mode="lm" if eval_generator is None else "gen",
        return_module=True,
    )
    compute_metrics = val_data_module.dataset.compute_metrics

    if debug:
        train_data_module.dataset = torch.utils.data.Subset(train_data_module.dataset, range(8))
        val_data_module.dataset = torch.utils.data.Subset(val_data_module.dataset, range(2))

    # Prefer non-reentrant checkpointing with PEFT bases
    _gc_kwargs = {"use_reentrant": False} if gradient_checkpointing else None

    # Optional optimizer/scheduler/rng saving via env
    _sos_env = str(os.environ.get("SAVE_OPTIMIZER_STATE", "")).lower()
    _save_optimizer_state = _sos_env in ("1", "true", "yes", "on")

    # DataLoader memory knobs (env)
    def _env_int(name: str, default: int) -> int:
        try:
            v = os.environ.get(name)
            return int(v) if v is not None else default
        except Exception:
            return default

    _prefetch = _env_int("DATALOADER_PREFETCH_FACTOR", 2)
    _pin_memory = _env_bool("DATALOADER_PIN_MEMORY", True)
    _persist_workers = _env_bool("DATALOADER_PERSISTENT_WORKERS", False)
    _eval_acc_steps = _env_int("EVAL_ACCUMULATION_STEPS", 128)

    trainer = GenericLMTrainer(
        model=model,
        train_dataset=train_data_module.dataset,
        tokenizer=tokenizer,
        args=GenericLMTrainingArguments(
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
            save_steps=(save_steps_override if save_steps_override is not None else int(eval_epochs * (len(train_data_module.dataset) // batch_size + (len(train_data_module.dataset) % batch_size > 0)))),
            eval_steps=(eval_steps_override if eval_steps_override is not None else int(eval_epochs * (len(train_data_module.dataset) // batch_size + (len(train_data_module.dataset) % batch_size > 0)))),
            dataloader_drop_last=True,
            report_to="none",
            seed=seed,
        ),
        compute_metrics=compute_metrics,
        data_collator=train_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        eval_generator=eval_generator,
        min_eval_metric_after_epoch=min_eval_metric_after_epoch,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--prec")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # env overrides (consistent with train.py)
    env = os.environ
    def _maybe(v, cast):
        return cast(v) if v is not None and v != "" else None

    data_env = env.get("HP_DATA")
    if data_env:
        cfg["data"] = data_env
    bs_env = _maybe(env.get("HP_BATCH_SIZE"), int)
    if bs_env is not None:
        cfg["batch_size"] = bs_env
    lr_env = _maybe(env.get("HP_LR"), float)
    if lr_env is not None:
        cfg["learning_rate"] = lr_env
    epochs_env = _maybe(env.get("HP_EPOCHS"), int)
    if epochs_env is not None:
        cfg["num_epochs"] = epochs_env
    prec_env = env.get("HP_PREC")
    if prec_env:
        cfg["prec"] = prec_env
    seed_env = _maybe(env.get("HP_SEED"), int)
    if seed_env is not None:
        cfg["seed"] = seed_env

    val_split_env = env.get("HP_VAL_SPLIT")
    if val_split_env in {"train", "val", "test"}:
        cfg["val_data_split"] = val_split_env

    # eval_gen defaults (HF path)
    def _truthy(x: str | None) -> bool:
        if x is None:
            return False
        return str(x).lower() in ("1", "true", "yes", "on")

    is_gen_task = any([
        str(cfg.get("data", "")).startswith("samsum"),
        str(cfg.get("data", "")).startswith("dart"),
        str(cfg.get("data", "")).startswith("spider"),
    ])
    force_eval_gen = _truthy(env.get("EVAL_GEN"))
    if (cfg.get("eval_gen") is None) and (is_gen_task or force_eval_gen):
        max_len = _maybe(env.get("EVAL_GEN_MAX_LENGTH"), int) or 1024
        min_len = _maybe(env.get("EVAL_GEN_MIN_LENGTH"), int) or 0
        num_beams = _maybe(env.get("EVAL_GEN_NUM_BEAMS"), int)  # None or int
        cfg["eval_gen"] = {
            "max_length": int(max_len),
            "min_length": int(min_len),
            "num_beams": (int(num_beams) if num_beams is not None else None),
        }

    # Prepare model & tokenizer (GLA only)
    model_id = args.model or cfg.get("model")
    if model_id is None:
        model_id = "fla-hub/gla-1.3B-100B"
    prec = args.prec or cfg.get("prec", "bf16")
    peft = cfg.get("peft", None)
    debug = bool(args.debug)

    model, tokenizer, _ = prepare_gla_model_and_tokenizer(
        model_id=model_id,
        prec=prec,
        debug=debug,
        peft_json_path=peft,
    )
    # Enforce decoder-only friendly padding policy
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    try:
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass

    # Steps / logging
    train_data_for_len = load_dataset(cfg["data"], tokenizer, "train", return_module=True)
    its_per_epoch = int((len(train_data_for_len.dataset) + cfg["batch_size"] - 1) // cfg["batch_size"])
    logging_steps = min(50, its_per_epoch)
    try:
        if env.get("HP_LOGGING_STEPS"):
            logging_steps = int(env.get("HP_LOGGING_STEPS"))
    except Exception:
        pass
    total_steps = int(cfg["num_epochs"] * its_per_epoch)
    try:
        if env.get("HP_MAX_STEPS"):
            total_steps = int(env.get("HP_MAX_STEPS"))
    except Exception:
        pass
    eval_steps_override = None
    save_steps_override = None
    try:
        if env.get("HP_EVAL_STEPS"):
            eval_steps_override = int(env.get("HP_EVAL_STEPS"))
    except Exception:
        pass
    try:
        if env.get("HP_SAVE_STEPS"):
            save_steps_override = int(env.get("HP_SAVE_STEPS"))
    except Exception:
        pass

    out_root = Path("/home/user/mzs_h/output/benchmark/glue")  # keep same convention
    yaml_stem = Path(args.cfg).stem
    output_dir = out_root / f"{cfg['data']}_seed{cfg.get('seed', 42)}" / yaml_stem

    build_and_run_trainer_gla_only(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        cfg=cfg,
        cfg_path=args.cfg,
        learning_rate=cfg["learning_rate"],
        total_steps=total_steps,
        logging_steps=logging_steps,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_data_workers=cfg.get("num_data_workers", 8),
        batch_size=cfg["batch_size"],
        eval_epochs=cfg.get("eval_epochs", 1),
        skip_eval=cfg.get("skip_eval", False),
        no_save=cfg.get("no_save", False),
        eval_steps_override=eval_steps_override,
        save_steps_override=save_steps_override,
        eval_gen=cfg.get("eval_gen"),
        resume_from_checkpoint=bool(args.resume),
        min_eval_metric_after_epoch=cfg.get("min_eval_metric_after_epoch"),
        seed=cfg.get("seed", 42),
        data=cfg["data"],
        val_data=cfg.get("val_data"),
        val_data_split=cfg.get("val_data_split", "val"),
        debug=debug,
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        logits_to_keep=cfg.get("logits_to_keep"),
    )


if __name__ == "__main__":
    main()


