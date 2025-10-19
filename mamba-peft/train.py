
import json
import os
import os
import shutil

from mamba_ssm_peft.peft import MambaPeftType
from mamba_ssm_peft.peft.sd_lora import SdLoraModel
os.environ["WANDB_PROJECT"] = "mamba-peft"

from pathlib import Path
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

import yaml
from mamba_ssm_peft import get_mamba_peft_model, get_trainable_parameters_ratio, load_mamba, load_tokenizer, print_trainable_parameter_names
from mamba_ssm_peft.utils.hf import load_gla, load_gla_tokenizer

from mamba_ssm_peft.utils.decoder import create_decoder
from dataset import load_dataset
from trainer.mamba_trainer import MambaTrainer, MambaTrainingArguments
from trainer.loraga_local import maybe_apply_loraga_ga_init


def _lock_share(name):
    path = Path("share/lock") / name
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "x"):
            pass
        return True
    except OSError:
        print(path, "exists")
        return False


def run_train(
    output_dir,
    cfg_path, 
    model,
    data, 
    val_data=None,
    val_data_split="val", 
    tokenizer="EleutherAI/gpt-neox-20b",
    num_epochs=10, 
    prec="bf16", 
    peft=None, 
    optim="adamw_torch", 
    learning_rate=5e-4, 
    gradient_accumulation_steps=1, 
    num_data_workers=8, 
    batch_size=4, 
    eval_gen=None, 
    backend="cuda", 
    debug=False, 
    resume=False, 
    overwrite=False, 
    lock=False, 
    no_save=False, 
    skip_eval=False, 
    eval_epochs=1, 
    min_eval_metric_after_epoch=None,
    seed=42,
    is_sdlora=False):
    
    if overwrite and is_sdlora:
        assert Path(output_dir).exists()

    cfg = {**locals()}

    if not overwrite:
        if lock and _lock_share(output_dir):
            return

        if (Path(output_dir) / "cfg.yaml").exists():
            if resume:
                resume_from_checkpoint = True
            else:
                assert False, str(Path(output_dir) / "cfg.yaml") + " exists!"
        else:
            resume_from_checkpoint = False
    else:
        # assert Path(output_dir).exists()
        resume_from_checkpoint = False

    if not (data.startswith("glue_") or data in ("glue_rte", "glue_mrpc", "glue_cola", "spider_1000")  or not (no_save and num_epochs > 1)):
        print("Training for more than one epoch without saving ckpts!")

    is_custom_tokenizer = tokenizer != "EleutherAI/gpt-neox-20b"

    # Check if model is GLA or Mamba
    is_gla_model = "gla" in model.lower() or "/gla-" in model.lower() or model.startswith("fla-hub/gla")

    if is_gla_model:
        print(f"Loading GLA model: {model}")

        model_kwargs = dict(
            dtype={"bf16": torch.bfloat16, "fp16": torch.bfloat16, "fp32": torch.float32}[prec],
            device="cuda" if not debug else "cpu",
            trust_remote_code=True,
        )

        gla_loaded = load_gla(model, **model_kwargs)
        model = gla_loaded["model"]
        # Always use the tokenizer packaged with the GLA checkpoint to avoid offline AutoTokenizer fetches
        tokenizer = gla_loaded["tokenizer"]
    else:
        print(f"Loading Mamba model: {model}")
        tokenizer = load_tokenizer(tokenizer)

        model_kwargs = dict(
            dtype={"bf16": torch.bfloat16, "fp16": torch.bfloat16, "fp32": torch.float32}[prec],
            device="cuda",
            backend=backend,
        )

        model = load_mamba(
            model,
            **model_kwargs
        )["model"]

    if peft is not None:
        if is_gla_model:
            # For GLA, use generic PEFT LoRA injection instead of Mamba-specific wrapper
            from peft import LoraConfig, get_peft_model
            with open(peft, "r") as f:
                peft_json = json.load(f)
            # Optional env overrides from launcher (highest precedence)
            r_env = os.environ.get("HP_PEFT_R")
            alpha_env = os.environ.get("HP_PEFT_ALPHA")
            drop_env = os.environ.get("HP_PEFT_DROPOUT")
            init_env = os.environ.get("HP_INIT")
            # Optional fast PiSSA: only upgrade when JSON explicitly requests PiSSA
            fast_pissa_env = os.environ.get("HP_PISSA_FAST")
            if r_env is not None:
                try:
                    peft_json["r"] = int(r_env)
                except Exception:
                    pass
            if alpha_env is not None:
                try:
                    peft_json["lora_alpha"] = int(alpha_env)
                except Exception:
                    pass
            if drop_env is not None:
                try:
                    peft_json["lora_dropout"] = float(drop_env)
                except Exception:
                    pass
            if init_env:
                # e.g., "pissa" or "pissa_niter_4"
                peft_json["init_lora_weights"] = init_env
            else:
                # If HP_PISSA_FAST is set, and config uses PiSSA, switch to fast SVD init
                try:
                    if fast_pissa_env and str(fast_pissa_env).lower() not in ("0", "false", "no", "off"): 
                        init_val = peft_json.get("init_lora_weights", None)
                        if isinstance(init_val, str) and init_val.lower() == "pissa":
                            peft_json["init_lora_weights"] = "pissa_niter_4"
                except Exception:
                    pass
            peft_cfg = LoraConfig(**peft_json)
            model = get_peft_model(model, peft_cfg)
        else:
            model, peft_cfg = get_mamba_peft_model(model, peft, return_peft_cfg=True, train_embedding=is_custom_tokenizer, no_print=True)
            assert (is_sdlora and isinstance(model.base_model, SdLoraModel)) or (not is_sdlora and not isinstance(model.base_model, SdLoraModel))
    else:
        peft_cfg = None

    print_trainable_parameter_names(model)

    print("Loaded model")

    train_data_module = load_dataset(data, tokenizer, "train", return_module=True)

    # Optional LoRA-GA initialization (module-only, stable-scaling & layerwise supported)
    maybe_apply_loraga_ga_init(model, train_data_module, peft, debug=debug)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(output_dir) / "cfg.yaml", "w") as f:
        yaml.dump(cfg, f)

    if eval_gen is not None:
        eval_generator = create_decoder(tokenizer, **eval_gen)
    else:
        eval_generator = None

    val_data_module = load_dataset(
        val_data if val_data is not None else data,
        tokenizer,
        val_data_split,
        mode="lm" if eval_gen is None else "gen",
        return_module=True)

    compute_metrics = val_data_module.dataset.compute_metrics

    if debug:
        train_data_module.dataset = torch.utils.data.Subset(train_data_module.dataset, range(8))
        val_data_module.dataset = torch.utils.data.Subset(val_data_module.dataset, range(2))
        num_epochs = 1

    its_per_epoch = int(np.ceil(len(train_data_module.dataset) / batch_size))
    # Allow runtime overrides to quickly constrain training length/frequency
    env = os.environ
    logging_steps = min(50, its_per_epoch)
    try:
        if env.get("HP_LOGGING_STEPS"):
            logging_steps = int(env.get("HP_LOGGING_STEPS"))
    except Exception:
        pass
    total_steps = int(num_epochs * its_per_epoch)
    try:
        if env.get("HP_MAX_STEPS"):
            total_steps = int(env.get("HP_MAX_STEPS"))
    except Exception:
        pass
    # Eval/save frequency overrides
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

    os.environ["WANDB_NAME"] = str(output_dir).replace("weights/", "")

    print("Dropping last batch")
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
            optim=optim,
            output_dir=output_dir,
            logging_steps=logging_steps,
            dataloader_num_workers=num_data_workers,
            dataloader_prefetch_factor=2,
            eval_accumulation_steps=128,
            info={
                "trainable_params": get_trainable_parameters_ratio(model),
                "cfg_path": cfg_path
            },
            save_strategy="steps" if not no_save else "no",
            evaluation_strategy="steps" if not skip_eval else "no",
            save_steps=(save_steps_override if save_steps_override is not None else int(eval_epochs * its_per_epoch)),
            eval_steps=(eval_steps_override if eval_steps_override is not None else int(eval_epochs * its_per_epoch)),
            dataloader_drop_last=True,
            report_to="none",
            # report_to="wandb",
            seed=seed,
        ),
        compute_metrics=compute_metrics,
        data_collator=train_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        eval_generator=eval_generator,
        min_eval_metric_after_epoch=min_eval_metric_after_epoch,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def get_output_path_for_cfg(cfg_path, cfg):
    """
    目标：
      /home/user/mzs_h/output/benchmark/glue/<data>_seed<seed>/<yaml_stem>
    回退（缺 data/seed 时）：
      /home/user/mzs_h/output/benchmark/glue/cola_gla/<yaml_stem>
    """
    yaml_stem = Path(cfg_path).stem
    data = cfg.get("data")
    seed = cfg.get("seed")

    if data and seed is not None:
        folder = f"{data}_seed{seed}"
        return Path("/home/user/mzs_h/output/benchmark/glue") / folder / yaml_stem
    # fallback 与旧逻辑一致
    return Path("/home/user/mzs_h/output/benchmark/glue/cola_gla") / yaml_stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--prec")
    parser.add_argument("--device")
    args = parser.parse_args()

    if args.device is not None:
        os.environ["VISIBLE_DEVICES"] = args.device

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply environment overrides (highest precedence)
    env = os.environ
    def _maybe(v, cast):
        return cast(v) if v is not None and v != "" else None
    # Data override: accept raw task (e.g., rte) or full id (glue-tvt_rte)
    data_env = env.get("HP_DATA")
    if data_env:
        cfg["data"] = data_env if data_env.startswith("glue") else f"glue-tvt_{data_env}"
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

    output_dir = get_output_path_for_cfg(args.cfg, cfg)

    train_args = {**cfg, **{k: v for k, v in vars(args).items() if v is not None}, "output_dir": str(output_dir)}
    train_args["cfg_path"] = train_args.pop("cfg")
    if "device" in train_args:
        del train_args["device"]

    is_sdlora = False
    if train_args["peft"] is not None:
        with open(train_args["peft"], "r") as f:
            peft_cfg = json.load(f)

        if peft_cfg["peft_type"] == MambaPeftType.SD_LORA:
            is_sdlora = True

    if is_sdlora:
        # assert not train_args["overwrite"], f"Cannot override SDLora checkpoint"
        if train_args["overwrite"]:
            if Path(train_args["output_dir"]).exists():
                shutil.rmtree(train_args["output_dir"])

        del train_args["overwrite"]
        run_train(**train_args, is_sdlora=True)  # warmup
        run_train(**train_args, is_sdlora=True, overwrite=True)  # training
    else:
        run_train(**train_args)


if __name__ == "__main__":
    main()
