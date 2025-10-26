
import sys
from pathlib import Path
# --- ensure local 'fla' submodule is importable when running from mamba-peft/ ---
try:
    import fla  # noqa: F401
except Exception:
    try:
        repo_root = Path(__file__).resolve().parents[1]  # .../zh-LAT-peft
        fla_symlink = repo_root / "fla"
        if fla_symlink.exists():
            sys.path.insert(0, str(repo_root))
            import fla  # noqa: F401
    except Exception:
        pass

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

# Adapters and shared trainer
from train_gla_adapter import prepare_gla_model_and_tokenizer
from train_mamba_adapter import prepare_mamba_model_and_tokenizer
from train_shared import build_and_run_trainer


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
    is_sdlora=False,
    gradient_checkpointing=False,
    logits_to_keep=None):
    
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
        model, tokenizer, _ = prepare_gla_model_and_tokenizer(
            model_id=model,
            prec=prec,
            debug=debug,
            peft_json_path=peft,
        )
    else:
        print(f"Loading Mamba model: {model}")
        model, tokenizer, _, is_sdlora_detected = prepare_mamba_model_and_tokenizer(
            model_id=model,
            tokenizer_id=tokenizer,
            prec=prec,
            backend=backend,
            is_custom_tokenizer=is_custom_tokenizer,
            peft_json_path=peft,
            no_print=True,
        )
        # Keep legacy assertion semantics intact
        assert (is_sdlora and is_sdlora_detected) or ((not is_sdlora) and (not is_sdlora_detected))

    if not is_gla_model:
        # Optional LoRA-GA initialization (module-only, stable-scaling & layerwise supported)
        train_data_module_for_ga = load_dataset(data, tokenizer, "train", return_module=True)
        maybe_apply_loraga_ga_init(model, train_data_module_for_ga, peft, debug=debug)

    its_per_epoch = int(np.ceil(len(load_dataset(data, tokenizer, "train", return_module=True).dataset) / batch_size))
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
    build_and_run_trainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        cfg=cfg,
        cfg_path=cfg_path,
        learning_rate=learning_rate,
        total_steps=total_steps,
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_data_workers=num_data_workers,
        batch_size=batch_size,
        eval_epochs=eval_epochs,
        skip_eval=skip_eval,
        no_save=no_save,
        eval_steps_override=eval_steps_override,
        save_steps_override=save_steps_override,
        eval_gen=eval_gen,
        resume_from_checkpoint=resume_from_checkpoint,
        min_eval_metric_after_epoch=min_eval_metric_after_epoch,
        seed=seed,
        data=data,
        val_data=val_data,
        val_data_split=val_data_split,
        debug=debug,
        logits_to_keep=logits_to_keep,
        gradient_checkpointing=gradient_checkpointing,
    )


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
