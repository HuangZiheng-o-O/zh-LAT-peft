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
import shutil
from typing import Optional, Dict

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader  # noqa: F401  # kept for compatibility

import yaml

os.environ["WANDB_PROJECT"] = "mamba-peft"

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


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return float(v) if v is not None else default
    except Exception:
        return default


def _lock_share(name: str) -> bool:
    """
    Acquire a simple filesystem lock under share/lock/<name>.
    Returns:
      True  -> lock already exists (another process holds it) – caller SHOULD skip.
      False -> lock created successfully – caller MAY proceed and SHOULD remove it after completion.
    """
    path = Path("share/lock") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[GLA][lock] {path} exists; skipping this run to avoid duplicate training.")
        return True
    try:
        with open(path, "x"):
            pass
        print(f"[GLA][lock] acquired {path}")
        return False
    except OSError:
        print(f"[GLA][lock] {path} exists; skipping this run.")
        return True


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
    """
    纯 GLA-only 的训练和评估入口：
      - 使用 GenericLMTrainer / GenericLMTrainingArguments
      - 生成评估统一走 HF-native model.generate()（create_gla_decoder）
      - 数据加载仍复用原 dataset.load_dataset 以及 Spider/GLUE 等模块
    """
    print_trainable_parameter_names(model, output_dir=output_dir, cfg_path=cfg_path)
    print("Loaded model")

    # # 在构建任何数据模块之前，优先依据环境变量强制左填充到传入的 tokenizer，避免生成期间出现右填充告警
    # try:
    #     _force_left = str(os.environ.get("GLA_FORCE_LEFT_PAD", "1")).lower() in ("1", "true", "yes", "on")
    #     if _force_left and hasattr(tokenizer, "padding_side"):
    #         tokenizer.padding_side = "left"
    #         if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
    #             tokenizer.pad_token = tokenizer.eos_token
    #         if str(os.environ.get("GLA_VERBOSE", "0")).lower() in ("1", "true", "yes", "on"):
    #             print("[GLA] Using left padding for decoder-only generation (GLA_FORCE_LEFT_PAD=1).")
    # except Exception as _e:
    #     print(f"[GLA][warn] Failed to enforce left padding policy early: {_e}")

    # 构建 train data module（真正用来训练的）
    train_data_module = load_dataset(data, tokenizer, "train", return_module=True)

    # 保存 cfg.yaml（保持与旧 train.py 一致）
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "cfg.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # 构造生成式评估 decoder（GLA HF-native）
    eval_generator = None
    if eval_gen is not None:
        _eval = dict(eval_gen)
        max_length = int(_eval.get("max_length", 1024))
        min_length = int(_eval.get("min_length", 5))
        eval_generator = create_gla_decoder(
            tokenizer,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )

    # 验证/评估 data module：生成任务用 "gen" 模式，否则 "lm"
    val_data_module = load_dataset(
        val_data if val_data is not None else data,
        tokenizer,
        val_data_split,
        mode="lm" if eval_generator is None else "gen",
        return_module=True,
    )
    compute_metrics = val_data_module.dataset.compute_metrics

    # debug 模式下截断数据规模
    if debug:
        train_data_module.dataset = torch.utils.data.Subset(
            train_data_module.dataset, range(8)
        )
        val_data_module.dataset = torch.utils.data.Subset(
            val_data_module.dataset, range(2)
        )

    # gradient checkpointing 参数（默认 non-reentrant，对 PEFT 更友好）
    _gc_kwargs = {"use_reentrant": False} if gradient_checkpointing else None

    # 是否保存 optimizer state：由 env 控制（与新代码保持一致）
    _sos_env = str(os.environ.get("SAVE_OPTIMIZER_STATE", "")).lower()
    _save_optimizer_state = _sos_env in ("1", "true", "yes", "on")

    # DataLoader 相关 env knob
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

    # Modern LR scheduler configuration (advanced scheduling with warmup)
    _lr_scheduler_type = os.environ.get("LR_SCHEDULER_TYPE", "constant")  # constant|linear|cosine|polynomial
    _warmup_steps = _env_int("LR_WARMUP_STEPS", None)
    _warmup_ratio = _env_float("LR_WARMUP_RATIO", 0.1)  # fallback if warmup_steps not set
    if _warmup_steps is None and _warmup_ratio > 0:
        _warmup_steps = int(_warmup_ratio * total_steps)

    # Optional SwanLab integration (controlled by env)
    callbacks = []
    _sl_enable = str(os.environ.get("SWANLAB_ENABLE", "")).lower() in ("1", "true", "yes", "on", "cloud", "local")
    if _sl_enable:
        try:
            # Filter out tokenizer padding warnings before SwanLab initialization
            import warnings
            warnings.filterwarnings("ignore", message=".*For correct generation results, please set.*padding_side.*left.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*decoder-only architecture is being used, but right-padding was detected.*", category=UserWarning)

            from swanlab.integration.transformers import SwanLabCallback
            sl_project = os.environ.get("SWANLAB_PROJECT", "gla-peft")
            exp_prefix = os.environ.get("SWANLAB_EXPERIMENT_PREFIX", "")
            exp_name = Path(output_dir).name
            if exp_prefix:
                exp_name = f"{exp_prefix}_{exp_name}"
            sl_mode = os.environ.get("SWANLAB_MODE", "")
            if sl_mode:
                callbacks.append(SwanLabCallback(project=sl_project, experiment_name=exp_name, mode=sl_mode))
            else:
                callbacks.append(SwanLabCallback(project=sl_project, experiment_name=exp_name))
            # Optional: register EmailCallback from YAML secrets (no code changes to SwanLab core)
            try:
                import swanlab  # type: ignore
                from swanlab.plugin.notification import EmailCallback  # type: ignore
                email_yaml = os.environ.get("SWANLAB_EMAIL_YAML", "dangerous/email_notify.yaml")
                if Path(email_yaml).is_file():
                    with open(email_yaml, "r") as _ef:
                        _ecfg = yaml.safe_load(_ef) or {}
                    if all(k in _ecfg for k in ("sender_email", "receiver_email", "password", "smtp_server", "port")):
                        _email_cb = EmailCallback(
                            sender_email=str(_ecfg["sender_email"]),
                            receiver_email=str(_ecfg["receiver_email"]),
                            password=str(_ecfg["password"]),
                            smtp_server=str(_ecfg["smtp_server"]),
                            port=int(_ecfg.get("port", 587)),
                            language=str(_ecfg.get("language", "zh")),
                        )
                        swanlab.register_callbacks([_email_cb])  # register alongside SwanLabCallback
                        # Optional immediate STARTED email (can be disabled by SWANLAB_EMAIL_ON_START=0)
                        _start_env = str(os.environ.get("SWANLAB_EMAIL_ON_START", "1")).lower()
                        if _start_env in ("1", "true", "yes", "on"):
                            try:
                                _msg = f"Output: {output_dir}\nData: {cfg.get('data')}\nSeed: {cfg.get('seed')}\nCfg: {cfg_path}"
                                _email_cb.send_email(subject=f"SwanLab | STARTED | {exp_name}", content=_msg)
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception as e:
            print(f"[GLA][swanlab][warn] Failed to initialize SwanLabCallback: {e}")

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
            lr_scheduler_type=_lr_scheduler_type,
            warmup_steps=_warmup_steps,
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
            save_steps=(
                save_steps_override
                if save_steps_override is not None
                else int(
                    eval_epochs
                    * (
                        len(train_data_module.dataset) // batch_size
                        + (len(train_data_module.dataset) % batch_size > 0)
                    )
                )
            ),
            eval_steps=(
                eval_steps_override
                if eval_steps_override is not None
                else int(
                    eval_epochs
                    * (
                        len(train_data_module.dataset) // batch_size
                        + (len(train_data_module.dataset) % batch_size > 0)
                    )
                )
            ),
            dataloader_drop_last=True,
            # 旧版默认是 wandb，通过 Trainer 的 report_to 控制；
            # 这里如果你后面想开 wandb，可以改成 "wandb"
            report_to="none",
            seed=seed,
        ),
        compute_metrics=compute_metrics,
        data_collator=train_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        callbacks=callbacks or None,
        eval_generator=eval_generator,
        min_eval_metric_after_epoch=min_eval_metric_after_epoch,
    )

    # Train with best-effort email notifications on failure/success (does not interfere with SwanLab core)
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        try:
            # Optional FINISHED email fallback (SWANLAB_EMAIL_ON_FINISH controls; default on)
            _fin_env = str(os.environ.get("SWANLAB_EMAIL_ON_FINISH", "1")).lower()
            if _sl_enable and _fin_env in ("1", "true", "yes", "on"):
                from scripts.utils.email_notify import send_event_email  # type: ignore
                send_event_email("FINISHED", group=Path(output_dir).name, details=f"Finished OK: {output_dir}")
        except Exception:
            pass
    except Exception as _e:
        try:
            from scripts.utils.email_notify import send_event_email  # type: ignore
            import traceback
            tb = "".join(traceback.format_exception_only(type(_e), _e))
            send_event_email("FAILED", group=Path(output_dir).name, details=f"Failed: {tb}")
        except Exception:
            pass
        raise


def run_train(
    output_dir,
    cfg_path,
    model,
    data,
    val_data=None,
    val_data_split="val",
    tokenizer="EleutherAI/gpt-neox-20b",  # 保留参数名以兼容旧 cfg；实际不再使用
    num_epochs=10,
    prec="bf16",
    peft=None,
    optim="adamw_torch",
    learning_rate=5e-4,
    gradient_accumulation_steps=1,
    num_data_workers=8,
    batch_size=4,
    eval_gen=None,
    backend="cuda",  # 保留参数，仅为兼容旧 cfg，不再使用
    debug=False,
    resume=False,
    overwrite=False,
    lock=False,
    no_save=False,
    skip_eval=False,
    eval_epochs=1,
    min_eval_metric_after_epoch=None,
    seed=42,
    is_sdlora=False,  # 保留字段但不再触发 SD-LoRA 两阶段逻辑
    gradient_checkpointing=False,
    logits_to_keep=None,
):
    """
    GLA-only 的 run_train：
      - 仍然保留旧 train.py 的 cfg/lock/resume/overwrite 语义
      - 但不再支持 Mamba 模型路径，也不再做 SD-LoRA warmup 两阶段
    """
    # 旧代码：如果 overwrite 且 is_sdlora，要求 output_dir 已存在
    # 现在我们不再走 SD-LoRA 两阶段，这个条件基本不会触发，保留不影响行为
    if overwrite and is_sdlora:
        assert Path(output_dir).exists()

    # 在任何后续修改前先 snapshot 一份 cfg（保持与旧逻辑一致）
    cfg = {**locals()}

    created_lock = False
    if not overwrite:
        if lock:
            # If a lock already exists, skip this run; otherwise acquire and remember to release after training.
            if _lock_share(str(output_dir)):
                return
            created_lock = True

        if (Path(output_dir) / "cfg.yaml").exists():
            if resume:
                resume_from_checkpoint = True
            else:
                assert False, str(Path(output_dir) / "cfg.yaml") + " exists!"
        else:
            resume_from_checkpoint = False
    else:
        # overwrite=True 时，总是从头训练
        resume_from_checkpoint = False

    # 旧版的安全提示：多 epoch 但不保存 ckpt 时警告
    if not (
        data.startswith("glue_")
        or data in ("glue_rte", "glue_mrpc", "glue_cola", "spider_1000")
        or not (no_save and num_epochs > 1)
    ):
        print("Training for more than one epoch without saving ckpts!")

    # -------------------------------
    # 纯 GLA 模型加载（不再有 Mamba 分支）
    # -------------------------------
    print(f"Loading GLA model: {model}")
    model_id = model
    model, tokenizer_obj, _ = prepare_gla_model_and_tokenizer(
        model_id=model_id,
        prec=prec,
        debug=debug,
        peft_json_path=peft,
    )

    # 强制 decoder-only 友好的左填充策略（可通过 GLA_FORCE_LEFT_PAD 控制，默认开启）
    force_left = str(os.environ.get("GLA_FORCE_LEFT_PAD", "1")).lower() in ("1", "true", "yes", "on")
    if force_left:
        try:
            tokenizer_obj.padding_side = "left"
            if getattr(tokenizer_obj, "pad_token_id", None) is None and getattr(tokenizer_obj, "eos_token", None) is not None:
                tokenizer_obj.pad_token = tokenizer_obj.eos_token
            print("[GLA] Using left padding for decoder-only generation (GLA_FORCE_LEFT_PAD=1).")
        except Exception as e:
            print(f"[GLA][warn] Failed to enforce left padding policy: {e}")
    else:
        print("[GLA] Respecting tokenizer's original padding policy (GLA_FORCE_LEFT_PAD=0).")

    # 预构建 train data module，仅用于计算长度和 steps
    train_data_module_for_len = load_dataset(
        data, tokenizer_obj, "train", return_module=True
    )

    its_per_epoch = int(
        np.ceil(len(train_data_module_for_len.dataset) / batch_size)
    )

    # 与旧 train.py 保持一致的 logging / steps / env override 逻辑
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

    # WANDB 名称（保持旧行为）
    os.environ["WANDB_NAME"] = str(output_dir).replace("weights/", "")

    print("Dropping last batch")

    # resume 语义：若请求 resume，必须确保存在有效的 checkpoint
    resume_arg = None
    if resume_from_checkpoint:
        last_ckpt = _find_last_checkpoint(Path(output_dir))
        if last_ckpt is None:
            raise RuntimeError(f"[GLA] --resume was set but no checkpoint-* found under {output_dir}")
        resume_arg = str(last_ckpt)
        print(f"[GLA] Resuming from checkpoint: {resume_arg}")

    try:
        build_and_run_trainer_gla_only(
            model=model,
            tokenizer=tokenizer_obj,
            output_dir=str(output_dir),
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
            resume_from_checkpoint=resume_arg,
            min_eval_metric_after_epoch=min_eval_metric_after_epoch,
            seed=seed,
            data=data,
            val_data=val_data,
            val_data_split=val_data_split,
            debug=debug,
            gradient_checkpointing=gradient_checkpointing,
            logits_to_keep=logits_to_keep,
        )
    finally:
        if created_lock:
            try:
                lock_path = Path("share/lock") / str(output_dir)
                lock_path.unlink(missing_ok=True)
                print(f"[GLA][lock] released {lock_path}")
            except Exception as e:
                print(f"[GLA][lock][warn] failed to remove lock: {e}")


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

def _find_last_checkpoint(root: Path) -> Optional[Path]:
    """
    扫描输出目录下的 checkpoint-* 子目录，返回步数最高的一个。
    """
    if not root.exists():
        return None
    try:
        candidates = [p for p in root.glob("checkpoint-*") if p.is_dir()]
        if not candidates:
            return None
        def step_of(p: Path) -> int:
            try:
                return int(p.name.split("-")[-1])
            except Exception:
                return -1
        candidates.sort(key=step_of)
        return candidates[-1] if candidates else None
    except Exception:
        return None


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

    # 兼容旧逻辑：允许通过 --device 设置 VISIBLE_DEVICES
    if args.device is not None:
        os.environ["VISIBLE_DEVICES"] = args.device

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply environment overrides (最高优先级，与旧 train.py 保持一致)
    env = os.environ

    def _maybe(v, cast):
        return cast(v) if v is not None and v != "" else None

    # HP_DATA：GLUE/spider 等任务的别名映射逻辑（完整保留）
    data_env = env.get("HP_DATA")
    if data_env:
        glue_tasks = {
            "rte",
            "mrpc",
            "cola",
            "sst2",
            "qnli",
            "qqp",
            "mnli",
            "wnli",
        }
        accepted_prefixes = (
            "glue",
            "samsum",
            "dart",
            "spider",
            "mnist",
            "cifar",
            "piqa",
            "boolq",
            "arc",
        )
        if data_env in glue_tasks:
            cfg["data"] = f"glue-tvt_{data_env}"
        elif data_env == "cifar":
            cfg["data"] = "cifar-tvt"
        elif data_env == "spider":
            cfg["data"] = "spider-tvt"
        else:
            cfg["data"] = (
                data_env
                if data_env.startswith(accepted_prefixes)
                else data_env
            )

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

    # Optional override of validation split via env (train|val|test)
    val_split_env = env.get("HP_VAL_SPLIT")
    if val_split_env in {"train", "val", "test"}:
        cfg["val_data_split"] = val_split_env

    # eval_gen 自动注入：保持与旧 train.py 一致（生成任务 + EVAL_GEN）
    def _truthy(x: Optional[str]) -> bool:
        if x is None:
            return False
        return str(x).lower() in ("1", "true", "yes", "on")

    data_name = str(cfg.get("data", ""))
    is_gen_task = any(
        [
            data_name.startswith("samsum"),
            data_name.startswith("dart"),
            data_name.startswith("spider"),  # e.g., spider-tvt
        ]
    )
    force_eval_gen = _truthy(env.get("EVAL_GEN"))
    if (cfg.get("eval_gen") is None) and (is_gen_task or force_eval_gen):
        max_len = _maybe(env.get("EVAL_GEN_MAX_LENGTH"), int) or 1024
        min_len = _maybe(env.get("EVAL_GEN_MIN_LENGTH"), int) or 5
        cfg["eval_gen"] = {
            "max_length": int(max_len),
            "min_length": int(min_len),
        }

    # 输出目录与旧 get_output_path_for_cfg 完全一致
    output_dir = get_output_path_for_cfg(args.cfg, cfg)

    # 将 cfg + CLI args 合并成 run_train 的参数
    train_args = {
        **cfg,
        **{k: v for k, v in vars(args).items() if v is not None},
        "output_dir": str(output_dir),
    }
    train_args["cfg_path"] = train_args.pop("cfg")
    if "device" in train_args:
        del train_args["device"]

    # 不再根据 peft_type 特判 SD-LoRA，
    # 也不再做两阶段 warmup，直接调用 run_train 一次
    run_train(**train_args)


if __name__ == "__main__":
    main()


