"""
Unified evaluation entry point for zh-LAT-peft.

Goals:
- Reuse the existing LAT model loading/adapter stack:
  ModelRegistry + lat_model_loader + lat_adapter + env_config
- Evaluate a trained PEFT adapter (LoRA/DoRA/RSLoRA/...) on multiple datasets
- Prefer local/offline datasets under ./data or $LAT_DATA_DIR / $DATA_DIR
- Write results to: mamba-peft/outputs/lm_eval/

Typical usage (after training):
  export EVAL_TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'
  python eval_lat.py --cfg cfg/my_lora_exp/yaml/E1_QKVO_plus_MLP_r8_alpha16.yaml --model-type retnet

You can also pass PEFT weights explicitly:
  python eval_lat.py --model-type retnet --model /path/to/base --peft-weights /path/to/checkpoint-XXXX --tasks boolq,piqa
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch

from dataset import load_dataset
from trainer.generic_lm_trainer import GenericLMTrainer, GenericLMTrainingArguments

from lat_adapter import prepare_lat_model_and_tokenizer, attach_peft_weights


REPO_ROOT = Path(__file__).resolve().parent.parent  # .../zh-LAT-peft
DEFAULT_EVAL_OUT_ROOT = Path(__file__).resolve().parent / "outputs" / "lm_eval"


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _dtype_from_prec(prec: str) -> torch.dtype:
    prec = str(prec).lower()
    if prec in ("bf16", "bfloat16"):
        return torch.bfloat16
    if prec in ("fp16", "float16", "half"):
        # Keep consistent with training behavior: fp16 maps to bf16 in adapter
        return torch.bfloat16
    if prec in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown prec={prec}")


def _find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    ckpts = []
    for p in output_dir.glob("checkpoint-*"):
        if p.is_dir():
            m = re.match(r"checkpoint-(\d+)$", p.name)
            step = int(m.group(1)) if m else -1
            ckpts.append((step, p))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def _looks_like_peft_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    # Common PEFT files
    for fname in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        if (p / fname).exists():
            return True
    return False


def _infer_peft_weights_dir(output_dir: Path) -> Optional[Path]:
    """
    Try common locations:
    - output_dir/ (some saves write adapter here)
    - output_dir/checkpoint-*/ (typical Trainer checkpoint)
    """
    if _looks_like_peft_dir(output_dir):
        return output_dir
    latest = _find_latest_checkpoint(output_dir)
    if latest and _looks_like_peft_dir(latest):
        return latest
    # fallback: maybe adapter saved in output_dir even if checkpoints exist
    if latest and _looks_like_peft_dir(output_dir):
        return output_dir
    return None


def _load_cfg(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mirror the minimal env override behavior used in train_lat.py for model/prec.
    """
    env = os.environ
    out = dict(cfg)
    # Model overrides
    model_env = env.get("LAT_MODEL") or env.get("GLA_MODEL")
    if model_env:
        out["model"] = model_env
    prec_env = env.get("LAT_PREC") or env.get("HP_PREC")
    if prec_env:
        out["prec"] = prec_env
    seed_env = env.get("HP_SEED")
    if seed_env:
        try:
            out["seed"] = int(seed_env)
        except Exception:
            pass
    data_env = env.get("HP_DATA")
    if data_env:
        # Keep train_lat.py mapping logic minimal here: allow raw string
        # Users can pass full names like glue-tvt_cola, etc.
        out["data"] = data_env
    val_split_env = env.get("HP_VAL_SPLIT")
    if val_split_env in {"train", "val", "test"}:
        out["val_data_split"] = val_split_env
    return out


def _compute_output_dir_from_train_lat(cfg_path: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """
    Reuse train_lat.py's output_dir logic to locate adapter checkpoints for this config.
    """
    try:
        import train_lat  # local import (mamba-peft/train_lat.py)

        return Path(train_lat.get_output_path_for_cfg(cfg_path, cfg))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation for LAT + PEFT adapters")
    parser.add_argument("--cfg", type=str, default=None, help="YAML config path (same as training)")
    parser.add_argument("--model-type", type=str, default="auto", help="gla|retnet|delta_net|mamba2|auto")
    parser.add_argument("--model", type=str, default=None, help="Base model id/path override")
    parser.add_argument("--prec", type=str, default=None, help="bf16|fp16|fp32 override")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated tasks")
    parser.add_argument("--split", type=str, default=None, help="val|test (dataset-dependent; default uses HP_VAL_SPLIT or val)")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Per-device eval batch size")
    parser.add_argument("--num-data-workers", type=int, default=None, help="Dataloader workers")
    parser.add_argument("--peft-weights", type=str, default=None, help="Explicit PEFT adapter dir (checkpoint or output dir)")
    parser.add_argument("--output-root", type=str, default=None, help="Where to write eval outputs (default: mamba-peft/outputs/lm_eval)")
    parser.add_argument("--debug", action="store_true", help="Force CPU for debugging")
    args = parser.parse_args()

    # Resolve model_type (env override like train_lat.py)
    if args.model_type == "auto":
        env_model_type = os.environ.get("MODEL_TYPE", "auto")
        if env_model_type != "auto":
            args.model_type = env_model_type

    # Tasks
    tasks_str = (
        args.tasks
        or os.environ.get("EVAL_TASKS")
        or "boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa"
    )
    tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]

    # Split
    split = args.split or os.environ.get("HP_VAL_SPLIT") or "val"
    if split not in {"train", "val", "test"}:
        split = "val"

    # Eval batch size
    eval_bs = args.eval_batch_size or int(os.environ.get("EVAL_BATCH_SIZE") or os.environ.get("HP_EVAL_BATCH_SIZE") or 64)
    num_workers = args.num_data_workers if args.num_data_workers is not None else int(os.environ.get("NUM_DATA_WORKERS") or 4)

    # Load cfg if provided
    cfg: Dict[str, Any] = {}
    if args.cfg:
        cfg = _apply_env_overrides(_load_cfg(args.cfg))

    # Resolve base model + prec from CLI/env/cfg
    model_id = args.model or cfg.get("model") or os.environ.get("LAT_MODEL") or os.environ.get("GLA_MODEL")
    if not model_id:
        raise ValueError("Base model is required: pass --model or set LAT_MODEL/GLA_MODEL or provide cfg with 'model'.")
    prec = args.prec or cfg.get("prec") or os.environ.get("LAT_PREC") or os.environ.get("HP_PREC") or "bf16"
    dtype = _dtype_from_prec(prec)

    # Determine PEFT weights dir
    peft_dir: Optional[Path] = Path(args.peft_weights).expanduser() if args.peft_weights else None
    if peft_dir is None and args.cfg:
        out_dir = _compute_output_dir_from_train_lat(args.cfg, cfg)
        if out_dir is not None:
            peft_dir = _infer_peft_weights_dir(out_dir)
            if peft_dir is None:
                print(f"[LAT][eval][warn] No PEFT adapter files found under {out_dir}. Running base model evaluation.")

    # Load base model via unified adapter (NO peft_json here; we attach trained adapter weights below)
    device = "cpu" if args.debug else "cuda"
    model, tokenizer, _ = prepare_lat_model_and_tokenizer(
        model_type=args.model_type,
        model_id=str(model_id),
        prec=str(prec),
        debug=(device == "cpu"),
        peft_json_path=None,
    )

    # Attach adapter weights if provided/found
    if peft_dir is not None:
        model = attach_peft_weights(model, str(peft_dir), torch_dtype=dtype)

    # Safety: disable caching in eval (matches GenericLMTrainer logic)
    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception:
        pass

    # Output root
    out_root = Path(args.output_root).expanduser() if args.output_root else DEFAULT_EVAL_OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    # Run name
    safe_model_type = str(args.model_type).replace("/", "_")
    safe_model_id = str(model_id).split("/")[-1].replace("/", "_")
    peft_tag = peft_dir.name if peft_dir is not None else "base"
    run_dir = out_root / f"{safe_model_type}_{safe_model_id}_{peft_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, Any] = {
        "model_type": args.model_type,
        "model": str(model_id),
        "prec": str(prec),
        "device": device,
        "peft_weights": str(peft_dir) if peft_dir is not None else None,
        "split": split,
        "tasks": tasks,
        "metrics": {},
    }

    for task in tasks:
        print(f"[LAT][eval] task={task} split={split} bs={eval_bs}")
        data_module = load_dataset(task, tokenizer, split, return_module=True)
        compute_metrics = data_module.dataset.compute_metrics

        task_out = run_dir / task
        task_out.mkdir(parents=True, exist_ok=True)

        trainer = GenericLMTrainer(
            model=model,
            args=GenericLMTrainingArguments(
                output_dir=str(task_out),
                per_device_eval_batch_size=int(eval_bs),
                per_device_train_batch_size=1,
                dataloader_num_workers=int(num_workers),
                report_to="none",
                evaluation_strategy="no",
                save_strategy="no",
                logging_steps=50,
                seed=int(cfg.get("seed", 42)) if isinstance(cfg.get("seed", 42), int) else 42,
                remove_unused_columns=False,
            ),
            tokenizer=tokenizer,
            data_collator=data_module.data_collator,
            eval_dataset=data_module.dataset,
            compute_metrics=compute_metrics,
        )

        metrics = trainer.evaluate()
        # Normalize keys
        all_metrics["metrics"][task] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}

        with open(task_out / "metrics.json", "w") as f:
            json.dump(all_metrics["metrics"][task], f, indent=2, ensure_ascii=False)

    with open(run_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"[LAT][eval] Done. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()


