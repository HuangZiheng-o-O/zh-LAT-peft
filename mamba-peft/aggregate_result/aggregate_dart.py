#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
聚合 DART 实验结果的工具脚本。

支持两种模式（通过环境变量 / 参数切换）：

1）最佳 checkpoint 模式（默认）：
    - 对每个实验目录（E*），读取最新的 trainer_state.json
    - 在 log_history 中找到所有含 eval_bleu 的 eval 行
    - 按以下规则选出最佳行：
        eval_bleu DESC
        eval_meteor DESC
        eval_chrf DESC
        eval_loss ASC
        step DESC
    - 输出每个实验一行的 summary 表

2）固定 step 模式（设置 DART_AGG_STEP 或 --fixed_step）：
    - 对每个实验目录，从 trainer_state.json 中找到 step == 固定值 的 eval 行
      （带 eval_bleu / eval_chrf / eval_meteor 的那条）
    - 可选地记录同一 step 的 train loss
    - 输出每个实验在该 step 下的指标表

用法示例：

    # 模式 A：最佳 checkpoint 模式
    conda activate mzsz
    python /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/aggregate_dart.py \
      --base_dir /home/user/mzs_h/output/benchmark/glue \
      --dataset dart \
      --output /home/user/mzs_h/output/benchmark/glue_agg_dart

    # 模式 B：固定 step 模式（以 80000 为例）
    conda activate mzsz
    export DART_AGG_STEP=80000
    python /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/aggregate_dart.py \
      --base_dir /home/user/mzs_h/output/benchmark/glue/dart_seed87 \
      --output /home/user/mzs_h/output/benchmark/glue_agg_dart

你也可以在保存到当前目录后使用：
    python -m aggregate_dart  ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _is_dataset_dir(path: Path) -> bool:
    """
    简单判定一个目录是不是“数据集目录”：
    - 子目录里是否有 E* 前缀的实验目录
    """
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("E"):
            return True
    return False


def find_dataset_dirs(base_dir: Path, dataset_filter: Optional[str] = None) -> List[Path]:
    """
    在 base_dir 下寻找数据集目录。

    规则：
    - 如果 base_dir 自己看起来像数据集目录（里面直接就是 E*），那就只返回它自己；
    - 否则，在 base_dir 下找所有子目录，按名字过滤（包含 dataset_filter 即可）。
    """
    if _is_dataset_dir(base_dir):
        return [base_dir]

    dirs = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if dataset_filter and dataset_filter not in child.name:
            continue
        if _is_dataset_dir(child):
            dirs.append(child)
    return dirs


def find_experiment_dirs(dataset_dir: Path) -> List[Path]:
    """
    在一个数据集目录下寻找所有实验目录（简单地认为名称以 'E' 开头的子目录）。
    """
    exps = []
    for child in dataset_dir.iterdir():
        if child.is_dir() and child.name.startswith("E"):
            exps.append(child)
    return sorted(exps, key=lambda p: p.name)


def find_latest_trainer_state(exp_dir: Path) -> Optional[Path]:
    """
    在实验目录下找到 step 最大的 checkpoint-*/trainer_state.json。

    假设每个 trainer_state.json 都包含完整的 log_history（HuggingFace 默认行为）。
    """
    best_step = None
    best_path = None

    for child in exp_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-")[1])
        except (IndexError, ValueError):
            continue

        ts_path = child / "trainer_state.json"
        if not ts_path.is_file():
            continue

        if best_step is None or step > best_step:
            best_step = step
            best_path = ts_path

    return best_path


def load_log_history(trainer_state_path: Path) -> List[Dict]:
    """
    读取 trainer_state.json 的 log_history。
    """
    with trainer_state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("log_history", [])
    if not isinstance(history, list):
        raise ValueError(f"log_history is not a list in {trainer_state_path}")
    return history


def select_best_row_for_dart(history: List[Dict]) -> Optional[Dict]:
    """
    从 log_history 中选出 DART 的最佳 eval 行。

    只考虑包含 eval_bleu 的行，排序规则：

        eval_bleu  DESC
        eval_meteor DESC
        eval_chrf DESC
        eval_loss ASC
        step DESC
    """
    eval_rows = [r for r in history if "step" in r and "eval_bleu" in r]
    if not eval_rows:
        return None

    df = pd.DataFrame(eval_rows)

    # 补全列
    if "eval_meteor" not in df.columns:
        df["eval_meteor"] = -1.0
    if "eval_chrf" not in df.columns:
        df["eval_chrf"] = -1.0
    if "eval_loss" not in df.columns:
        df["eval_loss"] = float("inf")

    for col in ["eval_bleu", "eval_meteor", "eval_chrf", "eval_loss", "step"]:
        if col not in df.columns:
            # 如果连 eval_bleu 或 step 都缺，那就没法选
            df[col] = float("nan")

    df_sorted = df.sort_values(
        ["eval_bleu", "eval_meteor", "eval_chrf", "eval_loss", "step"],
        ascending=[False, False, False, True, False],
    )

    best_row = df_sorted.iloc[0].to_dict()
    return best_row


def select_fixed_step_row_for_dart(
    history: List[Dict], fixed_step: int
) -> Tuple[Optional[Dict], Optional[float]]:
    """
    在 log_history 中找到指定 step 的 eval 行（含 eval_bleu/chrf/meteor），
    并尝试同时找到该 step 的训练 loss。

    返回：
        (eval_row_dict 或 None, train_loss 或 None)
    """
    eval_rows = [
        r
        for r in history
        if r.get("step") == fixed_step
        and ("eval_bleu" in r or "eval_chrf" in r or "eval_meteor" in r)
    ]
    if not eval_rows:
        return None, None

    # 一般只有一行，取最后一行更保险
    eval_row = eval_rows[-1]

    train_loss = None
    train_rows = [r for r in history if r.get("step") == fixed_step and "loss" in r]
    if train_rows:
        train_loss = train_rows[-1].get("loss")

    return eval_row, train_loss


def aggregate_dataset_best(
    dataset_dir: Path, output_root: Path
) -> pd.DataFrame:
    """
    对单个数据集目录（例如 dart_seed87）进行“最佳 ckpt”聚合。
    """
    dataset_name = dataset_dir.name
    out_dir = output_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []

    exps = find_experiment_dirs(dataset_dir)
    if not exps:
        print(f"[WARN] No experiments found in dataset_dir={dataset_dir}")
        return pd.DataFrame()

    for exp_dir in exps:
        exp_name = exp_dir.name
        ts_path = find_latest_trainer_state(exp_dir)
        if ts_path is None:
            print(f"[WARN] No trainer_state.json found in {exp_dir}")
            continue

        try:
            history = load_log_history(ts_path)
        except Exception as e:
            print(f"[WARN] Failed to load log_history from {ts_path}: {e}")
            continue

        best_row = select_best_row_for_dart(history)
        if best_row is None:
            print(f"[WARN] No eval_bleu rows in {ts_path}")
            continue

        step = int(best_row.get("step", -1))
        rec = {
            "dataset": dataset_name,
            "experiment": exp_name,
            "step": step,
            "eval_bleu": best_row.get("eval_bleu"),
            "eval_chrf": best_row.get("eval_chrf"),
            "eval_meteor": best_row.get("eval_meteor"),
            "eval_loss": best_row.get("eval_loss"),
            "checkpoint_path": str(exp_dir / f"checkpoint-{step}"),
            "trainer_state_path": str(ts_path),
        }
        records.append(rec)

    if not records:
        print(f"[WARN] No valid records for dataset={dataset_name}")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # 排序：和选 best 的规则相同
    df = df.sort_values(
        ["eval_bleu", "eval_meteor", "eval_chrf", "eval_loss", "step"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)

    # 输出
    csv_path = out_dir / "dataset_summary_best.csv"
    json_path = out_dir / "dataset_summary_best.json"
    md_path = out_dir / "dataset_summary_best.md"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    # 简单 Markdown 表
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# DART summary (best ckpt) for dataset `{dataset_name}`\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"[INFO] Saved best summary for dataset={dataset_name} to {csv_path}")
    return df


def aggregate_dataset_fixed_step(
    dataset_dir: Path, output_root: Path, fixed_step: int
) -> pd.DataFrame:
    """
    对单个数据集目录（例如 dart_seed87）进行“固定 step 聚合”。

    对每个实验，从 log_history 中取 step==fixed_step 的 eval 行。
    """
    dataset_name = dataset_dir.name
    out_dir = output_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []

    exps = find_experiment_dirs(dataset_dir)
    if not exps:
        print(f"[WARN] No experiments found in dataset_dir={dataset_dir}")
        return pd.DataFrame()

    for exp_dir in exps:
        exp_name = exp_dir.name
        ts_path = find_latest_trainer_state(exp_dir)
        if ts_path is None:
            print(f"[WARN] No trainer_state.json found in {exp_dir}")
            continue

        try:
            history = load_log_history(ts_path)
        except Exception as e:
            print(f"[WARN] Failed to load log_history from {ts_path}: {e}")
            continue

        eval_row, train_loss = select_fixed_step_row_for_dart(history, fixed_step)
        if eval_row is None:
            print(
                f"[WARN] No eval row with step={fixed_step} in {ts_path} "
                f"(experiment={exp_name})"
            )
            continue

        step = int(eval_row.get("step", fixed_step))
        rec = {
            "dataset": dataset_name,
            "experiment": exp_name,
            "step": step,
            "eval_bleu": eval_row.get("eval_bleu"),
            "eval_chrf": eval_row.get("eval_chrf"),
            "eval_meteor": eval_row.get("eval_meteor"),
            "eval_loss": eval_row.get("eval_loss"),
            "train_loss": train_loss,
            "checkpoint_path": str(exp_dir / f"checkpoint-{step}"),
            "trainer_state_path": str(ts_path),
        }
        records.append(rec)

    if not records:
        print(
            f"[WARN] No valid fixed-step records for dataset={dataset_name}, "
            f"step={fixed_step}"
        )
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # 排序：主要按 eval_bleu DESC，其次 eval_meteor / eval_chrf DESC
    df = df.sort_values(
        ["eval_bleu", "eval_meteor", "eval_chrf", "experiment"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    # 输出
    stem = f"dataset_summary_step{fixed_step}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(
            f"# DART summary (fixed step={fixed_step}) for dataset "
            f"`{dataset_name}`\n\n"
        )
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(
        f"[INFO] Saved fixed-step summary for dataset={dataset_name}, "
        f"step={fixed_step} to {csv_path}"
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate DART experiments (best ckpt or fixed step)."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help=(
            "训练输出根目录，或者单个数据集目录。"
            "例如：/home/user/mzs_h/output/benchmark/glue "
            "或 /home/user/mzs_h/output/benchmark/glue/dart_seed87"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名字过滤（子串匹配），例如 dart 或 dart_seed87。",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "聚合输出根目录，每个数据集会在下面生成一个子目录，"
            "例如：/home/user/mzs_h/output/benchmark/glue_agg_dart"
        ),
    )
    parser.add_argument(
        "--fixed_step",
        type=int,
        default=None,
        help=(
            "固定 step 聚合模式：指定 step（例如 80000）。"
            "如不指定，则可通过环境变量 DART_AGG_STEP 控制。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 模式判定：优先使用命令行 --fixed_step，其次环境变量 DART_AGG_STEP
    fixed_step: Optional[int] = args.fixed_step
    if fixed_step is None:
        env_step = os.getenv("DART_AGG_STEP")
        if env_step is not None:
            try:
                fixed_step = int(env_step)
            except ValueError:
                raise SystemExit(
                    f"环境变量 DART_AGG_STEP={env_step!r} 不是合法整数"
                )

    if fixed_step is not None:
        mode = "fixed_step"
        print(f"[INFO] Running in FIXED-STEP mode, step={fixed_step}")
    else:
        mode = "best"
        print("[INFO] Running in BEST-CHECKPOINT mode")

    # 找到所有数据集目录
    dataset_dirs = find_dataset_dirs(base_dir, dataset_filter=args.dataset)

    if not dataset_dirs:
        raise SystemExit(
            f"No dataset dirs found under base_dir={base_dir} "
            f"with filter={args.dataset!r}"
        )

    all_dfs: List[pd.DataFrame] = []

    for ds_dir in dataset_dirs:
        print(f"[INFO] Processing dataset_dir={ds_dir}")
        if mode == "fixed_step":
            df = aggregate_dataset_fixed_step(ds_dir, output_root, fixed_step=fixed_step)  # type: ignore[arg-type]
        else:
            df = aggregate_dataset_best(ds_dir, output_root)
        if not df.empty:
            all_dfs.append(df)

    # 如果 base_dir 包含多个 DART 数据集，可以顺便输出一个全局总表
    if all_dfs:
        big_df = pd.concat(all_dfs, ignore_index=True)
        merged_csv = output_root / (
            "all_datasets_summary_step{}.csv".format(fixed_step)
            if mode == "fixed_step"
            else "all_datasets_summary_best.csv"
        )
        big_df.to_csv(merged_csv, index=False)
        print(f"[INFO] Saved merged summary to {merged_csv}")
    else:
        print("[WARN] No summary generated.")


if __name__ == "__main__":
    main()