#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate combined_tidy.csv from the per-task CSVs in dataset_summaries2.

This reproduces the unified, tidy table used by the frontier/heatmap/ridge analyses.

Inputs (default):
  /mnt/data/dataset_summaries2/*.csv

Output (default):
  /mnt/data/combined_tidy.csv

Schema (key fields):
  model, task, exp, exp_norm, trainable_ratio, trainable_params, total_params,
  primary_metric, score, plus any metric columns present in the source CSVs.

Primary metric policy (matches existing combined_tidy.csv):
  - glue-tvt_cola   -> eval_matthews_correlation
  - glue-tvt_mrpc   -> eval_f1
  - glue-tvt_qqp    -> eval_f1
  - commonsense_170k -> eval_token_accuracy
  - otherwise       -> eval_accuracy
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List

import pandas as pd

from dataset_aliases import canonicalize_task_name

TASK_PRIMARY: Dict[str, str] = {
    "glue-tvt_cola": "eval_matthews_correlation",
    "glue-tvt_mrpc": "eval_f1",
    "glue-tvt_qqp": "eval_f1",
    "commonsense_170k": "eval_token_accuracy",
}


def infer_model_and_task(path: str) -> tuple[str, str]:
    """Infer model and task from filename like: gla_glue-tvt_sst2.csv"""
    base = os.path.basename(path)
    m = re.match(r"^(gla|retnet|delta_net)_(.+)\.csv$", base)
    if not m:
        raise ValueError(f"Unrecognized filename pattern: {base}")
    return m.group(1), m.group(2)


def normalize_exp_name(exp: str) -> str:
    """Remove leading E<digits>_ prefix if present."""
    return re.sub(r"^E\d+_", "", str(exp))


def choose_primary_metric(task: str, row: pd.Series) -> str:
    if task in TASK_PRIMARY:
        return TASK_PRIMARY[task]
    # fall back
    if "eval_accuracy" in row.index and pd.notna(row.get("eval_accuracy")):
        return "eval_accuracy"
    # last resort: pick the first eval_* column
    for c in row.index:
        if c.startswith("eval_"):
            return c
    raise ValueError(f"No eval_* metric columns found for task={task}")


def build_combined(input_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files match {input_glob}")

    frames: List[pd.DataFrame] = []
    for p in paths:
        model, task = infer_model_and_task(p)
        task = canonicalize_task_name(task)
        df = pd.read_csv(p)

        # Harmonize column names across sources
        if "experiment" in df.columns and "exp" not in df.columns:
            df = df.rename(columns={"experiment": "exp"})

        # required columns sanity
        required = ["exp", "trainable_ratio", "trainable_params", "total_params"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"{os.path.basename(p)} missing required column: {c}")

        df = df.copy()
        df["model"] = model
        df["task"] = task
        df["exp_norm"] = df["exp"].map(normalize_exp_name)

        # choose primary metric row-wise and compute score
        primary = []
        score = []
        for _, r in df.iterrows():
            pm = choose_primary_metric(task, r)
            primary.append(pm)
            val = r.get(pm)
            score.append(val)
        df["primary_metric"] = primary
        df["score"] = score

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # consistent column order: put key fields first, keep remaining as-is
    front = [
        "model",
        "task",
        "exp",
        "exp_norm",
        "trainable_ratio",
        "trainable_params",
        "total_params",
        "primary_metric",
        "score",
    ]
    cols = front + [c for c in out.columns if c not in set(front)]
    out = out[cols]

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", default="/Users/huangziheng/Documents/zotero附件/transformer改造/research_trackers/dataset_summaries2/*.csv")
    ap.add_argument("--output", default="/Users/huangziheng/Documents/zotero附件/transformer改造/research_trackers/combined_tidy.csv")
    args = ap.parse_args()

    df = build_combined(args.input_glob)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
