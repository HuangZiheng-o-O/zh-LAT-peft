#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ICML-style task-wise z-score heatmaps for multiple models.

For each model prefix (e.g., delta_net, gla, retnet), the script loads all
available per-dataset CSV logs under --data_dir. For each dataset and
configuration, it selects the best checkpoint by the dataset's primary metric,
computes per-dataset z-scores (column-wise), and renders an annotated heatmap.

Formatting edits applied:
  - Drop the suffix "_r8_alpha16" from configuration labels.
  - Drop the suffix "_170k" from dataset labels (commonsense_170k -> commonsense).
  - Remove the word "ONLY" in configuration display names (e.g., VONLY -> V).
  - Use a serif (Times-like) font and STIX mathtext for an ICML-compatible style.

Model-specific filtering:
  - RetNet: only the user-specified configurations.
  - GLA: only the user-specified configurations (with one row dropped).
  - DeltaNet: keep all configurations found in the CSVs.
"""

from __future__ import annotations

import argparse
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_aliases import canonicalize_task_name, commonsense_name_candidates

# -------------------------
# ICML-ish typography
# -------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


DATASET_ORDER = [
    "cola",
    "commonsense_170k",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
]

# Preferred row ordering
PREFERRED_CONFIG_ORDER_RAW = [
    "KONLY_r8_alpha16",
    "OONLY_r8_alpha16",
    "QONLY_r8_alpha16",
    "VONLY_r8_alpha16",
    "KVONLY_r8_alpha16",
    "QVONLY_r8_alpha16",
    "VOONLY_r8_alpha16",
    "QKVO_r8_alpha16",
    "MLPONLY_r8_alpha16",
    "OMLP_r8_alpha16",
    "QKVO_plus_MLP_r8_alpha16",
    "QKVO_plus_G_r8_alpha16",
    "QKVO_plus_G_plus_MLP_r8_alpha16",
]

# RetNet allowed
RETNET_ALLOWED_RAW = {
    "QKVO_plus_G_plus_MLP_r8_alpha16",
    "QKVO_plus_G_r8_alpha16",
    "QKVO_plus_MLP_r8_alpha16",
    "OMLP_r8_alpha16",
    "VOONLY_r8_alpha16",
    "QKVO_r8_alpha16",
    "KVONLY_r8_alpha16",
    "OONLY_r8_alpha16",
    "VONLY_r8_alpha16",
    "MLPONLY_r8_alpha16",
    "QVONLY_r8_alpha16",
    "KONLY_r8_alpha16",
    "QONLY_r8_alpha16",
}

# GLA allowed (ROUND_E12)
GLA_ALLOWED_RAW = {
    "QKVO_plus_MLP_r8_alpha16",
    "QKVO_r8_alpha16",
    "MLPONLY_r8_alpha16",
    "OMLP_r8_alpha16",
    "QONLY_r8_alpha16",
    "KONLY_r8_alpha16",
    "VONLY_r8_alpha16",
    "OONLY_r8_alpha16",
    "KVONLY_r8_alpha16",
    "QVONLY_r8_alpha16",
    "VOONLY_r8_alpha16",
    "QKVO_plus_G_r8_alpha16",
}

METRIC_PRIORITY = [
    "eval_matthews_correlation",
    "eval_f1",
    "eval_accuracy",
    "eval_token_accuracy",
]


def infer_dataset_from_filename(fname: str, model: str) -> str:
    base = os.path.basename(fname)
    prefix = f"{model}_"
    if base.startswith(prefix):
        base = base[len(prefix):]
    if base.startswith("glue-tvt_"):
        return base.split("glue-tvt_", 1)[1].replace(".csv", "")
    return base.replace(".csv", "")



def normalize_config_name(cfg: str) -> str:
    """Normalize config strings so that legacy prefixes (e.g., E1_) do not create duplicate rows."""
    s = str(cfg)
    # Collapse legacy experiment prefixes like E1_, E2_, ...
    s = re.sub(r"^E\d+_", "", s)
    # Drop trailing LoRA-style suffixes (e.g., _r8_alpha16) so variants map to the same config.
    s = re.sub(r"_r\d+_alpha\d+$", "", s)
    return s


def _canonicalize_sequence(values) -> list[str]:
    seen = set()
    out: list[str] = []
    for v in values:
        canon = normalize_config_name(v)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def _canonicalize_set(values) -> set[str]:
    return set(_canonicalize_sequence(values))


PREFERRED_CONFIG_ORDER = _canonicalize_sequence(PREFERRED_CONFIG_ORDER_RAW)
RETNET_ALLOWED = _canonicalize_set(RETNET_ALLOWED_RAW)
GLA_ALLOWED = _canonicalize_set(GLA_ALLOWED_RAW)
GLA_DROP_CONFIG = normalize_config_name("QKVO_plus_G_plus_MLP_r8_alpha16")


def pick_primary_metric(df: pd.DataFrame) -> str:
    for m in METRIC_PRIORITY:
        if m in df.columns and df[m].notna().any():
            return m
    raise ValueError(f"No recognized metric found. Columns: {list(df.columns)}")


def filter_configs_for_model(model: str, configs: pd.Series) -> pd.Series:
    m = model.lower()
    if m == "retnet":
        return configs[configs.isin(RETNET_ALLOWED)]
    if m == "gla":
        return configs[configs.isin(GLA_ALLOWED)]
    return configs


def _maybe_append_commonsense_file(files: list[str], data_dir: Path, model: str) -> None:
    for fname in commonsense_name_candidates(model):
        path = data_dir / fname
        if path.exists():
            files.append(str(path))
            return


def read_best_scores_for_model(data_dir: Path, model: str) -> pd.DataFrame:
    files = sorted([str(p) for p in data_dir.glob(f"{model}_glue-tvt_*.csv")])
    _maybe_append_commonsense_file(files, data_dir, model)

    rows = []
    for path in files:
        df = pd.read_csv(path)
        dataset = canonicalize_task_name(infer_dataset_from_filename(path, model))
        metric = pick_primary_metric(df)

        df["config_raw"] = df["experiment"].astype(str).str.split("__").str[0]
        df["config"] = df["config_raw"].apply(normalize_config_name)
        keep = filter_configs_for_model(model, df["config"])
        df = df.loc[keep.index]
        df = df[df["config"].isin(keep.unique())]

        for cfg, g in df.groupby("config", sort=False):
            g2 = g.dropna(subset=[metric])
            if g2.empty:
                continue
            idx = g2[metric].idxmax()
            rows.append({
                "dataset": dataset,
                "config": cfg,
                "score": float(df.loc[idx, metric]),
            })

    return pd.DataFrame(rows)


def build_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    mat = long_df.pivot(index="config", columns="dataset", values="score")
    mat = mat.reindex(columns=[c for c in DATASET_ORDER if c in mat.columns])

    present = set(mat.index)
    ordered = [c for c in PREFERRED_CONFIG_ORDER if c in present]
    remaining = [c for c in mat.index if c not in ordered]
    mat = mat.reindex(index=ordered + remaining)
    return mat


def zscore_columns(mat: pd.DataFrame) -> pd.DataFrame:
    z = mat.astype(float).copy()
    for col in z.columns:
        mu = np.nanmean(z[col])
        sigma = np.nanstd(z[col])
        z[col] = 0.0 if sigma == 0 or not np.isfinite(sigma) else (z[col] - mu) / sigma
    return z


def display_dataset(name: str) -> str:
    return name.replace("_170k", "") if name.endswith("_170k") else name


def display_config(name: str) -> str:
    """
    Presentation-only normalization:
      - remove '_r8_alpha16'
      - collapse '*ONLY' -> '*'
    """
    s = normalize_config_name(name)
    if s.endswith("ONLY"):
        s = s.replace("ONLY", "")
    return s


def plot_heatmap(z: pd.DataFrame, out_png: Path, out_pdf: Path, title: str) -> None:
    n_rows, n_cols = z.shape
    fig_h = max(5.8, 0.42 * n_rows + 1.8)

    plt.figure(figsize=(10.8, fig_h))
    im = plt.imshow(z.values, cmap="viridis", aspect="auto")

    plt.xticks(range(n_cols), [display_dataset(c) for c in z.columns], rotation=30, ha="right")
    plt.yticks(range(n_rows), [display_config(r) for r in z.index])

    for i in range(n_rows):
        for j in range(n_cols):
            v = z.iat[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.2f}",
                         ha="center", va="center",
                         fontsize=10,
                         color="black" if abs(v) < 1.2 else "white")

    plt.xlabel("Dataset")
    plt.ylabel("Configuration")
    plt.title(title)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("z score")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default="delta_net,gla,retnet")
    ap.add_argument("--sort_rows", choices=["none","mean_z"], default="mean_z", help="Row ordering: none or by average z-score across tasks.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in [m.strip() for m in args.models.split(",")]:
        df = read_best_scores_for_model(data_dir, model)
        mat = build_matrix(df)

        # requested earlier: drop this row only for GLA
        if model.lower() == "gla":
            mat = mat.drop(index=[GLA_DROP_CONFIG], errors="ignore")

        z = zscore_columns(mat)
        if args.sort_rows == "mean_z":
            row_mean = z.mean(axis=1, skipna=True)
            z = z.loc[row_mean.sort_values(ascending=False).index]

        plot_heatmap(
            z,
            out_dir / f"fig2_heatmap_task_zscores_{model}.png",
            out_dir / f"fig2_heatmap_task_zscores_{model}.pdf",
            title="Task-wise performance as z scores",
        )

        print(f"[OK] {model}")

    print(f"Done. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
