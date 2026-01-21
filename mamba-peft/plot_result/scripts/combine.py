#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structured attribution comparison across models (single figure).

- Reads CSVs named: {model}_{task}.csv from a directory
  model in {retnet, gla, delta_net}
- Computes task-normalized scores, fits a main-effects ridge regression
  over module indicators (Q, K, V, O, G, GK, MLP).
- Uses bootstrap over tasks to estimate 95% CIs.
- Produces a cross-model comparison forest plot with y-offsets.

Usage:
  python structured_attribution_comparison.py \
    --data_dir "/Users/huangziheng/Documents/zotero附件/transformer改造/research_trackers/强code/my_fig_code/data" \
    --out_prefix "structured_attribution_comparison"

Outputs:
  structured_attribution_comparison.png
  structured_attribution_comparison.pdf
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Plot style (paper-ish)
# ----------------------------
mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 320,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ----------------------------
# Task metric mapping
# ----------------------------
PRIMARY_METRIC: Dict[str, str] = {
    "commonsense_170k": "eval_token_accuracy",
    "glue-tvt_cola": "eval_matthews_correlation",
    "glue-tvt_sst2": "eval_accuracy",
    "glue-tvt_rte": "eval_accuracy",
    "glue-tvt_qnli": "eval_accuracy",
    "glue-tvt_mnli": "eval_accuracy",
    "glue-tvt_qqp": "eval_f1",
    "glue-tvt_mrpc": "eval_f1",
}


# ----------------------------
# Module parsing from experiment name
# ----------------------------
KNOWN_MODULES: Dict[str, Set[str]] = {
    "QONLY": {"Q"},
    "KONLY": {"K"},
    "VONLY": {"V"},
    "OONLY": {"O"},
    "VOONLY": {"V", "O"},
    "KVONLY": {"K", "V"},
    "QVONLY": {"Q", "V"},
    "QOONLY": {"Q", "O"},
    "KOONLY": {"K", "O"},
    "QKONLY": {"Q", "K"},
    "QKVO": {"Q", "K", "V", "O"},
    "MLPONLY": {"MLP"},
    "MLPUPDOWN": {"MLP_updown"},
    "MLPGATEONLY": {"MLP_gate"},
    "OMLP": {"O", "MLP"},
    "GPROJONLY": {"G"},
    "GKONLY": {"GK"},
    "GATINGONLY": {"G", "GK"},
}
MODULE_ORDER = ["Q", "K", "V", "O", "G", "GK", "MLP", "MLP_updown", "MLP_gate"]


def normalize_experiment(name: str) -> str:
    # keep it conservative; remove known suffixes while preserving semantics
    s = str(name)
    s = s.replace("_r8_alpha16", "")
    s = s.replace("__rep1", "")
    return s


def parse_modules_fallback(exp: str) -> Set[str]:
    mods: Set[str] = set()

    # projections
    if exp.startswith("QKVO") or exp.startswith("QKVO_plus"):
        mods.update(["Q", "K", "V", "O"])
    elif "ONLY" in exp:
        prefix = exp.split("ONLY")[0].replace("_", "")
        for ch in prefix:
            if ch in "QKVO":
                mods.add(ch)
    else:
        for ch in exp:
            if ch in "QKVO":
                mods.add(ch)

    # MLP variants
    if "MLPUPDOWN" in exp:
        mods.add("MLP_updown")
    if "MLPGATEONLY" in exp:
        mods.add("MLP_gate")
    if "MLP" in exp and ("MLPONLY" in exp or "plus_MLP" in exp or "OMLP" in exp):
        mods.add("MLP")

    # Gates
    if "GK" in exp:
        mods.add("GK")
    if "GPROJONLY" in exp:
        mods.add("G")
    if re.search(r"(^|_)G($|_)", exp) or "plus_G" in exp or exp.endswith("plus_G"):
        mods.add("G")
    if "GATINGONLY" in exp:
        mods.update(["G", "GK"])

    return mods


def modules_from_exp(exp_norm: str) -> Set[str]:
    if exp_norm in KNOWN_MODULES:
        return set(KNOWN_MODULES[exp_norm])

    if exp_norm.startswith("QKVO_plus_"):
        mods = {"Q", "K", "V", "O"}
        rest = exp_norm[len("QKVO_plus_") :]
        rest = rest.replace("G_plus_GK_plus_MLP", "G+GK+MLP")
        rest = rest.replace("G_plus_GK", "G+GK")
        rest = rest.replace("G_plus_MLP", "G+MLP")
        parts = rest.split("_plus_")
        for p in parts:
            p = p.replace("_", "")
            if p in ("G", "GK", "MLP"):
                mods.add(p)
            elif p == "G+GK":
                mods.update(["G", "GK"])
            elif p == "G+MLP":
                mods.update(["G", "MLP"])
            elif p == "G+GK+MLP":
                mods.update(["G", "GK", "MLP"])
            else:
                for tok in re.split(r"[^A-Za-z]+", p):
                    if tok in ("G", "GK", "MLP"):
                        mods.add(tok)
        return mods

    if exp_norm.startswith("OMLP_plus_"):
        mods = {"O", "MLP"}
        rest = exp_norm[len("OMLP_plus_") :]
        rest = rest.replace("G_plus_GK", "G+GK")
        parts = rest.split("_plus_")
        for p in parts:
            p = p.replace("_", "")
            if p in ("G", "GK"):
                mods.add(p)
            elif p == "G+GK":
                mods.update(["G", "GK"])
        return mods

    return parse_modules_fallback(exp_norm)


# ----------------------------
# Loading and scoring
# ----------------------------
def infer_score_column(df: pd.DataFrame, task: str) -> str:
    # Prefer known primary metric; otherwise pick a reasonable eval_* column.
    if task in PRIMARY_METRIC and PRIMARY_METRIC[task] in df.columns:
        return PRIMARY_METRIC[task]

    candidates = [c for c in df.columns if c.startswith("eval_")]
    # remove losses
    candidates = [c for c in candidates if "loss" not in c.lower()]
    if candidates:
        # common preference order
        pref = ["eval_f1", "eval_accuracy", "eval_token_accuracy", "eval_matthews_correlation"]
        for p in pref:
            if p in candidates:
                return p
        return candidates[0]

    raise ValueError(f"No usable eval metric column found for task {task}. Columns: {list(df.columns)}")


def load_all_runs(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    pat = re.compile(r"^(gla|retnet|delta_net)_(.+)\.csv$")

    rows: List[pd.DataFrame] = []
    for fp in files:
        fn = os.path.basename(fp)
        m = pat.match(fn)
        if not m:
            continue
        model, task = m.group(1), m.group(2)
        d = pd.read_csv(fp)
        d["model"] = model
        d["task"] = task
        score_col = infer_score_column(d, task)
        d["score"] = pd.to_numeric(d[score_col], errors="coerce")
        d["exp_norm"] = d["experiment"].map(normalize_experiment)
        d["modules"] = d["exp_norm"].map(modules_from_exp)
        rows.append(d)

    if not rows:
        raise RuntimeError(f"No CSV files matched the pattern in: {data_dir}")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["score"])
    return out


def zscore_within_task(df: pd.DataFrame) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        if sd < 1e-12:
            return s * 0.0
        return (s - mu) / sd
    return df.groupby(["model", "task"])["score"].transform(_z)


# ----------------------------
# Ridge with bootstrap CIs
# ----------------------------
def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    XtX = X.T @ X
    p = XtX.shape[0]
    return np.linalg.solve(XtX + lam * np.eye(p), X.T @ y)


def ridge_bootstrap_ci(
    df: pd.DataFrame,
    model: str,
    terms: List[str],
    lam: float = 5.0,
    n_boot: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    d = df[df["model"] == model].copy()
    d["z"] = zscore_within_task(d)

    # aggregate per task per configuration to reduce step noise
    d = (
        d.groupby(["task", "exp_norm"], as_index=False)
        .agg(z=("z", "mean"), mods=("modules", "first"))
    )

    # design matrix: main effects only
    X = np.zeros((len(d), len(terms)), dtype=float)
    for i, mods in enumerate(d["mods"].tolist()):
        for j, t in enumerate(terms):
            X[i, j] = 1.0 if (t in mods) else 0.0
    y = d["z"].to_numpy(dtype=float)

    coef = ridge_fit(X, y, lam)

    # bootstrap over tasks
    rng = np.random.default_rng(seed)
    tasks = d["task"].unique().tolist()
    boot = np.zeros((n_boot, len(terms)), dtype=float)

    for b in range(n_boot):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        sampled = pd.concat([d[d["task"] == t] for t in sampled_tasks], ignore_index=True)

        Xb = np.zeros((len(sampled), len(terms)), dtype=float)
        for i, mods in enumerate(sampled["mods"].tolist()):
            for j, t in enumerate(terms):
                Xb[i, j] = 1.0 if (t in mods) else 0.0
        yb = sampled["z"].to_numpy(dtype=float)
        boot[b] = ridge_fit(Xb, yb, lam)

    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)

    out = pd.DataFrame({
        "model": model,
        "term": terms,
        "coef": coef,
        "ci95_lo": lo,
        "ci95_hi": hi,
    })
    out["abs"] = out["coef"].abs()
    return out


# ----------------------------
# Plot cross-model comparison
# ----------------------------
def plot_structured_attribution_comparison(
    ridges: List[pd.DataFrame],
    out_prefix: str,
    top_k: int = 7,
) -> None:
    merged = pd.concat(ridges, ignore_index=True)

    # pick top terms by mean absolute coefficient across models
    score = merged.groupby("term")["coef"].apply(lambda s: float(np.mean(np.abs(s)))).sort_values(ascending=False)
    top_terms = score.head(top_k).index.tolist()

    models = ["retnet", "gla", "delta_net"]
    label_map = {"retnet": "RETNET", "gla": "GLA", "delta_net": "DELTA_NET"}

    # y-offsets
    offsets = {"retnet": -0.22, "gla": 0.0, "delta_net": 0.22}
    colors = {"retnet": "#0072B2", "gla": "#D55E00", "delta_net": "#009E73"}

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    y0 = np.arange(len(top_terms))

    for m in models:
        sub = merged[(merged["model"] == m) & (merged["term"].isin(top_terms))].set_index("term").reindex(top_terms)
        y = y0 + offsets[m]
        ax.hlines(y, sub["ci95_lo"], sub["ci95_hi"], linewidth=1.6, color=colors[m], alpha=0.95)
        ax.scatter(sub["coef"], y, s=60, color=colors[m], label=label_map[m], zorder=3)

    ax.axvline(0.0, color="#666666", linewidth=1.2)
    ax.set_yticks(y0)
    ax.set_yticklabels(top_terms)
    ax.set_xlabel("Ridge coefficient")
    ax.set_title("Structured attribution comparison across models")

    # Legend bottom-left as requested
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=320)
    fig.savefig(f"{out_prefix}.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing {model}_{task}.csv files")
    ap.add_argument("--out_prefix", required=True, help="Output file prefix without extension")
    ap.add_argument("--lam", type=float, default=5.0, help="Ridge regularization strength")
    ap.add_argument("--n_boot", type=int, default=200, help="Bootstrap samples")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--top_k", type=int, default=7, help="Number of terms to display")
    args = ap.parse_args()

    df = load_all_runs(args.data_dir)

    # Main effects only, no interaction terms (no colon terms)
    terms = ["Q", "K", "V", "O", "G", "GK", "MLP"]

    ridges = []
    for m in ["retnet", "gla", "delta_net"]:
        ridges.append(ridge_bootstrap_ci(df, m, terms=terms, lam=args.lam, n_boot=args.n_boot, seed=args.seed))

    plot_structured_attribution_comparison(ridges, out_prefix=args.out_prefix, top_k=args.top_k)


if __name__ == "__main__":
    main()
