#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared utilities for publication-quality figures for LoRA target sweeps.

This module is intentionally dependency-light (numpy/pandas/matplotlib only)
so it can run in minimal environments.

Data convention
- Input CSV file name: {model}_{task}.csv
  model in {gla, retnet, delta_net}

Key derived fields
- score: task primary metric
- exp_norm: experiment name normalized across tasks
- modules: parsed set of targeted modules
- mods_key: canonical '+''-joined module list
- category: coarse family label for plotting

Operator signature convention
- Read   : Q present
- Write  : K or V present
- Forget : G or GK present
- Post   : O or any MLP variant present

The signature is encoded as R{0/1}W{0/1}F{0/1}P{0/1}.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from dataset_aliases import canonicalize_task_name


# ----------------------
# Style
# ----------------------

def set_rcparams() -> None:
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


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

PALETTE = {
    "proj_only": "#0072B2",
    "proj+gate": "#D55E00",
    "proj+mlp": "#009E73",
    "proj+gate+mlp": "#CC79A7",
    "other": "#4D4D4D",
}

MARKERS = {
    "proj_only": "o",
    "proj+gate": "s",
    "proj+mlp": "^",
    "proj+gate+mlp": "D",
    "other": "X",
}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def normalize_experiment(name: str) -> str:
    name = name.replace("_r8_alpha16", "")
    name = name.replace("__rep1", "")
    return name


def parse_modules_fallback(exp: str) -> Set[str]:
    mods: Set[str] = set()

    # Base projections
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


def modules_from_exp(exp: str) -> Set[str]:
    if exp in KNOWN_MODULES:
        return set(KNOWN_MODULES[exp])

    if exp.startswith("QKVO_plus_"):
        mods = {"Q", "K", "V", "O"}
        rest = exp[len("QKVO_plus_") :]
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

    if exp.startswith("OMLP_plus_"):
        mods = {"O", "MLP"}
        rest = exp[len("OMLP_plus_") :]
        rest = rest.replace("G_plus_GK", "G+GK")
        parts = rest.split("_plus_")
        for p in parts:
            p = p.replace("_", "")
            if p in ("G", "GK"):
                mods.add(p)
            elif p == "G+GK":
                mods.update(["G", "GK"])
        return mods

    return parse_modules_fallback(exp)


def mods_key(mods: Set[str]) -> str:
    return "+".join([m for m in MODULE_ORDER if m in mods]) if mods else ""


def parse_rank(exp_name: str) -> int:
    if not isinstance(exp_name, str):
        return 0
    m = re.search(r"_r(\d+)", exp_name)
    return int(m.group(1)) if m else 0


def category_from_mods(mods: Set[str]) -> str:
    has_proj = len(mods.intersection({"Q", "K", "V", "O"})) > 0
    has_gate = len(mods.intersection({"G", "GK"})) > 0
    has_mlp = ("MLP" in mods) or ("MLP_updown" in mods) or ("MLP_gate" in mods)

    if not has_proj:
        return "other"
    if has_gate and has_mlp:
        return "proj+gate+mlp"
    if has_gate:
        return "proj+gate"
    if has_mlp:
        return "proj+mlp"
    return "proj_only"


def load_all_runs(in_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    pat = re.compile(r"^(gla|retnet|delta_net)_(.+)\.csv$")

    rows: List[pd.DataFrame] = []
    for fp in files:
        fn = os.path.basename(fp)
        m = pat.match(fn)
        if not m:
            continue
        model, task = m.group(1), canonicalize_task_name(m.group(2))
        d = pd.read_csv(fp)
        d["model"] = model
        d["task"] = task
        rows.append(d)

    if not rows:
        raise RuntimeError(f"No CSV files matched pattern in {in_dir}")

    df = pd.concat(rows, ignore_index=True)

    def _score_row(r: pd.Series) -> float:
        col = PRIMARY_METRIC.get(r["task"])
        return float(r.get(col, np.nan)) if col else np.nan

    df["score"] = df.apply(_score_row, axis=1)
    df["exp_norm"] = df["experiment"].map(normalize_experiment)
    df["modules"] = df["exp_norm"].map(modules_from_exp)
    df["mods_key"] = df["modules"].map(mods_key)
    df["rank"] = df["exp_norm"].map(parse_rank)
    df["category"] = df["modules"].map(category_from_mods)
    return df


def zscore_within_task(df: pd.DataFrame) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        if sd < 1e-12:
            return s * 0.0
        return (s - mu) / sd

    return df.groupby(["model", "task"])["score"].transform(_z)


def minmax_within_task(df: pd.DataFrame) -> pd.Series:
    def _mm(s: pd.Series) -> pd.Series:
        lo = float(s.min())
        hi = float(s.max())
        if abs(hi - lo) < 1e-12:
            return s * 0.0
        return (s - lo) / (hi - lo)

    return df.groupby(["model", "task"])["score"].transform(_mm)


def compute_agg(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["z"] = zscore_within_task(d)
    d["mm"] = minmax_within_task(d)

    agg = (
        d.groupby(["model", "exp_norm", "mods_key", "category"], as_index=False)
        .agg(
            trainable_ratio=("trainable_ratio", "mean"),
            trainable_params=("trainable_params", "mean"),
            total_params=("total_params", "mean"),
            mean_z=("z", "mean"),
            mean_mm=("mm", "mean"),
            n_tasks=("task", "nunique"),
            rank=("rank", "max"),
        )
    )
    return agg


def pareto_front_xmin_ymax(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    best_y = -np.inf
    keep = np.zeros_like(x, dtype=bool)
    for idx in order:
        if y[idx] > best_y + 1e-12:
            keep[idx] = True
            best_y = y[idx]
    return keep


def set_percent_log_ticks(ax):
    xt = np.array([5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    ax.set_xticks(xt)
    ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))


# ----------------------
# Per-model figures
# ----------------------

def plot_pareto_scatter(agg: pd.DataFrame, model: str, out_dir: str) -> pd.DataFrame:
    a = agg[agg["model"] == model].copy().sort_values("trainable_ratio")

    x = a["trainable_ratio"].to_numpy()
    y = a["mean_z"].to_numpy()
    keep = pareto_front_xmin_ymax(x, y)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    label_map = {
        "proj_only": "Projection only",
        "proj+gate": "Projection and gating",
        "proj+mlp": "Projection and MLP",
        "proj+gate+mlp": "Projection, gating, and MLP",
        "other": "Other",
    }

    for cat in ["proj_only", "proj+gate", "proj+mlp", "proj+gate+mlp", "other"]:
        sub = a[a["category"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub["trainable_ratio"],
            sub["mean_z"],
            s=26 + 6 * np.clip(sub["rank"], 0, 16),
            marker=MARKERS.get(cat, "o"),
            c=PALETTE.get(cat, "#4D4D4D"),
            edgecolors="white",
            linewidths=0.7,
            alpha=0.92,
            label=label_map[cat],
        )

    front = a[keep].sort_values("trainable_ratio")
    ax.plot(front["trainable_ratio"], front["mean_z"], linewidth=2.2, color="black", alpha=0.9, label="Pareto frontier")

    top = front.sort_values("mean_z", ascending=False).head(6)
    for _, r in top.iterrows():
        ax.scatter([r["trainable_ratio"]], [r["mean_z"]], s=105, facecolors="none", edgecolors="black", linewidths=1.1)
        ax.annotate(r["exp_norm"], (r["trainable_ratio"], r["mean_z"]), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameter ratio")
    ax.set_ylabel("Mean task z score")
    ax.set_title(f"{model.upper()} LoRA target sweep: Pareto frontier of performance and budget")
    set_percent_log_ticks(ax)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{model}_pareto_frontier.pdf"))
    fig.savefig(os.path.join(out_dir, f"{model}_pareto_frontier.png"), dpi=320)
    plt.close(fig)

    return front


def compute_best_by_param_bin(df: pd.DataFrame) -> pd.DataFrame:
    param_bins = [0, 900_000, 1_800_000, 2_600_000, 3_600_000, 5_200_000, 7_200_000, 10_000_000]
    labels = [f"{param_bins[i]/1e6:.1f}-{param_bins[i+1]/1e6:.1f}M" for i in range(len(param_bins) - 1)]

    d = df.copy()
    d["param_bin"] = pd.cut(d["trainable_params"], bins=param_bins, labels=labels, include_lowest=True)
    d["score_norm"] = minmax_within_task(d)

    best = (
        d.sort_values(["model", "task", "param_bin", "score_norm"], ascending=[True, True, True, False])
        .groupby(["model", "task", "param_bin"], as_index=False)
        .head(1)
    )
    return best


def plot_best_by_bin_heatmap(best: pd.DataFrame, model: str, out_dir: str) -> None:
    sub = best[best["model"] == model].copy()

    bins = list(sub["param_bin"].cat.categories)
    tasks = [
        "commonsense_170k",
        "glue-tvt_cola",
        "glue-tvt_mnli",
        "glue-tvt_mrpc",
        "glue-tvt_qnli",
        "glue-tvt_qqp",
        "glue-tvt_rte",
        "glue-tvt_sst2",
    ]
    tasks = [t for t in tasks if t in sub["task"].unique()]

    pivot = sub.pivot_table(index="task", columns="param_bin", values="score_norm", aggfunc="max").reindex(index=tasks, columns=bins)

    fig, ax = plt.subplots(figsize=(9.0, 5.1))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap="cividis", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(bins, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels([t.replace("glue-tvt_", "GLUE ").replace("commonsense_170k", "Commonsense 170K") for t in tasks])

    ax.set_title(f"{model.upper()} top normalized score by trainable parameter budget")
    ax.set_xlabel("Trainable parameter budget")
    ax.set_ylabel("Task")

    for i in range(len(tasks)):
        for j in range(len(bins)):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                continue
            ax.text(j, i, f"{float(val):.2f}", ha="center", va="center", fontsize=9, color="white")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Normalized score")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{model}_best_by_budget_heatmap.pdf"))
    fig.savefig(os.path.join(out_dir, f"{model}_best_by_budget_heatmap.png"), dpi=320)
    plt.close(fig)


# ----------------------
# Structured attribution: ridge with bootstrap CI
# ----------------------

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    XtX = X.T @ X
    p = XtX.shape[0]
    beta = np.linalg.solve(XtX + lam * np.eye(p), X.T @ y)
    return beta


def ridge_bootstrap_ci(df: pd.DataFrame, model: str, lam: float = 5.0, n_boot: int = 50, seed: int = 42) -> pd.DataFrame:
    d = df[df["model"] == model].copy()
    d["z"] = zscore_within_task(d)

    d = d.groupby(["task", "exp_norm"], as_index=False).agg(z=("z", "mean"), mods=("modules", "first"))

    terms = ["Q", "K", "V", "O", "G", "GK", "MLP"]
    X = np.zeros((len(d), len(terms)), dtype=float)
    for i, mods in enumerate(d["mods"].tolist()):
        for j, t in enumerate(terms):
            X[i, j] = 1.0 if (t in mods) else 0.0

    y = d["z"].to_numpy(dtype=float)
    coef = ridge_fit(X, y, lam)

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
        "term": terms,
        "coef": coef,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "model": model,
    })

    out["abs"] = out["coef"].abs()
    out = out.sort_values("abs", ascending=False)
    return out


def plot_ridge_forest(ridge: pd.DataFrame, model: str, out_dir: str) -> None:
    d = ridge.copy()
    d = d.sort_values("abs", ascending=False).head(7).sort_values("coef", ascending=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    y = np.arange(len(d))
    ax.hlines(y, d["ci95_lo"], d["ci95_hi"], color="#333333", linewidth=1.6)
    ax.scatter(d["coef"], y, color="#111111", s=34, zorder=3)
    ax.axvline(0, color="#666666", linewidth=1.1)

    ax.set_yticks(y)
    ax.set_yticklabels(d["term"].tolist())
    ax.set_xlabel("Ridge coefficient")
    ax.set_title(f"{model.upper()} structured attribution from ridge regression")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{model}_structured_attribution.pdf"))
    fig.savefig(os.path.join(out_dir, f"{model}_structured_attribution.png"), dpi=320)
    plt.close(fig)


def plot_ridge_comparison(ridges: List[pd.DataFrame], out_root: str) -> None:
    common = set(ridges[0]["term"].tolist())
    for r in ridges[1:]:
        common &= set(r["term"].tolist())
    common = sorted(common)
    if not common:
        return

    merged = pd.concat(ridges, ignore_index=True)
    merged = merged[merged["term"].isin(common)].copy()

    score = merged.groupby("term")["coef"].apply(lambda s: float(np.mean(np.abs(s)))).sort_values(ascending=False)
    top_terms = score.head(7).index.tolist()

    models = ["retnet", "gla", "delta_net"]
    offsets = {"retnet": -0.22, "gla": 0.0, "delta_net": 0.22}
    colors = {"retnet": "#0072B2", "gla": "#D55E00", "delta_net": "#009E73"}

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    y0 = np.arange(len(top_terms))

    for m in models:
        sub = merged[(merged["model"] == m) & (merged["term"].isin(top_terms))].set_index("term").reindex(top_terms)
        y = y0 + offsets[m]
        ax.hlines(y, sub["ci95_lo"], sub["ci95_hi"], linewidth=1.3, color=colors[m], alpha=0.9)
        ax.scatter(sub["coef"], y, s=30, color=colors[m], label=m.upper(), zorder=3)

    ax.axvline(0, color="#666666", linewidth=1.1)
    ax.set_yticks(y0)
    ax.set_yticklabels(top_terms)
    ax.set_xlabel("Ridge coefficient")
    ax.set_title("Structured attribution comparison across models")
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "structured_attribution_comparison.pdf"))
    fig.savefig(os.path.join(out_root, "structured_attribution_comparison.png"), dpi=320)
    plt.close(fig)


# ----------------------
# Operator signatures
# ----------------------

def operator_signature(mods: Set[str]) -> str:
    R = 1 if "Q" in mods else 0
    W = 1 if ("K" in mods or "V" in mods) else 0
    F = 1 if ("G" in mods or "GK" in mods) else 0
    P = 1 if ("O" in mods or "MLP" in mods or "MLP_updown" in mods or "MLP_gate" in mods) else 0
    return f"R{R}W{W}F{F}P{P}"


def operator_sig_to_label(sig: str) -> str:
    if not isinstance(sig, str) or not re.match(r"R\dW\dF\dP\d", sig):
        return str(sig)
    parts = []
    if sig[1] == "1":
        parts.append("Read")
    if sig[3] == "1":
        parts.append("Write")
    if sig[5] == "1":
        parts.append("Forget")
    if sig[7] == "1":
        parts.append("Post")
    return "+".join(parts) if parts else "None"


def compute_operator_freq(frontiers: Dict[str, pd.DataFrame], agg: pd.DataFrame) -> pd.DataFrame:
    rows: List[Tuple[str, str]] = []
    for model, front in frontiers.items():
        a = agg[agg["model"] == model].set_index("exp_norm")
        for exp in front["exp_norm"].tolist():
            if exp not in a.index:
                continue
            mk = str(a.loc[exp, "mods_key"])
            mods = set(mk.split("+")) if mk else set()
            rows.append((model, operator_signature(mods)))

    df = pd.DataFrame(rows, columns=["model", "sig"])
    if df.empty:
        return df

    out = df.groupby(["model", "sig"], as_index=False).size().rename(columns={"size": "count"})
    out["label"] = out["sig"].map(operator_sig_to_label)
    out["share"] = out.groupby("model")["count"].transform(lambda s: s / max(int(s.sum()), 1))
    return out


def plot_operator_signature_combined(freq: pd.DataFrame, out_root: str) -> None:
    if freq.empty:
        return

    models = ["retnet", "gla", "delta_net"]
    colors = {"retnet": "#0072B2", "gla": "#D55E00", "delta_net": "#009E73"}

    labels = sorted(freq["label"].unique())
    mat = np.zeros((len(models), len(labels)), dtype=float)
    for i, m in enumerate(models):
        sub = freq[freq["model"] == m].set_index("label")
        for j, lab in enumerate(labels):
            mat[i, j] = float(sub.loc[lab, "share"]) if lab in sub.index else 0.0

    order = np.argsort(mat.sum(axis=0))[::-1]
    labels = [labels[i] for i in order]
    mat = mat[:, order]

    labels = labels[:10]
    mat = mat[:, :10]

    y = np.arange(len(labels))
    bar_h = 0.22
    offs = {"retnet": -bar_h, "gla": 0.0, "delta_net": bar_h}

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    for i, m in enumerate(models):
        ax.barh(y + offs[m], mat[i], height=bar_h, color=colors[m], alpha=0.92, label=m.upper())

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Share of Pareto frontier configurations")
    ax.set_title("Operator signature distribution on the Pareto frontier")
    ax.legend(frameon=False, loc="lower right")

    for i, m in enumerate(models):
        for j in range(len(labels)):
            v = mat[i, j]
            if v <= 0:
                continue
            ax.text(v + 0.006, y[j] + offs[m], f"{v*100:.1f}%", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "operator_signature_comparison.pdf"))
    fig.savefig(os.path.join(out_root, "operator_signature_comparison.png"), dpi=320)
    plt.close(fig)
