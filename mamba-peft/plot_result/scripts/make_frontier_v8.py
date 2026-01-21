#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pareto frontier plots for parameter-performance trade-off.

v6 style:
  - Only subset variant (GLA uses ROUND_E12 subset).
  - All marks black; non-frontier as black 'x'.
  - Label only Pareto points; boundary-aware offsets; small font; white textbox.
  - Legend bottom-right: Pareto frontier vs Non-frontier.
  - X-axis log scale with percent tick labels (0.05%, 0.10%, ...).

Input:
  /mnt/data/unified_analysis/out_<model>/combined_tidy.csv
Output:
  /mnt/data/frontier_v7/variant_subset/<model>_parameter_performance_frontier.(png|pdf)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator, NullFormatter


ROUND_E12_GLA = {
    "QKVO_plus_G_plus_MLP",
    "QKVO_plus_MLP",
    "QKVO",
    "QKVO_plus_G",
    "MLPONLY",
    "OMLP",
    "QONLY",
    "KONLY",
    "VONLY",
    "OONLY",
    "KVONLY",
    "QVONLY",
    "VOONLY",
}


def pretty_operator_label(exp_norm: str) -> str:
    """Pretty label for Pareto points.

    The raw experiment names include adapter bookkeeping (ONLY, r8_alpha16, etc.).
    This function converts them into compact operator-style labels.
    """
    s = str(exp_norm)
    # Remove any leading experiment prefix
    s = __import__("re").sub(r"^E\d+_", "", s)
    # Strip LoRA suffixes
    s = __import__("re").sub(r"_r\d+_alpha\d+$", "", s)

    # Standardize separators
    s = s.replace("_plus_", "+").replace("_", "_")

    # Collapse ONLY variants
    only_map = {
        "QONLY": "Q",
        "KONLY": "K",
        "VONLY": "V",
        "OONLY": "O",
        "KVONLY": "KV",
        "QVONLY": "QV",
        "VOONLY": "VO",
        "MLPONLY": "MLP",
    }
    if s in only_map:
        return only_map[s]
    # Also handle cases like VONLY_r8_alpha16 that weren't stripped above
    for k, v in only_map.items():
        if s.startswith(k):
            return v

    # Compact a few common names
    s = s.replace("QKVO_plus_G_plus_MLP", "QKVO+G+MLP")
    s = s.replace("QKVO_plus_G", "QKVO+G")
    s = s.replace("QKVO_plus_MLP", "QKVO+MLP")

    return s


@dataclass
class FrontierPoint:
    exp_norm: str
    trainable_ratio: float
    mean_z: float


def task_zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _z(x: pd.Series) -> pd.Series:
        mu = x.mean()
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd <= 1e-12:
            return (x - mu) * 0.0
        return (x - mu) / sd

    out["z"] = out.groupby("task")["score"].transform(_z)
    return out


def aggregate_global(dfz: pd.DataFrame) -> pd.DataFrame:
    return (
        dfz.groupby("exp_norm", as_index=False)
        .agg(trainable_ratio=("trainable_ratio", "median"), mean_z=("z", "mean"))
        .dropna(subset=["trainable_ratio", "mean_z"])
    )


def pareto_frontier(agg: pd.DataFrame) -> List[FrontierPoint]:
    pts = agg.sort_values(["trainable_ratio", "mean_z"], ascending=[True, False]).reset_index(drop=True)
    best = -np.inf
    frontier: List[FrontierPoint] = []
    for _, r in pts.iterrows():
        y = float(r["mean_z"])
        if y > best + 1e-12:
            best = y
            frontier.append(FrontierPoint(str(r["exp_norm"]), float(r["trainable_ratio"]), y))
    return frontier


def adaptive_limits_logx(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    y = y[np.isfinite(y)]

    xmin, xmax = float(x.min()), float(x.max())
    # stronger right padding to keep labels inside
    xmin *= 0.55
    xmax *= 1.40

    ymin, ymax = float(y.min()), float(y.max())
    yr = ymax - ymin
    pad = 0.14 * (yr if yr > 1e-9 else 1.0)
    return (ymin - pad, ymax + pad), (xmin, xmax)


def percent_ticks(xmin: float, xmax: float) -> List[float]:
    """Choose a small set of nice log-spaced ticks and label as percentages."""
    candidates = np.array([5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    ticks = [float(v) for v in candidates if xmin <= v <= xmax]
    # fallback: logspace
    if len(ticks) < 3:
        ticks = list(np.geomspace(xmin, xmax, num=4))
    return ticks


def label_offsets_for_point(x: float, y: float, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> Tuple[int, int, str, str]:
    """Fixed label placement: always upper-left of the point."""
    dx, dy = -8, 8
    ha, va = "right", "bottom"
    return dx, dy, ha, va


def place_frontier_labels(ax: plt.Axes, pts: List[FrontierPoint], xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    """Place labels with simple overlap-avoidance and keep them inside the axes."""
    renderer = ax.figure.canvas.get_renderer()
    used = []

    for p in pts:
        dx, dy, ha, va = label_offsets_for_point(p.trainable_ratio, p.mean_z, xlim, ylim)
        txt = ax.annotate(
            pretty_operator_label(p.exp_norm),
            xy=(p.trainable_ratio, p.mean_z),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=10,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.9),
            clip_on=True,
            zorder=5,
        )

        ax.figure.canvas.draw()
        bb = txt.get_window_extent(renderer=renderer).expanded(1.02, 1.10)

        tries = 0
        while any(bb.overlaps(prev) for prev in used) and tries < 20:            # overlap nudge: only move further upward to keep consistent upper-left placement
            dy += 6
            txt.set_position((dx, dy))
            txt.set_va("bottom")
            ax.figure.canvas.draw()
            bb = txt.get_window_extent(renderer=renderer).expanded(1.02, 1.10)
            tries += 1

        used.append(bb)


def plot_one(model: str, combined_csv: str, out_dir: str) -> str:
    df = pd.read_csv(combined_csv)
    # merged_all_runs.csv contains all models
    df = df[df["model"] == model].copy()

    if df.empty:
        raise ValueError(f"No rows for model={model} in combined_csv={combined_csv}")

    dfz = task_zscore(df)
    agg = aggregate_global(dfz)

    frontier = pareto_frontier(agg)
    frontier_set = {p.exp_norm for p in frontier}

    non = agg[~agg["exp_norm"].isin(frontier_set)].copy()

    fx = np.array([p.trainable_ratio for p in frontier], dtype=float)
    fy = np.array([p.mean_z for p in frontier], dtype=float)

    nx = non["trainable_ratio"].to_numpy(dtype=float) if len(non) else np.array([], dtype=float)
    ny = non["mean_z"].to_numpy(dtype=float) if len(non) else np.array([], dtype=float)

    ylims, xlims = adaptive_limits_logx(
        np.concatenate([fx, nx]) if len(nx) else fx,
        np.concatenate([fy, ny]) if len(ny) else fy,
    )
    ymin, ymax = ylims
    xmin, xmax = xlims

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 6.2))

    # Non-frontier
    if len(non):
        alpha = 0.30 if len(non) >= 18 else 0.50
        ax.scatter(nx, ny, marker="x", s=50, linewidths=1.6, color="black", alpha=alpha, zorder=2)

    # Frontier
    ax.plot(fx, fy, color="black", linewidth=2.8, zorder=3)
    ax.scatter(fx, fy, s=140, facecolor="black", edgecolor="black", linewidths=0.0, zorder=4)

    # Axes
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Ticks as percentages (log spaced)
    xt = percent_ticks(xmin, xmax)
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.2f}%"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, which="major", linewidth=0.8, alpha=0.22)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.10)

    ax.set_xlabel("Trainable parameter ratio")
    ax.set_ylabel("Task-averaged standardized score")

    ax.set_title(f"{model.upper()} Parameter–Performance Frontier", pad=12)

    # Legend bottom-right
    legend_elems = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="black", markersize=7, linestyle="None", label="Pareto frontier"),
        Line2D([0], [0], marker="x", color="black", markersize=7, linestyle="None", label="Non-frontier"),
    ]
    ax.legend(
        handles=legend_elems,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        frameon=False,
        fontsize=9,
        handletextpad=0.6,
        borderpad=0.0,
    )

    # Labels (frontier only)
    fig.canvas.draw()
    place_frontier_labels(ax, frontier, (xmin, xmax), (ymin, ymax))

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{model}_parameter_performance_frontier_subset")
    fig.savefig(base + ".png", bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return base


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_csv", required=True, help="Path to merged_all_runs.csv (or combined_tidy.csv with required columns).")
    ap.add_argument("--out_dir", required=True, help="Output directory for frontier figures.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for model in ["gla", "retnet", "delta_net"]:
        plot_one(model=model, combined_csv=args.combined_csv, out_dir=args.out_dir)

    print(f"Done. Wrote figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
