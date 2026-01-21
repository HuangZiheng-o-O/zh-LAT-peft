#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def primary_score(dataset: str, df: pd.DataFrame) -> pd.Series:
    ds = dataset.lower()
    # CommonSense mixture
    if "commonsense" in ds:
        return df["eval_token_accuracy"]
    # GLUE
    if ds.endswith("cola"):
        return df["eval_matthews_correlation"]
    if ds.endswith("mrpc") or ds.endswith("qqp"):
        return df["eval_f1"]
    return df["eval_accuracy"]


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Pareto front for (min trainable_ratio, max score)."""
    s = df.sort_values("trainable_ratio", ascending=True).reset_index(drop=True)
    best = -np.inf
    keep = []
    for _, r in s.iterrows():
        if r["score"] > best + 1e-12:
            keep.append(r)
            best = float(r["score"])
    return pd.DataFrame(keep)


@dataclass
class ModelSpec:
    name: str
    pattern: str
    modules: List[str]


def parse_flags(experiment: str, modules: List[str]) -> Dict[str, int]:
    """Parse module inclusion from experiment name using conservative string rules."""
    e = experiment
    base = e.replace("_r8_alpha16", "")

    flags = {m: 0 for m in modules}

    # Helper: set if present
    def set_if(m: str, cond: bool):
        if m in flags and cond:
            flags[m] = 1

    # Handle explicit ONLY patterns
    # Basic atoms
    set_if("Q", base.startswith("QONLY"))
    set_if("K", base.startswith("KONLY"))
    set_if("V", base.startswith("VONLY"))
    set_if("O", base.startswith("OONLY"))

    # Pairs
    if base.startswith("QVONLY"):
        set_if("Q", True)
        set_if("V", True)
    if base.startswith("KVONLY"):
        set_if("K", True)
        set_if("V", True)
    if base.startswith("VOONLY"):
        set_if("V", True)
        set_if("O", True)
    if base.startswith("QKONLY"):
        set_if("Q", True)
        set_if("K", True)
    if base.startswith("QOONLY"):
        set_if("Q", True)
        set_if("O", True)
    if base.startswith("KOONLY"):
        set_if("K", True)
        set_if("O", True)

    # MLP variants
    # MLPONLY, MLPUPDOWN, MLPGATEONLY, *_plus_MLP
    if "MLP" in base:
        set_if("MLP", True)

    # O+MLP shorthand
    if base.startswith("OMLP"):
        set_if("O", True)
        set_if("MLP", True)

    # Full QKVO family
    if base.startswith("QKVO"):
        set_if("Q", True)
        set_if("K", True)
        set_if("V", True)
        set_if("O", True)

    # RetNet gate (G) and GLA gates (G, GK)
    # plus_G, GPROJONLY, GATINGONLY, etc.
    if "plus_G" in base or base.startswith("GPROJONLY") or base.startswith("GATINGONLY"):
        set_if("G", True)

    # GLA recurrent gate gk
    if "GK" in base:
        set_if("GK", True)

    # GLA gate-only variants that do not include 'plus_G'
    if base.startswith("GKONLY"):
        set_if("GK", True)
    if base.startswith("GPROJONLY"):
        set_if("G", True)
    if base.startswith("GATINGONLY"):
        # Typically indicates gating branch; treat as both if GK exists
        set_if("G", True)
        if "GK" in flags:
            set_if("GK", True)

    return flags


def load_family(csv_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No CSVs matched: {csv_glob}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        dataset = os.path.basename(p).replace(".csv", "")
        # strip leading family name if present
        # e.g., retnet_glue-tvt_mnli -> glue-tvt_mnli
        for prefix in ("retnet_", "gla_", "delta_net_"):
            if dataset.startswith(prefix):
                dataset = dataset[len(prefix):]
                break
        df = df.copy()
        df["dataset"] = dataset
        df["score"] = primary_score(dataset, df)
        frames.append(df[["dataset", "experiment", "trainable_ratio", "score"]])
    return pd.concat(frames, ignore_index=True)


def plot_incidence(P: pd.DataFrame, modules: List[str], out_png: str, out_pdf: str, title: str):
    n = len(P)
    if n == 0:
        raise RuntimeError("No Pareto points found.")

    counts = {m: 0 for m in modules}
    for exp in P["experiment"].tolist():
        flags = parse_flags(exp, modules)
        for m in modules:
            counts[m] += int(flags.get(m, 0))

    shares = [counts[m] / n for m in modules]

    fig = plt.figure(figsize=(5.2, 5.2), dpi=220)
    plt.bar(modules, shares)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Share of Pareto points")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return n, {m: float(s) for m, s in zip(modules, shares)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Directory containing CSVs for all families (e.g., extracted data folder).")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    specs = [
        ModelSpec("retnet", os.path.join(args.data_dir, "retnet_*.csv"), ["O", "V", "K", "Q", "MLP", "G"]),
        ModelSpec("gla", os.path.join(args.data_dir, "gla_*.csv"), ["O", "V", "K", "Q", "MLP", "G", "GK"]),
        ModelSpec("delta_net", os.path.join(args.data_dir, "delta_net_*.csv"), ["O", "V", "K", "Q", "MLP"]),
    ]

    summary_rows = []

    for spec in specs:
        D = load_family(spec.pattern)
        pareto_points = []
        for ds, g in D.groupby("dataset", sort=True):
            f = pareto_front(g)
            f["dataset"] = ds
            pareto_points.append(f)
        P = pd.concat(pareto_points, ignore_index=True)

        out_png = os.path.join(args.out_dir, f"fig_module_incidence_pareto_{spec.name}.png")
        out_pdf = os.path.join(args.out_dir, f"fig_module_incidence_pareto_{spec.name}.pdf")
        title = f"Module incidence on Pareto frontier: {spec.name}"

        n_pareto, shares = plot_incidence(P, spec.modules, out_png, out_pdf, title)

        row = {"model": spec.name, "pareto_points": n_pareto}
        for m in spec.modules:
            row[f"share_{m}"] = shares[m]
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "module_incidence_summary.csv"), index=False)


if __name__ == "__main__":
    main()
