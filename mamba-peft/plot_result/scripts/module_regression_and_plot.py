#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""End-to-end: fit a controlled OLS and plot module coefficients.

This produces the *single* module-effects figure (point estimate + robust 95% CI)
used for mechanism-oriented analysis.

Inputs
------
/mnt/data/linear_attention_lora_analysis/merged_long.csv
  Required columns:
    - metric_z (within-task standardized metric; see Notes)
    - trainable_ratio
    - task
    - model
    - Q,K,V,O,G,GK,MLP_gate,MLP_updown (0/1 indicators)

Outputs
-------
/mnt/data/module_effects_scientific.(png|pdf)
/mnt/data/regression_module_effects_hc3_recomputed.csv

Notes: within-task normalization
------------------------------
For each task, we standardize the chosen evaluation metric across *all* rows
(all models and all target sets) as
  metric_z = (m - mean_task(m)) / std_task(m).
This ensures that coefficients are comparable across tasks with different
metric scales (accuracy/F1/Matthews/etc.).

Regression specification
------------------------
We fit
  metric_z = sum_j beta_j * I_j + beta_b * log10(trainable_ratio)
            + delta_task + delta_model + eps,
where I_j are module indicators (Q,K,V,O,G,GK,MLP_gate,MLP_updown), and
(delta_task, delta_model) are fixed effects implemented via categorical dummies.
We report robust (HC3) standard errors and 95% confidence intervals.

"""

import numpy as np
import pandas as pd
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf


MODULES = ["Q", "K", "V", "O", "G", "GK", "MLP_gate", "MLP_updown"]


def fit_and_summarize(df: pd.DataFrame) -> pd.DataFrame:
    # Basic hygiene
    df = df.copy()
    for c in MODULES:
        df[c] = df[c].astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["metric_z", "trainable_ratio", "task", "model"] + MODULES)

    df["log_budget"] = np.log10(df["trainable_ratio"].astype(float))

    # OLS with task/model fixed effects
    rhs = " + ".join(MODULES) + " + log_budget + C(task) + C(model)"
    formula = f"metric_z ~ {rhs}"

    res = smf.ols(formula, data=df).fit(cov_type="HC3")

    rows = []
    for term in MODULES:
        coef = float(res.params.get(term, np.nan))
        se = float(res.bse.get(term, np.nan))
        lo = coef - 1.96 * se
        hi = coef + 1.96 * se
        rows.append({"term": term, "coef": coef, "se": se, "lo": lo, "hi": hi})

    out = pd.DataFrame(rows)
    # Sort by effect size (descending) to improve readability
    out = out.sort_values("coef", ascending=False).reset_index(drop=True)
    return out


def plot_effects(tbl: pd.DataFrame, out_png: str, out_pdf: str) -> None:
    # ICML-ish typography
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.2,
    })

    terms = tbl["term"].tolist()
    coef = tbl["coef"].to_numpy()
    lo = tbl["lo"].to_numpy()
    hi = tbl["hi"].to_numpy()

    y = np.arange(len(terms))
    xerr = np.vstack([coef - lo, hi - coef])

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.errorbar(
        coef,
        y,
        xerr=xerr,
        fmt="o",
        capsize=0,
        elinewidth=2.4,
        markersize=12,
    )

    ax.axvline(0.0, linewidth=1.8)
    ax.set_yticks(y)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()

    ax.set_title("Module effects on within-task normalized performance")
    ax.set_xlabel("OLS coefficient β on module indicator (within-task z-score), robust 95% CI")
    ax.grid(True, axis="x", linewidth=0.6, alpha=0.5)

    # Symmetric-ish x limits with padding
    xmin = float(np.min(lo))
    xmax = float(np.max(hi))
    pad = 0.12 * (xmax - xmin + 1e-9)
    ax.set_xlim(xmin - pad, xmax + pad)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Path to merged_long.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    tbl = fit_and_summarize(df)

    out_csv = os.path.join(args.out_dir, "regression_module_effects_hc3_recomputed.csv")
    tbl.to_csv(out_csv, index=False)

    out_png = os.path.join(args.out_dir, "module_effects_scientific.png")
    out_pdf = os.path.join(args.out_dir, "module_effects_scientific.pdf")
    plot_effects(tbl, out_png, out_pdf)

    print("Wrote:", out_csv)
    print("Wrote:", out_png)
    print("Wrote:", out_pdf)


if __name__ == "__main__":
    main()
