#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate cross-model comparison figures.

Requires that per-model analysis can be computed from the same input CSVs.
Creates
- operator_signature_comparison
- structured_attribution_comparison

This script does not generate per-model Pareto/heatmap/forest plots.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

import paper_figures_lib_v2 as lib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    lib.set_rcparams()
    lib.ensure_dir(args.out_dir)

    df = lib.load_all_runs(args.in_dir)
    agg = lib.compute_agg(df)

    # compute frontiers for each model
    frontiers = {}
    for model in ["retnet", "gla", "delta_net"]:
        a = agg[agg["model"] == model].copy().sort_values("trainable_ratio")
        keep = lib.pareto_front_xmin_ymax(a["trainable_ratio"].to_numpy(), a["mean_z"].to_numpy())
        frontiers[model] = a[keep][["exp_norm", "mean_z", "trainable_ratio"]].copy()

    freq = lib.compute_operator_freq(frontiers, agg)
    lib.plot_operator_signature_combined(freq, args.out_dir)

    ridges = [lib.ridge_bootstrap_ci(df, m, lam=5.0, n_boot=50, seed=42) for m in ["retnet", "gla", "delta_net"]]
    lib.plot_ridge_comparison(ridges, args.out_dir)

    agg.to_csv(os.path.join(args.out_dir, "aggregated_config_summary.csv"), index=False)
    freq.to_csv(os.path.join(args.out_dir, "operator_signature_frequency.csv"), index=False)
    pd.concat(ridges, ignore_index=True).to_csv(os.path.join(args.out_dir, "ridge_terms_with_ci.csv"), index=False)


if __name__ == "__main__":
    main()
