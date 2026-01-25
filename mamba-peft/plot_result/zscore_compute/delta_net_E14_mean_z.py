#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeltaNet (ROUND_E14) GLUE + Commonsense_170K aggregation.

This replicates the user's GLA pipeline:
1) Load per-task CSV.
2) For each configuration, pick the best checkpoint (max scalar score).
3) Keep only configurations present in all tasks (strict comparability).
4) Compute per-task z-scores across configurations.
5) Aggregate mean z across tasks; also report non-RTE mean z.
6) Print both Markdown and LaTeX tables.

Notes:
- The CSVs in this repo store experiment names WITHOUT the "E{n}_" prefix.
  We therefore canonicalize by stripping a leading r"^E\d+_" if present.
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Paths (DeltaNet)
# -----------------------------
BASE = Path("/mnt/data")

TASKS = {
    "cola": BASE / "delta_net_glue-tvt_cola.csv",
    "mrpc": BASE / "delta_net_glue-tvt_mrpc.csv",
    "qqp":  BASE / "delta_net_glue-tvt_qqp.csv",
    "qnli": BASE / "delta_net_glue-tvt_qnli.csv",
    "sst2": BASE / "delta_net_glue-tvt_sst2.csv",
    "mnli": BASE / "delta_net_glue-tvt_mnli.csv",
    "rte":  BASE / "delta_net_glue-tvt_rte.csv",
    "commonsense": BASE / "delta_net_commonsense_170k.csv",
}

# -----------------------------
# ROUND_E14 (DeltaNet) configs
# -----------------------------
ROUND_E14 = [
    "E1_QKVO_plus_MLP_r8_alpha16",
    "E1_QKVO_r8_alpha16",
    "E4_MLPONLY_r8_alpha16",
    "E2_OMLP_r8_alpha16",
    "E4_QONLY_r8_alpha16",
    "E4_KONLY_r8_alpha16",
    "E4_VONLY_r8_alpha16",
    "E11_OONLY_r8_alpha16",
    "E7_KVONLY_r8_alpha16",
    "E6_QVONLY_r8_alpha16",
    "E6_VOONLY_r8_alpha16",
]

def canonical(exp: str) -> str:
    """Strip optional leading E{n}_ prefix so names match the CSVs."""
    return re.sub(r"^E\d+_", "", exp)

ROUND = [canonical(x) for x in ROUND_E14]

# concise shorthands
NAME = {
    "QONLY_r8_alpha16": "Q",
    "KONLY_r8_alpha16": "K",
    "VONLY_r8_alpha16": "V",
    "OONLY_r8_alpha16": "O",
    "KVONLY_r8_alpha16": "KV",
    "QVONLY_r8_alpha16": "QV",
    "VOONLY_r8_alpha16": "VO",
    "QKVO_r8_alpha16": "QKVO",
    "QKVO_plus_MLP_r8_alpha16": "QKVO+MLP",
    "MLPONLY_r8_alpha16": "MLP",
    "OMLP_r8_alpha16": "O+MLP",
}

def task_score(df: pd.DataFrame, task: str) -> pd.Series:
    """
    Scalar score per row:
      - CoLA: MCC
      - SST2/QNLI/MNLI/RTE: accuracy
      - MRPC/QQP: (Acc + F1)/2
      - Commonsense: token accuracy
    """
    if task == "commonsense":
        return df["eval_token_accuracy"].astype(float)

    if task in ("mrpc", "qqp"):
        return 0.5 * df["eval_accuracy"].astype(float) + 0.5 * df["eval_f1"].astype(float)

    if "eval_matthews_correlation" in df.columns:
        return df["eval_matthews_correlation"].astype(float)

    return df["eval_accuracy"].astype(float)

def main() -> None:
    # 1) Load, filter, pick BEST checkpoint per task+config
    best: dict[str, dict[str, float]] = {t: {} for t in TASKS}
    meta: dict[str, dict[str, float]] = {}  # exp -> trainable_ratio etc.

    for task, path in TASKS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for task '{task}': {path}")

        df = pd.read_csv(path).copy()
        df["score"] = task_score(df, task)

        for exp in ROUND:
            sub = df[df["experiment"] == exp]
            if sub.empty:
                continue

            idx = sub["score"].idxmax()  # best checkpoint
            best[task][exp] = float(sub.loc[idx, "score"])

            if exp not in meta:
                meta[exp] = {
                    "trainable_ratio": float(sub.loc[idx, "trainable_ratio"]),
                    "trainable_params": float(sub.loc[idx, "trainable_params"]),
                    "total_params": float(sub.loc[idx, "total_params"]),
                }

    # report missing
    all_tasks = list(TASKS.keys())
    missing_by_task = {t: [e for e in ROUND if e not in best[t]] for t in all_tasks}
    print("Missing experiments by task:")
    for t in all_tasks:
        if missing_by_task[t]:
            print(f"  {t}: {missing_by_task[t]}")

    # keep only configs present in ALL tasks (strict comparability)
    valid = [e for e in ROUND if all(e in best[t] for t in all_tasks)]
    print("\nComparable configs (present in all tasks):", valid)

    # 2) Task-wise z-score across VALID configs (population std ddof=0)
    z: dict[str, dict[str, float]] = {}
    for t in all_tasks:
        vals = np.array([best[t][e] for e in valid], dtype=float)
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        z[t] = {e: ((best[t][e] - mu) / sd) if sd > 0 else 0.0 for e in valid}

    # 3) Aggregate non-RTE mean z (includes commonsense) and mean z(all)
    non_rte_tasks = [t for t in all_tasks if t != "rte"]  # includes commonsense

    rows = []
    for e in valid:
        non_rte_mean = float(np.mean([z[t][e] for t in non_rte_tasks]))
        all_mean = float(np.mean([z[t][e] for t in all_tasks]))
        rows.append({
            "Config": NAME.get(e, e),
            "Trainable %": meta[e]["trainable_ratio"] * 100.0,
            "non-RTE mean z": non_rte_mean,
            "mean z (all)": all_mean,
        })

    out = pd.DataFrame(rows).sort_values("mean z (all)", ascending=False).reset_index(drop=True)

    # 4) Print Markdown
    md = out.copy()
    for c in ["Trainable %", "non-RTE mean z", "mean z (all)"]:
        md[c] = md[c].map(lambda x: f"{x:.3f}")
    print("\nMarkdown table:\n")
    print(md.to_markdown(index=False))

    # 5) Print LaTeX
    latex = out.copy()
    latex["Trainable %"] = latex["Trainable %"].round(3)
    latex["non-RTE mean z"] = latex["non-RTE mean z"].round(3)
    latex["mean z (all)"] = latex["mean z (all)"].round(3)

    print("\nLaTeX table:\n")
    print(
        latex.to_latex(
            index=False,
            escape=True,
            column_format="lrrr",
            caption=("DeltaNet ROUND\\_E14 configurations sorted by mean z (all). "
                     "Z-scores are computed per task across comparable configurations "
                     "(present on all tasks), including Commonsense\\_170K."),
            label="tab:deltanet_mean_z_all_with_commonsense",
        )
    )

if __name__ == "__main__":
    main()
