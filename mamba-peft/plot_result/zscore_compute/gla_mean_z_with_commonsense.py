
"""
Reproduce GLA main-results table (mean z and non-RTE mean z),
using GLUE (7 tasks) + Commonsense_170K.

- Best checkpoint per task/config is selected by max score.
- Task-wise z-score is computed across comparable configurations.
- non-RTE mean z excludes RTE but includes Commonsense.
- mean z (all) includes all tasks.
"""

import pandas as pd
import numpy as np

# -----------------------------
# Paths
# -----------------------------
TASKS = {
    "cola": "/mnt/data/gla_glue-tvt_cola.csv",
    "mrpc": "/mnt/data/gla_glue-tvt_mrpc.csv",
    "qqp":  "/mnt/data/gla_glue-tvt_qqp.csv",
    "qnli": "/mnt/data/gla_glue-tvt_qnli.csv",
    "sst2": "/mnt/data/gla_glue-tvt_sst2.csv",
    "mnli": "/mnt/data/gla_glue-tvt_mnli.csv",
    "rte":  "/mnt/data/gla_glue-tvt_rte.csv",
    "commonsense": "/mnt/data/gla_commonsense_170k.csv",
}

# -----------------------------
# Config set (ROUND_E12)
# -----------------------------
ROUND_E12 = [
    "QKVO_plus_G_plus_MLP_r8_alpha16",
    "QKVO_plus_MLP_r8_alpha16",
    "QKVO_r8_alpha16",
    "QKVO_plus_G_r8_alpha16",
    "MLPONLY_r8_alpha16",
    "OMLP_r8_alpha16",
    "QONLY_r8_alpha16",
    "KONLY_r8_alpha16",
    "VONLY_r8_alpha16",
    "OONLY_r8_alpha16",
    "KVONLY_r8_alpha16",
    "QVONLY_r8_alpha16",
    "VOONLY_r8_alpha16",
]

# strict shorthand naming
NAME = {
    "QONLY_r8_alpha16": "Q",
    "KONLY_r8_alpha16": "K",
    "VONLY_r8_alpha16": "V",
    "OONLY_r8_alpha16": "O",
    "KVONLY_r8_alpha16": "KV",
    "QVONLY_r8_alpha16": "QV",
    "VOONLY_r8_alpha16": "VO",
    "QKVO_r8_alpha16": "QKVO",
    "QKVO_plus_G_r8_alpha16": "QKVO+G",
    "QKVO_plus_MLP_r8_alpha16": "QKVO+MLP",
    "MLPONLY_r8_alpha16": "MLP",
    "OMLP_r8_alpha16": "O+MLP",
    "QKVO_plus_G_plus_MLP_r8_alpha16": "QKVO+G+MLP",
}

def task_score(df: pd.DataFrame, task: str) -> pd.Series:
    if task == "commonsense":
        return df["eval_token_accuracy"].astype(float)
    if task in ("mrpc", "qqp"):
        return 0.5 * df["eval_accuracy"].astype(float) + 0.5 * df["eval_f1"].astype(float)
    if "eval_matthews_correlation" in df.columns:
        return df["eval_matthews_correlation"].astype(float)
    return df["eval_accuracy"].astype(float)

# -----------------------------
# Load + best checkpoint
# -----------------------------
best = {t: {} for t in TASKS}
meta = {}

for task, path in TASKS.items():
    df = pd.read_csv(path).copy()
    df["score"] = task_score(df, task)

    for exp in ROUND_E12:
        sub = df[df["experiment"] == exp]
        if sub.empty:
            continue
        idx = sub["score"].idxmax()
        best[task][exp] = float(sub.loc[idx, "score"])
        if exp not in meta:
            meta[exp] = {
                "trainable_ratio": float(sub.loc[idx, "trainable_ratio"]),
            }

# Keep only configs present on all tasks
all_tasks = list(TASKS.keys())
valid = [e for e in ROUND_E12 if all(e in best[t] for t in all_tasks)]

# -----------------------------
# Task-wise z-score
# -----------------------------
z = {}
for t in all_tasks:
    vals = np.array([best[t][e] for e in valid], dtype=float)
    mu, sd = vals.mean(), vals.std(ddof=0)
    z[t] = {e: ((best[t][e] - mu) / sd) if sd > 0 else 0.0 for e in valid}

# -----------------------------
# Aggregate
# -----------------------------
non_rte_tasks = [t for t in all_tasks if t != "rte"]

rows = []
for e in valid:
    rows.append({
        "Config": NAME.get(e, e),
        "Trainable %": meta[e]["trainable_ratio"] * 100.0,
        "non-RTE mean z": np.mean([z[t][e] for t in non_rte_tasks]),
        "mean z": np.mean([z[t][e] for t in all_tasks]),
    })

out = (
    pd.DataFrame(rows)
    .sort_values("mean z", ascending=False)
    .reset_index(drop=True)
)

pd.set_option("display.precision", 3)
print(out)
