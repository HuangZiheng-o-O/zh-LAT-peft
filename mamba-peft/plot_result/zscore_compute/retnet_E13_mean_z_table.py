import pandas as pd
import numpy as np
import re

# -----------------------------
# RetNet paths + commonsense
# -----------------------------
paths = [
    "/mnt/data/retnet_glue-tvt_cola.csv",
    "/mnt/data/retnet_glue-tvt_mrpc.csv",
    "/mnt/data/retnet_glue-tvt_qqp.csv",
    "/mnt/data/retnet_glue-tvt_qnli.csv",
    "/mnt/data/retnet_glue-tvt_sst2.csv",
    "/mnt/data/retnet_glue-tvt_mnli.csv",
    "/mnt/data/retnet_glue-tvt_rte.csv",
]
commonsense_path = "/mnt/data/retnet_commonsense_170k.csv"

TASKS = {
    "cola": paths[0],
    "mrpc": paths[1],
    "qqp":  paths[2],
    "qnli": paths[3],
    "sst2": paths[4],
    "mnli": paths[5],
    "rte":  paths[6],
    "commonsense": commonsense_path,
}

# -----------------------------
# ROUND_E13_RETNET (your list) -> strip ".yaml" and any "E{num}_" prefix
# -----------------------------
ROUND_E13 = [
  "E1_QKVO_plus_MLP_r8_alpha16.yaml",
  "E1_QKVO_r8_alpha16.yaml",
  "E4_MLPONLY_r8_alpha16.yaml",
  "E2_OMLP_r8_alpha16.yaml",
  "E4_QONLY_r8_alpha16.yaml",
  "E4_KONLY_r8_alpha16.yaml",
  "E4_VONLY_r8_alpha16.yaml",
  "E11_OONLY_r8_alpha16.yaml",
  "E7_KVONLY_r8_alpha16.yaml",
  "E6_QVONLY_r8_alpha16.yaml",
  "E6_VOONLY_r8_alpha16.yaml",
  "E1_QKVO_plus_G_plus_MLP_r8_alpha16.yaml",
  "E1_QKVO_plus_G_r8_alpha16.yaml",
]

def normalize_exp(x: str) -> str:
    x = x.replace(".yaml", "")
    x = re.sub(r"^E\d+_", "", x)  # RetNet CSV uses no E-prefix
    return x

ROUND_E13 = [normalize_exp(x) for x in ROUND_E13]

# strict shorthand naming (your notation)
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

# -----------------------------
# 1) Load, filter, pick BEST checkpoint per task+config
# -----------------------------
best = {t: {} for t in TASKS}
meta = {}  # exp -> trainable_ratio etc.

for task, path in TASKS.items():
    df = pd.read_csv(path).copy()
    df["score"] = task_score(df, task)

    for exp in ROUND_E13:
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

all_tasks = list(TASKS.keys())
missing_by_task = {t: [e for e in ROUND_E13 if e not in best[t]] for t in all_tasks}
print("Missing experiments by task:")
for t in all_tasks:
    if missing_by_task[t]:
        print(f"  {t}: {missing_by_task[t]}")

# keep only configs present in ALL tasks (strict comparability)
valid = [e for e in ROUND_E13 if all(e in best[t] for t in all_tasks)]
print("\nComparable configs (present in all tasks):", valid)

# -----------------------------
# 2) Task-wise z-score across VALID configs (population std ddof=0)
# -----------------------------
z = {}
for t in all_tasks:
    vals = np.array([best[t][e] for e in valid], dtype=float)
    mu = vals.mean()
    sd = vals.std(ddof=0)
    z[t] = {e: ((best[t][e] - mu) / sd) if sd > 0 else 0.0 for e in valid}

# -----------------------------
# 3) Aggregate non-RTE mean z (includes commonsense) and mean z(all)
# -----------------------------
non_rte_tasks = [t for t in all_tasks if t != "rte"]  # includes commonsense

rows = []
for e in valid:
    non_rte_mean = float(np.mean([z[t][e] for t in non_rte_tasks]))
    all_mean = float(np.mean([z[t][e] for t in all_tasks]))
    rows.append({
        "Config": NAME.get(e, e),
        "Trainable %": meta[e]["trainable_ratio"] * 100.0,
        "non-RTE mean z": non_rte_mean,
        "mean z": all_mean,
    })

out = pd.DataFrame(rows).sort_values("mean z", ascending=False).reset_index(drop=True)

md = out.copy()
for c in ["Trainable %", "non-RTE mean z", "mean z"]:
    md[c] = md[c].map(lambda x: f"{x:.3f}")

print("\nMarkdown table:\n")
print(md.to_markdown(index=False))

latex = out.copy()
latex["Trainable %"] = latex["Trainable %"].round(3)
latex["non-RTE mean z"] = latex["non-RTE mean z"].round(3)
latex["mean z"] = latex["mean z"].round(3)

print("\nLaTeX table:\n")
print(
    latex.to_latex(
        index=False,
        escape=True,
        column_format="lrrr",
        caption=("RetNet configurations sorted by mean z (all). "
                 "Z-scores are computed per task across comparable configurations "
                 "(present on all tasks), including Commonsense\\_170K."),
        label="tab:retnet_mean_z_all_with_commonsense",
    )
)
