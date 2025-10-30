from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .glue_metrics import normalize_dataset_name


@dataclass
class BestRow:
    idx: int
    step: int
    checkpoint_path: str
    metrics: Dict[str, float]


def _read_eval_only(exp_out_dir: Path) -> Optional[pd.DataFrame]:
    # expect <exp_name>_eval_only.csv under exp_out_dir
    exp = exp_out_dir.name
    csv = exp_out_dir / f"{exp}_eval_only.csv"
    if csv.is_file():
        return pd.read_csv(csv)
    # fallback to log_history.csv
    csv_all = exp_out_dir / f"{exp}_log_history.csv"
    if csv_all.is_file():
        df = pd.read_csv(csv_all)
        eval_cols = [c for c in df.columns if c.startswith("eval_")]
        if "step" in df.columns and eval_cols:
            df_eval = df[["step"] + eval_cols]
            df_eval = df_eval[df_eval[eval_cols].notna().any(axis=1)]
            return df_eval
    return None


def _cfg_path_for(exp_train_dir: Path) -> Optional[str]:
    p = exp_train_dir / "cfg.yaml"
    return str(p) if p.is_file() else None


def _checkpoint_path_for(exp_train_dir: Path, step: int) -> str:
    return str((exp_train_dir / f"checkpoint-{step}").resolve())


def _best_for_cola(df: pd.DataFrame, exp_train_dir: Path) -> BestRow:
    # primary: eval_matthews_correlation desc, tie -> eval_loss asc, tie -> step desc
    if "eval_matthews_correlation" not in df.columns:
        raise ValueError("CoLA eval_matthews_correlation not found")
    tmp = df.copy()
    if "eval_loss" not in tmp.columns:
        tmp["eval_loss"] = float("inf")
    tmp = tmp.sort_values(["eval_matthews_correlation", "eval_loss", "step"], ascending=[False, True, False])
    row = tmp.iloc[0]
    step = int(row["step"]) if "step" in row else -1
    ckpt = _checkpoint_path_for(exp_train_dir, step)
    metrics = {"eval_matthews_correlation": float(row["eval_matthews_correlation"]) }
    if "eval_loss" in row and pd.notna(row["eval_loss"]):
        metrics["eval_loss"] = float(row["eval_loss"])
    return BestRow(idx=int(row.name), step=step, checkpoint_path=ckpt, metrics=metrics)


def _best_for_mrpc_qqp(df: pd.DataFrame, exp_train_dir: Path) -> BestRow:
    # primary: eval_f1 desc; tie: eval_accuracy desc; tie: eval_loss asc; tie: step desc
    if "eval_f1" not in df.columns:
        raise ValueError("MRPC/QQP eval_f1 not found")
    tmp = df.copy()
    if "eval_accuracy" not in tmp.columns:
        tmp["eval_accuracy"] = -1.0
    if "eval_loss" not in tmp.columns:
        tmp["eval_loss"] = float("inf")
    tmp = tmp.sort_values(["eval_f1", "eval_accuracy", "eval_loss", "step"], ascending=[False, False, True, False])
    row = tmp.iloc[0]
    step = int(row["step"]) if "step" in row else -1
    ckpt = _checkpoint_path_for(exp_train_dir, step)
    metrics = {"eval_f1": float(row["eval_f1"]) }
    if "eval_accuracy" in row and pd.notna(row["eval_accuracy"]):
        metrics["eval_accuracy"] = float(row["eval_accuracy"])
    if "eval_loss" in row and pd.notna(row["eval_loss"]):
        metrics["eval_loss"] = float(row["eval_loss"])
    return BestRow(idx=int(row.name), step=step, checkpoint_path=ckpt, metrics=metrics)


def _best_for_stsb(df: pd.DataFrame, exp_train_dir: Path) -> BestRow:
    # report pearson & spearman; primary: pearson desc; tie: spearman desc; tie: eval_loss asc; tie: step desc
    pearson_col = "eval_pearson" if "eval_pearson" in df.columns else None
    if not pearson_col:
        raise ValueError("STS-B eval_pearson not found")
    spearman_col = "eval_spearman" if "eval_spearman" in df.columns else ("eval_spearmanr" if "eval_spearmanr" in df.columns else None)
    tmp = df.copy()
    if spearman_col is None:
        tmp["_spearman_tmp"] = -1.0
        spearman_col = "_spearman_tmp"
    if "eval_loss" not in tmp.columns:
        tmp["eval_loss"] = float("inf")
    tmp = tmp.sort_values([pearson_col, spearman_col, "eval_loss", "step"], ascending=[False, False, True, False])
    row = tmp.iloc[0]
    step = int(row["step"]) if "step" in row else -1
    ckpt = _checkpoint_path_for(exp_train_dir, step)
    metrics = {pearson_col: float(row[pearson_col])}
    if spearman_col in row and pd.notna(row[spearman_col]):
        metrics[spearman_col] = float(row[spearman_col])
    if "eval_loss" in row and pd.notna(row["eval_loss"]):
        metrics["eval_loss"] = float(row["eval_loss"])
    return BestRow(idx=int(row.name), step=step, checkpoint_path=ckpt, metrics=metrics)


def _best_for_mnli(df: pd.DataFrame, exp_train_dir: Path) -> BestRow:
    # report matched & mismatched accuracy; selection: if both present, maximize their average; else fallback eval_accuracy
    matched_col = None
    mismatched_col = None
    for c in df.columns:
        lc = c.lower()
        if "matched" in lc and "accuracy" in lc:
            matched_col = c
        if "mismatched" in lc and "accuracy" in lc:
            mismatched_col = c
    tmp = df.copy()
    if matched_col and mismatched_col:
        tmp["_mnli_avg_acc"] = (tmp[matched_col].astype(float) + tmp[mismatched_col].astype(float)) / 2.0
        score_cols = ["_mnli_avg_acc", "eval_loss", "step"]
        ascending = [False, True, False]
    else:
        if "eval_accuracy" not in tmp.columns:
            raise ValueError("MNLI accuracy columns not found")
        score_cols = ["eval_accuracy", "eval_loss", "step"]
        ascending = [False, True, False]
    if "eval_loss" not in tmp.columns:
        tmp["eval_loss"] = float("inf")
    tmp = tmp.sort_values(score_cols, ascending=ascending)
    row = tmp.iloc[0]
    step = int(row["step"]) if "step" in row else -1
    ckpt = _checkpoint_path_for(exp_train_dir, step)
    metrics: Dict[str, float] = {}
    if matched_col and pd.notna(row.get(matched_col, None)):
        metrics[matched_col] = float(row[matched_col])
    if mismatched_col and pd.notna(row.get(mismatched_col, None)):
        metrics[mismatched_col] = float(row[mismatched_col])
    if "eval_accuracy" in row and pd.notna(row.get("eval_accuracy", None)):
        metrics["eval_accuracy"] = float(row["eval_accuracy"])
    if "eval_loss" in row and pd.notna(row.get("eval_loss", None)):
        metrics["eval_loss"] = float(row["eval_loss"])
    return BestRow(idx=int(row.name), step=step, checkpoint_path=ckpt, metrics=metrics)


def _best_for_accuracy_tasks(df: pd.DataFrame, exp_train_dir: Path) -> BestRow:
    if "eval_accuracy" not in df.columns:
        raise ValueError("eval_accuracy not found")
    tmp = df.copy()
    if "eval_loss" not in tmp.columns:
        tmp["eval_loss"] = float("inf")
    tmp = tmp.sort_values(["eval_accuracy", "eval_loss", "step"], ascending=[False, True, False])
    row = tmp.iloc[0]
    step = int(row["step"]) if "step" in row else -1
    ckpt = _checkpoint_path_for(exp_train_dir, step)
    metrics = {"eval_accuracy": float(row["eval_accuracy"]) }
    if "eval_loss" in row and pd.notna(row["eval_loss"]):
        metrics["eval_loss"] = float(row["eval_loss"])
    return BestRow(idx=int(row.name), step=step, checkpoint_path=ckpt, metrics=metrics)


def choose_best_row_for_task(task: str, df: pd.DataFrame, exp_train_dir: Path) -> BestRow:
    t = normalize_dataset_name(task)
    if t == "cola":
        return _best_for_cola(df, exp_train_dir)
    if t in ("mrpc", "qqp"):
        return _best_for_mrpc_qqp(df, exp_train_dir)
    if t == "stsb":
        return _best_for_stsb(df, exp_train_dir)
    if t == "mnli":
        return _best_for_mnli(df, exp_train_dir)
    # default accuracy tasks
    return _best_for_accuracy_tasks(df, exp_train_dir)


def summarize_dataset(dataset_out_dir: Path, dataset_name: str, dataset_train_dir: Optional[Path] = None) -> Path:
    """Aggregate per-experiment best rows into a dataset-level summary table.
    dataset_out_dir: out_root/<dataset_dir>
    dataset_name: can be glue-tvt_cola_seed87 or cola
    """
    t = normalize_dataset_name(dataset_name)
    rows: List[Dict[str, object]] = []
    for exp_out in sorted([d for d in dataset_out_dir.iterdir() if d.is_dir()]):
        eval_df = _read_eval_only(exp_out)
        if eval_df is None or eval_df.empty:
            continue
        # resolve original training experiment dir for checkpoint/cfg paths if provided
        exp_train_dir = (dataset_train_dir / exp_out.name) if dataset_train_dir else exp_out
        best = choose_best_row_for_task(t, eval_df, exp_train_dir)
        record: Dict[str, object] = {}
        # include native eval_* columns present in best.metrics
        for k, v in best.metrics.items():
            record[k] = v
        record["experiment"] = exp_out.name
        record["step"] = best.step
        record["checkpoint_path"] = best.checkpoint_path
        cfgp = _cfg_path_for(exp_train_dir)
        if cfgp:
            record["cfg_path"] = cfgp
        rows.append(record)

    if not rows:
        raise RuntimeError(f"No experiments summarized under {dataset_out_dir}")

    df = pd.DataFrame(rows)
    # Task-specific final sorting for readability
    if t == "cola" and "eval_matthews_correlation" in df.columns:
        df = df.sort_values(["eval_matthews_correlation", "step"], ascending=[False, False])
    elif t in ("mrpc", "qqp") and "eval_f1" in df.columns:
        keys = ["eval_f1"] + (["eval_accuracy"] if "eval_accuracy" in df.columns else []) + ["step"]
        asc = [False] + ([False] if "eval_accuracy" in df.columns else []) + [False]
        df = df.sort_values(keys, ascending=asc)
    elif t == "stsb":
        # prefer pearson then spearman (if present)
        keys = [c for c in ["eval_pearson", "eval_spearman", "eval_spearmanr"] if c in df.columns] + ["step"]
        asc = [False] * (len(keys) - 1) + [False]
        df = df.sort_values(keys, ascending=asc)
    elif t == "mnli":
        # prefer both matched/mismatched if exist; otherwise eval_accuracy
        if "eval_accuracy" in df.columns and ("matched" not in "".join(df.columns).lower()):
            df = df.sort_values(["eval_accuracy", "step"], ascending=[False, False])
        else:
            # sort by average of matched/mismatched if both present
            mcols = [c for c in df.columns if "matched" in c.lower() and "accuracy" in c.lower()]
            if len(mcols) >= 2:
                df["_avg_mnli_acc"] = df[mcols].astype(float).mean(axis=1)
                df = df.sort_values(["_avg_mnli_acc", "step"], ascending=[False, False]).drop(columns=["_avg_mnli_acc"])
    else:
        if "eval_accuracy" in df.columns:
            df = df.sort_values(["eval_accuracy", "step"], ascending=[False, False])

    out_csv = dataset_out_dir / "dataset_summary.csv"
    out_json = dataset_out_dir / "dataset_summary.json"
    out_md = dataset_out_dir / "dataset_summary.md"
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump({"dataset": dataset_out_dir.name, "rows": df.to_dict(orient="records")}, f, indent=2)
    # markdown table
    with open(out_md, "w") as f:
        f.write(df.to_markdown(index=False))
    return out_csv


