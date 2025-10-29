from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .glue_metrics import get_task_spec, select_score


def load_log_history(trainer_state_json: Path) -> pd.DataFrame:
    with open(trainer_state_json, "r") as f:
        obj = json.load(f)
    logs = obj.get("log_history", [])
    if not logs:
        raise ValueError(f"Empty 'log_history' in {trainer_state_json}")
    df = pd.DataFrame(logs)
    return df


def latest_trainer_state_json(exp_dir: Path) -> Path:
    latest_step = -1
    latest = None
    for cp in exp_dir.glob("checkpoint-*/"):
        m = re.search(r"checkpoint-(\d+)", cp.name)
        if not m:
            continue
        step = int(m.group(1))
        js = cp / "trainer_state.json"
        if js.is_file() and step > latest_step:
            latest_step = step
            latest = js
    if not latest:
        raise FileNotFoundError(f"No trainer_state.json in {exp_dir}")
    return latest


def pick_best_checkpoint(df: pd.DataFrame, dataset_name: str) -> Tuple[int, Dict[str, Any]]:
    spec = get_task_spec(dataset_name)
    best_row = None
    best_score: Optional[float] = None
    best_step = -1
    for _, row in df.iterrows():
        if "step" not in row or pd.isna(row["step"]):
            continue
        score = select_score(row.to_dict(), spec)
        if score is None:
            continue
        step = int(row["step"])
        if best_row is None:
            best_row, best_score, best_step = row, score, step
            continue
        if spec.higher_is_better:
            better = (score > best_score) or (score == best_score and step > best_step)
        else:
            better = (score < best_score) or (score == best_score and step > best_step)
        if better:
            best_row, best_score, best_step = row, score, step
    if best_row is None:
        raise ValueError("No evaluation rows with required metrics found")
    return best_step, best_row.to_dict()


def plot_series(df: pd.DataFrame, x: str, y: str, out_png: Path, title: str, ylabel: str) -> Optional[str]:
    if y not in df.columns or not df[y].notna().any():
        return None
    d = df[df[y].notna()].sort_values(x)
    plt.figure()
    plt.plot(d[x], d[y], linestyle="-", marker="o", markersize=2)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close("all")
    return str(out_png)


def aggregate_experiment(exp_dir: Path, dataset_name: str, out_dir: Path) -> Dict[str, Any]:
    js = latest_trainer_state_json(exp_dir)
    df = load_log_history(js)

    # Train/eval split based on presence of columns
    df_train = df[df["loss"].notna()] if "loss" in df.columns else pd.DataFrame()
    eval_mask = df.columns.str.startswith("eval_")
    eval_cols = list(df.columns[eval_mask])
    df_eval = df[["step"] + eval_cols] if eval_cols else pd.DataFrame()
    df_eval = df_eval[df_eval[eval_cols].notna().any(axis=1)] if not df_eval.empty else df_eval

    best_step, best_row = pick_best_checkpoint(df, dataset_name)

    # Export CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_all = out_dir / f"{exp_dir.name}_log_history.csv"
    df.to_csv(csv_all, index=False)
    csv_eval = None
    if not df_eval.empty:
        csv_eval = out_dir / f"{exp_dir.name}_eval_only.csv"
        df_eval.to_csv(csv_eval, index=False)

    # Plots
    plots = {}
    if not df_train.empty:
        p = plot_series(df_train, "step", "loss", out_dir / f"{exp_dir.name}_train_loss.png", "Training Loss", "loss")
        if p: plots["train_loss"] = p
        if "grad_norm" in df_train.columns:
            p = plot_series(df_train, "step", "grad_norm", out_dir / f"{exp_dir.name}_grad_norm.png", "Grad Norm", "grad_norm")
            if p: plots["grad_norm"] = p
        if "learning_rate" in df_train.columns:
            p = plot_series(df_train, "step", "learning_rate", out_dir / f"{exp_dir.name}_lr.png", "Learning Rate", "lr")
            if p: plots["lr"] = p
    if not df_eval.empty:
        # try primary metric plot
        spec = get_task_spec(dataset_name)
        primary = spec.primary
        if primary in df_eval.columns:
            p = plot_series(df_eval, "step", primary, out_dir / f"{exp_dir.name}_{primary}.png", f"{primary}", primary)
            if p: plots[primary] = p
        if "eval_loss" in df_eval.columns:
            p = plot_series(df_eval, "step", "eval_loss", out_dir / f"{exp_dir.name}_eval_loss.png", "Eval Loss", "eval_loss")
            if p: plots["eval_loss"] = p

    report = {
        "experiment": exp_dir.name,
        "best_step": best_step,
        "best_metrics": best_row,
        "csv_all": str(csv_all),
        "csv_eval": str(csv_eval) if csv_eval else None,
        "plots": plots,
        "trainer_state": str(js),
    }
    with open(out_dir / f"{exp_dir.name}_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


