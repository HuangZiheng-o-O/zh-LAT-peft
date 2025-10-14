#!/usr/bin/env python3
"""
make_cola_figs_raw.py (updated)
   1. 进入 mamba-peft 项目目录：

   cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/

   2. 运行脚本（使用默认的CPU核心数）：
   python plot_result/draw_figs.py

   3. 或者，指定使用特定数量的核心（例如，使用4个核心）：

   python plot_result/draw_figs.py --workers 4
   
Changes in this version:
- Now runs as a parallel batch processor for all subdirectories in the base directory.
- Uses concurrent.futures.ProcessPoolExecutor to process experiments in parallel.
- If processing a subdirectory fails, it creates a zip with an error log instead of skipping.
- For each experiment:
  - Finds the latest trainer_state.json automatically.
  - Generates all default plots (loss, grad_norm, MCC, etc.) and CSVs.
  - Analyzes log history to find the top 3 checkpoints by MCC.
  - Adds a 'best_checkpoints.txt' file with top 3 info into the zip.
- All logic is integrated into this single, modifiable script.
"""

import argparse
import json
import os
import shutil
import re
import zipfile
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend, suitable for multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def coerce_numeric(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_log_history(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r") as f:
        js = json.load(f)
    logs = js.get("log_history", [])
    if not logs:
        raise ValueError("'log_history' is empty or not found in the JSON file.")
    df = pd.DataFrame(logs)
    num_cols = [
        "epoch", "step", "loss", "grad_norm", "learning_rate",
        "eval_loss", "eval_matthews_correlation", "eval_runtime",
        "eval_samples_per_second", "eval_steps_per_second"
    ]
    df = coerce_numeric(df, num_cols)
    return df


def split_train_eval(df: pd.DataFrame):
    df_train = df[df["loss"].notna()].copy()
    df_eval = df[df["eval_matthews_correlation"].notna()].copy().sort_values("step").reset_index(drop=True)
    return df_train, df_eval


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_all(figdir: Path, fname_base: str):
    """Save current matplotlib figure as PNG(300dpi), PDF, SVG."""
    figdir = Path(figdir)
    ensure_dir(figdir)
    png = figdir / f"{fname_base}.png"
    pdf = figdir / f"{fname_base}.pdf"
    svg = figdir / f"{fname_base}.svg"
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close('all') # Close all figures to free memory
    return {"png": str(png), "pdf": str(pdf), "svg": str(svg)}


def _sorted_by_step(df):
    return df.sort_values("step").reset_index(drop=True)


def _rolling_trend(y: pd.Series, length: int):
    if length < 5:
        return None
    window = max(5, int(round(length / 50)))
    if window % 2 == 0:
        window += 1
    return y.rolling(window=window, center=True, min_periods=max(3, window//3)).mean()


def plot_training_loss(df_train: pd.DataFrame, figdir: Path):
    if df_train.empty:
        return None
    d = _sorted_by_step(df_train)
    plt.figure()
    plt.plot(d["step"], d["loss"], linestyle="-", marker="o", markersize=2)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss (raw, connected)")
    return save_all(figdir, "training_loss_raw")


def plot_grad_norm(df_train: pd.DataFrame, figdir: Path):
    if "grad_norm" not in df_train.columns or not df_train["grad_norm"].notna().any():
        return None
    d = _sorted_by_step(df_train[df_train["grad_norm"].notna()])
    plt.figure()
    plt.plot(d["step"], d["grad_norm"], linestyle="none", marker="o", markersize=2, label="Raw points")
    trend = _rolling_trend(d["grad_norm"], len(d))
    if trend is not None and trend.notna().any():
        plt.plot(d["step"], trend, linestyle="-", label="Trend (rolling mean)")
        plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm (raw + trend)")
    return save_all(figdir, "grad_norm_raw")


def plot_learning_rate(df_train: pd.DataFrame, figdir: Path):
    if "learning_rate" not in df_train.columns or not df_train["learning_rate"].notna().any():
        return None
    d = _sorted_by_step(df_train)
    plt.figure()
    plt.plot(d["step"], d["learning_rate"])
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    return save_all(figdir, "learning_rate")


def plot_eval_mcc(df_eval: pd.DataFrame, figdir: Path):
    if df_eval.empty:
        return None
    d = _sorted_by_step(df_eval)
    plt.figure()
    plt.plot(d["step"], d["eval_matthews_correlation"], linestyle="-", marker="o", markersize=4)
    plt.xlabel("Step")
    plt.ylabel("Eval MCC")
    plt.title("CoLA Eval MCC (raw checkpoints)")
    return save_all(figdir, "eval_mcc_raw")


def plot_eval_loss(df_eval: pd.DataFrame, figdir: Path):
    if df_eval.empty:
        return None
    d = _sorted_by_step(df_eval)
    plt.figure()
    plt.plot(d["step"], d["eval_loss"], linestyle="-", marker="o", markersize=4)
    plt.xlabel("Step")
    plt.ylabel("Eval Loss")
    plt.title("CoLA Eval Loss (raw checkpoints)")
    return save_all(figdir, "eval_loss_raw")


def export_csvs(df: pd.DataFrame, df_eval: pd.DataFrame, out_dir: Path, exp_name: str):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    csv_all = out_dir / f"cola_{exp_name}_log_history_raw.csv"
    df.to_csv(csv_all, index=False)
    csv_eval = out_dir / f"cola_{exp_name}_eval_only_raw.csv"
    cols = ["epoch", "step", "eval_matthews_correlation", "eval_loss", "eval_runtime", "eval_samples_per_second"]
    cols = [c for c in cols if c in df_eval.columns]
    df_eval[cols].to_csv(csv_eval, index=False)
    return str(csv_all), str(csv_eval)


def make_zip_with_extras(figdir: Path, zip_path: Path, extra_files_dict: dict = None):
    """ Zips the contents of figdir and adds extra files from a dictionary. """
    zip_base = str(zip_path).removesuffix(".zip")
    shutil.make_archive(zip_base, "zip", figdir)
    if extra_files_dict:
        with zipfile.ZipFile(f"{zip_base}.zip", 'a') as zf:
            for filename, content in extra_files_dict.items():
                zf.writestr(filename, content)
    return f"{zip_base}.zip"


def find_latest_checkpoint_json(exp_dir: Path):
    all_checkpoints = list(exp_dir.glob("checkpoint-*"))
    if not all_checkpoints:
        raise FileNotFoundError(f"No checkpoint directories found in {exp_dir}")

    latest_step = -1
    latest_checkpoint_dir = None
    for cp_dir in all_checkpoints:
        try:
            step = int(re.search(r"checkpoint-(\d+)", cp_dir.name).group(1))
            if step > latest_step:
                latest_step = step
                latest_checkpoint_dir = cp_dir
        except (AttributeError, ValueError):
            continue
    
    if latest_checkpoint_dir:
        json_path = latest_checkpoint_dir / "trainer_state.json"
        if json_path.exists():
            return json_path
    raise FileNotFoundError(f"trainer_state.json not found in latest checkpoint directory {latest_checkpoint_dir}")


def safe_process_experiment(exp_dir: Path, output_dir: Path):
    """A wrapper to catch all exceptions for one experiment and return a status."""
    exp_name = exp_dir.name
    try:
        print(f"--- Processing: {exp_name} ---")
        latest_json = find_latest_checkpoint_json(exp_dir)
        
        temp_fig_dir = output_dir / f"{exp_name}_temp_files"
        ensure_dir(temp_fig_dir)

        df = load_log_history(latest_json)
        df_train, df_eval = split_train_eval(df)

        if df_eval.empty:
            raise ValueError("No evaluation data found in logs after splitting.")

        plot_training_loss(df_train, temp_fig_dir)
        plot_grad_norm(df_train, temp_fig_dir)
        plot_learning_rate(df_train, temp_fig_dir)
        plot_eval_mcc(df_eval, temp_fig_dir)
        plot_eval_loss(df_eval, temp_fig_dir)

        export_csvs(df, df_eval, temp_fig_dir, exp_name)

        top_3 = df_eval.nlargest(3, 'eval_matthews_correlation')
        top_3_info = f"Top 3 Checkpoints for {exp_name} (by eval_matthews_correlation):\n\n"
        for _, row in top_3.iterrows():
            step = int(row['step'])
            checkpoint_path = exp_dir / f"checkpoint-{step}"
            top_3_info += f"- Score: {row['eval_matthews_correlation']:.6f}\n"
            top_3_info += f"  Step: {step}\n"
            top_3_info += f"  Path: {checkpoint_path}\n\n"

        zip_path = output_dir / f"{exp_name}.zip"
        extra_files = {"best_checkpoints.txt": top_3_info}
        make_zip_with_extras(temp_fig_dir, zip_path, extra_files)

        shutil.rmtree(temp_fig_dir)
        print(f"  Successfully created report: {zip_path}")
        return None # Success

    except Exception as e:
        print(f"  ERROR processing {exp_name}: {e}")
        error_zip_path = output_dir / f"{exp_name}_error.zip"
        error_message = f"Failed to process experiment: {exp_name}\n\n"
        error_message += f"Error Type: {type(e).__name__}\n"
        error_message += f"Error Details: {str(e)}\n\n"
        error_message += "Traceback:\n"
        error_message += traceback.format_exc()
        
        with zipfile.ZipFile(error_zip_path, 'w') as zf:
            zf.writestr("error_log.txt", error_message)
        print(f"  Created error log at {error_zip_path}")
        # Clean up temp dir if it exists
        if 'temp_fig_dir' in locals() and temp_fig_dir.exists():
            shutil.rmtree(temp_fig_dir)
        return f"{exp_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Process experiment results in parallel, generate plots and reports.")
    parser.add_argument("--base_dir", type=str, default="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/", help="Base directory containing experiment folders.")
    parser.add_argument("--output_dir", type=str, default="/home/user/mzs_h/output/all_agg_results/experiment_result/", help="Directory to save the final zip files.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers. Defaults to number of CPU cores.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    experiment_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    if not experiment_dirs:
        print(f"No subdirectories found in {base_dir}")
        return

    print(f"Found {len(experiment_dirs)} subdirectories to process using up to {args.workers or os.cpu_count()} workers.")

    # Use a partial function to pass the constant output_dir argument
    task_function = partial(safe_process_experiment, output_dir=output_dir)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # executor.map is a simple way to run jobs in parallel.
        # It applies task_function to each item in experiment_dirs.
        results = executor.map(task_function, experiment_dirs)

    # Check for errors returned from the processes
    errors = [res for res in results if res is not None]
    if errors:
        print("\n--- Some experiments failed ---")
        for err in errors:
            print(f"- {err}")
    
    print("\n=== All experiments processed! ===")


if __name__ == "__main__":
    main()