#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path

import config

def run(cmd: list[str], env=None) -> None:
    print("\n>>>", " ".join(cmd))
    p = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if p.stdout:
        print(p.stdout)
    if p.returncode != 0:
        if p.stderr:
            print(p.stderr, file=sys.stderr)
        raise SystemExit(p.returncode)
    if p.stderr:
        print(p.stderr, file=sys.stderr)

def zip_dir(src_dir: Path, zip_path: Path) -> None:
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(src_dir)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=config.DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default=config.DEFAULT_OUTPUT_DIR)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    scripts = project_root / "scripts"

    data_dir = Path(args.data_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_combined = out_dir / "combined"
    out_frontier = out_dir / "frontier_v8" / "variant_subset"
    out_heatmap = out_dir / "task_zscore_heatmaps"
    out_incidence = out_dir / "module_incidence"
    out_combined_fig = out_dir / "combined_figures"
    out_struct = out_dir / "structured_attribution"
    out_reg = out_dir / "module_regression"

    for d in [out_combined, out_frontier, out_heatmap, out_incidence, out_combined_fig, out_struct, out_reg]:
        d.mkdir(parents=True, exist_ok=True)

    combined_tidy = out_combined / "combined_tidy.csv"
    run([sys.executable, str(scripts / "generate_combined_tidy.py"),
         "--input_glob", str(data_dir / "*.csv"),
         "--output", str(combined_tidy)])

    merged_all_runs = out_combined / "merged_all_runs.csv"
    shutil.copy2(combined_tidy, merged_all_runs)

    run([sys.executable, str(scripts / "make_frontier_v8.py"),
         "--combined_csv", str(merged_all_runs),
         "--out_dir", str(out_frontier)])

    run([sys.executable, str(scripts / "plot_task_zscore_heatmap_3models3.py"),
         "--data_dir", str(data_dir),
         "--out_dir", str(out_heatmap),
         "--models", "delta_net,gla,retnet"])

    run([sys.executable, str(scripts / "generate_module_incidence_all.py"),
         "--data_dir", str(data_dir),
         "--out_dir", str(out_incidence)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str((scripts / "code_combined_code").resolve()) + os.pathsep + env.get("PYTHONPATH","")
    run([sys.executable, str(scripts / "code_combined_code" / "make_combined_figures_v2.py"),
         "--in_dir", str(data_dir),
         "--out_dir", str(out_combined_fig)], env=env)

    run([sys.executable, str(scripts / "combine.py"),
         "--data_dir", str(data_dir),
         "--out_prefix", str(out_struct / "structured_attribution_comparison"),
         "--lam", "5.0",
         "--n_boot", "50",
         "--seed", "42",
         "--top_k", "7"])

    merged_long = out_reg / "merged_long.csv"
    run([sys.executable, str(scripts / "build_merged_long.py"),
         "--combined_tidy", str(combined_tidy),
         "--out_csv", str(merged_long)])

    run([sys.executable, str(scripts / "module_regression_and_plot.py"),
         "--input_csv", str(merged_long),
         "--out_dir", str(out_reg)])

    zip_path = project_root / "outputs_bundle.zip"
    zip_dir(out_dir, zip_path)

    print("\nAll done.")
    print("Outputs directory:", out_dir)
    print("Zipped outputs:", zip_path)

if __name__ == "__main__":
    main()
