#!/usr/bin/env python3
"""
process_and_aggregate.py

  请在您的项目根目录 (/Users/huangziheng/PycharmProjects/code/zh-LAT-peft/mamba-peft/) 中，使用以下命令来运行：
cd ~/mzs_h/code/zh-LAT-peft/mamba-peft/

python3 analysis_tools/process_and_aggregate.py
/home/user/mzs_h/output/benchmark/glue/cola_gla/
/home/user/mzs_h/output/all_agg_results/experiment_result/
/home/user/mzs_h/output/benchmark/glue/cola_gla/

This script orchestrates a two-stage pipeline:
1.  Processes raw experiment data to generate plots, CSVs, and checkpoint summary
    files (`best_checkpoints_top1.txt` and `best_checkpoints_top3.txt`) for each
    experiment, using the logic from `make_cola_figs_raw.py`.
2.  Aggregates the `best_checkpoints_top1.txt` from all processed experiments
    into a single summary file, using the logic from `aggregate_results.py`.

This script assumes that `make_cola_figs_raw.py` and `aggregate_results.py`
are in the same directory.
"""

import argparse
from pathlib import Path

# Import the main functions from the other scripts
import make_cola_figs_raw
import aggregate_results
import parse_aggregated_results

def main():
    parser = argparse.ArgumentParser(
        description="Run the full experiment processing and aggregation pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/user/mzs_h/output/benchmark/glue/cola_gla/",
        help="Base directory containing the raw experiment folders (e.g., with checkpoint-* subdirs)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user/mzs_h/output/all_agg_results/experiment_result/",
        help="Directory to save the processed experiment folders and the final aggregated report."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for processing experiments. Defaults to number of CPU cores."
    )
    args = parser.parse_args()

    # --- Stage 1: Process individual experiments ---
    print("--- Stage 1: Processing individual experiment results ---")
    make_cola_figs_raw.run_processing(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        workers=args.workers
    )

    # --- Stage 2: Aggregating top-1 checkpoint results ---
    print("\n--- Stage 2: Aggregating top-1 checkpoint results ---")
    aggregated_filename = "all_best_checkpoints_top1.txt"
    aggregate_results.aggregate_specific_text_files(
        target_dir=args.output_dir,
        filename_to_aggregate="best_checkpoints_top1.txt",
        output_filename=aggregated_filename
    )

    # --- Stage 3: Parse aggregated results into structured tables ---
    print("\n--- Stage 3: Parsing aggregated results to CSV and Excel ---")
    parse_aggregated_results.run_parsing(
        aggregated_file=str(Path(args.output_dir) / aggregated_filename),
        output_dir=args.output_dir,
        excel_filename="top1_checkpoints_summary.xlsx",
        csv_filename="top1_checkpoints_summary.csv"
    )

    print("\n=== Pipeline finished! ===")


if __name__ == "__main__":
    main()
