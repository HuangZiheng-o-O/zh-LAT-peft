#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parses the aggregated experiment result file (e.g., 'all_best_checkpoints_top1.txt')
and creates structured tables in CSV and Excel formats.
"""
import re
import argparse
from pathlib import Path
import pandas as pd

# Regex to capture each experiment block from the aggregated file
pattern = re.compile(
    r"Experiment:\s*(?P<experiment>[^\n]+).*?"
    r"Score:\s*(?P<score>-?[0-9.]+)\s*"
    r"Step:\s*(?P<step>\d+)\s*"
    r"Path:\s*(?P<path>[^\n]+)",
    re.DOTALL
)

def parse_aggregated_file(file_path: str) -> pd.DataFrame:
    """Reads and parses the aggregated results file."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    rows = []
    for m in pattern.finditer(text):
        rows.append({
            "Experiment": m.group("experiment").strip(),
            "Score": float(m.group("score")),
            "Step": int(m.group("step")),
            "Path": m.group("path").strip(),
        })
    df = pd.DataFrame(rows, columns=["Experiment", "Score", "Step", "Path"])
    # Sort for convenience: Score desc, Step asc
    if not df.empty:
        df = df.sort_values(["Score", "Step"], ascending=[False, True]).reset_index(drop=True)
    return df

def run_parsing(aggregated_file: str, output_dir: str, excel_filename: str, csv_filename: str):
    """
    Main function to parse the file and generate outputs.
    
    Args:
        aggregated_file (str): Path to the input aggregated text file.
        output_dir (str): Directory to save the output files.
        excel_filename (str): Filename for the output Excel file.
        csv_filename (str): Filename for the output CSV file.
    """
    aggregated_file = Path(aggregated_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not aggregated_file.exists():
        print(f"[ERROR] Aggregated file not found: {aggregated_file}")
        return

    print(f"--- Parsing aggregated file: {aggregated_file.name} ---")
    df = parse_aggregated_file(str(aggregated_file))

    if df.empty:
        print("[WARN] No data was parsed from the aggregated file. No outputs will be generated.")
        return

    # --- Save to CSV ---
    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote CSV with {len(df)} entries to: {csv_path}")

    # --- Save to Excel ---
    excel_path = output_dir / excel_filename
    try:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Top_Checkpoints", index=False)
        print(f"[OK] Wrote Excel file to: {excel_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write Excel file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Parse an aggregated experiment results file into CSV and Excel.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--aggregated_file",
        type=str,
        required=True,
        help="Path to the aggregated .txt file (e.g., all_best_checkpoints_top1.txt)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output CSV and Excel files."
    )
    parser.add_argument(
        "--excel_filename",
        type=str,
        default="aggregated_results.xlsx",
        help="Filename for the output Excel file."
    )
    parser.add_argument(
        "--csv_filename",
        type=str,
        default="aggregated_results.csv",
        help="Filename for the output CSV file."
    )
    args = parser.parse_args()

    run_parsing(
        aggregated_file=args.aggregated_file,
        output_dir=args.output_dir,
        excel_filename=args.excel_filename,
        csv_filename=args.csv_filename
    )

if __name__ == "__main__":
    main()
