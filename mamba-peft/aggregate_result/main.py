from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Optional

from .fs_utils import find_dataset_dirs, find_experiment_dirs
from .glue_metrics import normalize_dataset_name
from .aggregator import aggregate_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate GLUE experiments: CSVs, plots, summaries.")
    p.add_argument("--base_dir", type=str, required=True,
                   help="Base dir containing dataset folders (e.g., /home/user/mzs_h/output/benchmark/glue)")
    p.add_argument("--dataset", type=str, default=None,
                   help="Optional dataset filter (e.g., glue-tvt_rte or rte). If omitted, process all under base_dir.")
    p.add_argument("--output", type=str, required=True,
                   help="Output directory to write aggregated reports.")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel workers (defaults to cpu count)")
    return p.parse_args()


def resolve_dataset_filter(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = normalize_dataset_name(s)
    return t if t else None


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    ds_filter = resolve_dataset_filter(args.dataset)
    dataset_dirs = find_dataset_dirs(base_dir, dataset_filter=ds_filter)
    if not dataset_dirs:
        print(f"No dataset directories found under {base_dir} with filter={ds_filter}")
        return

    print(f"Found {len(dataset_dirs)} dataset dirs")
    for ds in dataset_dirs:
        print(f"  - {ds.name}")

    tasks = []
    for ds in dataset_dirs:
        ds_name = ds.name
        # derive normalized dataset (e.g., rte) for metric spec
        norm = normalize_dataset_name(ds_name)
        exp_dirs = find_experiment_dirs(ds)
        out_dir = out_root / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for exp in exp_dirs:
            tasks.append((exp, norm, out_dir / exp.name))

    def _run(exp_dir: Path, norm_ds: str, out_dir: Path):
        try:
            rep = aggregate_experiment(exp_dir, norm_ds, out_dir)
            return {"ok": True, "exp": exp_dir.name, "out": str(out_dir)}
        except Exception as e:
            return {"ok": False, "exp": exp_dir.name, "error": str(e)}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda t: _run(*t), tasks))

    ok, bad = 0, 0
    for r in results:
        if r["ok"]:
            ok += 1
        else:
            bad += 1
            print(f"ERROR {r['exp']}: {r['error']}")

    summary = {"ok": ok, "bad": bad, "total": len(results)}
    with open(out_root / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregate done. ok={ok} bad={bad} total={len(results)}")


if __name__ == "__main__":
    main()


