from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def is_experiment_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    return any((d / cp / "trainer_state.json").is_file() for cp in d.glob("checkpoint-*/"))


def find_dataset_dirs(base_dir: Path, dataset_filter: str | None = None) -> List[Path]:
    """Return list of dataset-level directories under base_dir.
    Examples of dataset_dir names: glue-tvt_rte_seed87
    """
    ds = []
    for d in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        name = d.name.lower()
        if dataset_filter is None:
            ds.append(d)
        else:
            if dataset_filter in name:
                ds.append(d)
    return ds


def find_experiment_dirs(dataset_dir: Path) -> List[Path]:
    return sorted([d for d in dataset_dir.iterdir() if is_experiment_dir(d)])


