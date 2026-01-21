# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path


def _resolve_default_data_dir() -> str:
    env_override = os.environ.get("DATA_DIR")
    if env_override:
        return env_override

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "dataset_summaries2",
        Path("/mnt/data/work/data/dataset_summaries2"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    # fall back to the historical default even if it doesn't exist yet
    return str(candidates[-1])


DEFAULT_DATA_DIR = _resolve_default_data_dir()
DEFAULT_OUTPUT_DIR = os.environ.get("OUT_DIR", "outputs")
