"""
Local-first HuggingFace dataset resolver for zh-LAT-peft.

Goal:
- Prefer loading datasets from a local directory (offline-friendly)
- Fall back to HuggingFace Hub dataset IDs when local data is not available

This is used by commonsense_reasoning datasets:
  BoolQ / PIQA / SocialIQA / HellaSwag / WinoGrande / ARC / OpenBookQA

Env:
- LAT_DATA_DIR / DATA_DIR: root directory containing dataset repos downloaded by `hf download`
  Example: /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# Map HF dataset IDs (as used in our code) to local directory names (as downloaded by user).
_DATASET_ID_TO_LOCAL_SUBDIR = {
    # BoolQ
    "google/boolq": "boolq",
    # PIQA
    "piqa": "piqa",
    "ybisk/piqa": "piqa",
    # SocialIQA
    "allenai/social_i_qa": "siqa",
    # HellaSwag
    "Rowan/hellaswag": "hellaswag",
    # WinoGrande
    "allenai/winogrande": "winogrande",
    # ARC
    "allenai/ai2_arc": "ai2_arc",
    # OpenBookQA
    "allenai/openbookqa": "openbookqa",
}


def _get_data_root() -> Optional[Path]:
    # Explicit env wins
    root = os.environ.get("LAT_DATA_DIR") or os.environ.get("DATA_DIR")
    if root:
        p = Path(root).expanduser()
        return p if p.is_dir() else None

    # Default: repo-relative ./data when present
    p = Path("data")
    return p if p.is_dir() else None


def resolve_dataset_path(dataset_id: str) -> str:
    """
    Resolve a dataset identifier to a local path if available.

    Returns:
        - Local directory path string if dataset exists locally
        - Otherwise returns original dataset_id (HF hub id)
    """
    root = _get_data_root()
    if root is None:
        return dataset_id

    # Known mapping first
    sub = _DATASET_ID_TO_LOCAL_SUBDIR.get(dataset_id)
    candidates = []
    if sub:
        candidates.append(root / sub)
    # Common fallbacks
    candidates.append(root / dataset_id.split("/")[-1])
    candidates.append(root / dataset_id.replace("/", "_"))

    for cand in candidates:
        try:
            if cand.is_dir():
                return str(cand)
        except Exception:
            continue

    return dataset_id


