#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility helpers for normalizing dataset/task names across plotting scripts.

Some experiment exports use the shorter name "commonsense" while others keep
the original "commonsense_170k" suffix. Downstream scripts expect a stable
identifier, so we canonicalize all known aliases to "commonsense_170k". This
prevents the commonsense results from being dropped when the filename lacks
the suffix.
"""
from __future__ import annotations

from typing import Iterable

COMMONSENSE_CANONICAL = "commonsense_170k"
COMMONSENSE_ALIASES = {
    "commonsense",
    "commonsense_170k",
    "commonsense170k",
    "commonsense-170k",
}


def canonicalize_task_name(name: str) -> str:
    """Normalize dataset identifiers so downstream filters stay consistent."""
    raw = str(name).strip()
    if raw.lower() in COMMONSENSE_ALIASES:
        return COMMONSENSE_CANONICAL
    return raw


def commonsense_name_candidates(model: str, extra_aliases: Iterable[str] | None = None) -> list[str]:
    """
    Helper to enumerate plausible commonsense CSV filenames for a model.

    Parameters
    ----------
    model:
        Model prefix, e.g., "gla".
    extra_aliases:
        Optional iterable of additional suffixes to consider.
    """
    aliases = [COMMONSENSE_CANONICAL]
    if extra_aliases:
        aliases.extend(extra_aliases)
    aliases.extend(COMMONSENSE_ALIASES - {COMMONSENSE_CANONICAL})
    # maintain deterministic order so globbing remains stable
    aliases = [a for i, a in enumerate(aliases) if a not in aliases[:i]]
    return [f"{model}_{alias}.csv" for alias in aliases]
