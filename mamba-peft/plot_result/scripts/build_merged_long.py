#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import pandas as pd
import numpy as np

MODULE_COLS = ["Q","K","V","O","G","GK","MLP_gate","MLP_updown"]

def _z(x: pd.Series) -> pd.Series:
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=0))
    if (not np.isfinite(sd)) or sd <= 1e-12:
        return (x - mu) * 0.0
    return (x - mu) / sd

def parse_modules(exp: str) -> dict:
    s = str(exp)
    flags = {c:0 for c in MODULE_COLS}

    # Attention modules
    if s.startswith("QKVO"):
        flags["Q"]=flags["K"]=flags["V"]=flags["O"]=1
    else:
        if s.startswith("QK"):
            flags["Q"]=1; flags["K"]=1
        if s.startswith("QV"):
            flags["Q"]=1; flags["V"]=1
        if s.startswith("QO"):
            flags["Q"]=1; flags["O"]=1
        if s.startswith("KV"):
            flags["K"]=1; flags["V"]=1
        if s.startswith("KO"):
            flags["K"]=1; flags["O"]=1
        if s.startswith("VO"):
            flags["V"]=1; flags["O"]=1
        if s.startswith("QONLY"):
            flags["Q"]=1
        if s.startswith("KONLY"):
            flags["K"]=1
        if s.startswith("VONLY"):
            flags["V"]=1
        if s.startswith("OONLY"):
            flags["O"]=1
        if s.startswith("OMLP"):
            flags["O"]=1

    # G / GK
    if "plus_GK" in s or s.startswith("GKONLY") or "_plus_GK" in s:
        flags["GK"]=1
    if "plus_G" in s or s.startswith("GPROJONLY") or s.startswith("GATINGONLY"):
        flags["G"]=1

    # MLP variants
    if s.startswith("MLPGATEONLY"):
        flags["MLP_gate"]=1
    elif s.startswith("MLPUPDOWN"):
        flags["MLP_updown"]=1
    elif "MLP" in s:  # includes MLPONLY, OMLP, plus_MLP
        flags["MLP_gate"]=1
        flags["MLP_updown"]=1

    return flags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_tidy", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.combined_tidy)
    need = ["model","task","exp_norm","trainable_ratio","score"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"combined_tidy missing columns: {missing}")

    df = df.copy()
    df["metric_z"] = df.groupby("task")["score"].transform(_z)

    mods = df["exp_norm"].apply(parse_modules).apply(pd.Series)
    for c in MODULE_COLS:
        if c not in mods.columns:
            mods[c]=0
    mods = mods[MODULE_COLS].astype(float)

    out = pd.concat(
        [df[["metric_z","trainable_ratio","task","model"]].reset_index(drop=True), mods.reset_index(drop=True)],
        axis=1,
    )
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
