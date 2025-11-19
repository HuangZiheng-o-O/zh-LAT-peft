#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Ensure imports work when called via absolute path
mp_dir = Path(__file__).resolve().parents[1]  # .../mamba-peft
if str(mp_dir) not in sys.path:
    sys.path.insert(0, str(mp_dir))

from dataset import load_dataset
from mamba_ssm_peft.utils.hf import load_gla_tokenizer


def count_contiguous_zeros_left_right(mask_row: torch.Tensor) -> tuple[int, int]:
    # mask_row: (T,) with 1 for tokens, 0 for pad
    t = mask_row.numel()
    # left zeros
    i = 0
    while i < t and mask_row[i].item() == 0:
        i += 1
    left_zeros = i
    # right zeros
    j = t - 1
    while j >= 0 and mask_row[j].item() == 0:
        j -= 1
    right_zeros = (t - 1) - j
    if right_zeros < 0:
        right_zeros = 0
    return left_zeros, right_zeros


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Tokenizer source (e.g., fla-hub/gla-1.3B-100B)")
    ap.add_argument("--data", required=True, help="Dataset alias (e.g., spider-tvt)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--samples", type=int, default=64, help="Number of batches to inspect")
    args = ap.parse_args()

    tok = load_gla_tokenizer(args.model, trust_remote_code=True)
    # Enforce decoder-only friendly left padding
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"[debug] tokenizer: padding_side={tok.padding_side}, pad_token_id={tok.pad_token_id}, eos_token_id={tok.eos_token_id}")

    dm = load_dataset(args.data, tok, args.split, return_module=True, mode="gen")
    dl = DataLoader(dm.dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dm.data_collator)

    total_batches = 0
    any_right = 0
    for batch in dl:
        total_batches += 1
        attn = batch["attention_mask"]
        right_tail_nonzero = (attn.size(1) > 0) and (attn[:, -1] == 0).any().item()
        if right_tail_nonzero:
            any_right += 1
            print(f"[debug] batch#{total_batches}: right-padding DETECTED (attention_mask has trailing zeros)")
        else:
            print(f"[debug] batch#{total_batches}: OK (no trailing pad)")
        # detailed per-row stats (first 8 rows)
        rows = min(attn.size(0), 8)
        for i in range(rows):
            lz, rz = count_contiguous_zeros_left_right(attn[i])
            print(f"  row {i}: left_pad={lz} right_pad={rz} seq_len={attn.size(1)}")
        if total_batches >= args.samples:
            break

    print(f"[summary] inspected_batches={total_batches}, batches_with_right_padding={any_right}")
    if any_right > 0:
        print("[summary] Found right-padding in evaluation loader. Ensure tokenizer.padding_side='left' and that this collator is being used in your eval path.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()


