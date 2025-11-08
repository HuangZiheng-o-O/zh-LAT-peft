#!/usr/bin/env python3
# tools/check_loader_workers.py
import os, sys, traceback

# Ensure repo root is on sys.path BEFORE importing project modules
CODE_ROOT = os.environ.get("CODE_ROOT", "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

os.environ.setdefault("DART_LOCAL_DIR", f"{CODE_ROOT}/data/GEM_dart")

from torch.utils.data import DataLoader
from dataset.dart_data import DartDataset
from dataset.collator import DataCollator
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    os.environ.get("TOKENIZER_DIR", "/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B"),
    trust_remote_code=True,
)

def probe(split, mode, batch_size=4, num_workers=4, steps=2):
    print(f"\n== {split}:{mode} bs={batch_size} workers={num_workers} ==")
    ds = DartDataset(tok, split=split, mode=mode, use_cache=True)
    print("len(ds)=", len(ds))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    collate_fn=DataCollator(tokenizer=tok), drop_last=True)
    it = iter(dl)
    for i in range(steps):
        try:
            batch = next(it)
            print(f"batch{i} keys:", list(batch.keys()))
            for k in ("input_ids","label_ids"): print(k, batch[k].shape)
        except Exception as e:
            print(f"[ERROR] batch{i}: {type(e).__name__}: {e}")
            traceback.print_exc()
            break

if __name__ == "__main__":
    workers = int(os.environ.get("NUM_DATA_WORKERS", "0"))
    probe("train","lm", batch_size=4, num_workers=workers)
    probe("val","gen",  batch_size=1, num_workers=workers)