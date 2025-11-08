# quick_check_dart_eval.py
import os, sys, traceback
from pathlib import Path

CODE_ROOT = os.environ.get("CODE_ROOT", "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")
DART_LOCAL_DIR = os.environ.get("DART_LOCAL_DIR", f"{CODE_ROOT}/data/GEM_dart")
TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR", "/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B")
SUBSET = int(os.environ.get("SUBSET_SIZE", "64"))

sys.path.insert(0, CODE_ROOT)
os.environ["DART_LOCAL_DIR"] = DART_LOCAL_DIR

print("="*80)
print("DART pipeline quick check")
print(f"CODE_ROOT     = {CODE_ROOT}")
print(f"DART_LOCAL_DIR= {DART_LOCAL_DIR}")
print(f"TOKENIZER_DIR = {TOKENIZER_DIR}")
print(f"SUBSET        = {SUBSET}")
print("="*80)

from transformers import AutoTokenizer
from dataset.dart_data import DartDataset
from dataset.collator import DataCollator
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

def show_df_brief(df: pd.DataFrame, k=3):
    cols = [c for c in ("tripleset","source","text") if c in df.columns]
    print(f"DataFrame shape={df.shape}, cols={list(df.columns)}")
    for i in range(min(k, len(df))):
        row = df.iloc[i]
        def typ(v):
            return f"{type(v).__name__}" + ("[ndarray]" if isinstance(v, np.ndarray) else "")
        print(f"[{i}] types: tripleset={typ(row.get('tripleset'))}, source={typ(row.get('source'))}, text={typ(row.get('text'))}")
        print(f"    text-preview: {row.get('text')[:1] if isinstance(row.get('text'), list) else row.get('text')}")

def check_split(tokenizer, split, mode, subset, use_cache):
    print("-"*80)
    print(f"Check split='{split}' mode='{mode}' use_cache={use_cache} subset={subset}")
    ds = DartDataset(tokenizer, split=split, mode=mode, use_cache=use_cache, subset_size=subset)
    print(f"len(DartDataset[{split},{mode}]) = {len(ds)}")
    # 读取 df 做结构自检
    df = ds.load_df()
    show_df_brief(df, k=3)
    if len(ds) > 0:
        s0 = ds[0]
        print(f"Sample[0] ok: input_ids={s0['input_ids'].shape}, label_ids={s0['label_ids'].shape}")
    return ds

def check_dataloader(tokenizer, ds, batch_size=2, steps=2):
    print("-"*80)
    print(f"Check DataLoader (batch_size={batch_size})")
    collate = DataCollator(tokenizer=tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    it = iter(dl)
    for i in range(steps):
        try:
            batch = next(it)
            if batch is None:
                print(f"Batch {i}: None (ERROR)")
                continue
            keys = list(batch.keys())
            print(f"Batch {i}: keys={keys}")
            for k in ("input_ids", "label_ids"):
                assert k in batch and batch[k] is not None, f"missing {k}"
                print(f"  {k}.shape={tuple(batch[k].shape)}")
        except StopIteration:
            print("No more batches.")
            break
        except Exception as e:
            print(f"Batch {i} error: {type(e).__name__}: {e}")
            traceback.print_exc()

def main():
    # 1) Tokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    print(f"tokenizer.sep_token={tok.sep_token} eos_token={tok.eos_token}")

    # 2) train(lm) - no cache
    ds_train = check_split(tok, "train", "lm", SUBSET, use_cache=False)
    # 3) val(gen)  - no cache
    ds_val_nc = check_split(tok, "val", "gen", SUBSET, use_cache=False)
    check_dataloader(tok, ds_val_nc, batch_size=2, steps=2)

    # 4) warm val(gen) cache
    print("-"*80)
    print("Warm val(gen) cache (full or subset, depending on SUBSET)")
    ds_val_cached = DartDataset(tok, split="val", mode="gen", use_cache=True, subset_size=SUBSET)
    print(f"len(val_gen cached) = {len(ds_val_cached)}")
    check_dataloader(tok, ds_val_cached, batch_size=2, steps=2)

if __name__ == "__main__":
    main()