import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import DataLoader
import yaml

# Local imports (repo-relative)
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# Also add the hyphenated package directory so that 'dataset' and 'mamba_ssm_peft' are importable.
mp_dir = repo_root / "mamba-peft"
if str(mp_dir) not in sys.path:
    sys.path.insert(0, str(mp_dir))

# Robustly load prepare_gla_model_and_tokenizer without relying on 'mamba_peft' package name.
def _load_prepare_gla_model_and_tokenizer():
    try:
        # First try direct import with mp_dir on sys.path
        from train_gla_adapter import prepare_gla_model_and_tokenizer  # type: ignore
        return prepare_gla_model_and_tokenizer
    except Exception:
        import importlib.util
        adapter_path = repo_root / "mamba-peft" / "train_gla_adapter.py"
        spec = importlib.util.spec_from_file_location("train_gla_adapter", str(adapter_path))
        if spec is None or spec.loader is None:
            raise
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return getattr(mod, "prepare_gla_model_and_tokenizer")

prepare_gla_model_and_tokenizer = _load_prepare_gla_model_and_tokenizer()

from dataset import load_dataset
from mamba_ssm_peft import print_trainable_parameter_names, get_trainable_parameters_ratio
from typing import Optional


def _set_local_spider_dir(local_dir: Optional[str]) -> None:
    if local_dir:
        os.environ["SPIDER_LOCAL_DIR"] = local_dir


def _print_tokenizer_info(tokenizer) -> None:
    print("=== Tokenizer Special Tokens ===")
    print("bos_token:", getattr(tokenizer, "bos_token", None), getattr(tokenizer, "bos_token_id", None))
    print("eos_token:", getattr(tokenizer, "eos_token", None), getattr(tokenizer, "eos_token_id", None))
    print("pad_token:", getattr(tokenizer, "pad_token", None), getattr(tokenizer, "pad_token_id", None))
    print("sep_token:", getattr(tokenizer, "sep_token", None), getattr(tokenizer, "sep_token_id", None))
    print("===============================")


def _sample_dataset_examples(dataset_module, n: int = 3) -> List[Tuple[str, str]]:
    ds = dataset_module.dataset
    out = []
    for i in range(min(n, len(ds))):
        # SpiderDataset.get_input_label constructs raw strings; reuse preproc pipeline for exact I/O
        batch = ds[i]
        # batch is dict(input_ids=Tensor, label_ids=Tensor)
        in_ids = batch["input_ids"]
        lbl_ids = batch["label_ids"]
        out.append((in_ids, lbl_ids))
    return out


def _decode_pair(tokenizer, input_ids: torch.Tensor, label_ids: torch.Tensor) -> Tuple[str, str]:
    # inputs: keep specials for visibility
    input_txt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    # labels: strip specials to reflect real SQL
    label_txt = tokenizer.decode(label_ids[label_ids != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return input_txt, label_txt


def _hf_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    min_new_tokens: int,
    num_beams: Optional[int],
    attention_mask: Optional[torch.Tensor] = None,
):
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=int(max(0, min_new_tokens)),
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=False,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)),
    )
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if num_beams is not None and num_beams > 1:
        gen_kwargs["num_beams"] = int(num_beams)
        gen_kwargs["do_sample"] = False
    outputs = model.generate(**gen_kwargs)
    seq = outputs.sequences if hasattr(outputs, "sequences") else outputs
    # Trim prompt
    if seq.dim() == 2:
        seq = seq[:, input_ids.shape[1]:]
    return seq


def _make_leftpad_collator(tokenizer):
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)

    def _left_pad(vec: torch.Tensor, val: int, max_len: int) -> torch.Tensor:
        pad_len = max_len - int(vec.shape[0])
        if pad_len <= 0:
            return vec
        return torch.cat([torch.full((pad_len,), val, dtype=vec.dtype), vec], dim=0)

    def _right_pad(vec: torch.Tensor, val: int, max_len: int) -> torch.Tensor:
        pad_len = max_len - int(vec.shape[0])
        if pad_len <= 0:
            return vec
        return torch.nn.functional.pad(vec, (0, pad_len), value=val)

    def collate(instances):
        input_ids = [inst["input_ids"] for inst in instances]
        label_ids = [inst["label_ids"] for inst in instances]
        max_len_in = max(int(x.shape[0]) for x in input_ids)
        max_len_lab = max(int(y.shape[0]) for y in label_ids)
        input_ids = [_left_pad(x, pad_id, max_len_in) for x in input_ids]
        attention_mask = [x.ne(pad_id) for x in input_ids]
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        label_ids = [_right_pad(y, -100, max_len_lab) for y in label_ids]
        label_ids = torch.stack(label_ids, dim=0)
        return dict(input_ids=input_ids, label_ids=label_ids, attention_mask=attention_mask)

    return collate


def _collect_generation(model, tokenizer, dataset_module, max_new_tokens: int = 128, min_length: int = 0, num_beams: int = 1, k: int = 5):
    # Force left-padding in this debug path to satisfy decoder-only generate() checks
    dl = DataLoader(
        dataset_module.dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_make_leftpad_collator(tokenizer),
    )
    records = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dl):
            if idx >= k:
                break
            input_ids = batch["input_ids"].to(next(iter(model.parameters())).device)
            label_ids = batch["label_ids"].to(input_ids.device)
            pred_ids = _hf_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_length,
                num_beams=(num_beams if (num_beams is not None and num_beams > 1) else None),
                attention_mask=batch.get("attention_mask", None).to(input_ids.device) if "attention_mask" in batch else None,
            ).to(input_ids.device)
            # Decode: inputs keep specials; preds/labels strip specials
            in_txt = tokenizer.decode(input_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            pred_txt = tokenizer.decode(pred_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            label_txt = tokenizer.decode(label_ids[0][label_ids[0] != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            records.append({
                "idx": idx,
                "input": in_txt,
                "pred": pred_txt,
                "label": label_txt,
            })
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="fla-hub/gla-1.3B-100B")
    parser.add_argument("--peft_json", type=str, default=None)
    parser.add_argument("--no_peft", action="store_true", help="Force using base model only (ignore --peft_json).")
    parser.add_argument("--prec", type=str, default="bf16")
    parser.add_argument("--local_spider_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_length", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--out", type=str, default="debug_generation.yaml")
    args = parser.parse_args()

    _set_local_spider_dir(args.local_spider_dir)

    # Decide effective PEFT path (allow base-only testing via --no_peft)
    effective_peft = None if (args.no_peft or not args.peft_json) else args.peft_json
    if effective_peft is None:
        print(">>> Running in BASELINE mode (no PEFT/LoRA loaded)")
    else:
        print(f">>> Running with PEFT: {effective_peft}")

    # Load model/tokenizer (+LoRA if provided and not disabled)
    model, tokenizer, _ = prepare_gla_model_and_tokenizer(
        model_id=args.model_id,
        prec=args.prec,
        debug=False,
        peft_json_path=effective_peft,
    )

    # Ensure correct padding behavior for decoder-only generation
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    try:
        if getattr(tokenizer, "pad_token_id", None) is None:
            # Fallback: use eos as pad to avoid runtime warnings/errors
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass

    # ---- Deep debug: print out exactly what base model/tokenizer and libs are used ----
    print("\n=== Runtime & Model Introspection ===")
    # Environment caches
    for k in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_HUB_CACHE", "HF_ENDPOINT"):
        v = os.environ.get(k)
        print(f"{k}={v}")
    # Library versions and locations
    try:
        import transformers as _tf
        print(f"transformers.__version__={getattr(_tf, '__version__', None)}  file={getattr(_tf, '__file__', None)}")
    except Exception as _e:
        print(f"transformers import info error: {_e}")
    try:
        import huggingface_hub as _hh
        print(f"huggingface_hub.__version__={getattr(_hh, '__version__', None)}  file={getattr(_hh, '__file__', None)}")
    except Exception as _e:
        print(f"huggingface_hub import info error: {_e}")
    try:
        import fla as _fla
        print(f"fla module file={getattr(_fla, '__file__', None)}")
        print(f"fla version attr={getattr(_fla, '__version__', None)}")
    except Exception as _e:
        print(f"fla import info error: {_e}")
    # Model & tokenizer identities
    print(f"model.class={model.__class__.__module__}.{model.__class__.__name__}")
    print(f"model.name_or_path={getattr(model, 'name_or_path', None)}")
    if hasattr(model, 'config'):
        try:
            cfg = getattr(model, 'config')
            print(f"model.config._name_or_path={getattr(cfg, '_name_or_path', None)}")
            for key in ('model_type', 'architectures', 'vocab_size', 'hidden_size', 'num_hidden_layers'):
                if hasattr(cfg, key):
                    print(f"model.config.{key}={getattr(cfg, key)}")
        except Exception as _e:
            print(f"config introspection error: {_e}")
    try:
        first_param = next(iter(model.parameters()))
        print(f"model.device={first_param.device}  dtype={first_param.dtype}")
    except Exception:
        pass
    print(f"tokenizer.class={tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}")
    print(f"tokenizer.name_or_path={getattr(tokenizer, 'name_or_path', None)}  vocab_size={getattr(tokenizer, 'vocab_size', None)}")
    # Try to resolve cached config file path for the model id (best-effort)
    try:
        from huggingface_hub.file_download import hf_hub_url, try_to_load_from_cache
        cfg_url = hf_hub_url(args.model_id, filename="config.json")
        cached_cfg = try_to_load_from_cache(args.model_id, "config.json")
        print(f"hf_hub_url(config)={cfg_url}")
        print(f"cached_config_path={cached_cfg}")
    except Exception as _e:
        print(f"cache resolution info error: {_e}")
    print("=== End Introspection ===\n")

    _print_tokenizer_info(tokenizer)
    print_trainable_parameter_names(model)
    print("Trainable ratio:", get_trainable_parameters_ratio(model))

    # Load Spider dataset
    dm = load_dataset("spider-tvt", tokenizer, args.split, mode="gen", return_module=True)
    # Sample and print a few raw pairs
    print("\n=== Sample (prompt,label) ===")
    samples = _sample_dataset_examples(dm, n=min(args.n, 3))
    for i, (in_ids, lbl_ids) in enumerate(samples):
        in_txt, lbl_txt = _decode_pair(tokenizer, in_ids, lbl_ids)
        print(f"[{i}] PROMPT (head 400):\n{in_txt[:400]}")
        print(f"[{i}] LABEL:\n{lbl_txt}\n")

    # Collect generation records
    recs = _collect_generation(
        model, tokenizer, dm,
        max_new_tokens=args.max_new_tokens,
        min_length=args.min_length,
        num_beams=args.num_beams,
        k=args.n,
    )
    with open(args.out, "w") as f:
        yaml.safe_dump(recs, f, sort_keys=False, allow_unicode=True)
    print(f"\nSaved generation samples to {args.out}")


if __name__ == "__main__":
    main()


