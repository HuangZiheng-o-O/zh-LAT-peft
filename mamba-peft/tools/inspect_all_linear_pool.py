"""
Inspect base candidate pools for Sparse Selective Tuning (strict, PEFT-safe).

This script loads a LAT model (optionally with PEFT/LoRA injected) and prints:
  - all modules (name -> class)
  - a chosen base pool (all_linear / from_peft_json / from_current_peft) (param_name -> shape)
  - duplicate detection (same Parameter object referenced by multiple pool keys)
  - omission hints (linear-like 2D weights that are NOT collected, excluding embeddings)

Usage (run from repo root or mamba-peft/):

python /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/tools/inspect_all_linear_pool.py \
  --model /home/user/mzs_h/model/retnet-1.3B-100B/ \
  --model-type retnet \
  --prec bf16 \
  --peft-json /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/peft/lora_vo_r8_alpha16.json \
  --base-pool all_linear \
  --out-dir /tmp/pool_all_linear


python /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/tools/inspect_all_linear_pool.py \
  --model /home/user/mzs_h/model/retnet-1.3B-100B/ \
  --model-type retnet \
  --prec bf16 \
  --peft-json /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/peft/lora_vo_r8_alpha16.json \
  --base-pool from_peft_json \
  --base-pool-peft-json /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/peft/lora_qkvo_plus_mlp_r8_alpha16.json \
  --out-dir /tmp/pool_qkvomlp


"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml


def _repo_root() -> Path:
    # .../zh-LAT-peft
    return Path(__file__).resolve().parents[2]


def _mamba_peft_root() -> Path:
    return _repo_root() / "mamba-peft"


def _resolve_mamba_peft_rel_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (_mamba_peft_root() / pp).resolve()


def _is_peft_lora_linear(module: torch.nn.Module) -> bool:
    return (
        module.__class__.__module__.startswith("peft.tuners.lora.layer")
        and module.__class__.__name__ == "Linear"
    )

def _get_base_linear_from_peft_linear(module: torch.nn.Module) -> torch.nn.Module:
    """
    Best-effort extraction of the wrapped base layer from a PEFT LoRA Linear wrapper.
    Mirrors the logic used in sparse_selective_engine, but kept local to avoid version coupling.
    """
    candidates = []
    candidates.append(getattr(module, "base_layer", None))
    candidates.append(getattr(module, "linear", None))
    if hasattr(module, "get_base_layer") and callable(getattr(module, "get_base_layer")):
        try:
            candidates.append(module.get_base_layer())  # type: ignore[attr-defined]
        except Exception:
            pass
    for base in candidates:
        if base is None:
            continue
        w = getattr(base, "weight", None)
        if isinstance(w, (torch.nn.Parameter, torch.Tensor)) and getattr(w, "dim", lambda: -1)() == 2:
            return base
    raise TypeError(f"PEFT Linear wrapper does not expose a usable base layer with 2D `.weight`: {type(module)}")


def iter_all_linear_pool(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    """
    Local implementation of 'all_linear' candidate pool:
      - iterate model.named_modules()
      - exclude names containing 'lora_'
      - include nn.Linear.weight
      - include PEFT LoRA wrapper's base_layer.weight

    IMPORTANT (PEFT):
      PEFT wrappers commonly expose child modules like "<wrapper>.base_layer" which may itself be an nn.Linear.
      If we include BOTH "<wrapper>.weight" (mapped to base_layer.weight) and "<wrapper>.base_layer.weight",
      sparse replacement will attempt to sparsify the same layer twice and crash.
      Therefore we SKIP nn.Linear modules that are direct children of a PEFT LoRA wrapper.
    """
    out: List[Tuple[str, torch.nn.Parameter]] = []
    name_to_mod: Dict[str, torch.nn.Module] = dict(model.named_modules())
    for module_name, module in model.named_modules():
        if "lora_" in module_name:
            continue
        if isinstance(module, torch.nn.Linear):
            if _is_shadow_base_layer(module_name, name_to_mod):
                continue
            out.append((f"{module_name}.weight", module.weight))
            continue
        if _is_peft_lora_linear(module):
            base = _get_base_linear_from_peft_linear(module)
            w = getattr(base, "weight", None)
            if isinstance(w, torch.nn.Parameter) and w.dim() == 2:
                out.append((f"{module_name}.weight", w))
            continue
    return out


def _match_targets(name: str, targets: List[str]) -> bool:
    return any(name.endswith(t) for t in targets)


def iter_target_linear_pool(model: torch.nn.Module, targets: List[str]) -> List[Tuple[str, torch.nn.Parameter]]:
    """
    Build a target-based pool (from_current_peft / from_peft_json).
    Mirrors sparse_selective_engine semantics:
      - match by suffix
      - accept nn.Linear or PEFT LoRA Linear wrapper
      - for PEFT wrapper, use base_layer.weight
    """
    name_to_mod: Dict[str, torch.nn.Module] = dict(model.named_modules())
    out: List[Tuple[str, torch.nn.Parameter]] = []
    for module_name, module in model.named_modules():
        if not _match_targets(module_name, targets):
            continue
        if "lora_" in module_name:
            continue
        if isinstance(module, torch.nn.Linear):
            if _is_shadow_base_layer(module_name, name_to_mod):
                continue
            out.append((f"{module_name}.weight", module.weight))
            continue
        if _is_peft_lora_linear(module):
            base = _get_base_linear_from_peft_linear(module)
            w = getattr(base, "weight", None)
            if isinstance(w, torch.nn.Parameter) and getattr(w, "dim", lambda: -1)() == 2:
                out.append((f"{module_name}.weight", w))
            continue
        raise TypeError(
            f"Target module '{module_name}' matched by targets={targets} is not nn.Linear or PEFT LoRA Linear wrapper (got {type(module)})"
        )
    return out


def _parent_name(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else ""


def _is_shadow_base_layer(name: str, name_to_mod: Dict[str, torch.nn.Module]) -> bool:
    """
    PEFT LoRA Linear wrapper commonly exposes a `.base_layer` submodule.
    The sparse engine intends to treat the WRAPPER as the unit, and sparsify its base_layer weight.
    If we also include `<wrapper>.base_layer` as a separate "linear", that is a likely duplicate view.
    """
    if not (name.endswith(".base_layer") or name.endswith(".linear")):
        return False
    parent = _parent_name(name)
    pm = name_to_mod.get(parent)
    return pm is not None and _is_peft_lora_linear(pm)

def _is_lora_internal_name(name: str) -> bool:
    """
    Exclude LoRA internal modules from the BASE pool expectation.

    PEFT typically creates module paths containing:
      - lora_A / lora_B
      - lora_dropout
      - lora_embedding_A / lora_embedding_B

    These are NOT part of the backbone/base-weight pool and should never be expected there.
    """
    return "lora_" in name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="", help="Optional YAML path (if provided, may contain `peft:` field)")
    ap.add_argument("--peft-json", default="", help="Optional PEFT JSON path (overrides cfg.peft when set)")
    ap.add_argument("--model", default="", help="HF model id or local path. If omitted, uses env LAT_MODEL/GLA_MODEL")
    ap.add_argument("--model-type", default="auto", help="gla|retnet|delta_net|mamba2|auto")
    ap.add_argument("--prec", default="bf16", help="bf16|fp16|fp32")
    ap.add_argument("--debug", action="store_true", help="CPU mode (no CUDA)")
    ap.add_argument(
        "--base-pool",
        default="all_linear",
        choices=["all_linear", "from_peft_json", "from_current_peft"],
        help="Which base pool to inspect (matches sparse_selective_engine semantics).",
    )
    ap.add_argument(
        "--base-pool-peft-json",
        default="",
        help="Required when --base-pool=from_peft_json. Path to a PEFT json defining target_modules.",
    )
    ap.add_argument("--out-dir", default="", help="Optional directory to write dumps")
    ap.add_argument("--max-print", type=int, default=200, help="Max entries to print per section (stdout)")
    args = ap.parse_args()

    model_id = str(args.model).strip()
    if model_id == "":
        model_id = str(os.environ.get("LAT_MODEL") or os.environ.get("GLA_MODEL") or "").strip()
    if model_id == "":
        raise ValueError("Missing model id/path. Provide --model, or set env LAT_MODEL (or GLA_MODEL).")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path: Path | None = None
    if str(args.cfg).strip() != "":
        cfg_path = Path(args.cfg).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (_repo_root() / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"--cfg not found: {cfg_path}")

    # Resolve PEFT JSON path: --peft-json > cfg.peft > None
    peft_path_abs: Path | None = None
    if str(args.peft_json).strip() != "":
        peft_path_abs = _resolve_mamba_peft_rel_path(str(args.peft_json).strip())
        if not peft_path_abs.exists():
            raise FileNotFoundError(f"--peft-json not found: {peft_path_abs}")
    elif cfg_path is not None:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        peft_json_path = cfg.get("peft")
        peft_path_abs = _resolve_mamba_peft_rel_path(peft_json_path) if peft_json_path else None
        if peft_path_abs is not None and not peft_path_abs.exists():
            raise FileNotFoundError(f"peft json not found (from cfg.peft): {peft_path_abs}")

    # Make sure local modules are importable (same trick as train_lat.py)
    import sys

    sys.path.insert(0, str(_repo_root()))
    sys.path.insert(0, str(_mamba_peft_root()))

    from lat_adapter import prepare_lat_model_and_tokenizer  # noqa: WPS433
    # Note: we intentionally do NOT import private helpers from sparse_selective_engine here,
    # because server environments may have a different version of that module.

    model, tokenizer, _peft_cfg = prepare_lat_model_and_tokenizer(
        model_type=args.model_type,
        model_id=model_id,
        prec=args.prec,
        debug=bool(args.debug),
        peft_json_path=str(peft_path_abs) if peft_path_abs is not None else None,
    )

    # -------------------------
    # 1) Dump all modules
    # -------------------------
    name_to_mod: Dict[str, torch.nn.Module] = dict(model.named_modules())
    module_rows: List[Tuple[str, str]] = [(n, f"{m.__class__.__module__}.{m.__class__.__name__}") for n, m in name_to_mod.items()]
    module_rows.sort(key=lambda x: x[0])

    print("\n=== MODEL MODULES (name -> class) ===")
    for n, cls in module_rows[: max(1, args.max_print)]:
        print(f"{n}\t{cls}")
    if len(module_rows) > args.max_print:
        print(f"... (truncated) total_modules={len(module_rows)}")

    # -------------------------
    # 2) base pool (chosen)
    # -------------------------
    pool_kind = str(args.base_pool).strip()
    if pool_kind == "all_linear":
        pool_items = list(iter_all_linear_pool(model))
    elif pool_kind == "from_peft_json":
        if str(args.base_pool_peft_json).strip() == "":
            raise ValueError("--base-pool-peft-json is required when --base-pool=from_peft_json")
        bpj = _resolve_mamba_peft_rel_path(str(args.base_pool_peft_json).strip())
        peft_json = json.loads(bpj.read_text())
        targets = list(peft_json.get("target_modules") or [])
        if not targets:
            raise ValueError(f"base_pool_peft_json has empty target_modules: {bpj}")
        pool_items = list(iter_target_linear_pool(model, targets))
    elif pool_kind == "from_current_peft":
        if cfg_path is None:
            raise ValueError("--cfg is required when --base-pool=from_current_peft (to read cfg.peft target_modules)")
        cfg_for_pool = yaml.safe_load(cfg_path.read_text()) or {}
        peft_rel = cfg_for_pool.get("peft")
        if not peft_rel:
            raise ValueError(f"--cfg has no 'peft:' field: {cfg_path}")
        peft_p = _resolve_mamba_peft_rel_path(str(peft_rel))
        peft_json = json.loads(peft_p.read_text())
        targets = list(peft_json.get("target_modules") or [])
        if not targets:
            raise ValueError(f"cfg.peft json has empty target_modules: {peft_p}")
        pool_items = list(iter_target_linear_pool(model, targets))
    else:
        raise ValueError(f"Unknown base pool: {pool_kind}")

    pool: Dict[str, torch.nn.Parameter] = dict(pool_items)

    print(f"\n=== BASE POOL ({pool_kind}) (param_name -> shape, dtype, device) ===")
    pool_rows = []
    for k, p in pool.items():
        shape = tuple(p.shape)
        pool_rows.append((k, shape, str(p.dtype), str(p.device)))
    pool_rows.sort(key=lambda x: x[0])
    for k, shape, dt, dev in pool_rows[: max(1, args.max_print)]:
        print(f"{k}\t{shape}\t{dt}\t{dev}")
    if len(pool_rows) > args.max_print:
        print(f"... (truncated) total_pool_entries={len(pool_rows)}")

    # -------------------------
    # 3) Duplicate detection by Parameter identity
    # -------------------------
    id_to_keys: Dict[int, List[str]] = {}
    for k, p in pool.items():
        id_to_keys.setdefault(id(p), []).append(k)
    dup_groups = [(pid, keys) for pid, keys in id_to_keys.items() if len(keys) > 1]
    dup_groups.sort(key=lambda x: (-len(x[1]), x[1][0]))

    print("\n=== DUPLICATES (same Parameter object appears under multiple pool keys) ===")
    if not dup_groups:
        print("no_duplicates_by_identity")
    else:
        for pid, keys in dup_groups[: max(1, args.max_print)]:
            print(f"param_id={pid} count={len(keys)}")
            for kk in keys[:20]:
                print(f"  - {kk}")
            if len(keys) > 20:
                print("  - ...")
        if len(dup_groups) > args.max_print:
            print(f"... (truncated) total_dup_groups={len(dup_groups)}")

    unique_param_objs = len(id_to_keys)
    print(
        f"\n=== SUMMARY ===\n"
        f"total_modules={len(module_rows)}\n"
        f"pool_entries(dict_keys)={len(pool_rows)}\n"
        f"pool_unique_parameter_objects={unique_param_objs}\n"
        f"duplicate_groups={len(dup_groups)}"
    )

    # -------------------------
    # 3.5) PEFT shadow-key preflight (the exact failure mode in sparse_selective_engine)
    # -------------------------
    shadow_keys = [k for k in pool.keys() if k.endswith(".base_layer.weight") or k.endswith(".linear.weight")]
    if shadow_keys:
        print("\n=== ERROR: SHADOW base_layer keys are present in the pool (this will crash sparse replacement) ===")
        for k in sorted(shadow_keys)[: max(1, args.max_print)]:
            print(f"SHADOW_KEY\t{k}")
        if len(shadow_keys) > args.max_print:
            print("... (truncated)")
        print("Fix: all_linear pool must NOT include '<wrapper>.base_layer.weight' when '<wrapper>' is a PEFT LoRA Linear wrapper.")
        raise SystemExit(2)

    # -------------------------
    # 4) Omission hints
    # -------------------------
    # Define "expected backbone linears" as:
    #   - nn.Linear modules that are NOT PEFT wrapper children (.base_layer/.linear under PEFT wrapper)
    #   - PEFT LoRA Linear wrapper modules
    expected_linear_modules: List[str] = []
    for n, m in name_to_mod.items():
        if _is_lora_internal_name(n):
            continue
        if _is_peft_lora_linear(m):
            expected_linear_modules.append(n)
            continue
        if isinstance(m, torch.nn.Linear):
            if _is_shadow_base_layer(n, name_to_mod):
                continue
            expected_linear_modules.append(n)

    expected_linear_modules = sorted(set(expected_linear_modules))
    pool_module_names = sorted({k.rsplit(".weight", 1)[0] for k in pool.keys() if k.endswith(".weight")})
    missing_expected = sorted(set(expected_linear_modules) - set(pool_module_names))

    print("\n=== COVERAGE CHECK ===")
    if pool_kind == "all_linear":
        print(f"[all_linear] expected_linear_modules={len(expected_linear_modules)} pool_module_names={len(pool_module_names)} missing_expected={len(missing_expected)}")
        for n in missing_expected[: max(1, args.max_print)]:
            m = name_to_mod.get(n)
            cls = f"{m.__class__.__module__}.{m.__class__.__name__}" if m is not None else "<?>"
            print(f"MISSING\t{n}\t{cls}")
        if len(missing_expected) > args.max_print:
            print("... (truncated)")
    else:
        # Recompute expected matches under target suffix rules for this pool.
        targets_for_pool: List[str] = []
        if pool_kind == "from_peft_json":
            bpj = _resolve_mamba_peft_rel_path(str(args.base_pool_peft_json).strip())
            targets_for_pool = list((json.loads(bpj.read_text()) or {}).get("target_modules") or [])
        elif pool_kind == "from_current_peft" and cfg_path is not None:
            cfg_for_pool = yaml.safe_load(cfg_path.read_text()) or {}
            peft_p = _resolve_mamba_peft_rel_path(str(cfg_for_pool.get("peft")))
            targets_for_pool = list((json.loads(peft_p.read_text()) or {}).get("target_modules") or [])
        expected_t: List[str] = []
        for mn, mod in name_to_mod.items():
            if "lora_" in mn:
                continue
            if _is_shadow_base_layer(mn, name_to_mod):
                continue
            if _match_targets(mn, targets_for_pool) and (isinstance(mod, torch.nn.Linear) or _is_peft_lora_linear(mod)):
                expected_t.append(mn)
        expected_set = set(expected_t)
        pool_set = set(pool_module_names)
        missing = sorted(expected_set - pool_set)
        extra = sorted(pool_set - expected_set)
        print(f"[{pool_kind}] expected_matches={len(expected_set)} pool_matches={len(pool_set)} missing={len(missing)} extra={len(extra)}")
        for n in missing[: max(1, args.max_print)]:
            m = name_to_mod.get(n)
            cls = f"{m.__class__.__module__}.{m.__class__.__name__}" if m is not None else "<?>"
            print(f"MISSING\t{n}\t{cls}")
        for n in extra[: max(1, args.max_print)]:
            m = name_to_mod.get(n)
            cls = f"{m.__class__.__module__}.{m.__class__.__name__}" if m is not None else "<?>"
            print(f"EXTRA\t{n}\t{cls}")

    # Extra: list 2D-weight modules that are not nn.Linear and not PEFT wrapper (excluding embeddings),
    # which may look like "linear-like" projections and thus be omissions by design.
    linear_like_omissions: List[Tuple[str, str, Tuple[int, int]]] = []
    for n, m in name_to_mod.items():
        if _is_lora_internal_name(n):
            continue
        if isinstance(m, torch.nn.Embedding):
            continue
        if isinstance(m, torch.nn.Linear) or _is_peft_lora_linear(m):
            continue
        w = getattr(m, "weight", None)
        if isinstance(w, (torch.nn.Parameter, torch.Tensor)) and getattr(w, "dim", lambda: -1)() == 2:
            linear_like_omissions.append((n, f"{m.__class__.__module__}.{m.__class__.__name__}", (int(w.shape[0]), int(w.shape[1]))))

    linear_like_omissions.sort(key=lambda x: x[0])
    print("\n=== 2D-WEIGHT NON-LINEAR MODULES (potential omissions by design) ===")
    print(f"count={len(linear_like_omissions)}")
    for n, cls, shape in linear_like_omissions[: max(1, args.max_print)]:
        print(f"{n}\t{cls}\t{shape}")
    if len(linear_like_omissions) > args.max_print:
        print("... (truncated)")

    # -------------------------
    # 5) Optional dump to files
    # -------------------------
    if out_dir is not None:
        (out_dir / "modules.tsv").write_text("\n".join([f"{n}\t{cls}" for n, cls in module_rows]) + "\n")
        (out_dir / "base_pool.tsv").write_text(
            "\n".join([f"{k}\t{tuple(p.shape)}\t{str(p.dtype)}\t{str(p.device)}" for k, p in pool.items()]) + "\n"
        )
        (out_dir / "duplicates.json").write_text(json.dumps({str(pid): keys for pid, keys in dup_groups}, indent=2) + "\n")
        (out_dir / "missing_expected.txt").write_text("\n".join(missing_expected) + "\n")
        (out_dir / "non_linear_2d_weight.tsv").write_text(
            "\n".join([f"{n}\t{cls}\t{shape}" for n, cls, shape in linear_like_omissions]) + "\n"
        )
        meta = {
            "cfg": str(cfg_path) if cfg_path is not None else None,
            "model": model_id,
            "model_type": args.model_type,
            "prec": args.prec,
            "debug": bool(args.debug),
            "peft_json": str(peft_path_abs) if peft_path_abs is not None else None,
            "base_pool": pool_kind,
            "base_pool_peft_json": str(args.base_pool_peft_json) if str(args.base_pool_peft_json).strip() != "" else None,
            "total_modules": len(module_rows),
            "pool_entries": len(pool_rows),
            "pool_unique_parameter_objects": unique_param_objs,
            "duplicate_groups": len(dup_groups),
            "expected_linear_modules": len(expected_linear_modules),
            "pool_module_names": len(pool_module_names),
            "missing_expected": len(missing_expected),
            "non_linear_2d_weight": len(linear_like_omissions),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"\nWrote dumps to: {out_dir}")


if __name__ == "__main__":
    main()
