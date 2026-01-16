"""
Audit Sparse-Selective trainable counting (RetNet/GLA/DeltaNet).

Goal:
  - Prove that our current "trainable budget" computations are correct by:
    1) Printing EXACT per-module contributions to K_ref for a reference LoRA config
    2) Printing the current config's LoRA trainables K_A the same way
    3) For each sparse scope, printing the expected final trainable count used for training:
         - lora_only/base_only/hybrid: expected_total = K_ref
         - lora_dense_base_sparse: expected_total = K_ref, with base_budget = K_ref - K_A
    4) Printing candidate pool sizes (in elements) to check feasibility (candidate >= needed budget)

This script is intentionally "audit-first":
  - No training
  - No datasets needed
  - Loads model to read actual weight shapes (2D) and match target_modules by suffix.

Recommended usage on server (RetNet example):

  conda activate mzsz
  cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft

  python tools/audit_sparse_trainables.py \
    --model-type retnet \
    --model-id /home/user/mzs_h/model/retnet-1.3B-100B/ \
    --prec bf16 \
    --current-yaml cfg/my_lora_exp/yaml/sparse_modes/VOONLY__SPARSE_LoraDenseBaseSparse_REF_QKVOMLP.yaml \
    --e31-suite


Absolute-path variant:

  python tools/audit_sparse_trainables.py \
    --model-type retnet \
    --model-id /home/user/mzs_h/model/retnet-1.3B-100B/ \
    --prec bf16 \
    --current-yaml /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp/yaml/sparse_modes/VOONLY__SPARSE_LoraDenseBaseSparse_REF_QKVOMLP.yaml \
    --e31-suite

Notes:
  - If PEFT init (PiSSA) causes injection issues in your environment, this script does NOT need
    to inject adapters to compute counts; it uses weight shapes directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import yaml

# ---------------------------------------------------------------------------
# Import-path bootstrap (server-friendly)
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
MBA_ROOT = THIS_DIR.parent  # .../mamba-peft
REPO_ROOT = MBA_ROOT.parent  # .../zh-LAT-peft


def _ensure_import_paths() -> None:
    """
    Make this script runnable from:
      - repo root
      - mamba-peft/
      - mamba-peft/tools/
      - arbitrary cwd (tmux runners, etc.)
    """
    candidates = [
        str(MBA_ROOT),
        str(REPO_ROOT),
        str(THIS_DIR),
        os.getcwd(),
    ]
    for p in candidates:
        if p and p not in sys.path:
            sys.path.insert(0, p)


_ensure_import_paths()

# Lazy imports after sys.path bootstrap
try:
    from mamba_ssm_peft.utils.lat_model_loader import load_lat_model  # type: ignore
except ModuleNotFoundError as e:
    # Second-chance: sometimes the script is copied and __file__/parents differ; add more parents.
    more = [str(THIS_DIR.parents[i]) for i in range(0, min(6, len(THIS_DIR.parents)))]
    for p in more:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    try:
        from mamba_ssm_peft.utils.lat_model_loader import load_lat_model  # type: ignore
    except ModuleNotFoundError:
        print("[FATAL] Cannot import 'mamba_ssm_peft'.")
        print(f"  __file__   = {__file__}")
        print(f"  THIS_DIR   = {THIS_DIR}")
        print(f"  MBA_ROOT   = {MBA_ROOT}  (exists={MBA_ROOT.is_dir()})")
        print(f"  REPO_ROOT  = {REPO_ROOT} (exists={REPO_ROOT.is_dir()})")
        print(f"  Expect dir = {MBA_ROOT / 'mamba_ssm_peft'} (exists={(MBA_ROOT / 'mamba_ssm_peft').is_dir()})")
        print("  sys.path (head):")
        for i, p in enumerate(sys.path[:25]):
            print(f"    [{i:02d}] {p}")
        raise e

from utils.sparse_selective_engine import estimate_lora_trainable_count  # noqa: E402


@dataclass(frozen=True)
class PeftSpec:
    name: str
    r: int
    target_modules: List[str]
    peft_json_path: Path


def _read_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text())


def _resolve_repo_relative(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (MBA_ROOT / p).resolve()


def _load_peft_json(peft_json_path: Path) -> Dict:
    return json.loads(peft_json_path.read_text())


def _peft_spec_from_yaml(yaml_path: Path) -> PeftSpec:
    cfg = _read_yaml(yaml_path)
    peft_rel = cfg.get("peft")
    if not peft_rel:
        raise ValueError(f"YAML has no 'peft:' field: {yaml_path}")
    peft_json_path = _resolve_repo_relative(peft_rel)
    peft_json = _load_peft_json(peft_json_path)
    targets = list(peft_json.get("target_modules") or [])
    r = int(peft_json.get("r") or 0)
    if not targets or r <= 0:
        raise ValueError(f"Invalid peft json: targets={targets} r={r} path={peft_json_path}")
    return PeftSpec(
        name=yaml_path.stem,
        r=r,
        target_modules=targets,
        peft_json_path=peft_json_path,
    )


def _iter_linear_like_weights_by_suffix(
    model: torch.nn.Module,
    suffixes: List[str],
) -> Iterable[Tuple[str, torch.Tensor, str]]:
    """
    Yield (module_name, weight2d, matched_suffix) for modules whose name endswith matched_suffix
    and expose a 2D `.weight`.
    """
    for module_name, module in model.named_modules():
        for sfx in suffixes:
            if not module_name.endswith(sfx):
                continue
            w = getattr(module, "weight", None)
            if isinstance(w, (torch.nn.Parameter, torch.Tensor)) and w.dim() == 2:
                yield module_name, w, sfx
            break


def _audit_lora_budget(
    *,
    model: torch.nn.Module,
    spec: PeftSpec,
    title: str,
    limit_list: int = 2000,
) -> Dict:
    """
    Print per-module contributions to LoRA trainable count for (targets,r).
    Returns summary dict including computed K and match counts.
    """
    # Compute K via the shared implementation used by sparse engine.
    K = int(estimate_lora_trainable_count(model, spec.target_modules, spec.r))

    # Now print breakdown by explicitly listing matched modules and shapes.
    matches = []
    total_by_breakdown = 0
    for module_name, w, sfx in _iter_linear_like_weights_by_suffix(model, spec.target_modules):
        out, in_ = int(w.shape[0]), int(w.shape[1])
        contrib = int(spec.r * (in_ + out))
        total_by_breakdown += contrib
        matches.append((module_name, sfx, out, in_, contrib))

    print("")
    print("=" * 90)
    print(f"[AUDIT] {title}")
    print(f"  peft_json = {spec.peft_json_path}")
    print(f"  r = {spec.r}")
    print(f"  target_modules (suffix match) = {spec.target_modules}")
    print(f"  matched_modules = {len(matches)}")
    print(f"  K_ref(estimate_lora_trainable_count) = {K:,d}")
    print(f"  K_ref(breakdown sum)               = {total_by_breakdown:,d}")
    if K != total_by_breakdown:
        print("  [WARN] K mismatch between estimator and explicit breakdown (this indicates a bug).")

    print("-" * 90)
    print("  Per-module contributions (module_name | matched_suffix | out x in | r*(in+out))")
    for i, (mn, sfx, out, in_, contrib) in enumerate(matches[:limit_list]):
        print(f"  [{i:04d}] {mn} | {sfx} | {out}x{in_} | {contrib}")
    if len(matches) > limit_list:
        print(f"  ... truncated ({len(matches) - limit_list} more)")

    return {
        "K": K,
        "K_breakdown": total_by_breakdown,
        "matched_modules": len(matches),
        "targets": spec.target_modules,
        "r": spec.r,
        "peft_json_path": str(spec.peft_json_path),
    }


def _candidate_base_pool_all_minus_gate(model: torch.nn.Module) -> int:
    """
    Approximate base candidate elems for lora_dense_base_sparse scope:
      include q/k/v/o + MLP (gate/up/down), exclude attention gates g_proj/gk_proj.
    This mirrors sparse_selective_engine._default_all_minus_gate_targets + gate filters.
    """
    include_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "query", "key", "value"]
    total = 0
    for module_name, w, _sfx in _iter_linear_like_weights_by_suffix(model, include_suffixes):
        nn = module_name.lower()
        if nn.endswith("g_proj") or ".g_proj" in nn:
            continue
        if nn.endswith("gk_proj") or ".gk_proj" in nn:
            continue
        total += int(w.numel())
    return int(total)


def _print_sparse_contract(
    *,
    model: torch.nn.Module,
    current: PeftSpec,
    reference: PeftSpec,
    scope: str,
) -> None:
    """
    Print the expected final trainables under each scope in match_reference mode.
    """
    K_ref = int(estimate_lora_trainable_count(model, reference.target_modules, reference.r))
    K_A = int(estimate_lora_trainable_count(model, current.target_modules, current.r))

    scope = scope.lower().strip()
    print("")
    print("=" * 90)
    print(f"[CONTRACT] match_reference scope={scope}")
    print(f"  Current(A) peft = {current.peft_json_path.name}  r={current.r}  targets={current.target_modules}")
    print(f"  Reference(B) yaml/peft = {reference.name} / {reference.peft_json_path.name}  r={reference.r}  targets={reference.target_modules}")
    print(f"  K_A (dense LoRA params by formula) = {K_A:,d}")
    print(f"  K_ref (reference dense LoRA params by formula) = {K_ref:,d}")

    if scope in ("lora_only", "base_only", "hybrid"):
        expected_total = K_ref
        print(f"  Expected FINAL trainables (post-sparse) = K_ref = {expected_total:,d}")
        # Feasibility: candidate pool size check
        if scope == "lora_only":
            candidate = K_A  # LoRA pool size equals dense LoRA param elements
        elif scope == "base_only":
            # base pool uses current YAML's target modules
            candidate = 0
            for _mn, w, _sfx in _iter_linear_like_weights_by_suffix(model, current.target_modules):
                candidate += int(w.numel())
        else:  # hybrid
            base_cand = 0
            for _mn, w, _sfx in _iter_linear_like_weights_by_suffix(model, current.target_modules):
                base_cand += int(w.numel())
            candidate = int(base_cand) + int(K_A)
        print(f"  Candidate elems (approx) = {candidate:,d}")
        if candidate < expected_total:
            print("  [FAIL] candidate < K_ref (this run must fail under strict match_reference).")
        else:
            print("  [OK] candidate >= K_ref (feasible under strict match_reference).")
        return

    if scope == "lora_dense_base_sparse":
        # Contract: K_base = K_ref - K_A, total = K_ref
        K_base = int(K_ref) - int(K_A)
        print(f"  Derived base sparse budget = K_ref - K_A = {K_base:,d}")
        base_candidate = _candidate_base_pool_all_minus_gate(model)
        print(f"  Base candidate elems (all-minus-gate, approx) = {base_candidate:,d}")
        if K_base <= 0:
            print("  [FAIL] K_base <= 0 (reference budget not larger than current LoRA).")
        elif base_candidate < K_base:
            print("  [FAIL] base_candidate < K_base (not enough base elems to fill budget).")
        else:
            print("  [OK] feasible: base_candidate >= K_base and K_base > 0.")
        print(f"  Expected FINAL trainables (post-sparse) = dense LoRA (K_A) + base_sparse (K_base) = K_ref = {K_ref:,d}")
        return

    print(f"  [WARN] Unknown scope '{scope}' (no contract printed).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", required=True, help="gla|retnet|delta_net|mamba2|auto")
    ap.add_argument("--model-id", required=True, help="HF model id or local path")
    ap.add_argument("--prec", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--inject-peft-json",
        default=None,
        help="Optional PEFT json path to inject BEFORE auditing (enables catching PEFT shadow-key issues).",
    )

    ap.add_argument("--current-yaml", help="A config YAML (contains peft:) to audit", default=None)
    ap.add_argument("--reference-yaml", help="B reference YAML (contains peft:) to audit", default=None)
    ap.add_argument("--scope", help="scope for contract print", default="lora_dense_base_sparse")
    ap.add_argument("--e31-suite", action="store_true", help="Audit the 4 E31 REF YAMLs contract-wise")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.prec]
    device = "cpu" if args.device == "cpu" else "cuda"

    if args.inject_peft_json:
        # Optional PEFT injection: helps catch PEFT wrapper shadow-key issues for all_linear.
        from lat_adapter import prepare_lat_model_and_tokenizer  # type: ignore

        print(f"[LOAD] Loading model WITH PEFT injection for audit: peft={args.inject_peft_json}")
        debug = bool(args.device == "cpu")
        model, _tok, _pcfg = prepare_lat_model_and_tokenizer(
            model_type=args.model_type,
            model_id=args.model_id,
            prec=args.prec,
            debug=debug,
            peft_json_path=str(_resolve_repo_relative(args.inject_peft_json)),
        )
        # Validate all_linear candidate pool strictly (this is the failure mode you hit in training).
        try:
            from utils.sparse_selective_engine import _resolve_base_pool_params, _validate_base_pool_strict  # type: ignore

            base_params = _resolve_base_pool_params(
                model=model,
                base_pool="all_linear",
                current_targets=[],
                base_pool_peft_json=None,
            )
            _validate_base_pool_strict(model=model, base_pool="all_linear", base_params=base_params, model_type="AUDIT")
            print("[AUDIT] all_linear base pool validation: OK (no duplicates / no PEFT shadow keys / complete coverage).")
        except Exception as e:
            print(f"[AUDIT][FAIL] all_linear base pool validation failed under PEFT injection: {e}")
            raise
    else:
        print("[LOAD] Loading base model (no PEFT injection) for shape-audit...")
        loaded = load_lat_model(
            model_type=args.model_type,
            model_id=args.model_id,
            trust_remote_code=True,
            device=device,
            dtype=dtype,
        )
        model = loaded["model"]
    model.eval()

    # E31 audit: print contract for the 4 YAMLs (match_reference oriented).
    if args.e31_suite:
        e31_yamls = [
            "cfg/my_lora_exp/yaml/sparse_modes/QKVOMLP__SPARSE_LoraOnly_REF_VO.yaml",
            "cfg/my_lora_exp/yaml/sparse_modes/QKVOMLP__SPARSE_BaseOnly_REF_VO.yaml",
            "cfg/my_lora_exp/yaml/sparse_modes/VOONLY__SPARSE_Hybrid_REF_QKVOMLP.yaml",
            "cfg/my_lora_exp/yaml/sparse_modes/VOONLY__SPARSE_LoraDenseBaseSparse_REF_QKVOMLP.yaml",
        ]
        for y in e31_yamls:
            yp = _resolve_repo_relative(y)
            cfg = _read_yaml(yp)
            cur = _peft_spec_from_yaml(yp)
            ss = cfg.get("sparse_selective") or {}
            scope = str(ss.get("scope") or "lora_only")
            ref_path = ss.get("reference_cfg")
            if not ref_path:
                print(f"[E31][SKIP] {yp.name}: missing sparse_selective.reference_cfg")
                continue
            ref = _peft_spec_from_yaml(_resolve_repo_relative(ref_path))
            _audit_lora_budget(model=model, spec=cur, title=f"E31 CURRENT (A): {yp.name}")
            _audit_lora_budget(model=model, spec=ref, title=f"E31 REFERENCE (B): {Path(ref_path).name}")
            _print_sparse_contract(model=model, current=cur, reference=ref, scope=scope)
        return

    if not args.current_yaml or not args.reference_yaml:
        raise SystemExit("Provide --current-yaml and --reference-yaml, or use --e31-suite")

    cur_yaml = _resolve_repo_relative(args.current_yaml)
    ref_yaml = _resolve_repo_relative(args.reference_yaml)
    cur = _peft_spec_from_yaml(cur_yaml)
    ref = _peft_spec_from_yaml(ref_yaml)

    _audit_lora_budget(model=model, spec=cur, title=f"CURRENT (A): {cur_yaml.name}")
    _audit_lora_budget(model=model, spec=ref, title=f"REFERENCE (B): {ref_yaml.name}")
    _print_sparse_contract(model=model, current=cur, reference=ref, scope=args.scope)


if __name__ == "__main__":
    main()
