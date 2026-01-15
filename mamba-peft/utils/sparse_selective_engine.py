"""
Sparse Selective Tuning Engine (Gradient + Static + Global Top-K).

This module implements the minimal, reusable pieces needed to add:
  - Sparse-LoRA  (mask within LoRA parameters)
  - Sparse-Base  (mask within selected base weights)
  - Sparse-Hybrid (union of LoRA + selected base weights)

Design goals:
  - Backward compatible by default (disabled unless env enables).
  - Static mask only: compute once at init, save to output_dir, reuse on resume.
  - Global top-K over the candidate pool (no per-layer budget splitting).
  - Works without modifying GenericLMTrainer: we apply grad hooks before trainer construction.

NOTE: This is unstructured (parameter-level) sparsity; it does not change inference structure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from trainer.loss import CrossEntropy


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


@dataclass(frozen=True)
class SparseSelectiveConfig:
    enabled: bool = False
    scope: str = "lora_only"  # lora_only | base_only | hybrid
    budget_mode: str = "fixed_ratio"  # fixed_ratio | fixed_count | match_reference
    rho: float = 0.3  # used when fixed_ratio
    k: Optional[int] = None  # used when fixed_count
    reference_cfg: Optional[str] = None  # used when match_reference (YAML path)
    score_samples: int = 1024
    # Engine fixed choices (per requirement)
    salience: str = "gradient"  # only supported
    ranking: str = "global"  # only supported

    @staticmethod
    def from_env() -> "SparseSelectiveConfig":
        # Default OFF for strict backward compatibility.
        enabled = _truthy(os.environ.get("HP_SPARSE_ENABLE") or os.environ.get("LAT_SPARSE_ENABLE"))
        scope = os.environ.get("HP_SPARSE_SCOPE") or os.environ.get("LAT_SPARSE_SCOPE") or "lora_only"
        budget_mode = os.environ.get("HP_SPARSE_BUDGET_MODE") or os.environ.get("LAT_SPARSE_BUDGET_MODE") or "fixed_ratio"
        rho = _env_float("HP_SPARSE_RHO", None)
        if rho is None:
            rho = _env_float("LAT_SPARSE_RHO", 0.3) or 0.3
        k = _env_int("HP_SPARSE_K", None)
        if k is None:
            k = _env_int("LAT_SPARSE_K", None)
        reference_cfg = os.environ.get("HP_SPARSE_REFERENCE_CFG") or os.environ.get("LAT_SPARSE_REFERENCE_CFG")
        score_samples = _env_int("HP_SPARSE_SCORE_SAMPLES", None)
        if score_samples is None:
            score_samples = _env_int("LAT_SPARSE_SCORE_SAMPLES", 1024) or 1024
        return SparseSelectiveConfig(
            enabled=enabled,
            scope=str(scope),
            budget_mode=str(budget_mode),
            rho=float(rho),
            k=k,
            reference_cfg=reference_cfg,
            score_samples=int(score_samples),
        )


def _match_targets(name: str, targets: List[str]) -> bool:
    # Match PEFT-style targets (often suffixes like "attn.k_proj" or "q_proj").
    return any(name.endswith(t) for t in targets)


def _iter_linear_weight_params(model: torch.nn.Module, targets: List[str]) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    """
    Yield (param_name, param) for base weight parameters whose module names match targets.
    We only include tensors that look like linear weights (endswith '.weight').
    """
    # We match modules by module name, then pick their `.weight`.
    for module_name, module in model.named_modules():
        if not _match_targets(module_name, targets):
            continue
        w = getattr(module, "weight", None)
        if isinstance(w, torch.nn.Parameter):
            # Resolve full parameter name by scanning named_parameters once per module.
            # Common: "<module_name>.weight"
            yield f"{module_name}.weight", w


def _iter_lora_params(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    """
    Yield LoRA trainable tensors (A/B etc.) by name heuristic.
    This project uses HF PEFT LoRA, so parameter names typically contain 'lora_'.
    """
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # HF PEFT LoRA params usually contain these substrings.
        if "lora_" in name or ".lora_A" in name or ".lora_B" in name:
            yield name, p


def _load_peft_targets_and_rank_from_yaml(yaml_path: str) -> Tuple[List[str], int]:
    """
    For match_reference: load reference YAML, then load its `peft` JSON and return (target_modules, r).
    """
    import yaml as _yaml

    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"reference_cfg not found: {yaml_path}")
    cfg = _yaml.safe_load(p.read_text())
    peft_path = cfg.get("peft")
    if not peft_path:
        raise ValueError(f"reference_cfg has no 'peft' field: {yaml_path}")
    peft_p = Path(peft_path)
    if not peft_p.is_absolute():
        # Interpret relative to mamba-peft/ (same as how YAML is typically written).
        peft_p = (Path(__file__).resolve().parents[1] / peft_p).resolve()
    peft_json = json.loads(peft_p.read_text())
    targets = peft_json.get("target_modules") or []
    r = int(peft_json.get("r") or 0)
    if not targets or r <= 0:
        raise ValueError(f"Invalid reference peft json (targets={targets}, r={r}) from: {peft_p}")
    return list(targets), r


def estimate_lora_trainable_count(model: torch.nn.Module, targets: List[str], r: int) -> int:
    """
    Estimate dense LoRA trainable param count (A+B) for a given (targets, r) on this instantiated model.
    For each matched linear weight W of shape [out, in], LoRA adds:
      A: [r, in]  and  B: [out, r]  => r*(in+out)
    """
    total = 0
    for module_name, module in model.named_modules():
        if not _match_targets(module_name, targets):
            continue
        w = getattr(module, "weight", None)
        if not isinstance(w, torch.nn.Parameter):
            continue
        if w.dim() != 2:
            continue
        out, in_ = int(w.shape[0]), int(w.shape[1])
        total += r * (in_ + out)
    return int(total)


def _set_requires_grad_for_scope(
    model: torch.nn.Module,
    scope: str,
    base_params: Dict[str, torch.nn.Parameter],
    lora_params: Dict[str, torch.nn.Parameter],
) -> None:
    """
    Enforce scope-level trainability. This is ONLY called when sparse is enabled.
    - lora_only: keep existing requires_grad as-is (PEFT already freezes base).
    - base_only: freeze everything, then unfreeze selected base params.
    - hybrid: keep existing (LoRA) and additionally unfreeze selected base params.
    """
    scope = scope.lower().strip()
    if scope == "lora_only":
        return
    if scope == "base_only":
        for _, p in model.named_parameters():
            p.requires_grad = False
        for p in base_params.values():
            p.requires_grad = True
        return
    if scope == "hybrid":
        # Keep existing LoRA trainable, additionally unfreeze base candidates.
        for p in base_params.values():
            p.requires_grad = True
        for p in lora_params.values():
            p.requires_grad = True
        return
    raise ValueError(f"Unknown sparse scope: {scope}")


def compute_gradient_salience_scores(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    candidate_params: Dict[str, torch.nn.Parameter],
    num_examples: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Accumulate |grad| over `num_examples` examples for candidate parameters.
    Returns CPU float32 tensors matching each param shape.
    """
    model.train()
    crit = CrossEntropy()

    scores: Dict[str, torch.Tensor] = {}
    for n, p in candidate_params.items():
        scores[n] = torch.zeros_like(p, dtype=torch.float32, device="cpu")

    seen = 0
    for batch in dataloader:
        # Batch schema: project datasets use {input_ids, label_ids, attention_mask?}
        if batch is None:
            continue
        input_ids = batch.get("input_ids")
        label_ids = batch.get("label_ids") or batch.get("labels")
        attention_mask = batch.get("attention_mask")
        if input_ids is None or label_ids is None:
            continue

        # Move to device
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        model.zero_grad(set_to_none=True)
        out = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits
        loss = crit(logits, label_ids)
        loss.backward()

        for n, p in candidate_params.items():
            if p.grad is None:
                continue
            scores[n] += p.grad.detach().abs().to(dtype=torch.float32, device="cpu")

        seen += int(input_ids.shape[0])
        if seen >= num_examples:
            break

    return scores


def global_topk_mask_from_scores(
    scores: Dict[str, torch.Tensor],
    k: int,
) -> Dict[str, torch.Tensor]:
    """
    Build per-parameter boolean masks (CPU) from per-parameter score tensors (CPU).
    Uses a two-level topk:
      - per-tensor topk with k (global) to reduce work for huge tensors
      - then global topk over the union
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    # Build union of candidates.
    parts: List[Tuple[str, int, int, torch.Tensor, torch.Tensor]] = []
    # (name, start, end, local_idx, local_vals)
    union_vals: List[torch.Tensor] = []
    pos = 0
    for name, s in scores.items():
        flat = s.flatten()
        n = int(flat.numel())
        local_k = min(k, n)
        if local_k == n:
            local_vals = flat
            local_idx = torch.arange(n, dtype=torch.int64)
        else:
            local_vals, local_idx = torch.topk(flat, local_k, sorted=False)
        start = pos
        end = pos + int(local_vals.numel())
        parts.append((name, start, end, local_idx, local_vals))
        union_vals.append(local_vals)
        pos = end

    all_vals = torch.cat(union_vals, dim=0)
    if k >= int(all_vals.numel()):
        # everything selected
        global_sel = torch.arange(int(all_vals.numel()), dtype=torch.int64)
    else:
        _, global_sel = torch.topk(all_vals, k, sorted=False)

    # Build masks per tensor.
    mask_dict: Dict[str, torch.Tensor] = {}
    global_sel = global_sel.to(torch.int64)
    for name, start, end, local_idx, _local_vals in parts:
        # indices into union that fall into [start,end)
        in_chunk = (global_sel >= start) & (global_sel < end)
        sel = global_sel[in_chunk] - start
        # map union positions -> local indices
        chosen_local_idx = local_idx[sel]

        # create flat mask
        flat_mask = torch.zeros(int(scores[name].numel()), dtype=torch.bool)
        if int(chosen_local_idx.numel()) > 0:
            flat_mask[chosen_local_idx] = True
        mask_dict[name] = flat_mask.view_as(scores[name])

    return mask_dict


def apply_gradient_mask_hooks(model: torch.nn.Module, mask_dict: Dict[str, torch.Tensor]) -> None:
    """
    Apply param.register_hook(grad * mask) for each named parameter in mask_dict.
    Mask tensors are registered as module buffers so they move with model.to(...).
    """
    name_to_param = dict(model.named_parameters())
    name_to_module = dict(model.named_modules())
    for full_name, mask_cpu in mask_dict.items():
        p = name_to_param.get(full_name)
        if p is None:
            continue
        # Ensure mask on same device/dtype as grad for multiplication.
        mask = mask_cpu.to(device=p.device)
        module_name, param_name = full_name.rsplit(".", 1)
        module = name_to_module.get(module_name)
        if module is None:
            continue
        buf_name = f"{param_name}__sparse_mask"
        # Avoid duplicate registration on resume.
        if not hasattr(module, buf_name):
            module.register_buffer(buf_name, mask)
        else:
            setattr(module, buf_name, mask)

        def _hook(grad, _module=module, _buf_name=buf_name):
            m = getattr(_module, _buf_name)
            # Ensure mask is on grad device (should already be)
            if m.device != grad.device:
                m = m.to(grad.device)
                setattr(_module, _buf_name, m)
            return grad * m

        p.register_hook(_hook)


def maybe_run_sparse_selective_tuning(
    *,
    model: torch.nn.Module,
    train_dataset,
    data_collator,
    batch_size: int,
    output_dir: str,
    cfg_path: str,
    model_type: str,
) -> Optional[Dict]:
    """
    Main entry: if enabled, compute/load mask, apply hooks, and persist metadata.
    Returns metadata dict (also written to disk), or None if disabled.
    """
    cfg = SparseSelectiveConfig.from_env()
    if not cfg.enabled:
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "sparse_selective_mask.pt"
    meta_path = out_dir / "sparse_selective_meta.json"

    # Determine candidate set for base weights from PEFT config (current cfg yaml contains peft json path).
    # We derive base target modules from the CURRENT YAML's peft json (not from defaults), because
    # the user wants candidate pool scoped to YAML target modules.
    import yaml as _yaml
    yaml_cfg = _yaml.safe_load(Path(cfg_path).read_text())
    peft_json_path = yaml_cfg.get("peft")
    targets: List[str] = []
    if peft_json_path:
        peft_p = Path(peft_json_path)
        if not peft_p.is_absolute():
            peft_p = (Path(__file__).resolve().parents[1] / peft_p).resolve()
        try:
            peft_json = json.loads(peft_p.read_text())
            targets = list(peft_json.get("target_modules") or [])
        except Exception:
            targets = []

    base_params = dict(_iter_linear_weight_params(model, targets)) if targets else {}
    lora_params = dict(_iter_lora_params(model))

    # Enforce scope (only when enabled)
    _set_requires_grad_for_scope(model, cfg.scope, base_params, lora_params)

    # Candidate pool is defined by scope.
    candidate_params: Dict[str, torch.nn.Parameter] = {}
    if cfg.scope.lower() == "lora_only":
        candidate_params = dict(lora_params)
    elif cfg.scope.lower() == "base_only":
        candidate_params = dict(base_params)
    elif cfg.scope.lower() == "hybrid":
        candidate_params = {**base_params, **lora_params}
    else:
        raise ValueError(f"Unknown sparse scope: {cfg.scope}")

    candidate_elems = int(sum(int(p.numel()) for p in candidate_params.values()))
    if candidate_elems <= 0:
        raise RuntimeError(f"[{model_type}] sparse enabled but candidate pool is empty (scope={cfg.scope}, targets={targets})")

    # Budget resolution
    budget_k: int
    if cfg.budget_mode == "fixed_ratio":
        budget_k = int(max(1, min(candidate_elems, int(cfg.rho * candidate_elems))))
    elif cfg.budget_mode == "fixed_count":
        if cfg.k is None or cfg.k <= 0:
            raise ValueError("fixed_count requires HP_SPARSE_K (positive int)")
        budget_k = int(min(candidate_elems, cfg.k))
    elif cfg.budget_mode == "match_reference":
        if not cfg.reference_cfg:
            raise ValueError("match_reference requires HP_SPARSE_REFERENCE_CFG (reference YAML path)")
        ref_targets, ref_r = _load_peft_targets_and_rank_from_yaml(cfg.reference_cfg)
        k_ref = estimate_lora_trainable_count(model, ref_targets, ref_r)
        if k_ref <= 0:
            raise RuntimeError(f"Computed K_ref=0 from reference_cfg={cfg.reference_cfg}")
        budget_k = int(min(candidate_elems, k_ref))
    else:
        raise ValueError(f"Unknown budget_mode: {cfg.budget_mode}")

    # Load or compute mask
    if mask_path.exists():
        saved = torch.load(mask_path, map_location="cpu")
        mask_dict = saved["mask_dict"]
        # ensure requires_grad for masked params (base scope)
        for n in mask_dict.keys():
            p = dict(model.named_parameters()).get(n)
            if p is not None:
                p.requires_grad = True
    else:
        # Build a small scoring dataloader (no workers, deterministic-ish)
        score_bs = max(1, min(batch_size, 4))
        dl = DataLoader(
            train_dataset,
            batch_size=score_bs,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=0,
        )
        device = next(model.parameters()).device
        scores = compute_gradient_salience_scores(
            model=model,
            dataloader=dl,
            candidate_params=candidate_params,
            num_examples=cfg.score_samples,
            device=device,
        )
        mask_dict = global_topk_mask_from_scores(scores, budget_k)
        torch.save(
            {
                "mask_dict": mask_dict,
                "budget_k": budget_k,
                "candidate_elems": candidate_elems,
                "scope": cfg.scope,
                "budget_mode": cfg.budget_mode,
                "rho": cfg.rho,
                "reference_cfg": cfg.reference_cfg,
                "score_samples": cfg.score_samples,
                "salience": cfg.salience,
                "ranking": cfg.ranking,
                "cfg_path": cfg_path,
            },
            mask_path,
        )

    # Apply hooks (grad *= mask)
    apply_gradient_mask_hooks(model, mask_dict)

    # Compute realized K for logging
    realized_k = int(sum(int(m.to(torch.int64).sum().item()) for m in mask_dict.values()))

    meta = {
        "enabled": True,
        "scope": cfg.scope,
        "budget_mode": cfg.budget_mode,
        "rho": cfg.rho,
        "budget_k": int(budget_k),
        "realized_k": int(realized_k),
        "candidate_elems": int(candidate_elems),
        "score_samples": int(cfg.score_samples),
        "salience": cfg.salience,
        "ranking": cfg.ranking,
        "reference_cfg": cfg.reference_cfg,
        "targets_from_current_peft": targets,
        "mask_path": str(mask_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"[{model_type}][sparse] enabled scope={cfg.scope} budget_mode={cfg.budget_mode} ranking=global salience=gradient")
    print(f"[{model_type}][sparse] candidate_elems={candidate_elems:,d} budget_k={budget_k:,d} realized_k={realized_k:,d}")
    print(f"[{model_type}][sparse] saved: {mask_path.name}, {meta_path.name}")

    return meta


