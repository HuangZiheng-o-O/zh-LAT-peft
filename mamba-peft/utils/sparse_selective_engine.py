"""
Sparse Selective Tuning Engine (Gradient + Static + Global Top-K) - Reparameterized.

This module implements the minimal, reusable pieces needed to add:
  - Sparse-LoRA  (mask within LoRA parameters)
  - Sparse-Base  (mask within selected base weights)
  - Sparse-Hybrid (union of LoRA + selected base weights)

Design goals:
  - Backward compatible by default (disabled unless env enables).
  - Static mask only: compute once at init, save to output_dir, reuse on resume.
  - Global top-K over the candidate pool (no per-layer budget splitting).
  - Works without modifying GenericLMTrainer: we replace modules before trainer construction.

IMPORTANT DIFFERENCE VS grad-mask:
  - This implementation performs *sparse re-parameterization* so optimizer state scales with K.
  - We do NOT keep dense trainable tensors and mask gradients; instead we replace each selected
    nn.Linear with a SparseDeltaLinear that has ONLY a length-K trainable vector and an index buffer.

NOTE: This is unstructured (parameter-level) sparsity; it does not change inference structure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from peft import PeftModel
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


def _is_peft_lora_linear(module: torch.nn.Module) -> bool:
    # Avoid importing PEFT internals directly; check qualified name.
    return (
        module.__class__.__module__.startswith("peft.tuners.lora.layer")
        and module.__class__.__name__ == "Linear"
    )


def _get_base_linear_from_peft_linear(module: torch.nn.Module) -> torch.nn.Linear:
    """
    PEFT LoRA wraps nn.Linear as peft.tuners.lora.layer.Linear and stores the original linear
    as `base_layer` (PEFT>=0.7). Some variants may use `linear`.
    """
    base = getattr(module, "base_layer", None)
    if isinstance(base, torch.nn.Linear):
        return base
    base2 = getattr(module, "linear", None)
    if isinstance(base2, torch.nn.Linear):
        return base2
    raise TypeError(
        f"PEFT Linear wrapper does not expose an nn.Linear base_layer/linear (got {type(module)})."
    )


def _iter_linear_weight_params(model: torch.nn.Module, targets: List[str]) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    """
    Yield (param_name, param) for base weight parameters whose module names match targets.
    We only include tensors that look like linear weights (endswith '.weight').
    """
    # We match modules by module name, then pick their `.weight`.
    for module_name, module in model.named_modules():
        if not _match_targets(module_name, targets):
            continue
        if isinstance(module, torch.nn.Linear):
            yield f"{module_name}.weight", module.weight
            continue
        if _is_peft_lora_linear(module):
            # When PEFT is enabled, target modules are wrapped; base sparsification should operate
            # on the underlying base_layer weight.
            base = _get_base_linear_from_peft_linear(module)
            yield f"{module_name}.weight", base.weight
            continue
        raise TypeError(
            f"Target module '{module_name}' matched by target_modules={targets} is not nn.Linear "
            f"(got {type(module)}). Sparse reparameterization requires nn.Linear or PEFT LoRA Linear wrapper."
        )


def _iter_lora_linear_weight_params(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    """
    Yield LoRA A/B *linear weight* parameters.
    We intentionally restrict to nn.Linear weights to support SparseDeltaLinear replacement.
    """
    name_to_module = dict(model.named_modules())
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not name.endswith(".weight"):
            continue
        # Typical PEFT structure: ...lora_A.<adapter>.weight / ...lora_B.<adapter>.weight
        # Or module path includes 'lora_A'/'lora_B'.
        if ("lora_A" not in name) and ("lora_B" not in name) and ("lora_" not in name):
            continue
        module_name = name.rsplit(".", 1)[0]
        mod = name_to_module.get(module_name)
        if mod is None:
            continue
        if not isinstance(mod, torch.nn.Linear):
            raise TypeError(
                f"LoRA parameter '{name}' is not owned by nn.Linear (owner={type(mod)}). "
                "Sparse reparameterization requires nn.Linear for lora_A/lora_B."
            )
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
        # Handle both bare nn.Linear and PEFT-wrapped linear.
        if isinstance(module, torch.nn.Linear):
            w = module.weight
        elif _is_peft_lora_linear(module):
            w = _get_base_linear_from_peft_linear(module).weight
        else:
            # Unknown module type matched by suffix; skip (fail-fast would be too strict for estimation).
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


def _freeze_all_params(model: torch.nn.Module) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False


def _dist_available() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _dist_rank() -> int:
    if not _dist_available():
        return 0
    import torch.distributed as dist
    return dist.get_rank()


def _dist_barrier() -> None:
    if _dist_available():
        import torch.distributed as dist
        dist.barrier()


def _assert_peft_adapter_saving_captures_sparse_params(model: PeftModel) -> None:
    """
    Fail-fast guard: if user is saving adapter-only (save_full_model=False),
    ensure PEFT's adapter state dict includes our SparseDeltaLinear trainables.
    """
    try:
        from peft.utils.save_and_load import get_peft_model_state_dict  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Cannot import peft.utils.save_and_load.get_peft_model_state_dict to validate adapter-only saving. "
            "Refuse to proceed because sparse params might not be saved. "
            "Set HP_SAVE_FULL_MODEL=1 to save full model.pt. "
            f"Import error: {e}"
        ) from e

    sd = get_peft_model_state_dict(model)
    # Our trainables are named like "...SparseDeltaLinear.delta" under modules that were lora_A/lora_B or base targets.
    # We check presence of any '.delta' key as minimal guarantee.
    has_delta = any(k.endswith(".delta") for k in sd.keys())
    if not has_delta:
        raise RuntimeError(
            "PEFT adapter-only state_dict does not include any SparseDeltaLinear '.delta' parameters. "
            "This means your sparse trainable params would NOT be saved by adapter saving. "
            "Set HP_SAVE_FULL_MODEL=1 (and keep save_total_limit=2) to save full weights safely."
        )


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


def global_topk_indices_from_scores(
    scores: Dict[str, torch.Tensor],
    k: int,
) -> Dict[str, torch.Tensor]:
    """
    Build per-parameter 1D index tensors (flattened positions, CPU int64) from per-parameter
    score tensors (CPU float32), using global top-k across the candidate pool.

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

    # Build selected indices per tensor.
    index_dict: Dict[str, torch.Tensor] = {}
    global_sel = global_sel.to(torch.int64)
    for name, start, end, local_idx, _local_vals in parts:
        # indices into union that fall into [start,end)
        in_chunk = (global_sel >= start) & (global_sel < end)
        sel = global_sel[in_chunk] - start
        # map union positions -> local indices
        chosen_local_idx = local_idx[sel]
        index_dict[name] = chosen_local_idx.to(torch.int64).cpu()

    return index_dict


class SparseDeltaLinear(torch.nn.Module):
    """
    A drop-in replacement for nn.Linear where the dense weight is frozen, and the only trainable
    parameters are a 1D vector of length K (sparse delta) plus an index buffer of length K.

    Forward computes:
      W_eff = W_base + scatter_add(delta at selected indices)
      y = x @ W_eff^T + b

    This matches ACL-2025 SPEFT's core idea (LinearSparse), and guarantees optimizer state O(K).
    """

    def __init__(
        self,
        *,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
        selected_idx_flat: torch.Tensor,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if base_weight.dim() != 2:
            raise ValueError(f"SparseDeltaLinear requires 2D weight, got {tuple(base_weight.shape)}")
        if selected_idx_flat.dtype != torch.int64:
            selected_idx_flat = selected_idx_flat.to(torch.int64)
        if selected_idx_flat.dim() != 1:
            raise ValueError("selected_idx_flat must be a 1D int64 tensor")

        self.in_features = int(base_weight.shape[1])
        self.out_features = int(base_weight.shape[0])
        self.alpha = float(alpha)

        # Frozen base weight/bias stored as buffers (saved in state_dict, not trainable).
        self.register_buffer("base_weight", base_weight.detach().clone(), persistent=True)
        if base_bias is not None:
            self.register_buffer("base_bias", base_bias.detach().clone(), persistent=True)
        else:
            self.base_bias = None  # type: ignore[assignment]

        # Selected indices (flattened into base_weight.view(-1))
        self.register_buffer("selected_idx", selected_idx_flat.detach().clone(), persistent=True)

        # Trainable sparse delta vector (optimizer state scales with its length).
        delta = torch.zeros(int(selected_idx_flat.numel()), device=base_weight.device, dtype=base_weight.dtype)
        self.delta = torch.nn.Parameter(delta, requires_grad=True)

        self.dropout = torch.nn.Dropout(p=float(dropout)) if float(dropout) > 0 else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_dtype = x.dtype
        x = x.to(self.base_weight.dtype)
        x = self.dropout(x)

        # Construct effective weight on-the-fly
        flat = self.base_weight.flatten()
        scaled = self.delta * self.alpha
        flat2 = torch.scatter_add(flat, dim=0, index=self.selected_idx, src=scaled)
        w_eff = flat2.view(self.out_features, self.in_features)

        b = self.base_bias if hasattr(self, "base_bias") else None
        out = F.linear(x, w_eff, b)
        return out.to(prev_dtype)


def save_sparse_delta_snapshot_if_present(model: torch.nn.Module, output_dir: str) -> Optional[str]:
    """
    Save a lightweight snapshot of all SparseDeltaLinear trainables (delta vectors + selected indices).
    This enables minimal checkpointing (O(K)) even when not saving full model weights.

    Writes: <output_dir>/sparse_delta.pt
    Returns the written path string if any SparseDeltaLinear exists, else None.
    """
    deltas: Dict[str, torch.Tensor] = {}
    indices: Dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SparseDeltaLinear):
            deltas[name] = mod.delta.detach().to("cpu")
            indices[name] = mod.selected_idx.detach().to("cpu")
    if not deltas:
        return None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "sparse_delta.pt"
    torch.save({"impl": "reparam_v1", "deltas": deltas, "indices": indices}, p)
    return str(p)


def load_sparse_delta_snapshot_strict(model: torch.nn.Module, checkpoint_dir: str) -> None:
    """
    Load <checkpoint_dir>/sparse_delta.pt into existing SparseDeltaLinear modules.
    Fail-fast if:
      - file missing
      - module keys mismatch
      - selected_idx mismatch
      - dtype/shape mismatch
    """
    p = Path(checkpoint_dir) / "sparse_delta.pt"
    if not p.exists():
        raise FileNotFoundError(f"sparse enabled resume requires sparse_delta.pt, missing: {p}")
    saved = torch.load(p, map_location="cpu")
    if saved.get("impl") != "reparam_v1":
        raise RuntimeError(f"Unsupported sparse_delta.pt impl={saved.get('impl')} at {p}")
    deltas: Dict[str, torch.Tensor] = saved.get("deltas") or {}
    indices: Dict[str, torch.Tensor] = saved.get("indices") or {}
    if not deltas:
        raise RuntimeError(f"Invalid sparse_delta.pt (no deltas) at {p}")

    # Build current module map.
    cur: Dict[str, SparseDeltaLinear] = {n: m for n, m in model.named_modules() if isinstance(m, SparseDeltaLinear)}
    if set(cur.keys()) != set(deltas.keys()):
        missing = sorted(set(cur.keys()) - set(deltas.keys()))
        extra = sorted(set(deltas.keys()) - set(cur.keys()))
        raise RuntimeError(
            f"sparse_delta.pt module key mismatch at {p}. missing={missing[:10]} extra={extra[:10]} "
            f"(missing_count={len(missing)} extra_count={len(extra)})"
        )

    for name, mod in cur.items():
        d = deltas[name]
        idx = indices.get(name)
        if idx is None:
            raise RuntimeError(f"sparse_delta.pt missing indices for module '{name}'")
        if idx.dtype != torch.int64:
            idx = idx.to(torch.int64)
        if idx.shape != mod.selected_idx.detach().cpu().shape:
            raise RuntimeError(f"indices shape mismatch for '{name}': saved={tuple(idx.shape)} cur={tuple(mod.selected_idx.shape)}")
        if not torch.equal(idx, mod.selected_idx.detach().cpu()):
            raise RuntimeError(f"indices value mismatch for '{name}' (selection differs); refuse to resume.")
        if d.numel() != mod.delta.numel():
            raise RuntimeError(f"delta numel mismatch for '{name}': saved={d.numel()} cur={mod.delta.numel()}")
        # Copy into module param (preserve device/dtype)
        mod.delta.data.copy_(d.to(device=mod.delta.device, dtype=mod.delta.dtype))


def _get_module_by_qualname(model: torch.nn.Module, qualname: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if qualname not in modules:
        raise KeyError(f"Module not found for qualname='{qualname}'")
    return modules[qualname]


def _set_module_by_qualname(model: torch.nn.Module, qualname: str, new_module: torch.nn.Module) -> None:
    """
    Replace a submodule by its dotted qualname.
    Supports ModuleDict / ModuleList / regular attributes.
    """
    if qualname == "":
        raise ValueError("Cannot replace root module")
    parts = qualname.split(".")
    parent = model
    for p in parts[:-1]:
        if isinstance(parent, torch.nn.ModuleDict):
            parent = parent[p]
        elif isinstance(parent, torch.nn.ModuleList):
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
        if not isinstance(parent, torch.nn.Module):
            raise TypeError(f"Traversal hit non-module at '{p}' while setting '{qualname}'")
    last = parts[-1]
    if isinstance(parent, torch.nn.ModuleDict):
        parent[last] = new_module
    elif isinstance(parent, torch.nn.ModuleList):
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _replace_linear_weight_with_sparse_delta(
    *,
    model: torch.nn.Module,
    param_full_name: str,
    selected_idx_flat: torch.Tensor,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> int:
    """
    Replace the nn.Linear module that owns '<module>.weight' with SparseDeltaLinear.
    Returns number of trainable parameters introduced (K for this module).
    """
    if not param_full_name.endswith(".weight"):
        raise ValueError(f"Expected a weight param name ending with '.weight', got: {param_full_name}")
    module_name = param_full_name.rsplit(".", 1)[0]
    module = _get_module_by_qualname(model, module_name)
    if not (isinstance(module, torch.nn.Linear) or _is_peft_lora_linear(module)):
        raise TypeError(
            f"Cannot sparsify non-linear module at '{module_name}' (type={type(module)}). "
            "For optimizer-state-O(K), targeted modules must be nn.Linear or PEFT LoRA Linear wrapper."
        )
    if selected_idx_flat.numel() == 0:
        return 0

    if isinstance(module, torch.nn.Linear):
        base_linear = module
        replace_mode = "replace_self"
    else:
        # PEFT wrapper: replace only its base_layer so LoRA path remains intact (needed for hybrid).
        base_linear = _get_base_linear_from_peft_linear(module)
        replace_mode = "replace_base_layer"

    base_w = base_linear.weight.detach()
    base_b = base_linear.bias.detach() if base_linear.bias is not None else None
    new_mod = SparseDeltaLinear(
        base_weight=base_w,
        base_bias=base_b,
        selected_idx_flat=selected_idx_flat.to(torch.int64),
        alpha=alpha,
        dropout=dropout,
    )
    new_mod.to(device=base_linear.weight.device, dtype=base_linear.weight.dtype)
    if replace_mode == "replace_self":
        _set_module_by_qualname(model, module_name, new_mod)
    else:
        # Keep the PEFT wrapper module; only swap its base_layer.
        setattr(module, "base_layer", new_mod)
    return int(selected_idx_flat.numel())


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
    Main entry: if enabled, compute/load selection indices, RE-PARAMETERIZE targeted linears so
    optimizer state scales with K, and persist metadata. Returns metadata dict (also written to disk),
    or None if disabled.

    Fail-fast policy:
      - If enabled and any expected module/param cannot be found or isn't nn.Linear, we raise.
      - No silent fallbacks to dense LoRA or grad-masking.
    """
    cfg = SparseSelectiveConfig.from_env()
    if not cfg.enabled:
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persisted selection + metadata for resume/repro.
    sel_path = out_dir / "sparse_selective_selection.pt"
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

    # IMPORTANT: Only build the candidate pools needed by the selected scope.
    # In lora_only, we must not scan base target modules (they may be PEFT-wrapped).
    lora_params = dict(_iter_lora_linear_weight_params(model))
    base_params: Dict[str, torch.nn.Parameter] = {}
    if cfg.scope.lower() in ("base_only", "hybrid"):
        base_params = dict(_iter_linear_weight_params(model, targets)) if targets else {}

    # Enforce scope for salience computation (only when enabled).
    # For base_only/hybrid we need base weights to have grads during scoring.
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

    # Optional: forbid base/hybrid on PeftModel unless user explicitly saves full model.
    # Reason: PEFT save_pretrained saves adapter weights; base weight changes would be lost.
    # LoRA-only is safe (changes are inside adapter modules).
    if isinstance(model, PeftModel) and cfg.scope.lower() in ("base_only", "hybrid"):
        save_full = str(os.environ.get("HP_SAVE_FULL_MODEL", "")).lower() in ("1", "true", "yes", "on")
        if not save_full:
            raise RuntimeError(
                f"[{model_type}] Sparse scope={cfg.scope} modifies base weights under a PEFT-wrapped model. "
                "PEFT adapter saving will NOT capture these changes. "
                "Set HP_SAVE_FULL_MODEL=1 (and ensure your training config enables save_full_model) "
                "or run base-only without PEFT wrapping."
            )

    # Distributed safety:
    # - rank0 computes and saves selection
    # - other ranks wait then load the exact same selection
    # This matches the spirit of other/speft (per-rank score accumulation + reduction),
    # but we keep it simple and deterministic: compute once and share via filesystem.
    is_rank0 = (_dist_rank() == 0)
    if sel_path.exists():
        # If file exists, everyone loads it (after barrier to avoid partial reads).
        _dist_barrier()
        saved = torch.load(sel_path, map_location="cpu")
        sel_dict: Dict[str, torch.Tensor] = saved["selection"]
        if int(saved.get("budget_k", -1)) <= 0:
            raise RuntimeError(f"[{model_type}] Invalid saved selection file: {sel_path}")
        budget_k = int(saved["budget_k"])
        candidate_elems_saved = int(saved.get("candidate_elems", candidate_elems))
        if candidate_elems_saved != candidate_elems:
            raise RuntimeError(
                f"[{model_type}] Candidate pool size mismatch on resume: saved={candidate_elems_saved}, now={candidate_elems}. "
                f"Refuse to proceed to avoid silent behavior drift. Delete {sel_path} to recompute."
            )
    elif not is_rank0 and _dist_available():
        # Non-rank0 waits for rank0 to compute.
        _dist_barrier()
        if not sel_path.exists():
            raise RuntimeError(f"[{model_type}] Expected rank0 to create {sel_path}, but file is missing.")
        saved = torch.load(sel_path, map_location="cpu")
        sel_dict = saved["selection"]
        budget_k = int(saved["budget_k"])
        candidate_elems_saved = int(saved.get("candidate_elems", candidate_elems))
        if candidate_elems_saved != candidate_elems:
            raise RuntimeError(
                f"[{model_type}] Candidate pool size mismatch on resume: saved={candidate_elems_saved}, now={candidate_elems}. "
                f"Refuse to proceed to avoid silent behavior drift. Delete {sel_path} to recompute."
            )
    else:
        # rank0 (or single-process) computes selection.
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
        sel_dict = global_topk_indices_from_scores(scores, budget_k)
        realized_k = int(sum(int(v.numel()) for v in sel_dict.values()))
        if realized_k != int(budget_k):
            raise RuntimeError(
                f"[{model_type}] Internal error: realized_k({realized_k}) != budget_k({budget_k}). "
                "Refusing to proceed."
            )
        torch.save(
            {
                "selection": sel_dict,
                "budget_k": int(budget_k),
                "candidate_elems": int(candidate_elems),
                "scope": cfg.scope,
                "budget_mode": cfg.budget_mode,
                "rho": float(cfg.rho),
                "reference_cfg": cfg.reference_cfg,
                "score_samples": int(cfg.score_samples),
                "salience": cfg.salience,
                "ranking": cfg.ranking,
                "cfg_path": cfg_path,
                "impl": "reparam_v1",
            },
            sel_path,
        )
        _dist_barrier()

    # Replace targeted linears with SparseDeltaLinear (optimizer state O(K))
    # Also force all other params to requires_grad=False (train only sparse delta vectors),
    # except for parameters not in candidate pool when scope demands.
    alpha = float(os.environ.get("HP_SPARSE_ALPHA", "1.0"))
    dropout = float(os.environ.get("HP_SPARSE_DROPOUT", "0.0"))
    introduced = 0
    _freeze_all_params(model)
    for pname, idxs in sel_dict.items():
        introduced += _replace_linear_weight_with_sparse_delta(
            model=model,
            param_full_name=pname,
            selected_idx_flat=idxs,
            alpha=alpha,
            dropout=dropout,
        )

    # Strict sanity checks: trainable count must equal budget_k exactly.
    trainable_after = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if trainable_after != int(budget_k):
        raise RuntimeError(
            f"[{model_type}] Trainable param mismatch after reparameterization: "
            f"trainable_after={trainable_after} vs budget_k={budget_k}. "
            "This would violate optimizer-state-O(K). Refusing to proceed."
        )

    realized_k = trainable_after

    # If this is a PEFT-wrapped model and user is NOT saving full model, verify adapter-only
    # saving actually captures SparseDeltaLinear trainables (fail-fast).
    save_full = str(os.environ.get("HP_SAVE_FULL_MODEL", "") or os.environ.get("LAT_SAVE_FULL_MODEL", "")).lower() in ("1", "true", "yes", "on")
    if isinstance(model, PeftModel) and not save_full:
        _assert_peft_adapter_saving_captures_sparse_params(model)

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
        "selection_path": str(sel_path),
        "impl": "reparam_v1",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"[{model_type}][sparse] enabled scope={cfg.scope} budget_mode={cfg.budget_mode} ranking=global salience=gradient")
    print(f"[{model_type}][sparse] candidate_elems={candidate_elems:,d} budget_k={budget_k:,d} realized_k={realized_k:,d}")
    print(f"[{model_type}][sparse] saved: {sel_path.name}, {meta_path.name}")

    return meta


