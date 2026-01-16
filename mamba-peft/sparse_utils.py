import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.parametrize import register_parametrization, is_parametrized, remove_parametrizations


class CandidateAccessor(Protocol):
    def build(self) -> List["CandidateView"]:
        ...


@dataclass
class CandidateView:
    name: str
    parameter: torch.nn.Parameter
    numel: int
    original_requires_grad: bool
    is_lora: bool = False
    shape: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class SparsityConfig:
    enabled: bool = False
    scope: str = "lora"  # lora | base | hybrid
    budget_mode: str = "fixed_ratio"  # fixed_ratio | fixed_count | match_reference
    fixed_ratio: Optional[float] = None
    fixed_count: Optional[int] = None
    match_count: Optional[int] = None
    match_reference: Optional[str] = None
    score_samples: int = 128
    sample_batch_size: int = 1
    base_include: List[str] = field(default_factory=list)
    base_exclude: List[str] = field(default_factory=lambda: ["embed", "lm_head"])
    mask_meta_name: str = "sparse_mask_meta.json"
    mask_tensor_name: str = "sparse_mask.pt"

    @classmethod
    def from_env_and_cfg(cls, cfg: Dict, env: Dict[str, str]) -> "SparsityConfig":
        cfg_block = cfg.get("sparsity", {}) or {}
        enabled = bool(
            str(env.get("HP_SPARSE_ENABLED", cfg_block.get("enabled", "0"))).lower()
            in ("1", "true", "yes", "on")
        )
        scope = env.get("HP_SPARSE_SCOPE", cfg_block.get("scope", "lora")).lower()
        budget_mode = env.get(
            "HP_SPARSE_BUDGET_MODE", cfg_block.get("budget_mode", "fixed_ratio")
        ).lower()
        fixed_ratio = cls._maybe_float(
            env.get("HP_SPARSE_FIXED_RATIO", cfg_block.get("fixed_ratio"))
        )
        fixed_count = cls._maybe_int(
            env.get("HP_SPARSE_FIXED_COUNT", cfg_block.get("fixed_count"))
        )
        match_count = cls._maybe_int(
            env.get("HP_SPARSE_MATCH_COUNT", cfg_block.get("match_count"))
        )
        match_reference = env.get(
            "HP_SPARSE_MATCH_REFERENCE", cfg_block.get("match_reference")
        )
        score_samples = cls._maybe_int(
            env.get("HP_SPARSE_SCORE_SAMPLES", cfg_block.get("score_samples"))
        )
        sample_batch_size = cls._maybe_int(
            env.get("HP_SPARSE_SAMPLE_BATCH_SIZE", cfg_block.get("sample_batch_size"))
        )
        base_include = cls._maybe_csv(
            env.get("HP_SPARSE_BASE_INCLUDE", cfg_block.get("base_include"))
        )
        base_exclude = cls._maybe_csv(
            env.get("HP_SPARSE_BASE_EXCLUDE", cfg_block.get("base_exclude"))
        )
        cfg_obj = cls(
            enabled=enabled,
            scope=scope,
            budget_mode=budget_mode,
            fixed_ratio=fixed_ratio,
            fixed_count=fixed_count,
            match_count=match_count,
            match_reference=match_reference,
            score_samples=score_samples or 128,
            sample_batch_size=sample_batch_size or 1,
            base_include=base_include or [],
            base_exclude=base_exclude or ["embed", "lm_head"],
        )
        return cfg_obj

    @staticmethod
    def _maybe_float(value: Optional[str]) -> Optional[float]:
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _maybe_int(value: Optional[str]) -> Optional[int]:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _maybe_csv(value: Optional[str]) -> Optional[List[str]]:
        if value in (None, ""):
            return None
        parts = [item.strip() for item in value.replace(",", " ").split()]
        return [p for p in parts if p]


class LoraCandidateAccessor:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def build(self) -> List[CandidateView]:
        candidates: List[CandidateView] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" not in name and "loraA" not in name and "lora_B" not in name:
                continue
            candidates.append(
                CandidateView(
                    name=name,
                    parameter=param,
                    numel=param.numel(),
                    original_requires_grad=True,
                    is_lora=True,
                    shape=tuple(param.shape),
                )
            )
        return candidates


class BaseCandidateAccessor:
    def __init__(
        self,
        model: torch.nn.Module,
        include_keys: Iterable[str],
        exclude_keys: Iterable[str],
    ):
        self.model = model
        self.include_keys = list(include_keys)
        self.exclude_keys = list(exclude_keys)

    def build(self) -> List[CandidateView]:
        candidates: List[CandidateView] = []
        for name, param in self.model.named_parameters():
            if "lora_" in name or name.endswith("lora_A.weight") or name.endswith("lora_B.weight"):
                continue
            if self._should_skip(name):
                continue
            original = bool(param.requires_grad)
            if not original:
                param.requires_grad_(True)
            candidates.append(
                CandidateView(
                    name=name,
                    parameter=param,
                    numel=param.numel(),
                    original_requires_grad=original,
                    is_lora=False,
                    shape=tuple(param.shape),
                )
            )
        return candidates

    def _should_skip(self, name: str) -> bool:
        lowered = name.lower()
        for token in self.exclude_keys:
            if token and token.lower() in lowered:
                return True
        if self.include_keys:
            return not any(token.lower() in lowered for token in self.include_keys)
        return False


class HybridCandidateAccessor:
    def __init__(self, lora: CandidateAccessor, base: CandidateAccessor):
        self.lora = lora
        self.base = base

    def build(self) -> List[CandidateView]:
        return self.lora.build() + self.base.build()


class BudgetStrategy(Protocol):
    def compute(self, total_params: int) -> int:
        ...


class FixedRatioBudget(BudgetStrategy):
    def __init__(self, ratio: float):
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"Invalid sparse ratio: {ratio}")
        self.ratio = ratio

    def compute(self, total_params: int) -> int:
        return max(1, int(total_params * self.ratio))


class FixedCountBudget(BudgetStrategy):
    def __init__(self, count: int):
        if count <= 0:
            raise ValueError("Fixed count must be positive.")
        self.count = count

    def compute(self, total_params: int) -> int:
        return min(self.count, total_params)


class MatchReferenceBudget(BudgetStrategy):
    def __init__(self, count: Optional[int], reference: Optional[str]):
        if count is None and not reference:
            raise ValueError(
                "match_reference mode requires HP_SPARSE_MATCH_COUNT or HP_SPARSE_MATCH_REFERENCE."
            )
        self.count = count
        self.reference = reference

    def compute(self, total_params: int) -> int:
        if self.count is not None:
            return min(self.count, total_params)
        raise RuntimeError(
            "Automatic reference budget calculation is not implemented; "
            "please set HP_SPARSE_MATCH_COUNT."
        )


class GradientSalienceScorer:
    def __init__(
        self,
        model: torch.nn.Module,
        candidates: List[CandidateView],
    ):
        self.model = model
        self.candidates = candidates

    def score(
        self,
        dataloader: DataLoader,
        max_samples: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        accumulator: Dict[str, torch.Tensor] = {}
        consumed = 0
        iterator = iter(dataloader)
        while consumed < max_samples:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            batch_size = None
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
                    if batch_size is None:
                        batch_size = value.size(0)
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss = loss / max_samples
            loss.backward()
            consumed += batch_size or 1
            for view in self.candidates:
                grad = view.parameter.grad
                if grad is None:
                    continue
                entry = accumulator.setdefault(
                    view.name, torch.zeros_like(grad, dtype=torch.float32, device=device)
                )
                entry.add_(grad.detach().abs())
            self.model.zero_grad(set_to_none=True)
            if consumed >= max_samples:
                break
        return accumulator


def build_scope_accessor(config: SparsityConfig, model: torch.nn.Module) -> CandidateAccessor:
    scope = config.scope
    if scope == "lora":
        return LoraCandidateAccessor(model)
    if scope == "base":
        return BaseCandidateAccessor(model, config.base_include, config.base_exclude)
    if scope == "hybrid":
        return HybridCandidateAccessor(
            LoraCandidateAccessor(model),
            BaseCandidateAccessor(model, config.base_include, config.base_exclude),
        )
    raise ValueError(f"Unsupported sparse scope: {scope}")


def build_budget_strategy(config: SparsityConfig) -> BudgetStrategy:
    mode = config.budget_mode
    if mode == "fixed_ratio":
        if config.fixed_ratio is None:
            raise ValueError("HP_SPARSE_FIXED_RATIO must be set for fixed_ratio mode.")
        return FixedRatioBudget(config.fixed_ratio)
    if mode == "fixed_count":
        if config.fixed_count is None:
            raise ValueError("HP_SPARSE_FIXED_COUNT must be set for fixed_count mode.")
        return FixedCountBudget(config.fixed_count)
    if mode == "match_reference":
        return MatchReferenceBudget(config.match_count, config.match_reference)
    raise ValueError(f"Unknown budget mode: {mode}")


def _build_sampling_loader(
    dataset,
    collator,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )


def _flatten_scores(
    candidates: List[CandidateView], score_dict: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, List[Tuple[str, Tuple[int, ...]]]]:
    flat_scores: List[torch.Tensor] = []
    mapping: List[Tuple[str, Tuple[int, ...]]] = []
    for view in candidates:
        score = score_dict.get(view.name)
        if score is None:
            score = torch.zeros_like(view.parameter.data, dtype=torch.float32)
        flat_scores.append(score.flatten())
        mapping.append((view.name, tuple(view.parameter.shape)))
    return torch.cat(flat_scores), mapping


def _build_masks_from_indices(
    candidates: List[CandidateView],
    mapping: List[Tuple[str, Tuple[int, ...]]],
    selected_indices: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    sorted_idx, _ = torch.sort(selected_indices.to("cpu", torch.int64))
    masks: Dict[str, torch.Tensor] = {}
    index_map: Dict[str, torch.Tensor] = {}
    cursor = 0
    for (name, shape), view in zip(mapping, candidates):
        length = view.numel
        left = torch.searchsorted(
            sorted_idx, torch.tensor([cursor], dtype=torch.int64), right=False
        ).item()
        right = torch.searchsorted(
            sorted_idx, torch.tensor([cursor + length], dtype=torch.int64), right=False
        ).item()
        local = sorted_idx[left:right] - cursor
        flat = torch.zeros(length, dtype=torch.float32)
        if local.numel() > 0:
            flat.scatter_(0, local.to(torch.int64), 1.0)
        masks[name] = flat.view(shape)
        index_map[name] = local.to(torch.int64)
        cursor += length
    return masks, index_map


def apply_sparse_training(
    config: SparsityConfig,
    model: torch.nn.Module,
    train_dataset,
    data_collator,
    output_dir: str,
) -> Optional[Dict[str, int]]:
    if not config.enabled:
        return None
    os.makedirs(output_dir, exist_ok=True)
    mask_meta_path = os.path.join(output_dir, config.mask_meta_name)
    mask_tensor_path = os.path.join(output_dir, config.mask_tensor_name)
    if os.path.isfile(mask_meta_path) and os.path.isfile(mask_tensor_path):
        with open(mask_meta_path, "r") as meta_f:
            meta = json.load(meta_f)
        stored = torch.load(mask_tensor_path, map_location="cpu")
        _apply_loaded_masks(model, stored)
        return meta

    accessor = build_scope_accessor(config, model)
    candidates = accessor.build()
    if not candidates:
        raise RuntimeError("Sparse selection is enabled but no candidate parameters were found.")
    total_params = sum(view.numel for view in candidates)
    budget_strategy = build_budget_strategy(config)
    budget = budget_strategy.compute(total_params)
    if budget <= 0:
        raise RuntimeError("Computed sparse budget is zero; please adjust configuration.")
    sampler = _build_sampling_loader(
        train_dataset, data_collator, config.sample_batch_size
    )
    device = next(model.parameters()).device
    scorer = GradientSalienceScorer(model, candidates)
    score_map = scorer.score(sampler, config.score_samples, device)
    flat_scores, name_mapping = _flatten_scores(candidates, score_map)
    if budget >= flat_scores.numel():
        selected = torch.arange(flat_scores.numel(), device=device)
    else:
        _, selected = torch.topk(flat_scores, budget)
    mask_tensors, index_map = _build_masks_from_indices(candidates, name_mapping, selected)
    _apply_masks(model, candidates, index_map)
    mask_meta = {
        "scope": config.scope,
        "budget_mode": config.budget_mode,
        "total_params": total_params,
        "budget": budget,
        "selected": int(budget),
        "score_samples": config.score_samples,
    }
    with open(mask_meta_path, "w") as meta_f:
        json.dump(mask_meta, meta_f, indent=2)
    torch.save({k: v.cpu() for k, v in index_map.items()}, mask_tensor_path)
    return mask_meta


class SparseParametrization(torch.nn.Module):
    def __init__(self, base_tensor: torch.Tensor, indices: torch.Tensor):
        super().__init__()
        self.register_buffer("base_tensor", base_tensor.detach().clone())
        self.register_buffer("indices", indices.to(torch.int64))
        initial = base_tensor.flatten()[indices]
        self.values = torch.nn.Parameter(initial.clone())
        self.shape = base_tensor.shape

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        flat = self.base_tensor.flatten().clone()
        if self.indices.numel() > 0:
            flat[self.indices] = self.values
        return flat.view(self.shape)


def _apply_masks(
    model: torch.nn.Module,
    candidates: List[CandidateView],
    index_map: Dict[str, torch.Tensor],
) -> None:
    for view in candidates:
        local_idx = index_map.get(view.name)
        module, param_name = _locate_module_and_param(model, view.name)
        if local_idx is None or local_idx.numel() == 0:
            param = getattr(module, param_name)
            if is_parametrized(module, param_name):
                remove_parametrizations(module, param_name, leave_parametrized=False)
            param.requires_grad_(False)
            continue
        param = getattr(module, param_name)
        if is_parametrized(module, param_name):
            remove_parametrizations(module, param_name, leave_parametrized=False)
        register_parametrization(
            module,
            param_name,
            SparseParametrization(param.data, local_idx),
        )
        getattr(module, param_name).requires_grad_(True)


def _normalize_loaded_indices(stored: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for name, tensor in stored.items():
        if tensor.dim() == 1 and tensor.dtype in (torch.int64, torch.int32):
            normalized[name] = tensor.to(torch.int64)
        else:
            flat = tensor.reshape(-1)
            idx = torch.nonzero(flat, as_tuple=False).view(-1)
            normalized[name] = idx.to(torch.int64)
    return normalized


def _apply_loaded_masks(model: torch.nn.Module, stored: Dict[str, torch.Tensor]) -> None:
    index_map = _normalize_loaded_indices(stored)
    for name, indices in index_map.items():
        module, param_name = _locate_module_and_param(model, name)
        param = getattr(module, param_name)
        if is_parametrized(module, param_name):
            remove_parametrizations(module, param_name, leave_parametrized=False)
        register_parametrization(
            module,
            param_name,
            SparseParametrization(param.data, indices.to(torch.int64)),
        )
        getattr(module, param_name).requires_grad_(True)


def _locate_module_and_param(model: torch.nn.Module, full_name: str):
    parts = full_name.split(".")
    module = model
    for p in parts[:-1]:
        if not hasattr(module, p):
            raise AttributeError(f"Module '{module}' has no attribute '{p}' for parameter '{full_name}'.")
        module = getattr(module, p)
    return module, parts[-1]
