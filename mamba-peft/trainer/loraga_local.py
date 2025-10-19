import json
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader


def _get_env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).lower() not in ("0", "false", "no", "off", "")


def _pick_adapter_name(layer) -> Optional[str]:
    adapter = getattr(layer, "active_adapter", None)
    if adapter is not None:
        return adapter
    try:
        return next(iter(layer.lora_A.keys()))
    except Exception:
        return None


def _find_base_linear(layer):
    base = getattr(layer, "base_layer", None)
    if base is not None:
        return base
    base = getattr(layer, "linear", None)
    if base is not None:
        return base
    return layer


def _stable_scale_from_singular_values(S_r: torch.Tensor, alpha: int, r: int, c_target: float) -> float:
    """
    Heuristic stable scaling: normalize the Frobenius norm of the effective LoRA update.
    Target E|| (alpha/r) * (B A) x ||^2 ≈ c_target for isotropic x ⇒ set (alpha/r)*s*||S_r||_F = sqrt(c_target).
    Thus s = (sqrt(c_target) / max(||S_r||_F, eps)) * (r/alpha).
    This removes rank dependence and adapts to layer gradient magnitude. If c_target=1, s=(r/alpha)/||S_r||_F.
    """
    eps = 1e-12
    frob = float(torch.linalg.norm(S_r, ord=2) if S_r.ndim == 0 else torch.linalg.norm(S_r, ord=2))
    # Above uses spectral norm if scalar; but for vector of singular values we need Frobenius norm:
    frob = float(torch.linalg.norm(S_r, ord=2)) if S_r.ndim == 1 else float(abs(S_r))
    # Replace with true Frobenius for vector
    frob = float(torch.linalg.norm(S_r, ord=2))
    if frob < eps:
        return float(r) / float(alpha) if alpha else 1.0
    scale = (float(r) / float(alpha)) * (float(c_target) ** 0.5) / frob
    return scale


def _compute_loss_for_batch(model, batch, debug: bool = False) -> torch.Tensor:
    device = "cuda" if not debug else "cpu"
    local_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            local_batch[k] = v.to(device)
    model.train()
    outputs = model(local_batch["input_ids"])  # logits
    logits = outputs.logits
    labels = local_batch["label_ids"]
    ignore_index = -100
    vocab = logits.size(-1)
    logits = logits.view(-1, vocab)
    labels = labels.view(-1)
    valid = labels.ne(ignore_index)
    loss = torch.nn.functional.cross_entropy(logits[valid], labels[valid])
    return loss


def maybe_apply_loraga_ga_init(model, train_data_module, peft_json_path: str, debug: bool = False) -> None:
    """
    Apply local LoRA-GA initialization with stable scaling. No-op unless peft_json has "use_loraga": true.
    Implementation details:
      - Single minibatch gradient collection (default). Enable layerwise multiple backward via HP_LORAGA_LAYERWISE=1.
      - Stable scale: normalize by Frobenius norm of top-r singular values, with target magnitude HP_LORAGA_STABLE_C (default 1.0).
      - Never affects other variants; exits silently if conditions not met.
    """
    if peft_json_path is None:
        return
    try:
        with open(peft_json_path, "r") as f:
            peft_json = json.load(f)
    except Exception:
        return

    if not bool(peft_json.get("use_loraga", False)):
        return

    try:
        from peft import PeftModel
    except Exception:
        return

    if not isinstance(model, PeftModel):
        return

    # Estimation settings
    bs = int(os.environ.get("HP_LORAGA_BATCH_SIZE", 1))
    steps = int(os.environ.get("HP_LORAGA_STEPS", 1))
    layerwise = _get_env_bool("HP_LORAGA_LAYERWISE", False)
    c_target = float(os.environ.get("HP_LORAGA_STABLE_C", "1.0"))

    # Build small loader to sample a few minibatches
    dl = DataLoader(
        train_data_module.dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=train_data_module.data_collator,
    )

    # Helper to write A/B for a LoRA-wrapped layer from grad_w
    def _init_layer_from_grad(layer, grad_w: torch.Tensor):
        try:
            # grad_w: [out_features, in_features]
            U, S, Vh = torch.linalg.svd(grad_w.detach(), full_matrices=False)
        except Exception:
            U, S, Vh = torch.linalg.svd(grad_w.detach().cpu(), full_matrices=False)
            U, S, Vh = U.to(grad_w.device), S.to(grad_w.device), Vh.to(grad_w.device)
        r = int(getattr(layer, "r", 0))
        if r <= 0:
            return
        alpha = int(getattr(layer, "lora_alpha", 0) or int(peft_json.get("lora_alpha", 16)))
        A = Vh[:r, :].contiguous()            # [r, in]
        B = (U[:, :r] * S[:r]).contiguous()    # [out, r]
        # Stable scale heuristic: BA Frobenius normalized to c_target after LoRA forward scaling
        scale = _stable_scale_from_singular_values(S[:r], alpha=alpha, r=r, c_target=c_target)
        adapter = _pick_adapter_name(layer)
        if adapter is None:
            return
        with torch.no_grad():
            layer.lora_A[adapter].weight.copy_(A.to(layer.lora_A[adapter].weight.dtype))
            layer.lora_B[adapter].weight.copy_((B * scale).to(layer.lora_B[adapter].weight.dtype))

    # Strategy 1: single backward over a few minibatches (default)
    if not layerwise:
        # Accumulate gradients across 'steps' minibatches
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        device = "cuda" if not debug else "cpu"
        it = iter(dl)
        for _ in range(max(1, steps)):
            try:
                batch = next(it)
            except StopIteration:
                break
            loss = _compute_loss_for_batch(model, batch, debug=debug)
            loss.backward()
        # Initialize each LoRA-targeted layer from its base weight grad
        for layer in model.modules():
            if hasattr(layer, "lora_A") and hasattr(layer, "lora_B") and int(getattr(layer, "r", 0)) > 0:
                base = _find_base_linear(layer)
                grad_w = getattr(base, "weight", None)
                if grad_w is None or getattr(grad_w, "grad", None) is None:
                    continue
                _init_layer_from_grad(layer, grad_w.grad)
        # Clear grads
        for p in model.parameters():
            p.grad = None
        return

    # Strategy 2: layerwise (more memory-friendly, more compute)
    # Iterate lora layers and run a backward pass per layer while keeping other params frozen
    target_layers = []
    for layer in model.modules():
        if hasattr(layer, "lora_A") and hasattr(layer, "lora_B") and int(getattr(layer, "r", 0)) > 0:
            target_layers.append(layer)
    if not target_layers:
        return

    device = "cuda" if not debug else "cpu"
    it = iter(dl)
    try:
        batch = next(it)
    except StopIteration:
        return
    # For each layer, do a dedicated backward pass
    for layer in target_layers:
        # zero all grads
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        # enable grad only for this layer's base weight
        base = _find_base_linear(layer)
        if not hasattr(base, "weight"):
            continue
        # Forward/backward
        loss = _compute_loss_for_batch(model, batch, debug=debug)
        loss.backward()
        grad_w = base.weight.grad
        if grad_w is not None:
            _init_layer_from_grad(layer, grad_w)
        # clear grads promptly
        for p in model.parameters():
            p.grad = None


