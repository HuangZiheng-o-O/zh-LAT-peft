"""
GLA SD-LoRA (Sparse Dimension LoRA) Implementation

This module implements SD-LoRA (Sparse Dimension Tuning + LoRA) for GLA
(Gated Linear Attention) models from the FLA library.

Key adaptations from Mamba SD-LoRA:
1. Target modules: gk_proj (gate projection) instead of A_log
2. No state dimension selection (GLA has matrix-valued state)
3. Channel dimension selection on key dimensions (head_k_dim)
4. Zero mask value uses large negative value for logsigmoid (not 10)

References:
- SD-LoRA Paper: "SD-LoRA: Scalable and Deployable LoRA Fine-tuning for Large Language Models"
- GLA Paper: "Gated Linear Attention Transformers with Hardware-Efficient Training"
- Original Mamba SD-LoRA: mamba-peft-sd_lora/mamba_ssm_peft/peft/sd_lora.py
"""

from dataclasses import dataclass, field
import enum
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora import Linear as LoraLinear

from mamba_ssm_peft.peft import MambaPeftType, register_peft_config, register_peft_tuner
from mamba_ssm_peft.peft.gla_base_tuner import GLABaseTuner
from utils.utils import find_layer_by_name, find_module_parent


class GLASelectMode(str, enum.Enum):
    """Selection mode for GLA dimension pruning."""
    CHANNELS_ONLY = "CHANNELS_ONLY"  # Only select key/channel dimensions
    CHANNELS_PER_HEAD = "CHANNELS_PER_HEAD"  # Select per-head key dimensions


@register_peft_config(MambaPeftType.GLA_SD_LORA)
@dataclass
class GlaSdLoraConfig(PeftConfig):
    """
    Configuration for GLA SD-LoRA.

    Attributes:
        select_mode: How to select dimensions for SDT (default: CHANNELS_ONLY)
        proj_lora_r: LoRA rank for projection layers (q_proj, k_proj, etc.)
        num_zero: Dict with "channel" key specifying fraction/count to zero
        num_freeze: Dict with "channel" key specifying fraction/count to freeze
        num_warmup_it: Number of warmup iterations for gradient accumulation
        target_modules: List of target module names for SDT
        lora_targets: List of target module names for LoRA
        finetune_parameters: Additional parameters to fine-tune
        sdlora_alpha: Scaling factor for SDT adaptation
    """
    select_mode: GLASelectMode = field(default=GLASelectMode.CHANNELS_ONLY)
    proj_lora_r: int = field(default=None)
    num_zero: Dict = field(default=None)
    num_freeze: Dict = field(default=None)
    num_warmup_it: int = field(default=None)
    target_modules: List[str] = field(default=None)
    lora_targets: List[str] = field(default=None)
    finetune_parameters: List[str] = field(default=None)
    sdlora_alpha: Dict = field(default=None)

    def __post_init__(self):
        self.peft_type = MambaPeftType.GLA_SD_LORA

        # Validate required configuration fields
        # These must be explicitly specified in the config file or passed at initialization

        # target_modules: SDT targets (e.g., ["gk_proj.1"])
        assert self.target_modules is not None, (
            "target_modules is required for GLA SD-LoRA. "
            "Must be specified in config file or at initialization. "
            "Example: target_modules=['gk_proj.1']"
        )
        assert isinstance(self.target_modules, list) and len(self.target_modules) > 0, (
            "target_modules must be a non-empty list"
        )

        # lora_targets: LoRA targets for linear projections
        assert self.lora_targets is not None, (
            "lora_targets is required for GLA SD-LoRA. "
            "Must be specified in config file or at initialization. "
            "Example: lora_targets=['q_proj', 'k_proj', 'v_proj', 'g_proj', 'o_proj']"
        )
        assert isinstance(self.lora_targets, list) and len(self.lora_targets) > 0, (
            "lora_targets must be a non-empty list"
        )

        # num_zero: Dimensions to zero (sparsify)
        assert self.num_zero is not None, (
            "num_zero is required for GLA SD-LoRA. "
            "Must be specified in config file or at initialization. "
            "Example: num_zero={'channel': 0.1}"
        )
        assert isinstance(self.num_zero, dict) and "channel" in self.num_zero, (
            "num_zero must be a dict with 'channel' key"
        )

        # num_freeze: Dimensions to freeze
        assert self.num_freeze is not None, (
            "num_freeze is required for GLA SD-LoRA. "
            "Must be specified in config file or at initialization. "
            "Example: num_freeze={'channel': 0.5}"
        )
        assert isinstance(self.num_freeze, dict) and "channel" in self.num_freeze, (
            "num_freeze must be a dict with 'channel' key"
        )

        # Validate that Train + Freeze + Zero ratios sum to <= 1.0
        if isinstance(self.num_zero["channel"], float) and isinstance(self.num_freeze["channel"], float):
            total_ratio = self.num_zero["channel"] + self.num_freeze["channel"]
            assert total_ratio <= 1.0, (
                f"num_zero + num_freeze must be <= 1.0, got {total_ratio}. "
                f"This leaves Train ratio = {1.0 - total_ratio}"
            )


@register_peft_tuner(MambaPeftType.GLA_SD_LORA)
class GlaSdLoraModel(GLABaseTuner):
    """
    GLA SD-LoRA Model wrapper.

    Wraps a GLA model with SD-LoRA adapters for sparse dimension tuning
    on gate projections and LoRA on linear projections.
    """
    prefix: str = "gla_sdlora_"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        self.last_mode = None
        super().__init__(model, peft_config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        """Prepare adapter configuration."""
        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """Mark only adapter parameters as trainable."""
        finetune_parameters = self.peft_config[self.active_adapter].finetune_parameters

        if finetune_parameters is None:
            finetune_parameters = []

        for n, p in model.named_parameters():
            if (self.prefix in n or
                any(n.endswith("." + fp) for fp in finetune_parameters) or
                (self.peft_config["default"].proj_lora_r is not None and "lora_" in n)):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _create_new_module(self, peft_config, adapter_name, target, target_name):
        """Create a new adapter module for the target."""
        # [FIX Issue #4] Improved module name resolution with error handling
        matching_names = [n for n, m in self.model.named_modules() if m is target]

        if not matching_names:
            raise RuntimeError(
                f"Cannot find module '{target_name}' in model. "
                f"Target object: {type(target).__name__}. "
                "This should not happen - the target was matched but cannot be found by identity."
            )

        if len(matching_names) > 1:
            # Multiple matches - use the one that ends with target_name
            filtered = [n for n in matching_names if n.endswith(target_name)]
            if len(filtered) == 1:
                module_name = filtered[0]
            else:
                # Ambiguous - warn and use first match
                print(f"[SD-LoRA] WARNING: Multiple modules match target '{target_name}': {matching_names}. "
                      f"Using first match: {matching_names[0]}")
                module_name = matching_names[0]
        else:
            module_name = matching_names[0]

        # Check if this is a LoRA target (applies to linear projection layers)
        # Note: g_proj may not exist if use_output_gate=False in the GLA layer
        lora_targets = peft_config.lora_targets or []
        if target_name in lora_targets and peft_config.proj_lora_r is not None:
            new_module = LoraLinear(
                target, adapter_name,
                r=peft_config.proj_lora_r,
                lora_alpha=peft_config.proj_lora_r,
                lora_dropout=0.1
            )
        else:
            # SDT target - find the parent GLA attention block
            # Navigate up to find GatedLinearAttention block
            block = self._find_gla_block_for_module(module_name)

            sdlora_alpha = 1
            if peft_config.sdlora_alpha is not None:
                sdlora_alpha = peft_config.sdlora_alpha.get(target_name, 1)
                sdlora_alpha *= peft_config.sdlora_alpha.get("global", 1)

            new_module = GlaSdLoraParameter(
                target, adapter_name, module_name, block, peft_config.select_mode,
                num_zero=peft_config.num_zero,
                num_freeze=peft_config.num_freeze,
                num_warmup_it=peft_config.num_warmup_it,
                sdlora_alpha=sdlora_alpha
            )

        return new_module

    def _find_gla_block_for_module(self, module_name):
        """Find the GLA attention block containing this module."""
        # Parse module name to find parent block
        # e.g., "model.layers.0.attn.gk_proj.1" -> "model.layers.0.attn"
        parts = module_name.split(".")
        for i in range(len(parts) - 1, -1, -1):
            block_name = ".".join(parts[:i])
            block = find_layer_by_name(self.model, block_name)
            if block is not None:
                # Check if this is a GLA attention block
                try:
                    from fla.layers.gla import GatedLinearAttention
                    if isinstance(block, GatedLinearAttention):
                        return block
                except ImportError:
                    pass
                # Also check for attn attribute
                if hasattr(block, 'gk_proj'):
                    return block
        return None

    def _get_sdlora_params(self):
        """Get all SD-LoRA parameter modules."""
        return [m for m in self.model.modules() if isinstance(m, GlaSdLoraParameter)]

    def get_sdlora_mode(self):
        """Get the current SD-LoRA mode (warmup or train).

        Raises:
            RuntimeError: If no GlaSdLoraParameter modules are found (configuration error).
        """
        # [FIX Issue #3] Raise error instead of silent fallback
        params = self._get_sdlora_params()
        if not params:
            raise RuntimeError(
                "No GlaSdLoraParameter modules found in the model. "
                "This indicates a configuration error - check that target_modules "
                f"(e.g., 'gk_proj.1') matches actual module names in the model. "
                f"Available modules: {[n for n, _ in self.model.named_modules() if 'gk_proj' in n][:10]}"
            )
        modes = [m.get_sdlora_mode() for m in params]
        if len(set(modes)) != 1:
            raise RuntimeError(
                f"Inconsistent SD-LoRA modes across modules: {dict(zip([m.module_name for m in params], modes))}. "
                "All modules must be in the same mode."
            )
        return modes[0]

    def load_config(self, path, required=False):
        """Load SD-LoRA configuration from path.

        Args:
            path: Directory containing .pkl config files
            required: If True, raise error if config files don't exist (for Phase 2)

        Returns:
            int: Number of config files successfully loaded

        Raises:
            RuntimeError: If required=True and no config files are found/loaded
        """
        params = self._get_sdlora_params()
        if not params:
            raise RuntimeError(
                "No GlaSdLoraParameter modules found. Cannot load config."
            )

        loaded_count = 0
        missing_files = []
        for m in params:
            success = m.load_config(path, required=required)
            if success:
                loaded_count += 1
            else:
                missing_files.append(m.module_name)

        if required and loaded_count == 0:
            raise RuntimeError(
                f"[SD-LoRA] FATAL: No config files loaded from {path}. "
                f"Missing files for modules: {missing_files}. "
                "Phase 2 cannot proceed without warmup gradient data."
            )

        if required and missing_files:
            raise RuntimeError(
                f"[SD-LoRA] FATAL: Some config files missing from {path}: {missing_files}. "
                f"Only {loaded_count}/{len(params)} modules loaded."
            )

        print(f"[SD-LoRA] Loaded {loaded_count}/{len(params)} config files from {path}")
        return loaded_count

    def save_config(self, path):
        """Save SD-LoRA configuration to path.

        Returns:
            int: Number of config files saved

        Raises:
            RuntimeError: If no modules found to save
        """
        params = self._get_sdlora_params()
        if not params:
            raise RuntimeError(
                "No GlaSdLoraParameter modules found. Cannot save config."
            )

        saved_count = 0
        for m in params:
            m.save_config(path)
            saved_count += 1

        print(f"[SD-LoRA] Saved {saved_count} config files to {path}")
        return saved_count

    def verify_train_mode(self):
        """Verify all SD-LoRA modules are in train mode.

        Returns:
            bool: True if all modules are in train mode

        Raises:
            RuntimeError: If any module is not in train mode
        """
        params = self._get_sdlora_params()
        if not params:
            raise RuntimeError("No GlaSdLoraParameter modules found.")

        not_in_train = [(m.module_name, m.get_sdlora_mode()) for m in params if m.get_sdlora_mode() != "train"]
        if not_in_train:
            raise RuntimeError(
                f"[SD-LoRA] FATAL: Some modules are not in train mode: {not_in_train}. "
                "This indicates load_config failed or warmup data is missing."
            )
        return True

    @property
    def should_training_stop(self):
        """Check if training should stop (warmup→train transition)."""
        if self.last_mode == "warmup" and self.get_sdlora_mode() == "train":
            self.last_mode = "train"
            res = True
        else:
            res = False

        if self.last_mode is None:
            self.last_mode = self.get_sdlora_mode()

        return res


class GlaSdLoraParameter(nn.Module, BaseTunerLayer):
    """
    GLA SD-LoRA Parameter wrapper.

    Wraps a GLA parameter (typically gk_proj weights) with sparse dimension
    tuning capability. Operates in two phases:
    1. Warmup: Accumulate gradients to determine dimension importance
    2. Train: Train only selected dimensions, freeze/zero others

    Key differences from Mamba SdLoraParameter:
    - No state dimension (only channel/key dimension)
    - Works with Linear layers (not raw parameters like A_log)
    - Zero mask uses large negative value for logsigmoid gate
    """

    # Large negative value for zeroing gate dimensions
    # In GLA: gate = exp(logsigmoid(gk) / gate_logit_normalizer)
    # where gate_logit_normalizer = 16 (default)
    #
    # To achieve near-zero decay (complete forgetting):
    #   gk = -100 → logsigmoid(-100)/16 ≈ -6.25 → exp(-6.25) ≈ 0.002 (0.2% retained)
    #
    # Note: Previous value -20 was insufficient:
    #   gk = -20 → logsigmoid(-20)/16 ≈ -1.25 → exp(-1.25) ≈ 0.29 (29% retained!)
    ZERO_MASK_VALUE = -100.0

    def __init__(
        self,
        base_layer,
        adapter_name,
        module_name,
        block,
        select_mode,
        num_zero,
        num_freeze,
        num_warmup_it,
        sdlora_alpha=1
    ) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        # Validate inputs
        if base_layer is None:
            raise ValueError(f"base_layer cannot be None for module {module_name}")

        if not module_name:
            raise ValueError("module_name cannot be empty")

        if num_warmup_it is None:
            raise ValueError(
                f"[{module_name}] num_warmup_it cannot be None. "
                "Set to 0 or positive value for warmup, or -1 to skip warmup."
            )

        self.base_layer = base_layer
        self.module_name = module_name.replace(".", "_")
        self.select_mode = select_mode
        self.num_zero = self._parse_dims(num_zero)
        self.num_freeze = self._parse_dims(num_freeze)
        self.num_train = self._compute_num_train()
        self.num_warmup_it = num_warmup_it
        self.sdlora_mode = None
        self.train_mask = None
        self.zero_mask = None
        self.sdlora_alpha = sdlora_alpha
        self.get_block = lambda: block

        # Validate dimension ratios
        param_info = self.get_model_param_info()
        total_channels = param_info.out_features
        total_dims = self.num_zero["channel"] + self.num_freeze["channel"] + self.num_train["channel"]

        if total_dims != total_channels:
            raise ValueError(
                f"[{self.module_name}] Dimension mismatch: "
                f"zero({self.num_zero['channel']}) + freeze({self.num_freeze['channel']}) + "
                f"train({self.num_train['channel']}) = {total_dims}, "
                f"but total channels = {total_channels}"
            )

        if self.num_train["channel"] <= 0:
            raise ValueError(
                f"[{self.module_name}] num_train must be positive, got {self.num_train['channel']}. "
                f"Check num_zero and num_freeze ratios (they sum to >= 1.0)"
            )

        print(f"[{self.module_name}] SD-LoRA config: "
              f"train={self.num_train['channel']}, freeze={self.num_freeze['channel']}, "
              f"zero={self.num_zero['channel']} (total={total_channels})")

        # Save in state dict
        self.register_buffer("it_counter", torch.tensor(0).long())

        # Create gradient accumulator and adapter parameters
        self.sdlora_grad = self._create_grad_param()
        self.sdlora_adapter = self._create_adapter_param()

        if self.sdlora_grad is None:
            raise RuntimeError(f"[{self.module_name}] Failed to create sdlora_grad")
        if self.sdlora_adapter is None:
            raise RuntimeError(f"[{self.module_name}] Failed to create sdlora_adapter")

        self.set_sdlora_mode("warmup" if self.training and self.num_warmup_it >= 0 else "train")
        self.set_adapter(self.active_adapters)

    def _parse_dims(self, dims):
        """Parse dimension configuration."""
        if dims is None:
            return {"channel": 0}

        param_info = self.get_model_param_info()
        channel_dim = dims.get("channel", 0)

        if isinstance(channel_dim, float):
            # Fraction of total channels
            channel_dim = int(round(channel_dim * param_info.out_features))

        return {"channel": channel_dim}

    def _compute_num_train(self):
        """Compute number of trainable dimensions."""
        param_info = self.get_model_param_info()
        total_channels = param_info.out_features
        train_channels = total_channels - self.num_zero["channel"] - self.num_freeze["channel"]
        return {"channel": max(0, train_channels)}

    @property
    def is_layer(self):
        """Check if base_layer is a Linear layer."""
        return isinstance(self.base_layer, nn.Linear)

    def get_model_param_info(self):
        """Get information about the model parameter."""
        if self.is_layer:
            weight = self.base_layer.weight
            return SimpleNamespace(
                shape=weight.shape,
                out_features=weight.shape[0],
                in_features=weight.shape[1],
                device=weight.device,
                dtype=weight.dtype
            )
        else:
            param = self.base_layer
            return SimpleNamespace(
                shape=param.shape,
                out_features=param.shape[0] if len(param.shape) > 0 else 1,
                in_features=param.shape[1] if len(param.shape) > 1 else 1,
                device=param.device,
                dtype=param.dtype
            )

    def _create_grad_param(self):
        """Create gradient accumulation parameter."""
        param_info = self.get_model_param_info()
        return nn.Parameter(torch.zeros(
            param_info.shape,
            device=param_info.device,
            dtype=param_info.dtype
        ))

    def _create_adapter_param(self):
        """Create sparse adapter parameter."""
        param_info = self.get_model_param_info()
        # Create parameter for trainable dimensions only
        num_train = self.num_train["channel"]
        if num_train <= 0:
            num_train = param_info.out_features  # Fallback to full

        shape = (num_train, param_info.in_features) if self.is_layer else (num_train,)
        return nn.Parameter(torch.zeros(
            shape,
            device=param_info.device,
            dtype=param_info.dtype
        ))

    def _get_cfg_file(self, path):
        """Get configuration file path."""
        return Path(path) / (self.module_name + ".pkl")

    def load_config(self, path, required=False):
        """Load configuration from file.

        Args:
            path: Directory containing the config file
            required: If True, raise error if file doesn't exist

        Returns:
            bool: True if config was loaded successfully, False otherwise

        Raises:
            RuntimeError: If required=True and file doesn't exist
            RuntimeError: If file exists but loading fails
        """
        cfg_path = self._get_cfg_file(path)

        if not cfg_path.exists():
            if required:
                raise RuntimeError(
                    f"[{self.module_name}] FATAL: Config file not found: {cfg_path}. "
                    "This is required for Phase 2 training. Did Phase 1 (warmup) complete successfully?"
                )
            # For Phase 1, missing file is OK (first run)
            return False

        if self.sdlora_grad is None:
            raise RuntimeError(
                f"[{self.module_name}] Cannot load config: sdlora_grad is None"
            )

        try:
            with open(cfg_path, "rb") as f:
                loaded_data = pickle.load(f)

            # Validate loaded data shape matches current parameter
            if loaded_data.shape != self.sdlora_grad.shape:
                raise RuntimeError(
                    f"[{self.module_name}] Config shape mismatch: "
                    f"loaded {loaded_data.shape} vs expected {self.sdlora_grad.shape}. "
                    "Model architecture may have changed since warmup."
                )

            with torch.no_grad():
                self.sdlora_grad.data[:] = loaded_data

            # Verify non-zero gradients were loaded (warmup should have accumulated something)
            grad_norm = self.sdlora_grad.data.norm().item()
            if grad_norm == 0:
                print(f"[{self.module_name}] WARNING: Loaded gradient has zero norm. "
                      "Warmup may not have accumulated meaningful gradients.")

            print(f"[{self.module_name}] Loaded config from {cfg_path} (grad_norm={grad_norm:.4f})")
            self.set_sdlora_mode("train")
            return True

        except Exception as e:
            raise RuntimeError(
                f"[{self.module_name}] Failed to load config from {cfg_path}: {e}"
            ) from e

    def save_config(self, path):
        """Save configuration to file.

        Saves the accumulated gradient data (sdlora_grad) which is used
        to determine dimension importance for Phase 2.

        Raises:
            RuntimeError: If sdlora_grad is None or save fails
        """
        cfg_path = self._get_cfg_file(path)

        if self.sdlora_grad is None:
            raise RuntimeError(
                f"[{self.module_name}] Cannot save config: sdlora_grad is None. "
                "This should not happen - sdlora_grad should be initialized in __init__."
            )

        grad_data = self.sdlora_grad.data
        grad_norm = grad_data.norm().item()

        # Warn if gradient norm is zero (warmup may not have run or failed)
        if grad_norm == 0:
            print(f"[{self.module_name}] WARNING: Saving zero-norm gradient. "
                  "Warmup may not have accumulated meaningful gradients. "
                  "Check if warmup training actually ran and updated parameters.")

        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            with open(cfg_path, "wb") as f:
                pickle.dump(grad_data, f)

            # Verify the file was written correctly
            if not cfg_path.exists():
                raise RuntimeError(f"File was not created: {cfg_path}")

            file_size = cfg_path.stat().st_size
            if file_size == 0:
                raise RuntimeError(f"File is empty: {cfg_path}")

            print(f"[{self.module_name}] Saved config to {cfg_path} "
                  f"(grad_norm={grad_norm:.4f}, size={file_size} bytes)")

        except Exception as e:
            raise RuntimeError(
                f"[{self.module_name}] Failed to save config to {cfg_path}: {e}"
            ) from e

    def get_sdlora_mode(self):
        """Get current SD-LoRA mode."""
        return self.sdlora_mode

    def set_sdlora_mode(self, sdlora_mode):
        """Set SD-LoRA mode."""
        if sdlora_mode != self.sdlora_mode:
            if sdlora_mode == "train":
                print(f"[{self.module_name}] Switching to train mode")
        self.sdlora_mode = sdlora_mode

    def get_importances(self, x, dim=0):
        """
        Compute importance scores for each channel.

        Uses L2 norm of gradient as importance metric.
        """
        norms = x.square().detach().sum(dim=1 if dim == 0 else 0)  # Sum over input dimension
        indices = torch.argsort(-norms)  # Sort descending
        return indices

    def select_channels(self, importance_order, channel_type):
        """
        Select channels based on importance and type.

        Args:
            importance_order: Indices sorted by importance (descending)
            channel_type: "train", "freeze", or "zero"

        Returns:
            Tensor of channel indices
        """
        num_train = self.num_train["channel"]
        num_freeze = self.num_freeze["channel"]
        num_zero = self.num_zero["channel"]

        if channel_type == "train":
            return importance_order[:num_train]
        elif channel_type == "freeze":
            return importance_order[num_train:num_train + num_freeze]
        elif channel_type == "zero":
            return importance_order[num_train + num_freeze:num_train + num_freeze + num_zero]
        else:
            raise ValueError(f"Unknown channel_type: {channel_type}")

    def get_mask(self, mask_type):
        """
        Build mask for train/zero dimensions based on gradient importance.

        Args:
            mask_type: "train" or "zero"

        Returns:
            Boolean mask tensor
        """
        # Get gradient for importance calculation
        grad = self.sdlora_grad.data

        param_info = self.get_model_param_info()
        mask = torch.zeros(param_info.shape, device=param_info.device, dtype=torch.bool)

        # Get channel importance order
        importance_order = self.get_importances(grad, dim=0)

        # Select channels
        channel_indices = self.select_channels(importance_order, mask_type)

        if len(channel_indices) > 0:
            # Mark selected channels in mask
            mask.index_fill_(0, channel_indices, True)

        return mask

    def build_train_param(self, param, adapter):
        """
        Build the training parameter with sparse adapter applied.

        Args:
            param: Original parameter
            adapter: Sparse adapter values

        Returns:
            Modified parameter with adapter applied to train dimensions
        """
        if self.train_mask is None:
            print(f"[{self.module_name}] Building trainable mask")
            self.train_mask = self.get_mask("train")
            print(f"  Train mask: {self.train_mask.sum().item()} / {self.train_mask.numel()} channels")

        if self.zero_mask is None:
            self.zero_mask = self.get_mask("zero")
            # Masks should not overlap
            assert torch.sum(self.train_mask & self.zero_mask).item() == 0
            print(f"  Zero mask: {self.zero_mask.sum().item()} / {self.zero_mask.numel()} channels")

        # Apply zero mask: set zeroed channels to large negative value
        # This makes logsigmoid(gk) very negative → gate ≈ 0 → state decays
        param_new = param.clone()
        if self.zero_mask.any():
            param_new = torch.where(self.zero_mask, torch.full_like(param, self.ZERO_MASK_VALUE), param_new)

        # Apply adapter to trainable channels
        if self.train_mask.any():
            # Scatter adapter values into the trainable positions
            bias = torch.zeros_like(param)
            bias[self.train_mask] = adapter.flatten()[:self.train_mask.sum().item()]
            param_new = param_new + self.sdlora_alpha * bias

        return param_new

    def update_layer(self, adapter_name):
        """Update layer (required by BaseTunerLayer)."""
        pass

    def forward(self, x):
        """
        Forward pass with SD-LoRA adaptation.

        During warmup: accumulate gradients on full parameter
        During train: apply sparse adapter to selected dimensions

        Note on it_counter:
            it_counter increments per forward pass, NOT per training step.
            With gradient_accumulation_steps=N, each training step has N forward passes.
            num_warmup_it should be set based on forward passes, and the training script
            should adjust total_steps accordingly (see train_lat.py:_run_sdlora_two_phase_training).
        """
        if not hasattr(self, "sdlora_alpha"):
            self.sdlora_alpha = 1

        # Check if warmup is complete
        if self.sdlora_mode == "warmup" and self.num_warmup_it >= 0 and self.it_counter > self.num_warmup_it:
            print(f"[{self.module_name}] Warmup complete after {self.it_counter} forward passes. "
                  f"Switching to train mode.")
            self.set_sdlora_mode("train")

        if self.sdlora_mode == "warmup" and not self.training:
            raise RuntimeError(
                f"[{self.module_name}] Cannot be in warmup mode during evaluation. "
                "This suggests the model was not properly switched to train mode after warmup."
            )

        if self.is_layer:
            weight = self.base_layer.weight
            bias = self.base_layer.bias

            if self.sdlora_mode == "warmup":
                # During warmup: add gradient accumulator to weight
                weight_new = weight + self.sdlora_alpha * self.sdlora_grad
            elif self.sdlora_mode == "train":
                # During train: apply sparse adapter
                weight_new = self.build_train_param(weight, self.sdlora_adapter)
            else:
                raise ValueError(f"Unknown mode: {self.sdlora_mode}")

            self.it_counter += 1

            return F.linear(x, weight_new, bias)
        else:
            # For non-Linear parameters (rare case)
            param = self.base_layer

            if self.sdlora_mode == "warmup":
                param_new = param + self.sdlora_alpha * self.sdlora_grad
            elif self.sdlora_mode == "train":
                param_new = self.build_train_param(param, self.sdlora_adapter)
            else:
                raise ValueError(f"Unknown mode: {self.sdlora_mode}")

            self.it_counter += 1

            return param_new
