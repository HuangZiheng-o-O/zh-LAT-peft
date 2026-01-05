"""
Runtime Patches for LAT Framework.

This module provides the Strategy Pattern implementation for applying
runtime patches to FLA (Flash Linear Attention) models.

The primary use case is disabling fused SwiGLU operations when they
cause compatibility issues with certain hardware or configurations.

Usage:
======
    from mamba_ssm_peft.utils.patches import apply_model_patches

    # Apply patches based on model type and config
    apply_model_patches(model_type="gla", config=config)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from .env_config import env_config


class PatchStrategy(ABC):
    """Base class for runtime patch strategies."""

    @abstractmethod
    def is_applicable(self, model_type: str) -> bool:
        """Check if this patch applies to the given model type."""
        pass

    @abstractmethod
    def apply(self) -> bool:
        """
        Apply the patch.

        Returns:
            True if patch was applied successfully, False otherwise
        """
        pass

    @abstractmethod
    def is_applied(self) -> bool:
        """Check if the patch has already been applied."""
        pass


class FLASwiGLUPatch(PatchStrategy):
    """
    Patch to replace fused SwiGLU operations with PyTorch implementations.

    This patch is necessary for models that use the FLA library's fused
    SwiGLU operations, which may cause issues on certain hardware or
    when specific configurations are used.

    Applies to: GLA, RetNet, and other FLA models with fuse_swiglu capability.
    """

    _applied: bool = False
    _applicable_models = frozenset({"gla", "retnet", "rwkv", "hgrn", "linear_attn"})

    def is_applicable(self, model_type: str) -> bool:
        return model_type.lower() in self._applicable_models

    def is_applied(self) -> bool:
        return FLASwiGLUPatch._applied

    def apply(self) -> bool:
        if self._applied:
            return True

        try:
            import torch.nn.functional as F
            from importlib import import_module

            _mlp = import_module("fla.modules.mlp")
            _act = import_module("fla.modules.activations")

            def _pt_swiglu(x, y):
                """PyTorch implementation of SwiGLU: SiLU(x) * y"""
                return F.silu(x) * y

            def _pt_swiglu_linear(x, y, weight, bias):
                """PyTorch implementation of SwiGLU + Linear."""
                return F.linear(F.silu(x) * y, weight, bias)

            # Apply patches
            _mlp.swiglu = _pt_swiglu
            _mlp.swiglu_linear = _pt_swiglu_linear
            _act.swiglu = _pt_swiglu
            _act.swiglu_linear = _pt_swiglu_linear

            FLASwiGLUPatch._applied = True

            if env_config.get_bool("VERBOSE"):
                print("[LAT] SwiGLU patch applied: using PyTorch implementations.")

            return True

        except ImportError as e:
            print(f"[LAT][warn] Failed to apply SwiGLU patch (module not found): {e}")
            return False
        except Exception as e:
            print(f"[LAT][warn] Failed to apply SwiGLU patch: {e}")
            return False


class NoOpPatch(PatchStrategy):
    """No-operation patch for models that don't need patching."""

    def is_applicable(self, model_type: str) -> bool:
        return True  # Fallback for any model

    def is_applied(self) -> bool:
        return True

    def apply(self) -> bool:
        return True


class PatchManager:
    """
    Manages and applies patches based on model type and configuration.

    This class implements the Strategy Pattern, selecting and applying
    the appropriate patches for each model type.
    """

    _patches = [
        FLASwiGLUPatch(),
        NoOpPatch(),  # Fallback
    ]

    @classmethod
    def apply_patches(
        cls,
        model_type: str,
        config: Optional[Any] = None,
        force: bool = False,
    ) -> None:
        """
        Apply all applicable patches for the given model type.

        Args:
            model_type: The model type (e.g., "gla", "retnet", "mamba2")
            config: Optional model config (used to check fuse_swiglu setting)
            force: If True, apply patches even if USE_FUSED_SWIGLU is enabled
        """
        # Check if fused operations should be used
        use_fused = env_config.get_bool("USE_FUSED_SWIGLU")
        if use_fused and not force:
            if env_config.get_bool("VERBOSE"):
                print(f"[LAT] Fused SwiGLU enabled for {model_type}, skipping patches.")
            return

        # Disable fuse_swiglu in config if present
        if config is not None and hasattr(config, "fuse_swiglu"):
            try:
                config.fuse_swiglu = False
            except Exception:
                pass

        # Apply applicable patches
        for patch in cls._patches:
            if patch.is_applicable(model_type) and not patch.is_applied():
                patch.apply()
                break  # Only apply first matching patch


def apply_model_patches(
    model_type: str,
    config: Optional[Any] = None,
    force: bool = False,
) -> None:
    """
    Convenience function to apply patches for a model type.

    Args:
        model_type: The model type (e.g., "gla", "retnet", "mamba2")
        config: Optional model config
        force: If True, apply patches even if USE_FUSED_SWIGLU is enabled
    """
    PatchManager.apply_patches(model_type, config, force)


def apply_swiglu_patch() -> bool:
    """
    Directly apply the SwiGLU patch.

    Returns:
        True if patch was applied successfully
    """
    return FLASwiGLUPatch().apply()
