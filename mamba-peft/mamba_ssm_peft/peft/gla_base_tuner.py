"""
GLA Base Tuner - Base class for GLA-specific PEFT methods.

This module provides the base tuner class adapted for GLA (Gated Linear Attention)
models from the FLA library, replacing the Mamba-specific MambaBaseTuner.

Key differences from MambaBaseTuner:
- Works with GatedLinearAttention layers instead of Mamba mixer blocks
- Accesses GLA-specific parameters (gk_proj, q_proj, k_proj, etc.)
- Handles GLA's matrix-valued state structure
"""

import torch
from torch import nn

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists


class GLABaseTuner(BaseTuner):
    """
    Base tuner class for GLA models.

    Provides common functionality for PEFT methods applied to GLA models,
    including module discovery, parameter access, and device/dtype handling.
    """

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        super().__init__(model, peft_config, adapter_name)

    @property
    def device(self):
        """Get the device of the first GLA attention layer."""
        gla_layers = self.get_gla_attention_layers()
        if gla_layers:
            return next(gla_layers[0].parameters()).device
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        """Get the dtype of the first GLA attention layer."""
        gla_layers = self.get_gla_attention_layers()
        if gla_layers:
            return next(gla_layers[0].parameters()).dtype
        return next(self.model.parameters()).dtype

    def get_gla_attention_layers(self):
        """
        Get all GatedLinearAttention layers from the model.

        Returns:
            List of GatedLinearAttention modules.
        """
        from fla.layers.gla import GatedLinearAttention

        gla_layers = []
        for module in self.model.modules():
            if isinstance(module, GatedLinearAttention):
                gla_layers.append(module)
        return gla_layers

    def get_gla_blocks(self):
        """
        Get all GLA blocks (GLABlock) from the model.

        This is the GLA equivalent of get_mamba_blocks().

        Returns:
            List of GLABlock modules.
        """
        try:
            from fla.models.gla.modeling_gla import GLABlock
            blocks = []
            for module in self.model.modules():
                if isinstance(module, GLABlock):
                    blocks.append(module)
            return blocks
        except ImportError:
            # Fallback: return attention layers
            return self.get_gla_attention_layers()

    @staticmethod
    def _check_target_module_exists(peft_config, key):
        return check_target_module_exists(peft_config, key)

    def _replace_module(self, parent, child_name, new_module, child):
        """Replace a module with a new module, handling device placement."""
        device = next(self.parameters()).device
        new_module = new_module.to(device)
        setattr(parent, child_name, new_module)

    def _create_and_replace(
        self,
        peft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        """Create a new adapter module and replace the target."""
        new_module = self._create_new_module(peft_config, adapter_name, target, target_name)
        if adapter_name != self.active_adapter:
            # Adding an additional adapter: it is not automatically trainable
            new_module.requires_grad_(False)

        if new_module is not None:
            self._replace_module(parent, target_name, new_module, target)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        return self.model._prepare_encoder_decoder_kwargs_for_generation(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
