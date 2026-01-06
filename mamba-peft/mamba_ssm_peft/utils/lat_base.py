"""
LAT Framework Base Types and Abstractions.

This module defines the core type definitions, protocols, and data classes
used throughout the LAT (Linear Attention Toolkit) framework.

Key Components:
===============
- ModelCapabilities: Describes model-specific behaviors and requirements
- ModelSpec: Complete specification for a model type in the registry
- ModelRegistry: Central registry for all supported model types

Usage:
======
    from mamba_ssm_peft.utils.lat_base import ModelRegistry, ModelCapabilities

    # Get model specification
    spec = ModelRegistry.get("gla")

    # Check capabilities
    if spec.capabilities.has_fuse_swiglu:
        apply_swiglu_patch()

    # Import model classes dynamically
    ConfigClass, ModelClass = spec.import_classes()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type
from importlib import import_module


@dataclass(frozen=True)
class ModelCapabilities:
    """
    Describes model-specific capabilities and requirements.

    This replaces the ad-hoc 'special_handling' dictionary with a
    strongly-typed, documented data class.

    Attributes:
        has_fuse_swiglu: Whether the model supports fused SwiGLU operations
        cache_type: Type of cache used ('past_key_values' or 'cache_params')
        inner_model_attr: Attribute name for inner model ('model' or 'backbone')
        supports_generation: Whether the model supports HF generate()
        requires_attention_mask: Whether attention_mask is required for correct behavior
        default_lora_targets: Default LoRA target modules for this model type
    """

    has_fuse_swiglu: bool = True
    cache_type: str = "past_key_values"  # past_key_values | cache_params
    inner_model_attr: str = "model"  # model | backbone
    supports_generation: bool = True
    requires_attention_mask: bool = True
    default_lora_targets: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    )

    @classmethod
    def for_gla(cls) -> "ModelCapabilities":
        """Standard capabilities for GLA models."""
        return cls(
            has_fuse_swiglu=True,
            cache_type="past_key_values",
            inner_model_attr="model",
            default_lora_targets=(
                "q_proj", "k_proj", "v_proj", "o_proj",
                "g_proj", "gk_proj",
                "gate_proj", "up_proj", "down_proj"
            ),
        )

    @classmethod
    def for_retnet(cls) -> "ModelCapabilities":
        """
        Standard capabilities for RetNet (Retentive Network) models.

        RetNet uses MultiScaleRetention layer with the following projections:
        - q_proj, k_proj, v_proj: Query, Key, Value projections
        - g_proj: Gate projection for output gating (swish gate)
        - o_proj: Output projection

        Note: RetNet does NOT have gk_proj (that's GLA-specific).
        RetNet uses RotaryEmbedding for position encoding instead of learned gating.

        MLP uses SwiGLU with gate_proj, up_proj, down_proj (same as GLA).
        """
        return cls(
            has_fuse_swiglu=True,
            cache_type="past_key_values",
            inner_model_attr="model",
            default_lora_targets=(
                # MultiScaleRetention projections (no gk_proj - that's GLA-specific)
                "q_proj", "k_proj", "v_proj", "o_proj", "g_proj",
                # MLP projections (SwiGLU)
                "gate_proj", "up_proj", "down_proj"
            ),
        )

    @classmethod
    def for_mamba2(cls) -> "ModelCapabilities":
        """Standard capabilities for Mamba2 models."""
        return cls(
            has_fuse_swiglu=False,
            cache_type="cache_params",
            inner_model_attr="backbone",
            default_lora_targets=("in_proj", "out_proj"),
        )

    @classmethod
    def for_delta_net(cls) -> "ModelCapabilities":
        """
        Standard capabilities for DeltaNet models.

        DeltaNet (Parallelizing Linear Transformers with the Delta Rule)
        Reference: https://arxiv.org/abs/2406.06484

        DeltaNet uses the delta rule for state updates instead of simple
        additive updates. Key architectural differences from GLA:

        DeltaNet Layer projections:
        - q_proj, k_proj, v_proj: Query, Key, Value projections
        - o_proj: Output projection
        - b_proj: Beta projection (writing strength, DeltaNet-specific)
                  Outputs num_heads scalars controlling write strength
        - g_proj: Gate projection (optional, when use_gate=True)

        Note: DeltaNet does NOT have gk_proj (that's GLA-specific).
              DeltaNet uses b_proj for beta (writing strength) instead.

        MLP uses SwiGLU with gate_proj, up_proj, down_proj (same as GLA).
        """
        return cls(
            has_fuse_swiglu=True,
            cache_type="past_key_values",
            inner_model_attr="model",
            default_lora_targets=(
                # DeltaNet layer projections
                # Note: b_proj outputs only num_heads scalars, may not be ideal for LoRA
                # We include it for completeness but q/k/v/o are the primary targets
                "q_proj", "k_proj", "v_proj", "o_proj",
                # MLP projections (SwiGLU)
                "gate_proj", "up_proj", "down_proj"
            ),
        )

    @classmethod
    def for_based(cls) -> "ModelCapabilities":
        """
        Standard capabilities for Based models.

        Based (Simple linear attention language models balance the recall-throughput tradeoff)
        Reference: https://arxiv.org/abs/2402.18668

        Based uses Taylor linear attention with 2nd-order Taylor approximation of softmax:
            φ(q)^T φ(k) = 1 + q^T k + (q^T k)^2 / 2

        Key differences from GLA:
        - NO gating mechanism (no g_proj, no gk_proj)
        - Uses TaylorFeatureMap instead of gated recurrence
        - feature_dim (e.g., 16) separate from head_dim (e.g., 64)

        Based Layer projections:
        - q_proj: Projects to feature_dim * num_heads
        - k_proj: Projects to feature_dim * num_heads
        - v_proj: Projects to num_kv_heads * head_dim
        - o_proj: Output projection

        Note: Based does NOT have g_proj or gk_proj (those are GLA-specific).
              Based is simpler - just Q/K/V/O projections with Taylor feature map.

        MLP uses SwiGLU with gate_proj, up_proj, down_proj (same as GLA).
        """
        return cls(
            has_fuse_swiglu=True,
            cache_type="past_key_values",
            inner_model_attr="model",
            default_lora_targets=(
                # Based layer projections (NO g_proj, NO gk_proj!)
                "q_proj", "k_proj", "v_proj", "o_proj",
                # MLP projections (SwiGLU)
                "gate_proj", "up_proj", "down_proj"
            ),
        )


@dataclass
class ModelSpec:
    """
    Complete specification for a model type in the registry.

    Attributes:
        model_type: Canonical model type string (e.g., "gla", "retnet")
        module_path: Python module path (e.g., "fla.models.gla")
        config_class_name: Name of the config class (e.g., "GLAConfig")
        model_class_name: Name of the model class (e.g., "GLAForCausalLM")
        capabilities: Model-specific capabilities
    """

    model_type: str
    module_path: str
    config_class_name: str
    model_class_name: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    def import_classes(self) -> Tuple[Type, Type]:
        """
        Dynamically import the config and model classes.

        Returns:
            Tuple of (ConfigClass, ModelClass)

        Raises:
            ImportError: If the module or classes cannot be imported
        """
        try:
            module = import_module(self.module_path)
            config_cls = getattr(module, self.config_class_name)
            model_cls = getattr(module, self.model_class_name)
            return config_cls, model_cls
        except ImportError as e:
            raise ImportError(
                f"[LAT] Failed to import {self.module_path}. "
                f"Ensure flash-linear-attention is installed. Error: {e}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"[LAT] Class not found in {self.module_path}. Error: {e}"
            ) from e


class ModelRegistry:
    """
    Central registry for all supported Linear Attention model types.

    This implements the Registry Pattern for model discovery and instantiation.
    New model types can be added via the register() class method.

    Usage:
        # Get a model spec
        spec = ModelRegistry.get("gla")

        # List all available models
        models = ModelRegistry.list_models()

        # Register a new model type
        ModelRegistry.register(ModelSpec(
            model_type="custom",
            module_path="custom.models",
            config_class_name="CustomConfig",
            model_class_name="CustomForCausalLM",
            capabilities=ModelCapabilities(...),
        ))
    """

    _registry: Dict[str, ModelSpec] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Initialize the registry with default model specs if not already done."""
        if cls._initialized:
            return

        # GLA: Gated Linear Attention
        cls.register(ModelSpec(
            model_type="gla",
            module_path="fla.models.gla",
            config_class_name="GLAConfig",
            model_class_name="GLAForCausalLM",
            capabilities=ModelCapabilities.for_gla(),
        ))

        # RetNet: Retentive Network
        cls.register(ModelSpec(
            model_type="retnet",
            module_path="fla.models.retnet",
            config_class_name="RetNetConfig",
            model_class_name="RetNetForCausalLM",
            capabilities=ModelCapabilities.for_retnet(),
        ))

        # Mamba2: State Space Model
        cls.register(ModelSpec(
            model_type="mamba2",
            module_path="fla.models.mamba2",
            config_class_name="Mamba2Config",
            model_class_name="Mamba2ForCausalLM",
            capabilities=ModelCapabilities.for_mamba2(),
        ))

        # DeltaNet: Linear Transformers with Delta Rule
        # Reference: https://arxiv.org/abs/2406.06484
        cls.register(ModelSpec(
            model_type="delta_net",
            module_path="fla.models.delta_net",
            config_class_name="DeltaNetConfig",
            model_class_name="DeltaNetForCausalLM",
            capabilities=ModelCapabilities.for_delta_net(),
        ))

        # Based: Simple linear attention with Taylor feature map
        # Reference: https://arxiv.org/abs/2402.18668
        # NOTE: Based 模型在 mamba_ssm_peft.models.based (避免修改 3rdparty)
        cls.register(ModelSpec(
            model_type="based",
            module_path="mamba_ssm_peft.models.based",
            config_class_name="BasedConfig",
            model_class_name="BasedForCausalLM",
            capabilities=ModelCapabilities.for_based(),
        ))

        cls._initialized = True

    @classmethod
    def register(cls, spec: ModelSpec) -> None:
        """
        Register a new model type.

        Args:
            spec: The ModelSpec to register
        """
        cls._registry[spec.model_type] = spec

    @classmethod
    def get(cls, model_type: str) -> ModelSpec:
        """
        Get the ModelSpec for a given model type.

        Args:
            model_type: The model type key (e.g., "gla", "retnet")

        Returns:
            The ModelSpec for the requested model type

        Raises:
            ValueError: If the model type is not registered
        """
        cls._ensure_initialized()

        if model_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"[LAT] Unknown model_type '{model_type}'. "
                f"Available: {available}"
            )
        return cls._registry[model_type]

    @classmethod
    def list_models(cls) -> List[str]:
        """Return list of all registered model types."""
        cls._ensure_initialized()
        return list(cls._registry.keys())

    @classmethod
    def has(cls, model_type: str) -> bool:
        """Check if a model type is registered."""
        cls._ensure_initialized()
        return model_type in cls._registry

    @classmethod
    def get_capabilities(cls, model_type: str) -> ModelCapabilities:
        """Get capabilities for a model type."""
        return cls.get(model_type).capabilities


# Config model type mapping (from config.json model_type to registry key)
CONFIG_MODEL_TYPE_MAP: Dict[str, str] = {
    "gla": "gla",
    "retnet": "retnet",
    "mamba2": "mamba2",
    "delta_net": "delta_net",
    "based": "based",
}
