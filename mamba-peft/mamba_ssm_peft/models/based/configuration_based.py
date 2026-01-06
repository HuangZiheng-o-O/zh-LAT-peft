# -*- coding: utf-8 -*-
"""
Based Model Configuration

Reference: https://arxiv.org/abs/2402.18668
"Simple linear attention language models balance the recall-throughput tradeoff"
"""

from __future__ import annotations

import warnings

from transformers.configuration_utils import PretrainedConfig


class BasedConfig(PretrainedConfig):
    """
    Configuration class for Based model.

    Based uses Taylor linear attention with 2nd-order Taylor approximation of softmax:
        φ(q)^T φ(k) = 1 + q^T k + (q^T k)^2 / 2

    Key differences from GLA:
        - No gating mechanism (no g_proj, no gk_proj)
        - Uses TaylorFeatureMap (pure mathematical transformation, no learnable params)
        - feature_dim (e.g., 16) separate from head_dim (e.g., 64)

    Args:
        hidden_size (int):
            Model hidden dimension. Default: 2048.
        feature_dim (int):
            Feature dimension for Q/K projections (before Taylor expansion).
            The Taylor feature map expands this to ~feature_dim^2. Default: 16.
        num_heads (int):
            Number of attention heads. Default: 16.
        num_kv_heads (int, optional):
            Number of key/value heads for MQA/GQA. Default: None (same as num_heads).
        hidden_ratio (int, optional):
            Ratio for MLP hidden size. Default: 4.
        intermediate_size (int, optional):
            MLP intermediate size. If None, computed from hidden_ratio. Default: None.
        num_hidden_layers (int):
            Number of transformer layers. Default: 24.
        attn_mode (str):
            Attention computation mode. Options: "parallel", "chunk", "fused_chunk".
            Default: "parallel".
        hidden_act (str):
            Activation function for MLP. Default: "swish".
        max_position_embeddings (int):
            Maximum sequence length. Default: 2048.
        elementwise_affine (bool):
            Whether to use elementwise affine in LayerNorm. Default: True.
        norm_eps (float):
            Epsilon for layer normalization. Default: 1e-6.
        attn (dict, optional):
            Configuration for hybrid attention layers (sliding window). Default: None.
        use_cache (bool):
            Whether to use KV cache during generation. Default: True.
        pad_token_id (int, optional):
            Padding token ID. Default: None.
        bos_token_id (int):
            Beginning of sequence token ID. Default: 1.
        eos_token_id (int):
            End of sequence token ID. Default: 2.
        tie_word_embeddings (bool):
            Whether to tie input/output embeddings. Default: False.
        initializer_range (float):
            Standard deviation for weight initialization. Default: 0.02.
        fuse_norm (bool):
            Whether to use fused RMSNorm. Default: True.
        fuse_swiglu (bool):
            Whether to use fused SwiGLU. Default: True.
        fuse_cross_entropy (bool):
            Whether to use fused cross entropy. Default: True.
        fuse_linear_cross_entropy (bool):
            Whether to use fused linear cross entropy. Default: False.
        use_l2warp (bool):
            Whether to use L2 warp. Default: False.
        vocab_size (int):
            Vocabulary size. Default: 32000.
    """

    model_type = 'based'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        feature_dim: int = 16,
        num_heads: int = 16,
        num_kv_heads: int | None = None,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 24,
        attn_mode: str = "parallel",
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        attn: dict | None = None,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_mode = attn_mode
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improve memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        # Hybrid attention configuration (for sliding window)
        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', 64)  # Based uses small window (64)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
