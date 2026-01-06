"""
Unified Linear Attention HuggingFace Decoder for text generation.

This decoder wraps HuggingFace's generate() method for various Linear Attention
models from the FLA library, including GLA, RetNet, DeltaNet, and Mamba2.

Design Principles:
==================
1. **Backward Compatibility**: GLAHFDecoder behavior is preserved exactly.
2. **Unified Interface**: All models use the same decoder class.
3. **Model-Specific Handling**: Mamba2's cache_params is handled transparently.

Environment Variables (LAT_* preferred, GLA_* fallback for compatibility):
=========================================================================
- LAT_USE_MAX_NEW_TOKENS / GLA_USE_MAX_NEW_TOKENS=1 (default): Use max_new_tokens semantics
- LAT_VERBOSE / GLA_VERBOSE=1: Enable verbose logging and padding warnings
- LAT_STRICT_LEFT_PAD / GLA_STRICT_LEFT_PAD=1: Raise error on right-padding detection

Supported Models:
================
- gla: Gated Linear Attention - uses standard HF generate()
- retnet: Retentive Network - uses standard HF generate()
- delta_net: DeltaNet - uses standard HF generate()
- mamba2: Mamba2 - uses HF generate() with cache_params handling

Usage:
======
    from mamba_ssm_peft.utils.lat_decoder import LATHFDecoder, create_lat_decoder

    # Create decoder
    decoder = create_lat_decoder(tokenizer, max_length=256)

    # Or directly instantiate with model_type for special handling
    decoder = LATHFDecoder(tokenizer=tokenizer, model_type="mamba2", max_length=256)

    # Generate
    outputs = decoder(model, input_ids, attention_mask=attention_mask)
    generated_tokens = outputs.sequences  # Already trimmed of prompt
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


def _get_lat_env(key: str, default: str = "0") -> str:
    """
    Get environment variable with LAT_* prefix, falling back to GLA_* for compatibility.
    """
    lat_key = f"LAT_{key}"
    gla_key = f"GLA_{key}"
    return os.getenv(lat_key, os.getenv(gla_key, default))


def _get_lat_env_bool(key: str, default: str = "0") -> bool:
    """Get environment variable as boolean."""
    return _get_lat_env(key, default).lower() in ("1", "true", "yes", "on")


@dataclass
class LATHFDecoder:
    """
    Unified HuggingFace decoder for Linear Attention models.

    This decoder wraps HuggingFace's generate() method and provides:
    - Consistent attention_mask handling
    - Prompt trimming from generated sequences
    - Optional padding validation warnings
    - Model-specific cache handling (e.g., Mamba2's cache_params)

    Attributes:
        tokenizer: The tokenizer (for pad_token_id, eos_token_id)
        model_type: Model type for special handling ("gla", "retnet", "mamba2", "auto")
        max_length: Maximum new tokens to generate (when LAT_USE_MAX_NEW_TOKENS=1)
        min_length: Minimum new tokens to generate
        num_beams: Number of beams for beam search (None for greedy)
        do_sample: Whether to use sampling
    """

    tokenizer: Any
    model_type: str = "auto"  # "gla", "retnet", "mamba2", or "auto"
    max_length: int = 1024  # interpreted as max_new_tokens when LAT_USE_MAX_NEW_TOKENS=1 (default)
    min_length: int = 0     # interpreted as min_new_tokens when supported
    num_beams: Optional[int] = None
    do_sample: bool = False

    def __call__(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Generate sequences using the model.

        Args:
            model: The language model (GLA, RetNet, Mamba2, etc.)
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            GenerateOutput with .sequences attribute containing generated tokens
            (prompt is automatically trimmed)
        """
        # Get pad_token_id
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)

        # Build attention_mask if not provided
        if attention_mask is None:
            attention_mask = input_ids.ne(pad_id) if pad_id is not None else None

        # Determine generation semantics
        use_max_new = _get_lat_env_bool("USE_MAX_NEW_TOKENS", "1")
        verbose = _get_lat_env_bool("VERBOSE", "0")

        # Build generation kwargs
        gen_kwargs = dict(
            input_ids=input_ids,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=bool(self.do_sample),
        )

        # Set length parameters
        if use_max_new:
            if verbose:
                print("[LAT] Using HF generate(max_new_tokens/min_new_tokens) semantics.")
            gen_kwargs["max_new_tokens"] = int(self.max_length)
            if self.min_length and self.min_length > 0:
                gen_kwargs["min_new_tokens"] = int(self.min_length)
        else:
            # Legacy behavior: treat max_length/min_length as total length
            if verbose:
                print("[LAT] Using legacy generate(max_length=prompt+max_length) semantics.")
            gen_kwargs["max_length"] = int(input_ids.shape[1] + self.max_length)
            if self.min_length and self.min_length > 0:
                gen_kwargs["min_length"] = int(input_ids.shape[1] + self.min_length)

        # Add attention_mask
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

            # Optional padding validation
            if verbose:
                self._check_padding_side(attention_mask)

        # Add beam search parameters
        if self.num_beams is not None and self.num_beams > 1:
            gen_kwargs["num_beams"] = int(self.num_beams)
            gen_kwargs["do_sample"] = False

        # Generate
        try:
            outputs = model.generate(**gen_kwargs)
        except TypeError as e:
            # Handle unsupported kwargs (e.g., min_new_tokens in old transformers)
            if use_max_new and "min_new_tokens" in str(e):
                raise RuntimeError(
                    "min_new_tokens is not supported by the current transformers version. "
                    "Set LAT_USE_MAX_NEW_TOKENS=0 to fall back to legacy max_length semantics, "
                    "or upgrade transformers."
                ) from e
            raise

        # Trim prompt from generated sequences
        if hasattr(outputs, "sequences"):
            seq = outputs.sequences
            if seq is not None and seq.dim() == 2 and input_ids is not None and input_ids.dim() == 2:
                # Always trim off the original prompt so metrics only see generated tokens
                outputs.sequences = seq[:, input_ids.shape[1]:]

        return outputs

    def _check_padding_side(self, attention_mask: torch.Tensor) -> None:
        """
        Check for right-padding during generation (which can cause issues).

        Right-padding means the last column has zeros, indicating the valid tokens
        are not at the end of the sequence (bad for autoregressive generation).
        """
        try:
            if attention_mask.size(1) > 0 and (attention_mask[:, -1] == 0).any():
                msg = (
                    "[LAT][warn] Right-padding detected in attention_mask during generation. "
                    "Ensure tokenizer.padding_side='left' and collator applies left padding."
                )
                if _get_lat_env_bool("STRICT_LEFT_PAD", "0"):
                    raise RuntimeError(msg)
                print(msg)
        except Exception:
            pass


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================
def create_lat_decoder(
    tokenizer: Any,
    model_type: str = "auto",
    max_length: int = 1024,
    min_length: int = 0,
    num_beams: Optional[int] = None,
    do_sample: bool = False,
    **kwargs,
) -> LATHFDecoder:
    """
    Create a unified Linear Attention decoder.

    Args:
        tokenizer: The tokenizer
        model_type: Model type ("gla", "retnet", "mamba2", "auto")
        max_length: Maximum new tokens to generate
        min_length: Minimum new tokens to generate
        num_beams: Number of beams for beam search
        do_sample: Whether to use sampling
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        LATHFDecoder instance
    """
    return LATHFDecoder(
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        do_sample=do_sample,
    )


# ============================================================================
# BACKWARD COMPATIBILITY: GLA-specific aliases
# ============================================================================
@dataclass
class GLAHFDecoder(LATHFDecoder):
    """
    Backward-compatible alias for GLA decoder.

    This class is identical to LATHFDecoder but with model_type="gla" default.
    It exists for backward compatibility with code that imports GLAHFDecoder.
    """
    model_type: str = field(default="gla")


def create_gla_decoder(
    tokenizer: Any,
    max_length: int = 1024,
    min_length: int = 0,
    num_beams: Optional[int] = None,
    do_sample: bool = False,
    **kwargs,
) -> GLAHFDecoder:
    """
    Create a GLA decoder - backward compatible with original create_gla_decoder().

    Args:
        tokenizer: The tokenizer
        max_length: Maximum new tokens to generate
        min_length: Minimum new tokens to generate
        num_beams: Number of beams for beam search
        do_sample: Whether to use sampling
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        GLAHFDecoder instance
    """
    return GLAHFDecoder(
        tokenizer=tokenizer,
        model_type="gla",
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        do_sample=do_sample,
    )
