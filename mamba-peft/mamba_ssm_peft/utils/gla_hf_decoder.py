import torch
from dataclasses import dataclass
from typing import Optional, Any
import os


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse environment variable as boolean."""
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Parse environment variable as integer."""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    """Parse environment variable as float."""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


@dataclass
class GLAHFDecoder:
    """
    GLA-compatible HuggingFace decoder for generation tasks.
    
    IMPORTANT: FLA (flash-linear-attention) models do NOT support beam search with 
    `use_cache=True` because the FLACache class lacks the `reorder_cache` method 
    required by HuggingFace's beam search implementation.
    
    To use beam search (num_beams > 1), this decoder automatically sets `use_cache=False`,
    which is slower but correct. For greedy/sampling decoding, `use_cache=True` is used
    for maximum speed.
    
    Environment variables:
        - EVAL_GEN_NUM_BEAMS: Override num_beams (default: 1 for greedy)
        - EVAL_GEN_MAX_LENGTH: Override max_new_tokens
        - EVAL_GEN_MIN_LENGTH: Override min_new_tokens
        - EVAL_GEN_LENGTH_PENALTY: Length penalty for beam search (default: 1.0)
        - EVAL_GEN_NO_REPEAT_NGRAM: No repeat n-gram size (default: 0, disabled)
        - EVAL_GEN_EARLY_STOPPING: Enable early stopping for beam search (default: True when num_beams > 1)
        - GLA_VERBOSE: Enable verbose logging
        - GLA_FORCE_USE_CACHE: Force use_cache=True even with beam search (WILL FAIL, for debugging only)
    """
    tokenizer: Any
    max_length: int = 128  # max_new_tokens (reduced default for DART-like tasks)
    min_length: int = 5    # min_new_tokens
    num_beams: Optional[int] = None
    do_sample: bool = False
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = True
    
    def __post_init__(self):
        """Apply environment variable overrides after initialization."""
        # Environment overrides (highest priority)
        env_beams = _env_int("EVAL_GEN_NUM_BEAMS")
        if env_beams is not None:
            self.num_beams = env_beams
        
        env_max = _env_int("EVAL_GEN_MAX_LENGTH")
        if env_max is not None:
            self.max_length = env_max
            
        env_min = _env_int("EVAL_GEN_MIN_LENGTH")
        if env_min is not None:
            self.min_length = env_min
            
        env_lp = _env_float("EVAL_GEN_LENGTH_PENALTY")
        if env_lp is not None:
            self.length_penalty = env_lp
            
        env_ngram = _env_int("EVAL_GEN_NO_REPEAT_NGRAM")
        if env_ngram is not None:
            self.no_repeat_ngram_size = env_ngram
            
        env_early = os.environ.get("EVAL_GEN_EARLY_STOPPING")
        if env_early is not None:
            self.early_stopping = _env_bool("EVAL_GEN_EARLY_STOPPING", True)

    def __call__(self, model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Generate text using HuggingFace's generate() method.
        
        For beam search (num_beams > 1):
            - use_cache=False is REQUIRED because FLA's Cache doesn't support reorder_cache
            - This is slower but produces correct results
            
        For greedy/sampling (num_beams <= 1):
            - use_cache=True for maximum speed
        """
        verbose = _env_bool("GLA_VERBOSE")
        
        # Determine effective num_beams
        effective_beams = self.num_beams if self.num_beams is not None else 1
        use_beam_search = effective_beams > 1
        
        # CRITICAL: FLA models cannot use cache with beam search
        # The FLACache class lacks reorder_cache() which HuggingFace beam search requires
        force_cache = _env_bool("GLA_FORCE_USE_CACHE")
        if use_beam_search and force_cache:
            raise RuntimeError(
                f"[GLA] FATAL: Cannot use beam search (num_beams={effective_beams}) with use_cache=True. "
                f"FLA's Cache does not implement reorder_cache() required by HuggingFace beam search. "
                f"Either: (1) Remove GLA_FORCE_USE_CACHE=1, or (2) Set EVAL_GEN_NUM_BEAMS=1 for greedy decoding."
            )
        
        use_cache = not use_beam_search  # Only use cache for greedy/sampling
        
        if verbose:
            mode_str = f"beam_search(beams={effective_beams})" if use_beam_search else "greedy"
            cache_str = "use_cache=True" if use_cache else "use_cache=False (required for beam search)"
            print(f"[GLA] Generation mode: {mode_str}, {cache_str}")
        
        # Build attention mask
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if attention_mask is None:
            attention_mask = input_ids.ne(pad_id) if pad_id is not None else None

        # Base generation kwargs
        gen_kwargs = dict(
            input_ids=input_ids,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=use_cache,  # CRITICAL: False for beam search
            max_new_tokens=int(self.max_length),
        )
        
        # Add min_new_tokens if supported
        if self.min_length and self.min_length > 0:
            gen_kwargs["min_new_tokens"] = int(self.min_length)

        # Add attention mask
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
            # Check for right-padding (problematic for decoder-only models)
            if verbose:
                try:
                    if attention_mask.size(1) > 0 and (attention_mask[:, -1] == 0).any():
                        print("[GLA][warn] Right-padding detected. Ensure tokenizer.padding_side='left'.")
                except Exception:
                    pass

        # Configure decoding strategy
        if use_beam_search:
            gen_kwargs["num_beams"] = effective_beams
            gen_kwargs["do_sample"] = False
            gen_kwargs["length_penalty"] = self.length_penalty
            gen_kwargs["early_stopping"] = self.early_stopping
            if self.no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
            if verbose:
                print(f"[GLA] Beam search config: num_beams={effective_beams}, "
                      f"length_penalty={self.length_penalty}, early_stopping={self.early_stopping}, "
                      f"no_repeat_ngram_size={self.no_repeat_ngram_size}")
        else:
            gen_kwargs["do_sample"] = bool(self.do_sample)

        # Generate
        try:
            outputs = model.generate(**gen_kwargs)
        except AttributeError as e:
            if "past_key_values" in str(e):
                # This should not happen if we correctly set use_cache=False for beam search
                raise RuntimeError(
                    f"[GLA] FATAL: Beam search failed due to cache manipulation. "
                    f"This indicates a bug: use_cache should be False for num_beams={effective_beams}. "
                    f"Current use_cache={use_cache}. Error: {e}"
                ) from e
            raise
        except TypeError as e:
            if "min_new_tokens" in str(e):
                # Retry without min_new_tokens for older transformers
                gen_kwargs.pop("min_new_tokens", None)
                if verbose:
                    print("[GLA] Retrying without min_new_tokens (older transformers version)")
                outputs = model.generate(**gen_kwargs)
            else:
                raise

        # Trim prompt from output sequences
        if hasattr(outputs, "sequences"):
            seq = outputs.sequences
            if seq is not None and seq.dim() == 2 and input_ids is not None and input_ids.dim() == 2:
                outputs.sequences = seq[:, input_ids.shape[1]:]
        
        return outputs


def create_gla_decoder(tokenizer, **kwargs) -> GLAHFDecoder:
    """
    Factory function to create a GLAHFDecoder.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum new tokens to generate (default: 128)
        min_length: Minimum new tokens to generate (default: 5)
        num_beams: Number of beams for beam search (default: None, uses greedy)
        do_sample: Whether to use sampling (only for greedy mode)
        length_penalty: Length penalty for beam search (default: 1.0)
        no_repeat_ngram_size: N-gram size for no-repeat constraint (default: 0)
        early_stopping: Enable early stopping for beam search (default: True)
        
    Environment variables can override these settings:
        EVAL_GEN_NUM_BEAMS, EVAL_GEN_MAX_LENGTH, EVAL_GEN_MIN_LENGTH, etc.
    """
    return GLAHFDecoder(tokenizer=tokenizer, **kwargs)
