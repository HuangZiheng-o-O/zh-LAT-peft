"""
Unified Environment Variable Configuration for LAT Framework.

This module provides a single source of truth for all LAT-related environment
variables, with automatic fallback from LAT_* to GLA_* for backward compatibility.

Usage:
======
    from mamba_ssm_peft.utils.env_config import env_config

    # Get string value
    verbose = env_config.get("VERBOSE")

    # Get boolean value
    force_left_pad = env_config.get_bool("FORCE_LEFT_PAD")

    # Get integer value
    stagger_minutes = env_config.get_int("LAUNCH_STAGGER_MINUTES", default=0)

Environment Variable Priority:
==============================
    LAT_* > GLA_* > default value

Supported Variables:
===================
    - FORCE_LEFT_PAD: Force left padding for decoder-only generation (default: "1")
    - USE_MAX_NEW_TOKENS: Use max_new_tokens semantics in generation (default: "1")
    - VERBOSE: Enable verbose logging (default: "0")
    - USE_FUSED_SWIGLU: Enable fused SwiGLU operations (default: "0")
    - STRICT_LEFT_PAD: Raise error on right-padding detection (default: "0")
    - LOG_PADDING_STATS: Log padding statistics during training (default: "0")
    - LAUNCH_STAGGER_MINUTES: Delay between consecutive launches (default: "0")
"""

import os
from typing import Optional


class LATEnvConfig:
    """
    Unified environment variable access layer.

    All LAT-related environment variables should be accessed through this class
    to ensure consistent fallback behavior and default values.
    """

    # All supported environment variables with their default values
    _DEFAULTS = {
        "FORCE_LEFT_PAD": "1",
        "USE_MAX_NEW_TOKENS": "1",
        "VERBOSE": "0",
        "USE_FUSED_SWIGLU": "0",
        "STRICT_LEFT_PAD": "0",
        "LOG_PADDING_STATS": "0",
        "LAUNCH_STAGGER_MINUTES": "0",
        "MODEL": "",
        "PREC": "",
        "OUTPUT_ROOT": "",
    }

    @classmethod
    def get(cls, key: str, default: Optional[str] = None) -> str:
        """
        Get environment variable with LAT_* > GLA_* > default fallback.

        Args:
            key: Variable name without prefix (e.g., "VERBOSE" not "LAT_VERBOSE")
            default: Override default value (uses _DEFAULTS if None)

        Returns:
            The environment variable value
        """
        if default is None:
            default = cls._DEFAULTS.get(key, "")
        return os.getenv(f"LAT_{key}", os.getenv(f"GLA_{key}", default))

    @classmethod
    def get_bool(cls, key: str, default: Optional[str] = None) -> bool:
        """
        Get environment variable as boolean.

        Truthy values: "1", "true", "yes", "on" (case-insensitive)
        """
        return cls.get(key, default).lower() in ("1", "true", "yes", "on")

    @classmethod
    def get_int(cls, key: str, default: int = 0) -> int:
        """
        Get environment variable as integer.

        Returns default if value cannot be parsed as int.
        """
        try:
            value = cls.get(key, str(default))
            return int(value) if value else default
        except ValueError:
            return default

    @classmethod
    def get_float(cls, key: str, default: float = 0.0) -> float:
        """
        Get environment variable as float.

        Returns default if value cannot be parsed as float.
        """
        try:
            value = cls.get(key, str(default))
            return float(value) if value else default
        except ValueError:
            return default


# Global singleton for convenient access
env_config = LATEnvConfig()


# Convenience functions for direct import
def get_lat_env(key: str, default: str = "0") -> str:
    """Get environment variable with LAT_* > GLA_* fallback."""
    return env_config.get(key, default)


def get_lat_env_bool(key: str, default: str = "0") -> bool:
    """Get environment variable as boolean."""
    return env_config.get_bool(key, default)


def get_lat_env_int(key: str, default: int = 0) -> int:
    """Get environment variable as integer."""
    return env_config.get_int(key, default)
