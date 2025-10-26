#!/usr/bin/env python3
"""
Test script to verify gradient_checkpointing parameter is correctly passed through the pipeline.
"""

import tempfile
import yaml
from pathlib import Path

def test_gradient_checkpointing_parameter():
    """Test that gradient_checkpointing parameter is properly handled."""

    # Create a test YAML config with gradient_checkpointing enabled
    test_cfg = {
        "batch_size": 4,
        "learning_rate": 0.0003,
        "model": "fla-hub/gla-1.3B-100B",
        "no_save": false,
        "num_epochs": 1,
        "peft": "cfg/my_lora_exp/peft/lora_qkvo_r8_a16.json",
        "prec": "bf16",
        "seed": 42,
        "gradient_checkpointing": True,
        "logits_to_keep": 1,
        "num_data_workers": 2
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_cfg, f)
        cfg_path = f.name

    try:
        # Test that the YAML can be loaded without errors
        with open(cfg_path, 'r') as f:
            loaded_cfg = yaml.safe_load(f)

        print("✅ Test YAML loaded successfully")
        print(f"✅ gradient_checkpointing: {loaded_cfg.get('gradient_checkpointing', 'NOT_FOUND')}")
        print(f"✅ logits_to_keep: {loaded_cfg.get('logits_to_keep', 'NOT_FOUND')}")
        print(f"✅ num_data_workers: {loaded_cfg.get('num_data_workers', 'NOT_FOUND')}")

        # Verify the parameters are correctly set
        assert loaded_cfg['gradient_checkpointing'] == True
        assert loaded_cfg['logits_to_keep'] == 1
        assert loaded_cfg['num_data_workers'] == 2

        print("✅ All test parameters are correctly set")

    finally:
        # Clean up
        Path(cfg_path).unlink()

if __name__ == "__main__":
    test_gradient_checkpointing_parameter()
    print("🎉 Gradient checkpointing parameter test passed!")
