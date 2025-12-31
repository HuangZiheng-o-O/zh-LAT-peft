#!/usr/bin/env python3
"""
Batch generate GLA SD-LoRA configuration files.

Usage:
    python generate_gla_sdlora_configs.py --output-dir cfg/my_lora_exp/sparse_peft
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


# Define lora_targets combinations
LORA_TARGETS_CONFIGS = {
    "kv": ["k_proj", "v_proj"],
    "v": ["v_proj"],
    "vo": ["v_proj", "o_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qkvog": ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj"],
    "qkvo_plus_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "mlp_up", "mlp_down"],
    "omlp": ["o_proj", "mlp_up", "mlp_down"],
}

# Train ratios configuration
# KV and QKVO: full range of train ratios
FULL_TRAIN_RATIOS = [1, 5, 10, 20, 30]
# Other configs: only 5%
SINGLE_TRAIN_RATIO = [5]

# Configs that get full train ratio range
FULL_RATIO_CONFIGS = ["kv", "qkvo"]

# Always use Zero=0% as per new strategy
ZERO_RATIO = 0.0
NUM_WARMUP_IT = 100


def generate_config(
    lora_targets: List[str],
    train_ratio: int,
    proj_lora_r: int = 8,
    num_warmup_it: int = NUM_WARMUP_IT,
) -> Dict[str, Any]:
    """
    Generate a single GLA SD-LoRA configuration.
    """
    train_fraction = train_ratio / 100.0
    zero_fraction = ZERO_RATIO
    freeze_fraction = 1.0 - train_fraction - zero_fraction

    assert freeze_fraction >= 0.0, f"Freeze fraction is negative: {freeze_fraction}"

    config = {
        "peft_type": "GLA_SD_LORA",
        "select_mode": "CHANNELS_ONLY",
        "proj_lora_r": proj_lora_r,
        "num_zero": {
            "channel": zero_fraction
        },
        "num_freeze": {
            "channel": freeze_fraction
        },
        "num_warmup_it": num_warmup_it,
        "target_modules": ["gk_proj.1"],
        "lora_targets": lora_targets,
        "finetune_parameters": None,
        "sdlora_alpha": {
            "global": 1.0,
            "gk_proj.1": 1.0
        },
        "_comment": (
            f"GLA SD-LoRA: Train={train_ratio}%, Freeze={freeze_fraction*100:.0f}%, Zero=0%. "
            f"LoRA on: {', '.join(lora_targets)}"
        )
    }

    return config


def save_config(config: Dict[str, Any], path: Path) -> None:
    """Save configuration to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Generated: {path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate GLA SD-LoRA configuration files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cfg/my_lora_exp/sparse_peft",
        help="Output directory for configuration files (flat structure)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GLA SD-LoRA Configuration Generator")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Zero ratio: 0% (fixed, never use)")
    print(f"Full ratio configs (KV, QKVO): train {FULL_TRAIN_RATIOS}")
    print(f"Other configs: train {SINGLE_TRAIN_RATIO}")
    print(f"{'='*70}\n")

    total_configs = 0
    generated_files = []

    # Generate configs for each combination
    for lora_target_name, lora_targets in LORA_TARGETS_CONFIGS.items():
        # Determine which train ratios to use
        if lora_target_name in FULL_RATIO_CONFIGS:
            train_ratios = FULL_TRAIN_RATIOS
        else:
            train_ratios = SINGLE_TRAIN_RATIO

        for train_ratio in train_ratios:
            config = generate_config(
                lora_targets=lora_targets,
                train_ratio=train_ratio,
                proj_lora_r=8,
                num_warmup_it=NUM_WARMUP_IT
            )

            # Flat naming: gla_sdlora_{target}_{train_ratio}.json
            filename = f"gla_sdlora_{lora_target_name}_train{train_ratio:02d}.json"
            config_file = output_dir / filename

            save_config(config, config_file)
            generated_files.append(filename)
            total_configs += 1

    print(f"\n{'='*70}")
    print(f"✓ Successfully generated {total_configs} configuration files")
    print(f"  All files in: {output_dir}/")
    print(f"{'='*70}\n")

    # Print summary by category
    print("Generated files:")
    print()
    print("  KV (5 files):")
    for f in sorted(generated_files):
        if "_kv_" in f:
            print(f"    {f}")

    print()
    print("  QKVO (5 files):")
    for f in sorted(generated_files):
        if "_qkvo_" in f and "_qkvog_" not in f and "_plus_" not in f:
            print(f"    {f}")

    print()
    print("  Other (5 files, only train05):")
    for f in sorted(generated_files):
        if "_v_" in f or "_vo_" in f or "_qkvog_" in f or "_plus_" in f or "_omlp_" in f:
            print(f"    {f}")

    print()
    print("LoRA targets summary:")
    for name, targets in LORA_TARGETS_CONFIGS.items():
        ratios = FULL_TRAIN_RATIOS if name in FULL_RATIO_CONFIGS else SINGLE_TRAIN_RATIO
        print(f"  {name:15} → {', '.join(targets):45} → train {ratios}")


if __name__ == "__main__":
    main()
