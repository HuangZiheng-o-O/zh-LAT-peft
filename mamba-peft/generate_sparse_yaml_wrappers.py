#!/usr/bin/env python3
"""
Generate YAML wrapper files for SD-LoRA JSON configs.

Each YAML file contains just a `peft:` field pointing to the corresponding JSON config.
"""

import os
from pathlib import Path

# Directories
SPARSE_PEFT_DIR = "cfg/my_lora_exp/sparse_peft"
YAML_SPARSE_DIR = "cfg/my_lora_exp/yaml_sparse"

def main():
    # Get all JSON files in sparse_peft directory
    peft_dir = Path(SPARSE_PEFT_DIR)
    yaml_dir = Path(YAML_SPARSE_DIR)
    yaml_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(peft_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON configs in {SPARSE_PEFT_DIR}")
    print(f"Creating YAML wrappers in {YAML_SPARSE_DIR}")
    print()

    for json_file in json_files:
        # Create YAML wrapper with same name but .yaml extension
        yaml_name = json_file.stem + ".yaml"
        yaml_path = yaml_dir / yaml_name

        # Relative path from mamba-peft root
        peft_path = f"{SPARSE_PEFT_DIR}/{json_file.name}"

        # Write YAML file
        with open(yaml_path, 'w') as f:
            f.write(f"peft: {peft_path}\n")

        print(f"  {yaml_name} -> {peft_path}")

    print()
    print(f"Created {len(json_files)} YAML wrapper files")

if __name__ == "__main__":
    main()
