This folder contains fully resolved experiment configs for GLA LoRA rounds 1-4.

Structure:
- 28 YAML files: round{N}_{EXPCODE}_r{R}_alpha{A}_seed{S}.yaml
- peft/: JSON target-module specs referenced by YAMLs

Notes:
- YAML defaults use data: glue-tvt_cola and seed: 42. The unified launcher
  script can produce runtime copies with TASK/SEED overridden without mutating
  these baselines.

