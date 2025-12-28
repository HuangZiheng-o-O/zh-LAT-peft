# DoRA Integration Guide

This document explains how the zh-LAT-peft training stack already supports **DoRA (Weight-Decomposed Low-Rank Adaptation)** and how to configure, launch, and validate DoRA experiments within the existing LAT pipeline.

## 1. Background

DoRA augments classic LoRA by decomposing the weight update into a direction-preserving vector and a learned magnitude scalar. This makes the adapter more expressive at the same parameter budget and typically improves stability on small datasets. DoRA requires support from the PEFT layer (via `peft.LoraConfig(use_dora=True)`) but otherwise reuses the same training loop, optimizer, and data pipeline as standard LoRA.

## 2. End-to-End Flow

The LAT pipeline already propagates DoRA configuration from YAML → JSON → Python trainer.

1. **Batch launcher** (`mamba-peft/scripts/train/new/lat_batch_tmux.sh`:32-113) exports `MODEL_TYPE`, `LAT_MODEL`, and precision overrides, then spawns `lat_round.sh` inside tmux.
2. **Round runner** (`mamba-peft/scripts/train/new/lat_round.sh`:10-474) injects dataset-specific YAML, calls `train_lat.py --model-type … --model … --prec …`, and forwards the `peft` path embedded in each YAML file.
3. **Training entry point** (`mamba-peft/train_lat.py`:632-708) reads env overrides, loads the YAML, and passes the `peft` JSON to `prepare_lat_model_and_tokenizer()`.
4. **Adapter loader** (`mamba-peft/lat_adapter.py`:180-248) loads the PEFT JSON, applies env overrides, builds `peft.LoraConfig`, and calls `peft.get_peft_model()`. Any DoRA flag inside the JSON (e.g., `"use_dora": true`) is honored transparently.

Because the adapter logic already delegates to Hugging Face PEFT, no additional wiring is required beyond selecting the correct PEFT JSON.

## 3. DoRA Configuration Files

DoRA-ready adapter configs live under `mamba-peft/cfg/my_lora_exp/peft/`. Each JSON is a plain `LoraConfig` payload enriched with `"use_dora": true`. Example (`lora_QKVO_plus_G_plus_GK_DoRA_r8_alpha16.json`:1-13):

```json
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": [
    "attn.g_proj",
    "attn.gk_proj.0",
    "attn.gk_proj.1",
    "attn.k_proj",
    "attn.o_proj",
    "attn.q_proj",
    "attn.v_proj"
  ],
  "use_dora": true
}
```

To enable a DoRA run, point a YAML file’s `peft:` field to one of these JSONs (see `mamba-peft/cfg/my_lora_exp/yaml/E1_QKVO_plus_G_plus_GK_DoRA_r8_alpha16.yaml`).

## 4. Launching a DoRA Experiment

1. Export the base model and precision once:
   ```bash
   export MODEL_TYPE=gla
   export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
   export LAT_PREC=bf16
   ```
2. Export task-specific HPs (batch size, LR, etc.) as usual.
3. Ensure the chosen YAML references a DoRA PEFT JSON.
4. Launch:
   ```bash
   ./lat_batch_tmux.sh \
     --suite E15 \
     --round all \
     --pairs "87:glue-tvt_mrpc" \
     --gpus "0 1 2 3" \
     --gpu-plan "1,1,1,1" \
     --model-type gla
   ```

The logs will list the injected YAML and confirm PEFT attachment. You can also inspect `output/.../cfg.yaml` to verify the `peft` path.

## 5. Customization & Overrides

- **Rank / Alpha / Dropout**: Override via env vars before launch, e.g. `HP_PEFT_R=4`, `HP_PEFT_ALPHA=64`, `HP_PEFT_DROPOUT=0.1`. These are applied in `_apply_peft_env_overrides()` (`lat_adapter.py`:116-175).
- **Target modules**: Leave them unset in the JSON to fall back to architecture-specific defaults (`lat_adapter.py`:240-244). Otherwise, list every projection that should receive DoRA updates.
- **Initialization**: Set `HP_INIT=pissa` or `HP_PISSA_FAST=1` to switch DoRA adapters to PiSSA initialization without editing JSON files.
- **Force DoRA/RSLoRA without new JSONs**: Export `HP_USE_DORA=1` (or `0` to disable) before launching. The env flag overrides the `use_dora` field in memory via `_apply_peft_env_overrides()`. Likewise, `HP_USE_RSLoRA` toggles RSLoRA. This lets a single base PEFT JSON cover LoRA, DoRA, and RSLoRA experiments with no file duplication.

## 6. Validation Checklist

1. **Console output**: At startup, `train_lat.py` prints the path to the PEFT JSON. Ensure it matches the DoRA config.
2. **Parameter counts**: `mamba-peft/train_lat.py` calls `print_trainable_parameter_names()`; the output file in the experiment directory should list only the intended DoRA modules.
3. **Saved config**: `output/.../cfg.yaml` mirrors the injected YAML; confirm the `peft` entry points to a `use_dora` JSON.
4. **PEFT metadata**: The `output/.../parameter_counts.json` contains `trainable%`; DoRA typically matches LoRA of the same rank.

## 7. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No adapters appear in `parameter_counts.json` | YAML `peft:` path missing or typo | Point YAML to an existing DoRA JSON |
| Training fails with `KeyError: use_dora` | Using an outdated PEFT version | Upgrade `peft` to ≥0.6.0, which includes DoRA support |
| Gradients blow up on small tasks | LR too high for DoRA scale | Lower `HP_LR`, increase `lora_dropout`, or reduce rank |
| Target modules mismatch | Architecture updated (e.g., new layer names) | Update `target_modules` list or rely on defaults from `_get_target_modules_for_model()` |

## 8. Extending to New DoRA Variants

To add a new DoRA configuration:
1. Copy an existing JSON in `cfg/my_lora_exp/peft/` and edit `target_modules`, `r`, or `lora_alpha` as needed.
2. Reference the new JSON from a YAML under `cfg/my_lora_exp/yaml/`.
3. Launch via `lat_batch_tmux.sh`; no code changes are required as long as the JSON remains a valid `LoraConfig` payload.

If you only need to flip DoRA/RSLoRA on top of an existing LoRA JSON, set `HP_USE_DORA=1` (or `HP_USE_RSLoRA=1`) instead of cloning files.

## 9. References

- DoRA paper: *Weight-Decomposed Low-Rank Adaptation* (ICML 2023). Introduces the direction-magnitude decomposition that improves LoRA expressiveness.
- Hugging Face PEFT docs: describe `use_dora`, `use_rslora`, and PiSSA initialization flags.

By following this guide, you can run DoRA adapters across all supported LAT models without modifying the training scripts—just select the appropriate PEFT JSON and launch your experiment.
