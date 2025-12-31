DoRA and RSLoRA are “silky” toggles:

  - Set HP_USE_DORA=1 (or 0) and/or HP_USE_RSLoRA=1 before calling lat_batch_tmux.sh.
  - Those env vars survive the tmux hop and are consumed inside _apply_peft_env_overrides() (mamba-peft/lat_adapter.py:133-148), which edits the PEFT JSON in-memory before the LoraConfig is instantiated.
  - So a single base LoRA JSON can now act as plain LoRA, DoRA, or RSLoRA just by flipping env flags—no file duplication.

  Other low-effort LoRA variants you can leverage with the current stack:

  1. PiSSA / PiSSA-fast initialization – already supported via HP_INIT=pissa or HP_PISSA_FAST=1 (same function). This often improves convergence on small datasets.
  2. Rank / alpha / dropout sweeps – HP_PEFT_R, HP_PEFT_ALPHA, HP_PEFT_DROPOUT let you explore low-rank vs. higher-rank adapters without editing JSON.
  3. Target-module auto-fill – leave target_modules empty and let _get_target_modules_for_model() inject architecture-aware defaults (useful when adding new layer types).
  4. LoRA scaling tweaks – lora_alpha, lora_dropout, PiSSA init adjustments can emulate “LoRA+” style behaviors at negligible engineering cost.

  Beyond these, other PEFT variants (AdaLoRA, LoHA, QLoRA) would require new loader logic (bitsandbytes integration, scheduler changes, etc.), so they’re not “low-cost” yet. For quick wins, stick with DoRA/RSLoRA/PiSSA via env switches, and
  continue tuning ranks/alphas/dropouts through the existing HP_* overrides.
 
