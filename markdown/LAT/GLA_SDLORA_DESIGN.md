# GLA SD-LoRA Design Document

## 1. Mamba vs GLA Architecture Comparison

### 1.1 Mamba SSM Structure

```
Mamba SSM Recurrence:
  h_t = Ā h_{t-1} + B̄ x_t    (state update)
  y_t = C h_t + D x_t         (output)

Where:
  - A_log ∈ ℝ^{D×N}  (state matrix, D=channel, N=state_dim)
  - B ∈ ℝ^{D×N}      (input-to-state projection)
  - C ∈ ℝ^{D×N}      (state-to-output projection)
  - Δ ∈ ℝ^{D}        (timestep, scalar per channel)
  - Ā = exp(Δ ⊙ A)   (discretized state matrix)

State: h_t ∈ ℝ^{D×N} (vector-valued per channel)
```

### 1.2 GLA Structure

```
GLA Recurrence:
  S_t = Diag(α_t) S_{t-1} + k_t^T v_t    (matrix-valued state update)
  o_t = q_t S_t                           (output)

Where:
  - α_t = sigmoid(gk_proj(x_t))^{1/τ}    (data-dependent gate, shape: H×K)
  - q_t ∈ ℝ^{H×K}   (query, from q_proj)
  - k_t ∈ ℝ^{H×K}   (key, from k_proj)
  - v_t ∈ ℝ^{H×V}   (value, from v_proj)

State: S_t ∈ ℝ^{H×K×V} (matrix-valued per head)
```

### 1.3 Key Architectural Differences

| Aspect | Mamba | GLA | Implication for SDT |
|--------|-------|-----|---------------------|
| **State decay** | A_log (parameter) | gk_proj output (computed) | SDT must target **projection weights**, not parameter directly |
| **State shape** | Vector (D×N) | Matrix (H×K×V) | Two dimensions (K, V) vs one (N) |
| **Gate mechanism** | Global Δ per channel | Per-dimension α_t | Finer-grained control possible |
| **Decay parameterization** | A_log directly | logsigmoid(gk)/τ | Different numerical behavior |
| **Input projection** | B learned directly | k_proj linear | Standard linear layer |
| **Output projection** | C learned directly | q_proj linear | Standard linear layer |

## 2. Parameter Mapping

### 2.1 Functional Equivalence

| Mamba Parameter | GLA Equivalent | Role |
|-----------------|----------------|------|
| `A_log` (D×N) | `gk_proj` weights | **State decay/forgetting gate** |
| `in_proj_x` (D→2D) | `q_proj` (hidden→key_dim) | Query/input expansion |
| `in_proj_z` | `g_proj` (output gate) | Output gating |
| `x_proj` → B | `k_proj` (hidden→key_dim) | Input-to-state mapping |
| `x_proj` → C | (implicit in Q·S) | State-to-output mapping |
| `out_proj` | `o_proj` | Final output projection |

### 2.2 GLA Layer Parameters (from fla/layers/gla.py)

```python
GatedLinearAttention:
  ├── q_proj: Linear(hidden_size, key_dim)           # Query projection
  ├── k_proj: Linear(hidden_size, key_dim_per_group) # Key projection
  ├── v_proj: Linear(hidden_size, value_dim_per_group) # Value projection
  ├── g_proj: Linear(hidden_size, value_dim)         # Output gate (optional)
  ├── gk_proj: Sequential(                           # **Gate projection (critical)**
  │       Linear(hidden_size, gate_low_rank_dim),    #   - Low-rank compression
  │       Linear(gate_low_rank_dim, key_dim_per_group)  # - Expansion to gate dims
  │   )
  ├── o_proj: Linear(value_dim, hidden_size)         # Output projection
  └── [conv1d layers if use_short_conv]
```

### 2.3 Dimension Analysis

```
GLA Dimensions (typical config):
  hidden_size = 2048
  num_heads (H) = 4
  expand_k = 0.5 → key_dim = 1024
  expand_v = 1.0 → value_dim = 2048
  head_k_dim = key_dim / num_heads = 256
  head_v_dim = value_dim / num_heads = 512
  gate_low_rank_dim = 16 (default)

State shape: S_t ∈ ℝ^{H × head_k_dim × head_v_dim} = ℝ^{4 × 256 × 512}

gk_proj output: (B, T, key_dim_per_group) → after reshape: (B, T, H, head_k_dim)
```

## 3. SD-LoRA Design for GLA

### 3.1 Critical Insight: The Key Difference

**Mamba**: A_log is a **direct parameter** → SDT selects which (channel, state) dimensions to train/freeze/zero

**GLA**: Gate values come from `gk_proj(x)` → SDT must work on **projection weights**

This means:
1. We cannot directly mask gate dimensions like Mamba's A_log
2. We must identify which **output dimensions** of gk_proj are important
3. Dimension selection based on gradient flow through gk_proj weights

### 3.2 Proposed GLA SDT Targets

```
Priority 1 - Gate Projection (analogous to Mamba's A_log):
  └── gk_proj[1].weight: (key_dim_per_group, gate_low_rank_dim)
      - Controls per-key-dimension gating
      - Select which key dimensions get trained/frozen/zeroed
      - Output dimension corresponds to head_k_dim after reshape

Priority 2 - Input Projections (analogous to Mamba's B, C):
  ├── k_proj.weight: (key_dim_per_group, hidden_size)
  └── q_proj.weight: (key_dim, hidden_size)

Priority 3 - Standard LoRA Targets:
  ├── v_proj.weight
  ├── o_proj.weight
  └── g_proj.weight (if use_output_gate)
```

### 3.3 Dimension Selection Strategy

For GLA, the SDT dimension selection should focus on:

1. **Channel dimension** = head_k_dim (per-head key dimension)
   - This controls which dimensions of the state matrix S_t are important
   - Analogous to Mamba's channel dimension D

2. **State dimension** = Not directly applicable to GLA
   - GLA's matrix state S_t ∈ ℝ^{K×V} doesn't have the same N-state structure
   - Instead, we can consider V (value) dimensions

### 3.4 Implementation Approach

```python
# GLA SD-LoRA Target Configuration
GLA_SDLORA_TARGETS = {
    # SDT targets (sparse dimension tuning)
    "sdt_targets": [
        "gk_proj.1",   # Second layer of gate projection - primary SDT target
    ],

    # LoRA targets (standard low-rank adaptation)
    "lora_targets": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "g_proj",      # if use_output_gate
    ],

    # Dimension configuration
    "num_zero": {
        "channel": 0.1,   # Zero 10% of key dimensions
    },
    "num_freeze": {
        "channel": 0.5,   # Freeze 50% of key dimensions
    },
    # Remaining 40% are trainable
}
```

## 4. Key Implementation Differences from Mamba

### 4.1 Mamba sd_lora.py Adaptations Needed

| Mamba Implementation | GLA Adaptation |
|---------------------|----------------|
| `A_log` transpose handling | Not needed - gk_proj is standard linear |
| State (N) dimension selection | Not applicable - GLA has different state structure |
| `_is_layer_of("A_log")` checks | Replace with `_is_layer_of("gk_proj")` |
| `build_train_param()` with zero mask | Adapt for projection output dimensions |
| Block-level gradient collection | Collect gradients from gk_proj[1].weight |

### 4.2 What to Copy vs What to Redesign

**Copy from mamba-peft-sd_lora** (unchanged):
- `SdLoraConfig` structure (with different target_modules)
- `SdLoraModel` class structure
- Two-phase training logic (warmup → train)
- `should_training_stop` mechanism
- Gradient accumulation during warmup
- Mask-based sparse training

**Redesign for GLA**:
- Target module detection (`gk_proj.1` instead of `A_log`)
- Dimension parsing (only channel dimension, no state)
- Gradient importance calculation (from gk_proj weights)
- Mask building (for output dimensions of gk_proj)
- Parameter wrapping (Linear layer instead of raw parameter)

### 4.3 GLA-Specific Considerations

1. **Low-rank gate projection**: gk_proj is already low-rank (hidden→16→key_dim)
   - SDT on the second layer captures most important dimensions

2. **Per-head structure**: Gate values are reshaped to (B, T, H, head_k_dim)
   - Selection should consider head structure

3. **logsigmoid normalization**: `F.logsigmoid(gk) / gate_logit_normalizer`
   - Zero mask should map to very negative gk values (not 10 like Mamba)

## 5. Proposed File Structure

```
mamba-peft/
├── mamba_ssm_peft/
│   └── peft/
│       ├── sd_lora.py          # Original Mamba SD-LoRA (keep for reference)
│       ├── gla_sd_lora.py      # NEW: GLA-specific SD-LoRA
│       └── lat_sd_lora.py      # FUTURE: Unified LAT SD-LoRA
│
├── train_gla_sdlora.py         # NEW: GLA SD-LoRA training entry
├── configs/
│   └── gla_sdlora/
│       └── default.json        # Default config for GLA SD-LoRA
```

## 6. Next Steps

1. Implement `gla_sd_lora.py` based on sd_lora.py with GLA-specific adaptations
2. Create `GlaSdLoraConfig` with appropriate target modules
3. Implement dimension selection for gk_proj weights
4. Add two-phase training support to existing LAT training infrastructure
5. Test on GLA-1.3B with GLUE benchmarks

## 7. Usage Instructions

### 7.1 Training with GLA SD-LoRA

```bash
# Basic training with default config
python train_gla_sdlora.py --cfg configs/gla_sdlora/glue_cola.yaml

# With custom SD-LoRA config
python train_gla_sdlora.py --cfg configs/gla_sdlora/glue_cola.yaml --peft configs/gla_sdlora/aggressive.json

# Override model and precision
python train_gla_sdlora.py --cfg configs/gla_sdlora/glue_cola.yaml --model fla-hub/gla-1.3B-100B --prec bf16
```

### 7.2 Environment Variable Overrides

```bash
# Override warmup iterations
HP_WARMUP_IT=200 python train_gla_sdlora.py --cfg configs/gla_sdlora/glue_cola.yaml

# Override zero/freeze ratios
HP_ZERO_RATIO=0.5 HP_FREEZE_RATIO=0.3 python train_gla_sdlora.py --cfg configs/gla_sdlora/glue_cola.yaml
```

### 7.3 Config Files

**SD-LoRA Config (JSON)**:
```json
{
    "peft_type": "GLA_SD_LORA",
    "select_mode": "CHANNELS_ONLY",
    "proj_lora_r": 8,
    "num_zero": {"channel": 0.1},
    "num_freeze": {"channel": 0.5},
    "num_warmup_it": 100,
    "target_modules": ["gk_proj.1"],
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```

**Training Config (YAML)**:
```yaml
model: "fla-hub/gla-1.3B-100B"
data: "glue-tvt_cola"
num_epochs: 10
batch_size: 8
learning_rate: 5e-4
prec: "bf16"
seed: 42
```

---

## 8. Open Questions

1. Should we also apply SDT to k_proj/q_proj (analogous to B/C)?
   - Mamba applies SDT to B/C via x_proj
   - For GLA, these are separate projections

2. How to handle MQA (Multi-Query Attention) with num_kv_heads < num_heads?
   - gk is broadcast to all heads: `repeat(gk, '... (h d) -> ... (h g) d')`
   - Selection should happen before broadcast

3. Zero mask value for GLA gates:
   - Mamba uses 10 for A_log (making exp(Δ·A) ≈ 0)
   - GLA uses logsigmoid: `F.logsigmoid(gk)` → very negative = gate ≈ 0
   - Should use large negative value (e.g., -20) instead of 10


Dependency Verification ✅

  Files in mamba-peft (Self-Contained)
 
  | File                                     | Status     | Notes                                                                |
  |------------------------------------------|------------|----------------------------------------------------------------------|
  | mamba_ssm_peft/peft/gla_sd_lora.py       | ✅ Created | GLA SD-LoRA core implementation                                      |
  | mamba_ssm_peft/peft/gla_base_tuner.py    | ✅ Created | GLA base tuner                                                       |
  | mamba_ssm_peft/peft/__init__.py          | ✅ Updated | Added GLA_SD_LORA enum + import                                      |
  | utils/utils.py                           | ✅ Exists  | Has find_layer_by_name, find_module_parent                           |
  | trainer/generic_lm_trainer.py            | ✅ Exists  | Has SD-LoRA support (should_training_stop, load_config, save_config) |
  | dataset/                                 | ✅ Exists  | All dataset modules                                                  |
  | mamba_ssm_peft/utils/lat_model_loader.py | ✅ Exists  | Model loading                                                        |
  | mamba_ssm_peft/utils/lat_decoder.py      | ✅ Exists  | Generation decoder                                                   |
 
  Import Chain (Verified)

```
  train_gla_sdlora.py
  ├── dataset.load_dataset ✅
  ├── trainer.generic_lm_trainer.GenericLMTrainer ✅
  ├── mamba_ssm_peft.* ✅
  ├── mamba_ssm_peft.utils.lat_model_loader ✅
  ├── mamba_ssm_peft.utils.lat_decoder ✅
  ├── mamba_ssm_peft.peft.gla_sd_lora ✅
  │   ├── peft.config.PeftConfig ✅ (external)
  │   ├── peft.tuners.*.BaseTunerLayer ✅ (external)
  │   ├── mamba_ssm_peft.peft.MambaPeftType ✅
  │   ├── mamba_ssm_peft.peft.gla_base_tuner ✅
  │   └── utils.utils.find_layer_by_name ✅
  └── peft.get_peft_model ✅ (external)
```
  References to mamba-peft-sd_lora

  Only in documentation (no code dependencies):
  - docs/GLA_SDLORA_DESIGN.md - Design reference
  - docs/SD_LORA_ANALYSIS.md - Analysis document
  - gla_sd_lora.py line 16 - Comment only

  Key Feature: GenericLMTrainer Already Supports SD-LoRA

```py
  # generic_lm_trainer.py:102-104 - Load config at init
  if hasattr(model, "load_config"):
      model.load_config(self.args.output_dir)

  # generic_lm_trainer.py:169-175 - Phase transition support
  if getattr(model, "should_training_stop", False):
      if hasattr(model, "save_config"):
          model.save_config(self.args.output_dir)
      self.control.should_training_stop = True
```