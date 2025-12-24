# **DART Dataset Debugging Timeline & Clean GLA Route Complete Documentation**

## **Overview**

This document provides a comprehensive timeline of the DART dataset debugging process and a complete reading guide for the **clean GLA (Gated Linear Attention) route** implementation. The clean GLA route is designed to be free from Mamba decoder pollution, focusing solely on GLA + LoRA + HF Trainer pipeline.

---

## **Part 1: DART Dataset Debugging Timeline**

### **Initial Issue (Week 1-2)**

**Error:** DART dataset loading failures with `KeyError: 'triple'` and corrupted cache files.

**Root Cause:**
- DART dataset structure mismatch between expected format and actual HF dataset
- Cache corruption from incomplete preprocessing runs
- Parallel processing race conditions in `ParallelProcessorFS`

**Solutions Applied:**
1. **Fixed DART data structure mapping** in `dataset/dart_data.py`:
   ```python
   # Before: Incorrect triple extraction
   triple = example['triple']

   # After: Proper triple extraction with fallback
   triple = example.get('triple', example.get('triples', ''))
   ```

2. **Implemented cache validation** in `dataset/base.py`:
   ```python
   # Added cache integrity checks
   def _validate_cache_file(self, cache_file):
       try:
           with open(cache_file, "rb") as f:
               data = pickle.load(f)
           return len(data) > 0
       except Exception:
           return False
   ```

3. **Enhanced parallel processing robustness** in `utils/parallel_processor_fs.py`:
   - Added atomic writes with temporary files
   - Implemented worker failure recovery
   - Added progress tracking and error aggregation

**Scripts Used:**
- `change_logs/debug_dart_files/debug_dart_loading.py`
- `change_logs/debug_dart_files/fix_dart_remote.sh`
- `change_logs/debug_dart_files/test_dartdataset_full.py`

---

### **Secondary Issues (Week 3)**

**Error:** Memory exhaustion during large dataset preprocessing.

**Root Cause:**
- Loading entire dataset into memory before parallel processing
- Insufficient worker process management
- Large intermediate data structures

**Solutions:**
1. **Streaming dataset loading**:
   ```python
   # Implemented lazy loading in DatasetBase
   def __len__(self):
       if self.data is None:
           return len(self.get_hf_dataset())
       return len(self.data)
   ```

2. **Memory-efficient preprocessing**:
   ```python
   # Added subset_size validation
   if num_parallel_workers > 0:
       assert subset_size is None  # Force in-memory processing for large datasets
   ```

3. **Enhanced error handling** in preprocessing scripts.

**Commands Used:**
```bash
# Remote debugging commands
rsync -avz --exclude='*.pyc' /local/path user@remote:/remote/path
cd /remote/path && python debug_dart_loading.py
```

---

### **Final Resolution (Week 4)**

**Error:** Intermittent cache corruption and inconsistent preprocessing results.

**Root Cause:**
- Race conditions in parallel file writes
- Inconsistent environment variables across workers
- Missing atomic operations for cache files

**Final Solutions:**
1. **Atomic cache operations**:
   ```python
   # In ParallelProcessorFS.aggregate_result()
   tmp_path = self.output_file.with_suffix(self.output_file.suffix + f".tmp.{os.getpid()}")
   with open(tmp_path, "wb") as f:
       pickle.dump(output_all, f)
   os.replace(tmp_path, self.output_file)  # Atomic move
   ```

2. **Environment propagation**:
   ```python
   # In gla_batch_tmux_clean.sh
   printf 'export FORCE_SEED=%q\n' "${FORCE_SEED:-}"
   printf 'export DATA=%q\n' "${DATA:-}"
   ```

3. **Comprehensive validation**:
   ```python
   # Added post-processing validation
   def verify_dart_integrity(self):
       """Verify all DART samples have required fields"""
       for idx, sample in enumerate(self.data):
           if not all(k in sample for k in ['input_ids', 'label_ids']):
               raise ValueError(f"Sample {idx} missing required fields")
   ```

---

## **Part 2: Clean GLA Route - Complete Reading Guide**

### **Architecture Overview**

The clean GLA route implements a **pure GLA + LoRA + HF Trainer pipeline** without Mamba decoder dependencies:

```
Shell Scripts → tmux Batch → train_gla_only.py → GLA Loader → Dataset → GenericLMTrainer → GLAHFDecoder → Metrics
```

### **1. Shell → tmux → Script Entry Points**

#### **Batch Scheduling Scripts**
- `scripts/train/new/gla_batch_tmux_clean.sh`: **Main batch launcher**
  - Manages multiple sequential GLA training jobs in single tmux session
  - Handles GPU allocation, environment propagation, logging
  - Key features: automatic session naming, job sequencing, failure handling

- `scripts/train/new/gla_round_clean.sh`: **Single round executor**
  - Defines experimental suites (E1-E15) with different LoRA configurations
  - Manages parallel GPU training with dynamic round slicing
  - Handles YAML injection, seed management, cache management

- `scripts/train/new/gla_round_glaonly.sh`: **Simplified single-job launcher**
  - Minimal interface for quick GLA-only testing
  - Sequential job execution (one process at a time)

#### **Key Configuration Suites**
```bash
# Example usage
./gla_batch_tmux_clean.sh --suite E1 --round all --pairs "127:glue-tvt_rte,87:glue-tvt_cola" --gpus "0,1"
```

### **2. Python Training Entry: `train_gla_only.py`**

#### **Core Architecture**
- **Pure GLA pipeline** without Mamba dependencies
- **HF Trainer integration** with custom `GenericLMTrainer`
- **Environment-driven configuration** with extensive override support

#### **Key Functions**

**`build_and_run_trainer_gla_only()`**:
- Sets up GLA model with left padding for decoder-only generation
- Configures SwanLab logging and email notifications
- Initializes `GenericLMTrainer` with HF-compatible arguments
- Handles generation tasks vs classification tasks

**`run_train()`**:
- Lock-based training to prevent duplicate runs
- Checkpoint resume logic
- Environment variable processing for hyperparameters
- Output path construction

#### **Environment Variable Integration**
```python
# Extensive env var support
HP_SEED, HP_BATCH_SIZE, HP_LR, HP_EPOCHS, HP_EVAL_STEPS, HP_SAVE_STEPS
GLA_FORCE_LEFT_PAD, GLA_USE_MAX_NEW_TOKENS, GLA_VERBOSE
SWANLAB_ENABLE, SWANLAB_PROJECT, SWANLAB_EMAIL_YAML
```

### **3. GLA Model Loading & LoRA Injection**

#### **`train_gla_adapter.py`**

**`prepare_gla_model_and_tokenizer()`**:
- Loads GLA model via `fla.models.gla.GLAForCausalLM`
- Applies HF PEFT LoRA configuration
- Handles environment-based LoRA parameter overrides
- Returns (model, tokenizer, peft_config)

#### **Key Features**
- **Robust GLA loading** with automatic path resolution
- **Environment-driven LoRA tuning**:
  ```python
  # HP_PEFT_R, HP_PEFT_ALPHA, HP_PEFT_DROPOUT overrides
  if r_env is not None:
      peft_json["r"] = int(r_env)
  ```

#### **`mamba_ssm_peft/utils/hf.py`**

**`load_gla()`**:
- Primary GLA model loader using `fla.models.gla.GLAConfig.from_pretrained()`
- Disables fused SwiGLU for PyTorch compatibility
- Handles device placement and dtype conversion
- Robust fallback for different GLA model sources

**Key Safety Features**:
```python
# Runtime SwiGLU patching for compatibility
try:
    import torch.nn.functional as F
    def _pt_swiglu(x, y):
        return F.silu(x) * y
    _mlp.swiglu = _pt_swiglu
except Exception:
    pass
```

### **4. GLA HF Decoder: `mamba_ssm_peft/utils/gla_hf_decoder.py`**

#### **`GLAHFDecoder` Class**
- **HF-native generation** using `model.generate()`
- **Flexible token semantics** (max_new_tokens vs max_length)
- **Attention mask handling** with right-padding detection
- **Beam search support**

#### **Key Features**
```python
# Smart length handling
if use_max_new:
    gen_kwargs["max_new_tokens"] = int(self.max_length)
    if self.min_length and self.min_length > 0:
        gen_kwargs["min_new_tokens"] = int(self.min_length)
```

### **5. Trainer System**

#### **`trainer/generic_lm_trainer.py`**

**`GenericLMTrainer`** (extends HF Trainer):
- **Cross-entropy loss** with ignore_index handling
- **Generation evaluation** support
- **Gradient checkpointing** compatibility
- **Custom metrics computation**

**Key Methods**:
- `compute_loss()`: Standard LM loss computation
- `prediction_step()`: Validation step with proper masking
- `evaluate_generation()`: Text generation evaluation pipeline
- `generation_step()`: Decoder integration

#### **`trainer/trainer_utils.py`**

**`MambaEvalPrediction`**: Enhanced evaluation prediction handling
- Token decoding with EOS handling
- Text prediction storage
- YAML serialization for caching

**Early Stopping Classes**:
- `TrainLossEarlyStop`: Monitors training loss for NaN/inf detection
- `BadEvalEarlyStop`: Stops training on evaluation metric degradation

#### **`trainer/loss.py`**

**`CrossEntropy`**: Standard cross-entropy with ignore_index
**`Accuracy`**: Classification accuracy metric

### **6. Dataset System**

#### **Core Architecture**
- **`dataset/__init__.py`**: Dataset factory with string-based dispatch
- **`dataset/base.py`**: Abstract base classes for all datasets
- **`dataset/collator.py`**: Data collation with attention mask generation

#### **Key Dataset Types**

**Classification Tasks**:
- `GlueDataModule`: GLUE benchmark with task-specific prompts
- `ArcDataModule`, `BoolQDataModule`, `PiqaDataModule`: QA/choice tasks

**Generation Tasks**:
- `DartDataModule`: Data-to-text generation
- `SamSumDataModule`: Summarization
- `SpiderDataModule`: SQL generation

#### **Base Classes**

**`DatasetBase`**: Core dataset functionality
- **Caching system** with parallel processing support
- **Lazy loading** with materialization on demand
- **Subset sampling** for debugging

**`NluDatasetBase`**: Classification datasets
- Label string/int conversion
- Prompt prefixing

**`NlgDatasetBase`**: Generation datasets
- EOS token handling
- Input-label formatting

#### **Parallel Processing**
**`utils/parallel_processor_fs.py`**:
- Multi-worker preprocessing with atomic writes
- Progress tracking and error aggregation
- Memory-efficient large dataset handling

### **7. Configuration & Experiment Management**

#### **LoRA Configurations**
Located in `cfg/my_lora_exp/`:

**YAML Experiment Definitions** (`yaml/`):
- `E1_QKVO_r8_alpha16.yaml`: Standard QKVO LoRA baseline
- `E2_OMLP_r8_alpha16.yaml`: Output MLP targeting
- `E3_GATINGONLY_r8_alpha16.yaml`: Gate-only adaptation
- Various DoRA, RSLoRA, PiSSA variants

**JSON PEFT Configurations** (`peft/`):
- `lora_qkvo_r8_a16.json`: Standard LoRA config
- `lora_QKVO_DoRA_r8_alpha16.json`: DoRA variant
- `lora_qkvo_rs_r8_alpha16.json`: RSLoRA variant

#### **Key LoRA Targeting Strategies**
```yaml
# QKVO: Query, Key, Value, Output projections
target_modules: ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"]

# Gating: Gate projections only
target_modules: ["attn.g_proj", "attn.gk_proj"]

# O+MLP: Output projection + MLP layers
target_modules: ["attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
```

### **8. FLA (Flash Linear Attention) Kernel Implementation**

#### **Core Location**
`3rdparty/flash-linear-attention/fla/`

#### **Key Components**

**Ops** (`fla/ops/gla/`):
- `chunk.py`: Chunk-wise GLA computation
- `fused_recurrent.py`: Fused recurrent GLA operations
- `naive.py`: Reference GLA implementation

**Models** (`fla/models/gla/`):
- `configuration_gla.py`: GLA model configuration
- `modeling_gla.py`: GLA model implementation with HF compatibility

**Layers** (`fla/layers/`):
- `gla.py`: GatedLinearAttention layer implementation
- Attention mechanisms with gating and feature maps

#### **GLA Architecture Highlights**
```python
class GatedLinearAttention(nn.Module):
    def __init__(self,
                 mode="chunk",           # chunk/fused_recurrent/naive
                 hidden_size=2048,
                 expand_k=0.5,           # Key expansion ratio
                 expand_v=1.0,           # Value expansion ratio
                 num_heads=4,
                 use_gk=True,            # Use gate for keys
                 use_gv=False,           # Use gate for values
                 feature_map=None,      # Feature map for attention
                 use_short_conv=False,   # Short convolution
                 conv_size=4,
                 use_output_gate=True,
                 gate_fn="swish"):
```

#### **GLA Computation Modes**

**1. Chunk Mode (`fla.ops.gla.chunk`)**
```python
def chunk_gla(
    q: torch.Tensor,           # Query: [B, H, T, D]
    k: torch.Tensor,           # Key: [B, H, T, D]
    v: torch.Tensor,           # Value: [B, H, T, D]
    gk: torch.Tensor,          # Key gate: [B, H, T, 1]
    gv: torch.Tensor,          # Value gate: [B, H, T, 1]
    chunk_size: int = 64       # Processing chunk size
) -> torch.Tensor:
    """
    Chunk-wise GLA computation for memory efficiency.
    Processes sequence in chunks to avoid O(T^2) memory usage.
    """
```

**2. Fused Recurrent Mode (`fla.ops.gla.fused_recurrent`)**
```python
def fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor
) -> torch.Tensor:
    """
    Fused CUDA operations for recurrent GLA computation.
    Optimized for GPU execution with custom kernels.
    """
```

**3. Naive Mode (`fla.ops.gla.naive`)**
```python
def naive_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor
) -> torch.Tensor:
    """
    Reference implementation for validation.
    Standard attention-style computation without optimizations.
    """
```

#### **GLA Model Architecture**

**GLABlock Structure** (`fla.models.gla.modeling_gla.py`):
```python
class GLABlock(nn.Module):
    def __init__(self, config: GLAConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = GatedLinearAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            num_heads=config.num_heads,
            use_gk=config.use_gk,
            use_gv=config.use_gv,
            # ... other config parameters
        )
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = GLAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attn_weights, past_kv = self.attn(
            hidden_states, attention_mask=attention_mask, **kwargs
        )

        # Residual connection + MLP
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, past_kv
```

**GLAForCausalLM Structure**:
```python
class GLAForCausalLM(PreTrainedModel):
    def __init__(self, config: GLAConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GLABlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight
```

#### **Key GLA Innovations**

**1. Gated Linear Attention Mechanism**:
- **Linear Complexity**: O(T) instead of O(T²) for attention
- **Gating**: Separate gates for keys (gk) and values (gv) control information flow
- **Feature Maps**: Optional feature transformations for improved expressivity

**2. Memory-Efficient Computation**:
- **Chunking**: Process long sequences in fixed-size chunks
- **Fused Operations**: Custom CUDA kernels for efficiency
- **Gradient Checkpointing**: Support for training very long sequences

**3. Hybrid Attention Support**:
- **Configurable Layers**: Mix GLA with standard attention in different layers
- **RoPE Integration**: Rotary position embeddings for positional awareness
- **Multi-Head Support**: Parallel attention heads with different configurations

#### **GLA Configuration Options**

```python
@dataclass
class GLAConfig(PretrainedConfig):
    # Architecture
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_heads: int = 4
    num_kv_heads: int = None

    # GLA-specific
    expand_k: float = 0.5      # Key expansion ratio
    expand_v: float = 1.0      # Value expansion ratio
    attn_mode: str = "chunk"   # chunk/fused_recurrent/naive
    use_gk: bool = True        # Gate keys
    use_gv: bool = False       # Gate values
    feature_map: str = None    # Feature map type

    # Efficiency
    fuse_norm: bool = True     # Fused RMSNorm
    fuse_swiglu: bool = True   # Fused SwiGLU activation
    use_short_conv: bool = False  # Short convolution for locality
    conv_size: int = 4         # Convolution kernel size

    # Hybrid attention
    attn: dict = None          # Layer-specific attention configs
```

#### **FLA Ecosystem Integration**

**Available GLA Variants**:
- **GLA**: Standard Gated Linear Attention
- **HGRN**: Hierarchical Gated Recurrent Network
- **Mamba**: State Space Model (different family)
- **RetNet**: Retention Network
- **DeltaNet**: Delta rule-based attention

**Benchmarking & Evaluation** (`fla/benchmarks/`):
- Performance benchmarks across different attention mechanisms
- Memory usage comparisons
- Speed benchmarks for different sequence lengths

**Testing Suite** (`fla/tests/`):
- Unit tests for individual operations
- Integration tests for full models
- Numerical correctness validation

### **9. Metrics & Evaluation**

#### **Spider SQL Evaluation**
`metrics/spider/`:
- `evaluation.py`: SQL execution-based evaluation
- `process_sql.py`: SQL processing utilities
- `spider.py`: SpiderMetric class with database execution

#### **GLUE Classification Metrics**
Built into `dataset/glue.py`:
- `_compute_glue_metrics_local()`: Offline-safe GLUE metrics
- Task-specific metrics (accuracy, F1, Matthews correlation)

### **10. Preprocessing Scripts**

Located in `scripts/preproc/`:
- `preproc_glue.py`: GLUE dataset preprocessing
- `preproc_dart.py`: DART data-to-text preprocessing
- `preproc_spider.py`: Spider SQL dataset preprocessing
- `preproc_samsum.py`: Summarization preprocessing

#### **Key Features**
- Parallel processing integration
- Cache management
- Subset sampling support

---

## **Part 3: Complete Call Sequence Analysis**

### **Detailed Call Sequence & Code Flow**

#### **Complete Training Pipeline Flow**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SHELL ENTRY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ gla_batch_tmux_clean.sh ──→ tmux session creation ──→ Sequential jobs      │
│   ├── Parse --suite E1 --round all --pairs "127:glue-tvt_rte"               │
│   ├── GPU allocation (GPU_IDS="0,1", GPU_PLAN="1,1")                       │
│   ├── Environment propagation (FORCE_SEED=127, DATA=glue-tvt_rte)          │
│   └── Launch: bash gla_round_clean.sh E1 all                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ROUND EXECUTOR                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ gla_round_clean.sh E1 all                                                  │
│   ├── Load ROUND_E1=(E1_QKVO_r8_alpha16.yaml ...)                          │
│   ├── Dynamic round slicing (N_SLOTS=2, N_ROUNDS=ceil(len/2))              │
│   ├── YAML injection: data=glue-tvt_rte, num_data_workers=8                │
│   ├── Parallel GPU launch:                                                 │
│   │   CUDA_VISIBLE_DEVICES=0 HP_SEED=127 python train_gla_only.py --cfg    │
│   │   CUDA_VISIBLE_DEVICES=1 HP_SEED=127 python train_gla_only.py --cfg    │
│   └── Monitor completion and aggregate logs                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PYTHON TRAINING ENTRY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ train_gla_only.py --cfg cfg/my_lora_exp/yaml/E1_QKVO_r8_alpha16.yaml       │
│   ├── YAML config loading and env overrides                               │
│   │   HP_SEED=127, HP_DATA=glue-tvt_rte, GLA_FORCE_LEFT_PAD=1             │
│   ├── Lock acquisition: share/lock/glue-tvt_rte_seed127_E1_QKVO_r8_alpha16│
│   ├── Checkpoint detection: checkpoint-* directories                       │
│   └── run_train() → build_and_run_trainer_gla_only()                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MODEL & TOKENIZER SETUP                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ prepare_gla_model_and_tokenizer()                                          │
│   ├── load_gla(model_id="fla-hub/gla-1.3B-100B")                          │
│   │   ├── GLAConfig.from_pretrained()                                     │
│   │   ├── GLAForCausalLM.from_pretrained(dtype=bf16, device=cuda)         │
│   │   └── Runtime patches: fuse_swiglu=False                             │
│   ├── PEFT LoRA loading: peft/lora_qkvo_r8_a16.json                       │
│   │   ├── LoraConfig(r=8, alpha=16, target_modules=[qkvo])               │
│   │   └── get_peft_model(model, peft_config)                             │
│   └── Left padding enforcement: tokenizer.padding_side = "left"           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATASET PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ load_dataset("glue-tvt_rte", tokenizer, "train")                          │
│   ├── Dataset factory dispatch: GlueDataModule                           │
│   │   ├── task="rte", name="rte", has_test_split=True                    │
│   │   └── prompts["rte"] = entailment classification prompt              │
│   ├── GlueDataset.__init__()                                             │
│   │   ├── Cache check: data/glue_rte_tvt/cache_rte_tvt_val.pkl           │
│   │   ├── Parallel preprocessing (16 workers)                            │
│   │   └── Materialize: input_ids, label_ids pairs                        │
│   └── DataCollator(tokenizer) with left padding                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINER SETUP                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ GenericLMTrainer initialization                                           │
│   ├── GenericLMTrainingArguments:                                        │
│   │   ├── learning_rate=5e-4, max_steps=total_steps                      │
│   │   ├── per_device_train_batch_size=4, gradient_accumulation_steps=1  │
│   │   ├── lr_scheduler_type="cosine", warmup_steps=0.1*total_steps      │
│   │   └── save_strategy="steps", eval_strategy="steps"                  │
│   ├── SwanLab integration (if SWANLAB_ENABLE=1)                          │
│   │   ├── SwanLabCallback(project="gla-peft", experiment_name=...)      │
│   │   └── EmailCallback for notifications                                │
│   └── Custom components:                                                 │
│       ├── train_crit = CrossEntropy()                                   │
│       ├── val_crits = [Accuracy()]                                      │
│       └── eval_generator = GLAHFDecoder(tokenizer)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ trainer.train(resume_from_checkpoint)                                    │
│   ├── Training step:                                                    │
│   │   ├── _forward(): model(input_ids) → lm_logits                      │
│   │   ├── compute_loss(): CrossEntropy(lm_logits, label_ids)           │
│   │   └── optimizer_step()                                              │
│   ├── Validation step:                                                  │
│   │   ├── prediction_step(): logits_valid extraction                    │
│   │   └── compute_metrics(): GLUE metrics computation                   │
│   ├── Checkpointing: save every eval_steps                              │
│   └── Early stopping: TrainLossEarlyStop, BadEvalEarlyStop              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EVALUATION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ evaluate_generation() or evaluate()                                      │
│   ├── For generation tasks (Spider, DART, SamSum):                      │
│   │   ├── GLAHFDecoder.__call__():                                       │
│   │   │   ├── model.generate(max_new_tokens=1024, do_sample=False)      │
│   │   │   └── Trim prompt: outputs.sequences[:, input_len:]             │
│   │   ├── EvalPredictionWithText:                                        │
│   │   │   ├── tokenizer.batch_decode() with skip_special_tokens         │
│   │   │   └── YAML caching: predictions-{step}.yaml                      │
│   │   └── compute_metrics(): task-specific metrics                       │
│   └── For classification tasks (GLUE):                                  │
│       ├── Standard HF evaluation                                        │
│       └── _compute_glue_metrics_local(): accuracy/f1/mcc                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        METRICS & RESULTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Task-specific metric computation                                         │
│   ├── SpiderMetric: SQL execution accuracy                              │
│   ├── GlueMetric: accuracy, f1, matthews_correlation                    │
│   ├── DartMetric: BLEU, ROUGE, METEOR                                   │
│   └── SwanLab logging: metrics, hyperparameters, system info            │
│ Email notifications (optional):                                         │
│   ├── STARTED: output_dir, data, seed, cfg_path                         │
│   ├── FINISHED: success confirmation                                    │
│   └── FAILED: traceback and error details                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### **Key Code Integration Points**

**1. GLA Model Loading (`mamba_ssm_peft/utils/hf.py`)**
```python
def load_gla(model_id, **kwargs):
    from fla.models.gla import GLAForCausalLM, GLAConfig
    config = GLAConfig.from_pretrained(model_id)
    # Disable incompatible fused operations
    config.fuse_swiglu = False
    model = GLAForCausalLM.from_pretrained(model_id, config=config, **kwargs)
    return {"model": model, "tokenizer": AutoTokenizer.from_pretrained(model_id)}
```

**2. LoRA Configuration Loading (`train_gla_adapter.py`)**
```python
def prepare_gla_model_and_tokenizer(model_id, peft_json_path, **kwargs):
    gla_loaded = load_gla(model_id, **kwargs)
    model, tokenizer = gla_loaded["model"], gla_loaded["tokenizer"]

    if peft_json_path:
        with open(peft_json_path, "r") as f:
            peft_json = json.load(f)
        # Apply environment overrides
        peft_json["r"] = int(os.environ.get("HP_PEFT_R", peft_json["r"]))
        peft_config = LoraConfig(**peft_json)
        model = get_peft_model(model, peft_config)

    return model, tokenizer, peft_config
```

**3. Dataset Processing (`dataset/base.py`)**
```python
class DatasetBase:
    def __init__(self, tokenizer, path, split, use_cache=True, **kwargs):
        self.tokenizer = tokenizer
        cache_file = Path("data") / f"{path}_{split}.pkl"

        if use_cache and cache_file.exists():
            with open(cache_file, "rb") as f:
                self.data = pickle.load(f)
        else:
            # Parallel preprocessing
            if num_parallel_workers > 0:
                self.data = ParallelProcessorFS(self.preproc, len(self), num_parallel_workers, cache_file).run()
```

**4. Trainer Integration (`trainer/generic_lm_trainer.py`)**
```python
class GenericLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, label_ids, lm_logits = self._forward(model, inputs)
        lm_loss = self.train_crit(lm_logits, label_ids)
        return lm_loss

    def evaluate_generation(self, generator, **kwargs):
        # Generation evaluation pipeline
        dataloader = self.get_eval_dataloader()
        for inputs in dataloader:
            pred_ids, label_ids = self.generation_step(generator, self.model, inputs)
            # Process predictions and compute metrics
```

**5. Decoder Implementation (`mamba_ssm_peft/utils/gla_hf_decoder.py`)**
```python
class GLAHFDecoder:
    def __call__(self, model, input_ids, attention_mask=None):
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_length,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "do_sample": self.do_sample,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        outputs = model.generate(**gen_kwargs)
        # Trim prompt from generated sequences
        outputs.sequences = outputs.sequences[:, input_ids.shape[1]:]
        return outputs
```

### **Key Integration Points**

#### **GLA ↔ HF Integration**
- FLA models inherit from `PreTrainedModel`
- Generation uses standard HF `model.generate()`
- Tokenization uses `AutoTokenizer`

#### **LoRA ↔ HF PEFT Integration**
- Standard PEFT `LoraConfig` and `get_peft_model()`
- Compatible with HF Trainer's PEFT support
- Environment-based parameter overrides

#### **Dataset ↔ Trainer Integration**
- Standard PyTorch Dataset interface
- HF DataCollator compatibility
- Custom metric computation hooks

---

## **Part 4: Debugging & Troubleshooting Guide**

### **Common Issues & Solutions**

#### **GLA Model Loading Issues**
```python
# Issue: fuse_swiglu compatibility
# Solution: Disable in config or apply runtime patch
config.fuse_swiglu = False
```

#### **LoRA Configuration Issues**
```python
# Issue: Target modules not found
# Solution: Verify GLA model structure
print(model)  # Inspect available modules
```

#### **Memory Issues**
```python
# Issue: OOM during generation
# Solution: Use logits_to_keep parameter
logits_to_keep=1  # Only keep last token logits
```

#### **Generation Issues**
```python
# Issue: Right padding causing issues
# Solution: Enforce left padding
GLA_FORCE_LEFT_PAD=1
```

### **Performance Optimization**

#### **Training Optimizations**
- Gradient checkpointing: `gradient_checkpointing=True`
- LoRA dropout tuning: `lora_dropout=0.05`
- Batch size optimization based on GPU memory

#### **Inference Optimizations**
- `use_cache=True` for generation
- Appropriate `max_new_tokens` limits
- Beam search for quality (`num_beams=4`)

---

## **Part 4.5: Parallel Preprocessing System Deep Dive**

### **Data Pipeline Architecture**

The GLA training pipeline uses a sophisticated parallel preprocessing system designed to handle large datasets efficiently across multiple workers while maintaining data integrity and caching consistency.

#### **Core Components**

**1. ParallelProcessorFS (`utils/parallel_processor_fs.py`)**
```python
class ParallelProcessorFS:
    def __init__(self, func, size, n_workers, output_file):
        """
        Distributed preprocessing coordinator.

        Args:
            func: Preprocessing function (e.g., DatasetBase.preproc)
            size: Total number of items to process
            n_workers: Number of parallel workers
            output_file: Path to final cached output file
        """
        self.func = func
        self.size = size
        self.n = n_workers
        self.output_file = Path(output_file)
        self.cache_path = self.output_file.parent / "parts"
        self.worker_files = [self.cache_path / f"{output_file.stem}_part_{i:03d}.pkl"
                           for i in range(n_workers)]
```

**Worker Process Logic**:
```python
def _worker(self, worker_idx, counter):
    """Individual worker process implementation."""
    out = {}

    while True:
        with counter.get_lock():
            idx = counter.value
            if idx >= self.size:
                break
            counter.value += 1

        # Process one item at a time
        try:
            result = self.func(idx)
            out[idx] = result
        except Exception as e:
            print(f"[Worker {worker_idx}] Error processing idx={idx}: {e}")
            out[idx] = None  # Mark as failed

    # Atomic write of worker results
    self._atomic_write_worker_results(worker_idx, out)
```

**Result Aggregation**:
```python
def aggregate_result(self):
    """Collect and merge results from all workers."""
    output_all = [None] * self.size

    # Load results from each worker
    for worker_file in self.worker_files:
        with open(worker_file, "rb") as f:
            worker_results = pickle.load(f)
        for idx, result in worker_results.items():
            output_all[idx] = result

    # Filter out failed items and create final cache
    valid_results = [r for r in output_all if r is not None]
    self._atomic_write_final_cache(valid_results)

    return valid_results
```

#### **DatasetBase Integration**

**Cache Management Strategy**:
```python
class DatasetBase:
    def __init__(self, use_cache=True, num_parallel_workers=16, **kwargs):
        if use_cache:
            cache_file = self._get_cache_file_path()

            if cache_file.exists() and not self._is_corrupted(cache_file):
                # Fast path: load existing cache
                with open(cache_file, "rb") as f:
                    self.data = pickle.load(f)
            else:
                # Slow path: parallel preprocessing
                if num_parallel_workers > 0:
                    self.data = ParallelProcessorFS(
                        self.preproc, len(self), num_parallel_workers, cache_file
                    ).run()
                else:
                    # Fallback: single-threaded processing
                    self.data = [self.preproc(idx) for idx in tqdm(range(len(self)))]
```

**Cache Integrity Checks**:
```python
def _validate_cache_integrity(self, cache_file):
    """Ensure cache file is not corrupted."""
    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return isinstance(data, list) and len(data) > 0
    except (pickle.UnpicklingError, EOFError, FileNotFoundError):
        return False
```

#### **Preprocessing Scripts**

**GLUE Preprocessing** (`scripts/preproc/preproc_glue.py`):
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="+")  # Dataset names
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--subset_size", type=int)

    for dataset_name in args.name:
        tokenizer = load_mamba_tokenizer()  # Note: Uses Mamba tokenizer for compatibility
        for split in ["train", "val", "test"]:
            dataset = GlueDataset(
                tokenizer=tokenizer,
                name=dataset_name,
                split=split,
                num_parallel_workers=args.workers,
                subset_size=args.subset_size,
                has_test_split=True
            )
            # Preprocessing happens during DatasetBase.__init__
```

**DART Preprocessing** (`scripts/preproc/preproc_dart.py`):
```python
# DART-specific preprocessing with error handling
try:
    dataset = DartDataModule(
        tokenizer=tokenizer,
        split=split,
        subset_size=args.subset_size
    )
except Exception as e:
    print(f"DART preprocessing failed: {e}")
    # Continue with other datasets
```

#### **Data Format Standardization**

**Unified Data Structure**:
All datasets produce standardized `(input_ids, label_ids)` tuples:

```python
# Classification datasets (GLUE, etc.)
input_ids = tokenizer.encode("Premise [SEP] Hypothesis")
label_ids = tokenizer.encode("0")  # Class label as token

# Generation datasets (DART, Spider, etc.)
input_ids = tokenizer.encode("Generate summary: [INPUT]")
label_ids = tokenizer.encode("[TARGET_SUMMARY]")
```

**Batch Collation** (`dataset/collator.py`):
```python
class DataCollator:
    def __call__(self, instances):
        # Extract tensors
        input_ids = [torch.tensor(inst["input_ids"]) for inst in instances]
        label_ids = [torch.tensor(inst["label_ids"]) for inst in instances]

        # Pad sequences
        input_ids = self._pad_sequences(input_ids, self.tokenizer.pad_token_id,
                                       self.padding_side)
        label_ids = self._pad_sequences(label_ids, -100, self.padding_side)

        # Generate attention masks
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "attention_mask": attention_mask
        }
```

#### **Performance Optimizations**

**1. Memory Management**:
- **Subset Sampling**: Process only subset of large datasets during development
- **Lazy Loading**: Materialize data only when needed
- **Worker Isolation**: Each worker process has independent memory space

**2. I/O Optimization**:
- **Atomic Writes**: Prevent partial/corrupted cache files
- **Compressed Storage**: Use pickle for efficient serialization
- **Incremental Processing**: Process items one at a time to control memory usage

**3. Error Handling**:
- **Graceful Degradation**: Continue processing when individual items fail
- **Detailed Logging**: Track which items failed and why
- **Recovery Mechanisms**: Ability to resume interrupted preprocessing

#### **Dataset-Specific Considerations**

**GLUE Datasets**:
- **Task-Specific Prompts**: Different prompts for each GLUE task
- **Label Tokenization**: Ensure class labels map to single tokens
- **Validation Splits**: Handle train/val/test split logic

**Generation Datasets**:
- **Sequence Length Limits**: Truncate overly long sequences
- **Special Token Handling**: Proper EOS/BOS token placement
- **Diverse Formats**: Handle different data formats (DART triples, Spider SQL, etc.)

**Cross-Task Consistency**:
- **Unified Interface**: All datasets implement the same DatasetBase interface
- **Standardized Metrics**: Consistent metric computation across tasks
- **Tokenizer Compatibility**: Ensure all datasets work with GLA tokenizer

---

## **Part 5: File Dependencies & Relationships**

### **Complete File Dependencies & Relationships**

#### **Core Architecture Map**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ENTRY POINTS                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│ scripts/train/new/                                                              │
│ ├── gla_batch_tmux_clean.sh          # Multi-job batch scheduler               │
│ ├── gla_round_clean.sh               # Single round executor                   │
│ └── gla_round_glaonly.sh             # Simplified single job launcher          │
│                                                                                 │
│ train_gla_only.py                      # Main Python training entry             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MODEL LOADING SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ train_gla_adapter.py                   # GLA + LoRA integration                 │
│ ├── prepare_gla_model_and_tokenizer()  # Main loading function                 │
│ └── mamba_ssm_peft/utils/hf.py         # FLA integration layer                  │
│     ├── load_gla()                     # FLA model loader                      │
│     ├── load_gla_tokenizer()           # FLA tokenizer loader                 │
│     └── FLA path resolution            # 3rdparty/flash-linear-attention       │
│                                                                                 │
│ 3rdparty/flash-linear-attention/fla/   # FLA kernel implementations             │
│ ├── models/gla/                        # GLA model definitions                 │
│ │   ├── configuration_gla.py          # GLAConfig class                       │
│ │   └── modeling_gla.py               # GLAForCausalLM, GLABlock               │
│ ├── layers/                            # Core attention layers                 │
│ │   ├── gla.py                        # GatedLinearAttention                  │
│ │   └── attn.py                       # General attention utilities            │
│ └── ops/gla/                           # CUDA kernels                          │
│     ├── chunk.py                      # Chunk-wise GLA                         │
│     ├── fused_recurrent.py            # Fused operations                      │
│     └── naive.py                      # Reference implementation               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DECODER SYSTEM                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│ mamba_ssm_peft/utils/gla_hf_decoder.py  # HF-compatible generation              │
│ └── GLAHFDecoder                        # Wraps model.generate()               │
│     ├── __call__()                      # Generation interface                 │
│     ├── Token limit handling           # max_new_tokens vs max_length          │
│     └── Attention mask validation      # Left padding enforcement              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TRAINER SYSTEM                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│ trainer/generic_lm_trainer.py          # HF Trainer extension                   │
│ ├── GenericLMTrainer                   # Main trainer class                     │
│ │   ├── compute_loss()                 # Cross-entropy loss                     │
│ │   ├── prediction_step()              # Validation step                       │
│ │   ├── evaluate_generation()          # Text generation eval                  │
│ │   └── generation_step()              # Decoder integration                   │
│ ├── GenericLMTrainingArguments         # HF args extension                     │
│ └── Custom components:                                                        │
│     ├── CrossEntropy loss              # Standard LM loss                       │
│     ├── Accuracy metric                # Classification accuracy               │
│     ├── Early stopping                 # TrainLossEarlyStop, BadEvalEarlyStop   │
│     └── SwanLab integration            # Logging and notifications             │
│                                                                                 │
│ trainer/trainer_utils.py               # Evaluation utilities                   │
│ ├── MambaEvalPrediction                # Prediction storage                     │
│ └── EvalPredictionWithText             # Text-based evaluation                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DATASET SYSTEM                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│ dataset/__init__.py                    # Dataset factory                        │
│ └── load_dataset()                     # String → DataModule dispatch           │
│                                                                                 │
│ dataset/base.py                        # Abstract base classes                  │
│ ├── DatasetBase                        # Core dataset functionality             │
│ │   ├── __init__()                     # Cache management                       │
│ │   ├── preproc()                      # Sample preprocessing                   │
│ │   ├── __getitem__()                  # PyTorch Dataset interface              │
│ │   └── compute_metrics()              # Abstract metric computation           │
│ ├── NluDatasetBase                     # Classification datasets                │
│ └── NlgDatasetBase                     # Generation datasets                    │
│                                                                                 │
│ dataset/collator.py                    # Data collation                          │
│ └── DataCollator                       # HF-compatible collation                │
│     ├── __call__()                     # Batch creation                         │
│     └── Attention mask generation      # Left/right padding support            │
│                                                                                 │
│ Specific dataset implementations:                                              │
│ ├── glue.py                            # GLUE benchmark datasets                │
│ ├── dart_data.py                       # DART data-to-text                      │
│ ├── spider_data.py                     # Spider text-to-SQL                     │
│ ├── samsum_data.py                     # SamSum summarization                   │
│ ├── alpaca.py                          # Alpaca instruction tuning             │
│ └── arc.py, boolq.py, piqa.py         # QA datasets                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       PREPROCESSING SYSTEM                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ utils/parallel_processor_fs.py        # Distributed preprocessing               │
│ ├── ParallelProcessorFS               # Multi-worker processing                 │
│ │   ├── __init__()                    # Worker setup                           │
│ │   ├── _worker()                     # Individual worker logic                │
│ │   └── aggregate_result()            # Result collection                      │
│ └── Atomic file operations           # Prevent cache corruption                │
│                                                                                 │
│ scripts/preproc/                      # Preprocessing scripts                   │
│ ├── preproc_glue.py                  # GLUE preprocessing                       │
│ ├── preproc_dart.py                  # DART preprocessing                       │
│ ├── preproc_spider.py                # Spider preprocessing                    │
│ └── preproc_samsum.py                # SamSum preprocessing                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION SYSTEM                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│ cfg/my_lora_exp/                      # LoRA experiment configurations          │
│ ├── yaml/                             # Experiment YAML configs                │
│ │   ├── E1_QKVO_r8_alpha16.yaml       # Standard QKVO LoRA                     │
│ │   ├── E2_OMLP_r8_alpha16.yaml       # Output MLP targeting                   │
│ │   ├── E3_GATINGONLY_r8_alpha16.yaml # Gate-only adaptation                   │
│ │   └── E*_DoRA_*.yaml                # DoRA variants                          │
│ └── peft/                             # PEFT JSON configurations               │
│     ├── lora_qkvo_r8_a16.json         # Standard LoRA config                   │
│     ├── lora_QKVO_DoRA_r8_alpha16.json # DoRA config                          │
│     └── lora_qkvo_rs_r8_alpha16.json  # RSLoRA config                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        METRICS SYSTEM                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│ metrics/spider/                        # Spider SQL evaluation                 │
│ ├── spider.py                         # SpiderMetric class                     │
│ ├── evaluation.py                     # SQL execution evaluation               │
│ └── process_sql.py                    # SQL processing utilities               │
│                                                                                 │
│ dataset/glue.py                       # GLUE metrics (built-in)                │
│ └── _compute_glue_metrics_local()     # Offline-safe GLUE metrics             │
│                                                                                 │
│ dataset/dart_data.py                  # DART metrics                            │
│ dataset/samsum_data.py                # SamSum metrics                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### **Data Flow Architecture**

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐
│  Raw HF     │ -> │  DatasetBase    │ -> │  Parallel       │ -> │   Cached    │
│  Dataset    │    │  .preproc()     │    │  Processing     │    │   Data      │
│             │    │  Tokenization   │    │  (16 workers)   │    │   (pickle)  │
└─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘
         │                   │                       │                   │
         ▼                   ▼                       ▼                   ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐
│  Input      │ -> │  DataCollator   │ -> │  GenericLM      │ -> │  Training   │
│  IDs +      │    │  .__call__()    │    │  Trainer        │    │  Loop       │
│  Label IDs  │    │  Left padding   │    │  .train()       │    │             │
└─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘
         │                   │                       │                   │
         ▼                   ▼                       ▼                   ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐
│  Model      │ -> │  GLAHFDecoder   │ -> │  Generation     │ -> │  Metrics    │
│  .generate()│    │  .__call__()    │    │  Evaluation     │    │  Computation│
│             │    │  Post-process   │    │  Pipeline       │    │             │
└─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘
```

#### **Configuration Flow**

```
YAML Experiment Config
        │
        ▼
Environment Overrides (HP_* variables)
        │
        ▼
PEFT JSON Loading
        │
        ▼
LoraConfig Construction
        │
        ▼
get_peft_model() Wrapping
        │
        ▼
GLA + LoRA Model Ready
```

#### **Key Integration Interfaces**

**1. HF Trainer Extension Points**
```python
class GenericLMTrainer(Trainer):
    # Override key methods for GLA compatibility
    def compute_loss(self, model, inputs, return_outputs=False):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    def evaluate_generation(self, generator, **kwargs):
```

**2. Dataset Interface**
```python
class DatasetBase(torch.utils.data.Dataset):
    def __len__(self):  # Dataset size
    def __getitem__(self, idx):  # PyTorch Dataset interface
    def preproc(self, idx):  # Preprocessing logic
    def compute_metrics(self, eval_preds):  # Metric computation
```

**3. Decoder Interface**
```python
class GLAHFDecoder:
    def __call__(self, model, input_ids, attention_mask=None):
        # Returns HF Generate outputs with trimmed prompts
```

**4. PEFT Integration**
```python
# Standard HF PEFT workflow
peft_config = LoraConfig(**json_config)
model = get_peft_model(base_model, peft_config)
```

---

## **Summary**

The clean GLA route provides a robust, Mamba-free implementation of GLA + LoRA training with:

1. **Complete HF ecosystem integration** (Trainer, PEFT, Tokenizers)
2. **Comprehensive experiment management** (YAML configs, batch scheduling)
3. **Scalable data processing** (parallel preprocessing, caching)
4. **Flexible evaluation** (generation + classification tasks)
5. **Production monitoring** (SwanLab, email notifications)

The debugging timeline demonstrates the importance of:
- Atomic file operations in parallel processing
- Comprehensive error handling and validation
- Environment variable propagation across distributed jobs
- Robust caching with integrity checks

This implementation serves as a solid foundation for GLA-based language model fine-tuning with LoRA adaptation.

---

## **Conclusion & Key Takeaways**

### **Clean GLA Route Achievements**

1. **Complete Mamba-Free Implementation**: Successfully decoupled GLA training from Mamba decoder dependencies, creating a pure GLA + LoRA + HF Trainer pipeline.

2. **Scalable Architecture**: Built a robust system that handles:
   - **Large-scale batch training** with tmux orchestration
   - **Parallel data preprocessing** with atomic caching
   - **Comprehensive experiment management** with YAML/JSON configurations
   - **Production monitoring** with SwanLab integration

3. **Comprehensive Task Coverage**: Support for diverse NLP tasks:
   - **Classification**: GLUE benchmark (RTE, MRPC, CoLA, SST-2, QNLI, QQP, MNLI)
   - **Generation**: DART (data-to-text), Spider (text-to-SQL), SamSum (summarization)
   - **QA Tasks**: ARC, BoolQ, PIQA, MMLU

4. **Robust Debugging Framework**: Resolved critical issues in:
   - **Dataset integrity** with proper caching and validation
   - **Parallel processing** with atomic file operations
   - **Memory management** for large-scale preprocessing
   - **Environment propagation** across distributed jobs

### **Technical Innovations**

1. **FLA Integration**: Seamless integration of Flash Linear Attention kernels with HF ecosystem
2. **LoRA Ecosystem**: Comprehensive support for multiple PEFT methods (LoRA, DoRA, RSLoRA, PiSSA)
3. **Generation Pipeline**: HF-native text generation with proper prompt trimming
4. **Evaluation Framework**: Unified metrics computation across all task types

### **Production Readiness**

The clean GLA route provides:
- **Reproducible experiments** with configuration versioning
- **Fault tolerance** with comprehensive error handling
- **Monitoring & alerting** with email notifications
- **Scalable deployment** with tmux-based job management

### **Future Extensions**

The architecture supports easy extension to:
- **New attention mechanisms** via FLA ecosystem
- **Additional PEFT methods** through HF PEFT
- **Custom datasets** via DatasetBase interface
- **Advanced training techniques** (curriculum learning, multi-task, etc.)

This implementation serves as a solid foundation for state-of-the-art GLA-based language model fine-tuning with efficient parameter adaptation.
