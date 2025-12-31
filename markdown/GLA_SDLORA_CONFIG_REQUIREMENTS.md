# GLA SD-LoRA 配置管理与验证

## 改进总结

已将GLA SD-LoRA从**硬编码默认值**改进为**配置文件驱动**，并添加了**严格的assert验证**。这确保了：

1. ✓ 配置错误会立即报错，而不是隐含使用默认值
2. ✓ 所有必需的配置都必须在YAML/JSON中显式指定
3. ✓ 配置值的合法性会自动检查
4. ✓ 错误信息清晰，易于排查问题

---

## 配置要求

### 必需字段

GLA SD-LoRA需要在配置文件中显式指定以下字段：

#### 1. **target_modules** (SDT目标)
```json
"target_modules": ["gk_proj.1"]
```
- **类型**：List[str]
- **含义**：哪些模块应用Sparse Dimension Tuning
- **对GLA**：通常只有`gk_proj.1`（α_t的直接输出层）
- **为什么必需**：不同模型的动力学参数位置不同

#### 2. **lora_targets** (LoRA目标)
```json
"lora_targets": ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"]
```
- **类型**：List[str]
- **含义**：哪些线性投影层应用LoRA
- **对GLA**：所有投影层（Q/K/V/G/O）
- **为什么必需**：不同模型的投影层名字和数量不同

#### 3. **num_zero** (零化维度)
```json
"num_zero": {"channel": 0.1}
```
- **类型**：Dict[str, float|int]
- **含义**：多少比例（或数量）的维度被Zero（置为-100导致快速衰减）
- **范围**：0.0-1.0（比例）或正整数（绝对数量）
- **为什么必需**：SDT的关键超参数，影响剪枝程度

#### 4. **num_freeze** (冻结维度)
```json
"num_freeze": {"channel": 0.5}
```
- **类型**：Dict[str, float|int]
- **含义**：多少比例（或数量）的维度被Freeze（保持预训练权重）
- **范围**：0.0-1.0（比例）或正整数（绝对数量）
- **为什么必需**：SDT的关键超参数，影响知识保留程度

### 自动验证

代码在`__post_init__`中自动进行以下验证：

```python
# 1. 字段非空检查
assert self.target_modules is not None
assert self.lora_targets is not None
assert self.num_zero is not None
assert self.num_freeze is not None

# 2. 类型和结构检查
assert isinstance(self.target_modules, list) and len(self.target_modules) > 0
assert isinstance(self.lora_targets, list) and len(self.lora_targets) > 0
assert isinstance(self.num_zero, dict) and "channel" in self.num_zero
assert isinstance(self.num_freeze, dict) and "channel" in self.num_freeze

# 3. 逻辑一致性检查
total_ratio = self.num_zero["channel"] + self.num_freeze["channel"]
assert total_ratio <= 1.0  # Train = 1 - total_ratio
```

---

## 配置文件示例

### default.json（推荐配置）
```json
{
    "peft_type": "GLA_SD_LORA",
    "select_mode": "CHANNELS_ONLY",
    "proj_lora_r": 8,
    "num_zero": {
        "channel": 0.1
    },
    "num_freeze": {
        "channel": 0.5
    },
    "num_warmup_it": 100,
    "target_modules": ["gk_proj.1"],
    "lora_targets": ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"],
    "finetune_parameters": null,
    "sdlora_alpha": {
        "global": 1.0,
        "gk_proj.1": 1.0
    },
    "_comment": "Default GLA SD-LoRA: Train=40%, Freeze=50%, Zero=10%. All linear projections use LoRA."
}
```

**配置说明：**
- Train维度 = 1 - 0.1 - 0.5 = **40%**（可训练）
- Freeze维度 = **50%**（保留预训练权重）
- Zero维度 = **10%**（剪枝，快速衰减）

### aggressive.json（激进配置）
```json
{
    "peft_type": "GLA_SD_LORA",
    "select_mode": "CHANNELS_ONLY",
    "proj_lora_r": 8,
    "num_zero": {
        "channel": 0.4
    },
    "num_freeze": {
        "channel": 0.4
    },
    "num_warmup_it": 150,
    "target_modules": ["gk_proj.1"],
    "lora_targets": ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"],
    "finetune_parameters": null,
    "sdlora_alpha": {
        "global": 1.0,
        "gk_proj.1": 1.0
    },
    "_comment": "Aggressive GLA SD-LoRA: Train=20%, Freeze=40%, Zero=40%. All linear projections use LoRA."
}
```

**配置说明：**
- Train维度 = 1 - 0.4 - 0.4 = **20%**（仅微调关键维度）
- Freeze维度 = **40%**（保留更多预训练知识）
- Zero维度 = **40%**（激进剪枝）
- **适用场景**：参数预算极紧、任务与预训练任务相似

---

## 错误处理示例

### 错误1：缺少target_modules
```python
# 配置文件中没有target_modules
AssertionError: target_modules is required for GLA SD-LoRA.
Must be specified in config file or at initialization.
Example: target_modules=['gk_proj.1']
```

**修复**：在config.json中添加：
```json
"target_modules": ["gk_proj.1"]
```

### 错误2：缺少g_proj
```python
# 旧配置（已修复）
"lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**症状**：output gate不被微调，可能影响性能

**修复**：在配置中添加`"g_proj"`：
```json
"lora_targets": ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"]
```

### 错误3：Train比例为负
```python
# 配置中
"num_zero": {"channel": 0.6},
"num_freeze": {"channel": 0.5}
```

**出错**：
```
AssertionError: num_zero + num_freeze must be <= 1.0, got 1.1.
This leaves Train ratio = -0.1
```

**修复**：调整使得 zero + freeze <= 1.0
```json
"num_zero": {"channel": 0.4},
"num_freeze": {"channel": 0.5}
```

---

## 配置使用流程

### 1. 加载配置文件
```python
from peft import PeftConfig

# 从default.json加载
config = PeftConfig.from_pretrained("configs/gla_sdlora/default.json")
# ← 会自动执行__post_init__()中的所有assert检查
```

### 2. 或从dict初始化
```python
from mamba_ssm_peft.peft import GlaSdLoraConfig

config_dict = {
    "peft_type": "GLA_SD_LORA",
    "target_modules": ["gk_proj.1"],
    "lora_targets": ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"],
    "num_zero": {"channel": 0.1},
    "num_freeze": {"channel": 0.5},
    "num_warmup_it": 100,
    "proj_lora_r": 8,
}

config = GlaSdLoraConfig(**config_dict)
# ← 同样会执行assert检查，如缺少必需字段会立即失败
```

### 3. 使用配置初始化模型
```python
from peft import get_peft_model

# 必须提供完整的有效配置
peft_model = get_peft_model(model, config)
```

---

## 最佳实践

### ✓ 推荐做法

1. **所有配置都应在文件中显式声明**
   ```json
   {
       "target_modules": ["gk_proj.1"],
       "lora_targets": [...],
       "num_zero": {...},
       "num_freeze": {...}
   }
   ```

2. **使用配置文件而不是代码默认值**
   ```python
   # ✓ 推荐
   config = PeftConfig.from_pretrained("path/to/config.json")

   # ✗ 不推荐
   config = GlaSdLoraConfig()  # 会失败，因为没有默认值
   ```

3. **添加注释说明配置含义**
   ```json
   {
       "_comment": "Default: Train=40%, Freeze=50%, Zero=10%",
       ...
   }
   ```

4. **版本控制配置文件**
   ```
   configs/
   ├── gla_sdlora/
   │   ├── default.json           # 标准配置
   │   ├── aggressive.json        # 激进剪枝
   │   ├── conservative.json      # 保守配置
   │   └── task_specific/
   │       ├── glue.json
   │       └── code_gen.json
   ```

### ✗ 避免做法

1. **在代码中硬编码默认值**
   ```python
   # ✗ 坏做法（已移除）
   if self.target_modules is None:
       self.target_modules = ["gk_proj.1"]
   ```

2. **允许配置部分缺失**
   ```python
   # ✗ 坏做法（已改进）
   if self.lora_targets is None:
       self.lora_targets = [...]  # 隐性行为
   ```

3. **不验证配置的一致性**
   ```python
   # ✗ 坏做法（已改进）
   # 不检查 zero + freeze <= 1.0
   ```

---

## 配置文件总结表

| 字段 | 类型 | 必需 | 示例 | 含义 |
|------|------|------|------|------|
| peft_type | str | ✓ | "GLA_SD_LORA" | PEFT类型 |
| target_modules | List[str] | ✓ | ["gk_proj.1"] | SDT目标 |
| lora_targets | List[str] | ✓ | ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"] | LoRA目标 |
| num_zero | Dict | ✓ | {"channel": 0.1} | 零化比例 |
| num_freeze | Dict | ✓ | {"channel": 0.5} | 冻结比例 |
| select_mode | str | ✗ | "CHANNELS_ONLY" | 选择模式 |
| proj_lora_r | int | ✗ | 8 | LoRA秩 |
| num_warmup_it | int | ✗ | 100 | 预热迭代数 |
| sdlora_alpha | Dict | ✗ | {"global": 1.0} | 适应缩放因子 |
| finetune_parameters | List[str] | ✗ | null | 额外微调参数 |
| _comment | str | ✗ | "Train=40%" | 注释说明 |

---

## 总结

**配置驱动的PEFT框架的优势：**

1. **显式化**：所有配置都明确在文件中，无隐式默认值
2. **可验证**：自动检查配置的完整性和一致性
3. **可追溯**：配置更改可在版本控制中跟踪
4. **易扩展**：添加新配置预设无需修改代码
5. **错误早发现**：配置错误立即报错，而不是在训练时才暴露

