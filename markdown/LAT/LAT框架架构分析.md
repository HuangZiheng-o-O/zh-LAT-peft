Linear Attention（LAT）框架：架构与设计
================================

本文档描述了**统一 LAT（Linear ATtention）框架**的架构设计，这是一个支持多种线性注意力模型（GLA、RetNet、Mamba2等）的可插拔训练框架。

**核心设计原则**：

1.  **单一入口点**：所有线性注意力模型共用 `train_lat.py` 入口
2.  **注册表模式**：通过 `ModelRegistry` 管理模型类型，易于扩展
3.  **统一环境变量**：使用 `LAT_*` 前缀，自动回退到 `GLA_*` 以保持向后兼容
4.  **策略模式**：SwiGLU patch 等运行时修改采用策略模式

* * *

第一部分：架构总览（Architecture Overview）
--------------------------------

### 1.1 框架流程

```
lat_batch_tmux.sh (with MODEL_TYPE env)
    |
    +-> lat_round.sh
            |
            +-> train_lat.py --model-type <gla|retnet|mamba2|auto>
                    |
                    +-> lat_adapter.py::prepare_lat_model_and_tokenizer()
                    |       |
                    |       +-> lat_model_loader.py::load_lat_model()
                    |               |
                    |               +-> ModelRegistry lookup (lat_base.py)
                    |               +-> Dynamic import (fla.models.*)
                    |               +-> patches.py::apply_model_patches()
                    |
                    +-> lat_decoder.py::create_lat_decoder()
                    |
                    +-> GenericLMTrainer (trainer/generic_lm_trainer.py)
```

* * *

### 1.2 核心模块结构

```
mamba-peft/
├── train_lat.py                         # 统一训练入口
├── lat_adapter.py                       # 模型适配器
│
├── mamba_ssm_peft/
│   └── utils/
│       ├── env_config.py                # 统一环境变量配置 (NEW)
│       ├── lat_base.py                  # 类型定义与 ModelRegistry (NEW)
│       ├── patches.py                   # SwiGLU patch 策略 (NEW)
│       ├── lat_model_loader.py          # 统一模型加载器 (REFACTORED)
│       ├── lat_decoder.py               # 统一解码器
│       └── hf.py                        # HuggingFace 工具 (SIMPLIFIED)
│
├── trainer/
│   └── generic_lm_trainer.py            # 通用训练器 (UPDATED)
│
└── scripts/train/new/
    ├── lat_batch_tmux.sh                # 批量训练脚本 (SIMPLIFIED)
    └── lat_round.sh                     # 轮次训练脚本 (SIMPLIFIED)
```

* * *

### 1.3 关键设计原则

1.  **统一接口**：所有线性注意力模型共用单一入口
2.  **注册表模式**：`ModelRegistry` 管理所有支持的模型类型
3.  **策略模式**：`patches.py` 中的 `PatchManager` 管理运行时 patch
4.  **环境变量统一**：`env_config.py` 提供单一真相源，优先级 `LAT_*` > `GLA_*`

* * *

第二部分：核心组件详解
-----------

### 2.1 ModelRegistry（模型注册表）

```python
from mamba_ssm_peft.utils.lat_base import ModelRegistry, ModelCapabilities

# 获取模型规格
spec = ModelRegistry.get("gla")

# 检查能力
if spec.capabilities.has_fuse_swiglu:
    apply_swiglu_patch()

# 动态导入模型类
ConfigClass, ModelClass = spec.import_classes()

# 列出支持的模型
models = ModelRegistry.list_models()  # ["gla", "retnet", "mamba2"]
```

### 2.2 LATEnvConfig（统一环境变量）

```python
from mamba_ssm_peft.utils.env_config import env_config

# 获取环境变量（自动回退 LAT_* > GLA_*）
verbose = env_config.get_bool("VERBOSE")  # 检查 LAT_VERBOSE 或 GLA_VERBOSE
force_left_pad = env_config.get_bool("FORCE_LEFT_PAD")
stagger_min = env_config.get_int("LAUNCH_STAGGER_MINUTES", default=0)
```

### 2.3 PatchManager（运行时 Patch）

```python
from mamba_ssm_peft.utils.patches import apply_model_patches

# 根据模型类型应用 patch
apply_model_patches(model_type="gla", config=config)
```

* * *

第三部分：环境变量参考
-----------

所有环境变量使用 `LAT_*` 前缀，自动回退到 `GLA_*` 以保持向后兼容。

| 环境变量 | 默认值 | 描述 |
| --- | --- | --- |
| `LAT_FORCE_LEFT_PAD` | `1` | 强制左填充 |
| `LAT_USE_MAX_NEW_TOKENS` | `1` | 使用 max_new_tokens 语义 |
| `LAT_VERBOSE` | `0` | 详细日志 |
| `LAT_USE_FUSED_SWIGLU` | `0` | 启用融合 SwiGLU（默认禁用） |
| `LAT_LOG_PADDING_STATS` | `0` | 记录填充统计 |
| `LAT_LAUNCH_STAGGER_MINUTES` | `0` | 启动延迟分钟数 |

* * *

第四部分：支持的模型类型
-----------

| 模型类型 | 描述 | 论文 |
| --- | --- | --- |
| `gla` | Gated Linear Attention | [arXiv:2312.06635](https://arxiv.org/abs/2312.06635) |
| `retnet` | Retentive Network | [arXiv:2307.08621](https://arxiv.org/abs/2307.08621) |
| `mamba2` | Mamba2 State Space Model | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |

* * *

第五部分：使用示例
---------

### 5.1 命令行使用

```bash
# GLA 训练（默认）
./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_cola"

# RetNet 训练
./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_cola" --model-type retnet

# Mamba2 训练
MODEL_TYPE=mamba2 ./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_cola"

# 自动检测模型类型
./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_cola" --model-type auto
```

### 5.2 Python API 使用

```python
from mamba_ssm_peft.utils.lat_model_loader import load_lat_model

# 加载 GLA 模型
result = load_lat_model("gla", "fla-hub/gla-1.3B-100B")
model = result["model"]
tokenizer = result["tokenizer"]

# 自动检测模型类型
result = load_lat_model("auto", "fla-hub/gla-1.3B-100B")
```

* * *

第六部分：扩展新模型
---------

要添加新的线性注意力模型，只需在 `lat_base.py` 中注册：

```python
from mamba_ssm_peft.utils.lat_base import ModelRegistry, ModelSpec, ModelCapabilities

# 注册新模型
ModelRegistry.register(ModelSpec(
    model_type="rwkv",
    module_path="fla.models.rwkv",
    config_class_name="RWKVConfig",
    model_class_name="RWKVForCausalLM",
    capabilities=ModelCapabilities(
        has_fuse_swiglu=True,
        cache_type="past_key_values",
        inner_model_attr="model",
    ),
))

# 更新 CONFIG_MODEL_TYPE_MAP
CONFIG_MODEL_TYPE_MAP["rwkv"] = "rwkv"
```

* * *

第七部分：已删除的遗留文件
-------------

以下文件已被删除，由新的统一实现替代：

| 删除的文件 | 替代方案 |
| --- | --- |
| `train_gla_only.py` | `train_lat.py` |
| `train_gla_adapter.py` | `lat_adapter.py` |
| `gla_hf_decoder.py` | `lat_decoder.py` |
| `gla_batch_tmux_clean.sh` | `lat_batch_tmux.sh` |
| `gla_round_clean.sh` | `lat_round.sh` |

* * *

最终结论
----

LAT 框架通过以下设计模式实现了可扩展的线性注意力训练框架：

1. **注册表模式**：`ModelRegistry` 管理所有模型类型
2. **策略模式**：`patches.py` 管理运行时 patch
3. **依赖注入**：`env_config.py` 提供统一的配置访问
4. **工厂模式**：`load_lat_model()` 和 `create_lat_decoder()` 提供统一接口

新增模型只需在 `ModelRegistry` 中注册，无需修改核心训练逻辑。
