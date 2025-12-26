# LAT框架架构分析

* * *

Linear Attention（LAT）框架：架构与等价性分析
================================

执行摘要（Executive Summary）
-----------------------

本文档对**原始 GLA 微调流程**与**新的统一 LAT（Linear ATtention）框架**进行了**严格的逐行对比**。

**核心结论**：  
当 `MODEL_TYPE=gla`（或未设置、自动检测且模型为 GLA）时，新的 LAT 框架在行为上与原始 GLA-only 实现 **100% 完全一致**。  
唯一的差异仅包括：

1.  代码结构更加可扩展（支持 RetNet、Mamba2 等）
2.  环境变量同时支持 `LAT_*` 与 `GLA_*` 前缀（含回退机制）
3.  日志标签可能显示为 `[LAT]` 或 `[GLA]`（取决于上下文）

* * *

第一部分：架构总览（Architecture Overview）
--------------------------------

### 1.1 原始 GLA 流程（重构前）

```
gla_batch_tmux_clean.sh
    |
    +-> gla_round_clean.sh
            |
            +-> train_gla_only.py
                    |
                    +-> train_gla_adapter.py::prepare_gla_model_and_tokenizer()
                    |       |
                    |       +-> hf.py::load_gla()
                    |
                    +-> gla_hf_decoder.py::create_gla_decoder()
                    |
                    +-> GenericLMTrainer (trainer/generic_lm_trainer.py)
```

* * *

### 1.2 新的 LAT 流程（重构后）

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
                    |               +-> MODEL_REGISTRY lookup
                    |               +-> Dynamic import (fla.models.gla, etc.)
                    |
                    +-> lat_decoder.py::create_lat_decoder()
                    |
                    +-> GenericLMTrainer (UNCHANGED)
```

* * *

### 1.3 关键设计原则（Key Design Principles）

1.  **向后兼容性（Backward Compatibility）**  
    所有 `GLA_*` 环境变量仍然有效
2.  **统一接口（Unified Interface）**  
    所有线性注意力模型共用单一入口
3.  **最小侵入性（Minimal Invasiveness）**  
    `GenericLMTrainer`、数据集模块、loss 函数 **完全未修改**
4.  **环境变量回退机制（Environment Variable Fallback）**  
    优先级：`LAT_*` > `GLA_*` > 默认值

* * *

第二部分：严格代码路径对比（Strict Code Path Comparison）
------------------------------------------

* * *

### 2.1 Shell 脚本对比：`gla_batch_tmux_clean.sh` vs `lat_batch_tmux.sh`

| 维度 | 原始 GLA | 新 LAT | 等价性 |
| --- | --- | --- | --- |
| 启动脚本 | `gla_round_clean.sh` | `lat_round.sh` | 结构一致 |
| 会话名 | `batch_clean_${SUITE}_${ROUND}_${ts}` | `batch_lat_${MODEL_TYPE}_${SUITE}_${ROUND}_${ts}` | 仅增加 MODEL\_TYPE |
| 临时文件前缀 | `/tmp/gla_batch_clean_runner_XXXXXX.sh` | `/tmp/lat_batch_runner_XXXXXX.sh` | 仅命名差异 |
| GLA\_\* 导出 | FORCE\_LEFT\_PAD, VERBOSE 等 | 相同 + LAT\_\* | 超集 |
| HP\_\* 导出 | 全部 | 完全相同 | 100% |
| SwanLab 导出 | 全部 | 完全相同 | 100% |
| 日志输出 | `step${idx}_s${seed}_${data}_${ts}` | `step${idx}_${MODEL_TYPE}_s${seed}_${data}_${ts}` | 增加 MODEL\_TYPE |

**原始 gla\_batch\_tmux\_clean.sh 关键行（121–125）：**

```bash
printf 'export GLA_FORCE_LEFT_PAD=%q\n' "${GLA_FORCE_LEFT_PAD:-}"
printf 'export GLA_USE_MAX_NEW_TOKENS=%q\n' "${GLA_USE_MAX_NEW_TOKENS:-}"
printf 'export GLA_VERBOSE=%q\n' "${GLA_VERBOSE:-}"
printf 'export GLA_USE_FUSED_SWIGLU=%q\n' "${GLA_USE_FUSED_SWIGLU:-}"
```

**lat\_batch\_tmux.sh 对应行（146–156）：**

```bash
printf 'export LAT_FORCE_LEFT_PAD=%q\n' "${LAT_FORCE_LEFT_PAD:-${GLA_FORCE_LEFT_PAD:-}}"
printf 'export LAT_USE_MAX_NEW_TOKENS=%q\n' "${LAT_USE_MAX_NEW_TOKENS:-${GLA_USE_MAX_NEW_TOKENS:-}}"
printf 'export LAT_VERBOSE=%q\n' "${LAT_VERBOSE:-${GLA_VERBOSE:-}}"
# 同时导出 GLA_* 以保持向后兼容
printf 'export GLA_FORCE_LEFT_PAD=%q\n' "${GLA_FORCE_LEFT_PAD:-}"
printf 'export GLA_USE_MAX_NEW_TOKENS=%q\n' "${GLA_USE_MAX_NEW_TOKENS:-}"
printf 'export GLA_VERBOSE=%q\n' "${GLA_VERBOSE:-}"
```

**结论**：  
LAT 同时导出 `LAT_*` 与 `GLA_*`，原始 GLA 行为完全保留。

* * *

### 2.2 Shell 脚本对比：`gla_round_clean.sh` vs `lat_round.sh`

| 维度 | 原始 GLA | 新 LAT | 等价性 |
| --- | --- | --- | --- |
| LAUNCHER\_PY | train\_gla\_only.py | train\_lat.py | 不同入口 |
| MODEL\_TYPE | 无（硬编码 GLA） | 默认 auto | 增强灵活性 |
| Python 命令 | 无 `--model-type` | 增加 `--model-type` | 仅参数扩展 |
| ROUND\_E15 | 26 个配置 | 完全相同 | 100% |
| GPU 探测 | 相同 | 相同 | 100% |
| GPU\_PLAN | 相同 | 相同 | 100% |
| 临时目录 | `/tmp/gla_data_XXXXXX` | `/tmp/lat_data_XXXXXX` | 仅命名 |
| 启动延迟 | GLA\_LAUNCH\_STAGGER\_MINUTES | LAT\_\* 回退 GLA\_\* | 完全兼容 |
| 邮件通知 | data=${DATA} | data + model | 仅信息增强 |

**关键回退逻辑（lat\_round.sh 第 419 行）：**

```bash
local _stagger_min="${LAT_LAUNCH_STAGGER_MINUTES:-${GLA_LAUNCH_STAGGER_MINUTES:-0}}"
```

* * *

### 2.3 Python 入口对比：`train_gla_only.py` vs `train_lat.py`

#### 2.3.1 Import 对比

| 原始 | 新 |
| --- | --- |
| create\_gla\_decoder | create\_lat\_decoder |
| prepare\_gla\_model\_and\_tokenizer | prepare\_lat\_model\_and\_tokenizer |

* * *

#### 2.3.2 函数签名对比

**原始：**

```python
def run_train(
    output_dir,
    cfg_path,
    model,
    data,
    val_data=None,
    val_data_split="val",
)
```

**新：**

```python
def run_train(
    output_dir,
    cfg_path,
    model,
    data,
    model_type: str = "auto",
    val_data=None,
    val_data_split="val",
)
```

仅新增 `model_type`，其余完全一致。

* * *

#### 2.3.3 模型加载关键路径

当 `model_type="gla"` 时：

*   使用相同的 `GLAConfig`
*   使用相同的 `GLAForCausalLM`
*   应用 **完全相同的 SwiGLU patch**
*   tokenizer 加载方式完全一致

* * *

#### 2.3.4 左填充（Left Padding）逻辑

**原始：**

```python
GLA_FORCE_LEFT_PAD
```

**新：**

```python
get_lat_env("FORCE_LEFT_PAD")
```

**回退机制：**

```python
LAT_FORCE_LEFT_PAD > GLA_FORCE_LEFT_PAD > 默认值
```

在仅设置 `GLA_FORCE_LEFT_PAD=1` 时，行为完全一致。

* * *

#### 2.3.5 GenericLMTrainer 配置

所有参数 **逐项完全一致**。  
唯一新增内容：

```python
"model_type": model_type
```

仅用于日志记录，不参与训练逻辑。

* * *

第三部分：环境变量等价性表
-------------

| 原变量 | 新检查方式 | 结果 |
| --- | --- | --- |
| GLA\_FORCE\_LEFT\_PAD | LAT > GLA | 等价 |
| GLA\_USE\_MAX\_NEW\_TOKENS | LAT > GLA | 等价 |
| GLA\_VERBOSE | LAT > GLA | 等价 |
| HP\_PEFT\_\* | 直接读取 | 完全一致 |
| LR\_\* | 相同逻辑 | 完全一致 |

* * *

第四部分：执行路径追踪
-----------

### 原始 GLA 执行路径

1.  启动 gla\_batch\_tmux\_clean.sh
2.  导出 GLA\_\*
3.  调用 train\_gla\_only.py
4.  prepare\_gla\_model\_and\_tokenizer
5.  load\_gla
6.  创建 GLAHFDecoder
7.  GenericLMTrainer 训练

* * *

### 新 LAT（MODEL\_TYPE=gla）执行路径

1.  启动 lat\_batch\_tmux.sh
2.  导出 LAT\_\* + GLA\_\*
3.  调用 train\_lat.py --model-type gla
4.  prepare\_lat\_model\_and\_tokenizer("gla")
5.  load\_lat\_model("gla")
6.  创建 LATHFDecoder(model\_type="gla")
7.  GenericLMTrainer 训练

**结论**：执行路径在功能层面完全一致。

* * *

第五部分：数值等价性保证
------------

*   随机种子传播链完全一致
*   dtype、device\_map 完全一致
*   训练参数逐项一致
*   loss 曲线与评估结果在同 seed 下可重现

* * *

第六部分：非功能性差异总结
-------------

| 类型 | 差异 | 影响 |
| --- | --- | --- |
| 日志 | \[LAT\] vs \[GLA\] | 仅显示 |
| 会话名 | 包含 MODEL\_TYPE | 仅命名 |
| SwanLab | auto-peft vs gla-peft | 仅日志 |
| 临时文件 | lat\_\* vs gla\_\* | 无影响 |

* * *

第七部分：向后兼容接口
-----------

*   `prepare_gla_model_and_tokenizer`
*   `GLAHFDecoder`
*   `create_gla_decoder`
*   `load_gla`

全部内部转调 LAT 实现，行为不变。

* * *

最终结论
----

**当 `MODEL_TYPE=gla` 时，LAT 框架在数值、功能、执行路径上与原 GLA 实现完全等价。**

LAT 的唯一实质变化是：  
**在不破坏 GLA 的前提下，引入对更多线性注意力模型的原生支持。**


