# GLA PEFT实施指南：代码位置与参数配置

本文档详细说明如何在实际代码中实施"投影层用LoRA，动力学层用SDT"的分工策略。

---

## 第一部分：代码文件地图

### 核心文件结构

```
zh-LAT-peft/
├── mamba-peft/
│   ├── lat_adapter.py                    [★ 关键：配置处理]
│   ├── train_lat.py                      [入口点，环境变量文档]
│   ├── mamba_ssm_peft/peft/
│   │   ├── gla_sd_lora.py               [★ 关键：SDT实现]
│   │   ├── sd_lora.py                   [Mamba SD-LoRA，参考实现]
│   │   └── gla_base_tuner.py            [GLA基础调优器]
│   ├── configs/gla_sdlora/
│   │   └── default.json                 [配置模板]
│   ├── scripts/train/new/
│   │   ├── lat_batch_tmux.sh            [★ 关键：环境变量导出]
│   │   └── lat_round.sh                 [调试echo列表]
│   └── 3rdparty/flash-linear-attention/fla/
│       └── layers/gla.py               [★ GLA模型代码]
```

---

## 第二部分：GLA模型代码分析

### 2.1 GLA层参数详解

**文件**: `3rdparty/flash-linear-attention/fla/layers/gla.py` (第24-299行)

#### 初始化阶段 (第71-171行)

```python
class GatedLinearAttention(nn.Module):
    def __init__(self, ...):
        # ========== 线性投影层 ==========
        # [适合LoRA]

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        # 作用：x_t → q_t (查询)
        # 维度：hidden_size → key_dim (默认: 1024 → 512)
        # LoRA秩建议：r=8

        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        # 作用：x_t → k_t (键)
        # 维度：hidden_size → key_dim/num_heads (默认: 1024 → 128)
        # LoRA秩建议：r=8

        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        # 作用：x_t → v_t (值)
        # 维度：hidden_size → value_dim/num_heads (默认: 1024 → 128)
        # LoRA秩建议：r=8 或 r=12 (V梯度最大)

        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            # 作用：x_t → r_t (输出门)
            # 维度：hidden_size → value_dim (默认: 1024 → 1024)
            # LoRA秩建议：r=8

        # ========== 动力学参数 ==========
        # [SDT适配]

        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),      # .0
            nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True) # .1
        )
        # 作用：x_t → gk_t (衰减门控)
        # 核心：gk_t被转换为 α_t = sigmoid(gk) / normalizer
        # 维度：
        #   输入：hidden_size (1024)
        #   中间：gate_low_rank_dim (默认16)
        #   输出：key_dim_per_group (128)
        # SDT目标：gk_proj.1 (第二层，16→128)
        # 原因：第二层直接生成α_t，决定通道遗忘

        # ========== 输出投影 ==========
        # [适合LoRA]

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        # 作用：最后一个线性投影回原维度
        # 维度：value_dim → hidden_size (默认: 1024 → 1024)
        # LoRA秩建议：r=8
```

### 2.2 前向传播流程 (第172-299行)

```python
def forward(self, hidden_states, ...):
    # ========== 投影 ==========
    q = self.q_proj(hidden_states)           # [batch, seq, key_dim]
    k = self.k_proj(hidden_states)           # [batch, seq, key_dim_per_group]
    v = self.v_proj(hidden_states)           # [batch, seq, value_dim_per_group]
    gk = self.gk_proj(hidden_states)         # [batch, seq, key_dim_per_group]
                                              # ← 关键：gk是衰减因子来源

    # ========== 门控计算 ==========
    gk = F.logsigmoid(gk) / self.gate_logit_normalizer  # ← 转换为α_t
    # 重要：gate_logit_normalizer=16 (论文中固定)
    # α_t[i] = exp(logsigmoid(gk[i]) / 16)

    # ========== GLA核心计算 ==========
    # 三种模式选择：fused_recurrent, fused_chunk, chunk
    if mode == 'fused_recurrent':
        o, recurrent_state = fused_recurrent_gla(
            q=q, k=k, v=v,
            gk=gk,  # ← 衰减因子α_t = exp(logsigmoid(gk)/16)
            ...
        )
    # 内部计算：S_t = (α_t^T · 1) ⊙ S_{t-1} + k_t^T v_t

    # ========== 输出门 ==========
    if self.use_output_gate:
        g = self.g_proj(hidden_states)  # [batch, seq, value_dim]
        # o *= sigmoid(g)  或 o *= swish(g)

    # ========== 最终投影 ==========
    o = self.o_proj(o)                  # [batch, seq, hidden_size]
```

---

## 第三部分：PEFT实施代码

### 3.1 GLA SD-LoRA实现

**文件**: `mamba-peft/mamba_ssm_peft/peft/gla_sd_lora.py`

#### 配置类 (第48-88行)

```python
@dataclass
class GlaSdLoraConfig(PeftConfig):
    # ========== 标识 ==========
    peft_type: str = "GLA_SD_LORA"

    # ========== 目标模块 ==========
    # 这里定义哪些模块适配
    target_modules: list = field(default_factory=lambda: ["gk_proj.1"])
    # 仅对gk_proj.1做SDT
    # 理由：控制状态衰减的关键参数

    # ========== LoRA部分（对其他投影层） ==========
    lora_targets: list = field(default_factory=lambda: [
        "q_proj",   # 查询投影
        "k_proj",   # 键投影
        "v_proj",   # 值投影
        "o_proj"    # 输出投影
        # 注意：g_proj 可选，如果需要完整 ≈ lora_targets + ["g_proj"]
    ])

    proj_lora_r: int = 8              # LoRA秩
    proj_lora_alpha: int = 16         # LoRA缩放系数
    proj_lora_dropout: float = 0.1    # LoRA Dropout

    # ========== SDT部分 ==========
    select_mode: str = "CHANNELS_ONLY"    # 仅对通道维度选择

    # 维度选择（训练、冻结、零化）
    num_zero: Dict = field(default_factory=lambda: {"channel": 0.1})
    # 10% 的通道被完全遗忘（设为-100）

    num_freeze: Dict = field(default_factory=lambda: {"channel": 0.5})
    # 50% 的通道保持预训练权重

    # 隐含：num_train = 1 - num_zero - num_freeze = 0.4 (40%)

    num_warmup_it: int = 100              # 梯度累积轮数
    sdlora_alpha: Dict = field(
        default_factory=lambda: {
            "global": 1.0,
            "gk_proj.1": 1.0
        }
    )
```

#### 默认值设置 (第83-88行)

```python
if self.num_zero is None:
    self.num_zero = {"channel": 0.1}     # ← 10% Zero
if self.num_freeze is None:
    self.num_freeze = {"channel": 0.5}   # ← 50% Freeze
# ← 隐含 40% Train
```

### 3.2 参数包装器 (GlaSdLoraParameter)

**文件**: `mamba_peft/mamba_ssm_peft/peft/gla_sd_lora.py` (第230-417行)

#### 关键数据成员

```python
class GlaSdLoraParameter:
    def __init__(self, ..., num_zero, num_freeze, ...):
        # ========== 维度解析 ==========
        self.num_zero = self._parse_dims(num_zero)        # {"channel": N_zero}
        self.num_freeze = self._parse_dims(num_freeze)    # {"channel": N_freeze}
        self.num_train = total_dim - N_zero - N_freeze    # 计算训练维度

        # ========== 两阶段模式 ==========
        self.sdlora_mode = "warmup"  # or "train"

        # Phase 1 (warmup): 所有维度参与，累积梯度
        self.sdlora_grad = torch.zeros_like(base_weight)
        self.it_counter = 0

        # Phase 2 (train): 根据梯度重要性划分维度
        self.train_mask = ...  # [True] * num_train + [False] * (num_freeze + num_zero)
        self.freeze_mask = ... # 冻结维度
        self.zero_mask = ...   # 零化维度

        self.sdlora_adapter = ...  # 训练维度的LoRA适配器
```

#### 前向传播 (第364-404行)

```python
def forward(self, x):
    # ========== 获取基础权重 ==========
    base_weight = self.get_base_weight()  # 原始预训练权重

    if self.sdlora_mode == "warmup":
        # ========== Phase 1: 梯度累积 ==========
        # 所有维度都参与，用于评估重要性
        weight_new = base_weight + self.sdlora_alpha * self.sdlora_grad

    elif self.sdlora_mode == "train":
        # ========== Phase 2: 稀疏训练 ==========
        weight_new = base_weight.clone()

        # 1. 零化维度：设为-100
        weight_new[self.zero_mask] = -100.0
        # ← 核心：这使得 exp(logsigmoid(-100)/16) ≈ 0.002
        # → 该通道完全遗忘

        # 2. 冻结维度：保持原值
        # weight_new[self.freeze_mask] = base_weight[self.freeze_mask]  (隐含)

        # 3. 训练维度：加上LoRA适配
        weight_new[self.train_mask] += self.sdlora_alpha * self.sdlora_adapter[self.train_mask]

    # ========== 应用到线性层 ==========
    return F.linear(x, weight_new, self.bias)
```

---

## 第四部分：环境变量与配置

### 4.1 环境变量传递链

**lat_batch_tmux.sh** (第200-206行)
```bash
# SD-LoRA specific parameters
# Dimension ratios: Train + Freeze + Zero = 100%
# Default: Train=40%, Freeze=50%, Zero=10%
printf 'export HP_WARMUP_IT=%q\n' "${HP_WARMUP_IT:-}"
printf 'export HP_TRAIN_RATIO=%q\n' "${HP_TRAIN_RATIO:-}"
printf 'export HP_FREEZE_RATIO=%q\n' "${HP_FREEZE_RATIO:-}"
printf 'export HP_ZERO_RATIO=%q\n' "${HP_ZERO_RATIO:-}"
```

↓ 传递到 lat_round.sh

**lat_round.sh** (第193行)
```bash
HP_WARMUP_IT HP_TRAIN_RATIO HP_FREEZE_RATIO HP_ZERO_RATIO HP_USE_DORA HP_USE_RSLoRA \
```

↓ 传递到 train_lat.py

**train_lat.py**
```python
# 文档中记录 (第32-38行)
SD-LoRA Specific Environment Variables:
- HP_WARMUP_IT: Override warmup iterations (default: 100)
- HP_TRAIN_RATIO: Override train dimension ratio (default: 0.4)
  If set, HP_ZERO_RATIO is auto-computed as: 1 - train - freeze
- HP_FREEZE_RATIO: Override freeze dimension ratio (default: 0.5)
- HP_ZERO_RATIO: Override zero dimension ratio (default: 0.1)
```

### 4.2 lat_adapter.py中的处理逻辑

**文件**: `lat_adapter.py` (第188-264行)

```python
def _apply_sdlora_env_overrides(peft_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    应用环境变量覆盖。

    维度比例逻辑：Train + Freeze + Zero = 100%
    默认：Train=40%, Freeze=50%, Zero=10%

    如果设置了 HP_TRAIN_RATIO，自动计算 HP_ZERO_RATIO：
      HP_ZERO_RATIO = 1 - HP_TRAIN_RATIO - HP_FREEZE_RATIO
    """

    # ========== 默认值 ==========
    default_train = 0.4   # 40%
    default_freeze = 0.5  # 50%
    default_zero = 0.1    # 10%

    # ========== 读取环境变量 ==========
    train_ratio_env = os.environ.get("HP_TRAIN_RATIO")
    freeze_ratio_env = os.environ.get("HP_FREEZE_RATIO")
    zero_ratio_env = os.environ.get("HP_ZERO_RATIO")

    # ========== 计算冻结比例 ==========
    if freeze_ratio_env is not None:
        freeze_ratio = float(freeze_ratio_env)
    else:
        freeze_ratio = peft_json.get("num_freeze", {}).get("channel", default_freeze)

    # ========== 智能计算零化比例 ==========
    if train_ratio_env is not None and zero_ratio_env is None:
        # 如果设置了 HP_TRAIN_RATIO，自动计算零化比例
        train_ratio = float(train_ratio_env)
        zero_ratio = max(0.0, 1.0 - train_ratio - freeze_ratio)
        print(f"[SD-LoRA] HP_TRAIN_RATIO={train_ratio:.2f} set, "
              f"auto-computed zero_ratio={zero_ratio:.2f} (freeze={freeze_ratio:.2f})")
    elif zero_ratio_env is not None:
        # 显式设置了零化比例
        zero_ratio = float(zero_ratio_env)
    else:
        # 使用配置文件的默认值
        zero_ratio = peft_json.get("num_zero", {}).get("channel", default_zero)

    # ========== 打印有效比例 ==========
    train_ratio_effective = 1.0 - zero_ratio - freeze_ratio
    print(f"[SD-LoRA] Effective ratios: train={train_ratio_effective:.1%}, "
          f"freeze={freeze_ratio:.1%}, zero={zero_ratio:.1%}")
```

### 4.3 配置文件

**文件**: `configs/gla_sdlora/default.json`

```json
{
    "peft_type": "GLA_SD_LORA",
    "select_mode": "CHANNELS_ONLY",

    "proj_lora_r": 8,
    "proj_lora_alpha": 16,

    "num_zero": {
        "channel": 0.1          # 10% 通道被零化
    },
    "num_freeze": {
        "channel": 0.5          # 50% 通道被冻结
    },                          # 隐含 40% 通道被训练

    "num_warmup_it": 100,

    "target_modules": ["gk_proj.1"],   # ← SDT目标
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],  # ← LoRA目标

    "sdlora_alpha": {
        "global": 1.0,
        "gk_proj.1": 1.0
    }
}
```

---

## 第五部分：使用示例

### 5.1 基础用法

#### 方式1：使用默认配置（Train=40%, Freeze=50%, Zero=10%）

```bash
HP_PEFT_TYPE=sdlora ./lat_batch_tmux.sh \
    --suite E15 \
    --round all \
    --pairs "87:glue-tvt_cola"
```

**结果**：
```
[SD-LoRA] Effective ratios: train=40.0%, freeze=50.0%, zero=10.0%
```

#### 方式2：自定义训练比例（自动计算零化比例）

```bash
HP_PEFT_TYPE=sdlora HP_TRAIN_RATIO=0.6 ./lat_batch_tmux.sh \
    --suite E15 --round all --pairs "87:glue-tvt_cola"
```

**结果**：
```
[SD-LoRA] HP_TRAIN_RATIO=0.60 set, auto-computed zero_ratio=0.05 (freeze=0.50)
[SD-LoRA] Effective ratios: train=60.0%, freeze=50.0%, zero=5.0%
```

#### 方式3：显式设置所有三个比例

```bash
HP_PEFT_TYPE=sdlora \
    HP_TRAIN_RATIO=0.5 \
    HP_FREEZE_RATIO=0.3 \
    HP_ZERO_RATIO=0.2 \
    ./lat_batch_tmux.sh --suite E15 --round all --pairs "87:glue-tvt_cola"
```

**结果**：
```
[SD-LoRA] Effective ratios: train=50.0%, freeze=30.0%, zero=20.0%
```

### 5.2 对比实验

```bash
# 实验1：纯LoRA（不做SDT）
HP_PEFT_TYPE=lora ./lat_batch_tmux.sh --pairs "87:glue-tvt_cola"

# 实验2：LoRA + SDT（默认比例）
HP_PEFT_TYPE=sdlora ./lat_batch_tmux.sh --pairs "87:glue-tvt_cola"

# 实验3：LoRA + 更激进的SDT
HP_PEFT_TYPE=sdlora HP_TRAIN_RATIO=0.6 HP_ZERO_RATIO=0.2 \
    ./lat_batch_tmux.sh --pairs "87:glue-tvt_cola"
```

---

## 第六部分：关键代码位置速查

### 当需要修改时...

| 需要修改 | 文件位置 | 代码位置 |
|---------|---------|---------|
| **默认比例** | gla_sd_lora.py | Line 83-88 (`num_zero`, `num_freeze`) |
| **SDT目标模块** | gla_sd_lora.py | Line 76 (`target_modules`) |
| **LoRA目标** | gla_sd_lora.py | Line 77 (`lora_targets`) |
| **环境变量处理** | lat_adapter.py | Line 188-264 (`_apply_sdlora_env_overrides`) |
| **零值掩码** | gla_sd_lora.py | Line 389-390 或搜索 `ZERO_MASK_VALUE = -100` |
| **门控归一化** | gla.py (FLA) | Line 170 (`gate_logit_normalizer = 16`) |
| **零化的计算** | gla_sd_lora.py | Line 389 (`weight_new[self.zero_mask] = -100.0`) |

### 关键数值

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| Train比例 | 0.4 (40%) | lat_adapter.py:207 | 可训练维度 |
| Freeze比例 | 0.5 (50%) | lat_adapter.py:208 | 冻结维度 |
| Zero比例 | 0.1 (10%) | lat_adapter.py:209 | 零化维度 |
| gate_logit_normalizer | 16 | gla.py:170 | 门控归一化系数 |
| ZERO_MASK_VALUE | -100.0 | gla_sd_lora.py:241 | 零化掩码值 |
| Warmup轮数 | 100 | gla_sd_lora.py:64 | 梯度累积轮数 |

---

## 第七部分：调试与诊断

### 检查当前配置

```bash
# 启用verbose模式查看详细日志
LAT_VERBOSE=1 HP_PEFT_TYPE=sdlora ./lat_batch_tmux.sh --pairs "87:glue-tvt_cola"
```

输出应包含：
```
[SD-LoRA] Effective ratios: train=40.0%, freeze=50.0%, zero=10.0%
[GLA SD-LoRA] Warmup phase...
[GLA SD-LoRA] Training phase...
```

### 验证目标模块

查看 `lat_adapter.py` 第322-326行：
```python
def _get_target_modules_for_model(model_type: str, peft_type: str) -> Optional[list]:
    if peft_type == "GLA_SD_LORA":
        return ["gk_proj.1"]  # ← SDT只目标化gk_proj的第二层
```

### 梯度检查

在 train_lat.py 中添加：
```python
# 在warmup阶段，打印gk_proj.1的梯度
for name, param in model.named_parameters():
    if 'gk_proj.1' in name and param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"[DEBUG] {name} grad_norm: {grad_norm:.6f}")
```

---

## 总结：当前实现状态

✅ **已完成**：
- [x] gk_proj.1作为SDT目标
- [x] q/k/v/o_proj作为LoRA目标
- [x] Train=40%, Freeze=50%, Zero=10%的默认比例
- [x] 环境变量导出与处理 (lat_batch_tmux.sh → lat_round.sh → train_lat.py)
- [x] 自动计算逻辑（设置HP_TRAIN_RATIO时自动算Zero）
- [x] ZERO_MASK_VALUE=-100实现彻底遗忘

⚠️ **可选优化**：
- [ ] g_proj也加入LoRA目标
- [ ] V向量更大的LoRA秩 (r=12)
- [ ] 全局Top-K选择 (不用固定比例)
- [ ] 通道级梯度度量优化

---

## 参考
- GLA论文：第4.4节 GLA Transformer
- GLA代码：fla/layers/gla.py
- SD-LoRA代码：gla_sd_lora.py
- 配置系统：lat_adapter.py，train_lat.py
