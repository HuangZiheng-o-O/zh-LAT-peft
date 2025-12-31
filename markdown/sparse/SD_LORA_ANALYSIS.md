# SD-LoRA实现深度分析

## 执行流程概览

### 命令入口
```bash
DEVICES=${1:-0}
python run_all.py train.py --device $DEVICES --cfg cfg/exps/glue/*/* --override
```

**执行流程**: `run_all.py` → `train.py` → `run_train()` → 两阶段训练

---

## 一、核心组件架构

### 1.1 多设备并行调度器 (`run_all.py`)

**职责**: 将多个训练任务分配到多个GPU并行执行

```python
# 关键流程
1. 解析参数: --device指定GPU列表, --cfg指定配置文件glob模式
2. 展开配置: cfg/exps/glue/*/* → 多个.yaml文件
3. 生成命令队列: 每个配置文件对应一条 "python train.py --cfg <path>" 命令
4. 并行执行:
   - 每个GPU一个worker线程
   - 从队列中取任务并在对应GPU上执行 (CUDA_VISIBLE_DEVICES=<gpu_id>)
   - 支持GPU空闲等待 (wait_gpu_free=True时会等待GPU利用率为0)
```

**核心代码逻辑**:
```python
# run_all.py:82-89
for i in (range(num_tasks) if not args.reversed else reversed(range(num_tasks))):
    cmd = ["python", script]
    for group in other_args_var_grouped:
        cmd += [group[0], group[i+1]]  # --cfg <config_i>
    cmd += other_args_const  # --override
    cmds.append(cmd)

# run_all.py:94-96
qu = Queue()
for cmd in cmds:
    qu.put(cmd)

# run_all.py:34-47 (worker线程)
def worker_func(qu: Queue, device, wait_free=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    if wait_free:
        wait_gpu_free(int(device))

    while True:
        try:
            proc = qu.get(block=False)
        except Empty:
            return
        subprocess.run(proc, env=env)  # 执行 python train.py --cfg <path> --override
```

---

### 1.2 单任务训练入口 (`train.py`)

**职责**: 检测SD-LoRA配置并执行两阶段训练

```python
# train.py:214-232 (main函数关键逻辑)
is_sdlora = False
if train_args["peft"] is not None:
    with open(train_args["peft"], "r") as f:
        peft_cfg = json.load(f)

    if peft_cfg["peft_type"] == MambaPeftType.SD_LORA:
        is_sdlora = True

if is_sdlora:
    if train_args["overwrite"]:
        if Path(train_args["output_dir"]).exists():
            shutil.rmtree(train_args["output_dir"])  # 删除旧checkpoint

    del train_args["overwrite"]
    run_train(**train_args, is_sdlora=True)  # 阶段1: warmup
    run_train(**train_args, is_sdlora=True, overwrite=True)  # 阶段2: fine-tuning
else:
    run_train(**train_args)  # 普通训练
```

**关键点**:
1. **SD-LoRA检测**: 通过读取PEFT配置文件的`peft_type`字段判断
2. **两阶段调用**:
   - 第一次调用 `run_train(..., is_sdlora=True)`: warmup阶段
   - 第二次调用 `run_train(..., is_sdlora=True, overwrite=True)`: fine-tuning阶段
3. **checkpoint管理**: 第二阶段前删除第一阶段的checkpoint目录

---

### 1.3 配置文件层次结构

#### 实验配置 (Experiment Config)
**示例**: `cfg/exps/benchmark/glue/cola/000_full.yaml`
```yaml
batch_size: 4
data: glue-tvt_cola
learning_rate: 0.0001
model: state-spaces/mamba-130m
num_epochs: 10
peft: null  # 或指向PEFT配置文件路径
prec: bf16
```

#### PEFT配置 (SD-LoRA Config)
**示例**: `cfg/peft/sd_lora/500it/n0.95_d0.99.json`
```json
{
    "peft_type": "SD_LORA",
    "num_zero": {
        "state": 0,      // 状态维度零化数量 (绝对值或比例)
        "channel": 0     // 通道维度零化数量
    },
    "num_freeze": {
        "state": 0.95,   // 状态维度冻结比例 (95%)
        "channel": 0.99  // 通道维度冻结比例 (99%)
    },
    "target_modules": [
        "A_log",         // Mamba的SSM参数矩阵 A (对数空间)
        "x_proj_B",      // 投影矩阵 B
        "x_proj_C",      // 投影矩阵 C
        "out_proj"       // 输出投影
    ],
    "proj_lora_r": 8,    // 投影层的LoRA秩
    "num_warmup_it": 499 // warmup迭代次数 (从0开始计数,故实际为500次)
}
```

**配置参数详解**:
- `num_zero`: 完全置零的参数位置 (掩码填充为+∞,在softmax中变为0)
- `num_freeze`: 冻结的参数位置 (使用预训练权重,不更新梯度)
- `num_train = total - num_zero - num_freeze`: 可训练的参数位置
- `num_warmup_it`: warmup阶段的迭代次数,用于梯度累积来选择重要通道/状态

---

## 二、SD-LoRA核心实现

### 2.1 两阶段训练机制

#### 阶段1: Warmup (梯度累积)
**目标**: 通过梯度幅值选择重要的通道和状态

```python
# train.py:229 (第一次调用)
run_train(**train_args, is_sdlora=True)

# train.py:63-149 (run_train函数关键逻辑)
def run_train(..., is_sdlora=False):
    # 1. 检查checkpoint
    if overwrite and is_sdlora:
        assert Path(output_dir).exists()  # 第二阶段必须有第一阶段的config

    # 2. 加载模型和PEFT
    model = load_mamba(model, ...)["model"]
    if peft is not None:
        model, peft_cfg = get_mamba_peft_model(model, peft, return_peft_cfg=True, ...)
        assert (is_sdlora and isinstance(model.base_model, SdLoraModel)) or \
               (not is_sdlora and not isinstance(model.base_model, SdLoraModel))

    # 3. 创建Trainer
    trainer = MambaTrainer(model=model, ...)

    # 4. 开始训练
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

**Warmup阶段行为** (在 `SdLoraParameter` 中实现):
```python
# mamba_ssm_peft/peft/sd_lora.py:422-423
if self.sdlora_mode == "warmup":
    param_new = param + self.sdlora_alpha * (self.sdlora_grad if self.sdlora_grad is not None else 0)
```
- **初始化**: `sdlora_mode = "warmup"` (sd_lora.py:149)
- **梯度累积**: `sdlora_grad` 参数会在每次迭代中累积梯度
- **迭代计数**: `it_counter` 从0开始递增
- **模式切换**: 当 `it_counter > num_warmup_it` 时,切换到 `"train"` 模式 (sd_lora.py:408-409)

#### 阶段2: Fine-tuning (结构化稀疏训练)
**目标**: 只训练选中的重要通道/状态,冻结或置零其他位置

```python
# train.py:230 (第二次调用)
run_train(**train_args, is_sdlora=True, overwrite=True)

# train.py:65-66
if overwrite and is_sdlora:
    assert Path(output_dir).exists()  # 第一阶段必须已完成
```

**Fine-tuning阶段行为**:
```python
# mamba_ssm_peft/peft/sd_lora.py:424-425
elif self.sdlora_mode == "train":
    param_new = self.build_train_param(param, self.sdlora_adapter)

# sd_lora.py:386-401 (build_train_param)
def build_train_param(self, param, adapter):
    if self.train_mask is None:
        print("Building trainable mask")
        self.train_mask = self.get_mask("train")  # 基于warmup累积的梯度选择重要位置

    if self._is_layer_of("A_log"):
        if self.zero_mask is None:
            self.zero_mask = self.get_mask("zero")  # 选择零化位置
            assert torch.sum(self.train_mask & self.zero_mask).item() == 0  # 确保不重叠

        param = torch.masked_fill(param, self.zero_mask, 10)  # 将零化位置填充为10 (近似+∞)

    bias = torch.masked_scatter(torch.zeros_like(param), self.train_mask, adapter)
    return param + self.sdlora_alpha * bias  # adapter只在train_mask位置非零
```

---

### 2.2 掩码选择算法 (Structured Sparsity)

**核心思想**: 基于warmup阶段累积的梯度幅值,选择最重要的通道和状态

```python
# sd_lora.py:345-384 (get_mask方法)
def get_mask(self, mask_type):
    grad = self.get_grad_for_sel()  # 获取累积的梯度 (A_log的sdlora_grad)

    param = self.get_model_param_info()
    mask = torch.zeros(param.shape, device=param.device, dtype=torch.bool)

    match self.select_mode:
        case SelectMode.CHANNELS_PER_STATE_CHANNELS:
            channel_indices = self.select_rows(grad, "CHANNEL", mask_type)

            if mask_type == "train":
                if self._is_layer_of(("in_proj_x", "in_proj_z", "out_proj")):
                    # 投影层: 选择整列/行
                    match self.proj_select_mode:
                        case SelectMode.CHANNELS_ALL_STATES:
                            mask.index_fill_(
                                1 if self._is_layer_of("out_proj") else 0,
                                channel_indices, True
                            )
                else:
                    # SSM参数: 每个通道内选择重要的状态
                    state_indices_per_row = self.select_rows(
                        grad[:, channel_indices], "STATE", mask_type, per_row=True
                    )
                    n = state_indices_per_row.shape[0]
                    mask.T[channel_indices.repeat(n), state_indices_per_row.reshape(-1)] = True

            elif mask_type == "zero":
                # 零化整个通道 (仅对A_log)
                assert self._is_layer_of("A_log")
                mask.index_fill_(1, channel_indices, True)

    return mask

# sd_lora.py:306-314 (get_importances - 重要性评分)
def get_importances(self, x, dim, per_row=False):
    norms = x.square().detach()  # L2范数的平方 (梯度幅值)
    if per_row:
        ind = torch.argsort(-norms, dim=dim)  # 按行排序
    else:
        dim = 1 - dim
        norms = norms.sum(dim)  # 求和后再排序
        ind = torch.argsort(-norms)
    return ind

# sd_lora.py:322-335 (select_rows - 根据重要性选择索引)
def select_rows(self, x, dim, row_type=None, per_row=False):
    dim = self._dim_name_to_idx(dim)
    imp = self.get_importances(x, dim, per_row=per_row)

    row_types = {
        "train": imp[0:self.num_train[dim]],  # 前num_train个最重要的
        "freeze": imp[self.num_train[dim]:self.num_train[dim]+self.num_freeze[dim]],
        "zero": imp[self.num_train[dim]+self.num_freeze[dim]:...]
    }

    if row_type is None:
        return row_types
    else:
        return row_types[row_type]
```

**选择策略可视化** (以 `num_freeze={"state": 0.95, "channel": 0.99}` 为例):

对于 `A_log` 参数 (shape: [states=16, channels=2560]):
1. **通道维度选择** (CHANNEL dim):
   - `num_train[channel] = 2560 * (1 - 0.99) = 25.6 ≈ 26` (可训练通道)
   - `num_freeze[channel] = 2560 * 0.99 = 2534.4 ≈ 2534` (冻结通道)
   - 根据梯度幅值排序,选择前26个通道作为可训练

2. **状态维度选择** (STATE dim, per-channel):
   - 对每个选中的通道,在其16个状态中:
     - `num_train[state] = 16 * (1 - 0.95) = 0.8 ≈ 1` (每通道可训练1个状态)
     - `num_freeze[state] = 16 * 0.95 = 15.2 ≈ 15` (每通道冻结15个状态)
   - 结果: `26 channels × 1 state/channel = 26` 个可训练参数

对于 `out_proj` 参数 (shape: [hidden_size, channels]):
- 只选择通道维度 (CHANNELS_ALL_STATES模式)
- 选中的26个通道对应的整列都可训练

---

### 2.3 两阶段状态转换机制

**状态转换触发** (在 `MambaTrainer.compute_loss` 中检测):
```python
# trainer/mamba_trainer.py:104-107
if getattr(model, "should_training_stop", False):
    if hasattr(model, "save_config"):
        model.save_config(self.args.output_dir)  # 保存warmup梯度到checkpoint
        self.control.should_training_stop = True  # 停止训练
```

**`should_training_stop` 属性** (在 `SdLoraModel` 中实现):
```python
# mamba_ssm_peft/peft/sd_lora.py:107-118
@property
def should_training_stop(self):
    if self.last_mode == "warmup" and self.get_sdlora_mode() == "train":
        self.last_mode = "train"
        res = True  # 刚从warmup切换到train,需要停止保存config
    else:
        res = False

    if self.last_mode is None:
        self.last_mode = self.get_sdlora_mode()

    return res
```

**Config保存/加载** (用于跨阶段传递梯度信息):
```python
# sd_lora.py:166-173 (save_config - 在warmup结束时调用)
def save_config(self, path):
    cfg_path = self._get_cfg_file(path)  # e.g., "output_dir/mixer_layers_0_mixer_A_log_adapter.pkl"
    grad = self.sdlora_grad
    if grad is not None:
        grad = grad.data
    with open(cfg_path, "wb") as f:
        pickle.dump(grad, f)  # 保存warmup累积的梯度
    print(f"Saved {cfg_path}")

# sd_lora.py:156-164 (load_config - 在fine-tuning开始时调用)
def load_config(self, path):
    cfg_path = self._get_cfg_file(path)
    if cfg_path.exists():
        if self.sdlora_grad is not None:
            with open(cfg_path, "rb") as f:
                with torch.no_grad():
                    self.sdlora_grad.data[:] = pickle.load(f)  # 加载warmup梯度
        print(f"Loaded {cfg_path}")
        self.set_sdlora_mode("train")  # 直接进入train模式

# trainer/mamba_trainer.py:62-63 (在Trainer初始化时调用)
if hasattr(model, "load_config"):
    model.load_config(self.args.output_dir)
```

**完整流程**:
```
阶段1 (warmup):
  1. SdLoraParameter初始化: sdlora_mode="warmup", it_counter=0
  2. 前向传播: param_new = param + alpha * sdlora_grad (全参数)
  3. it_counter递增,当超过num_warmup_it时:
     - set_sdlora_mode("train")
     - should_training_stop返回True
  4. Trainer检测到should_training_stop:
     - 调用model.save_config(output_dir) → 保存sdlora_grad到.pkl文件
     - 设置control.should_training_stop=True
  5. 训练循环退出,run_train()返回

阶段2 (fine-tuning):
  1. 再次调用run_train(..., overwrite=True)
  2. Trainer初始化时调用model.load_config(output_dir):
     - 从.pkl文件加载sdlora_grad
     - 直接set_sdlora_mode("train")
  3. 前向传播首次调用时:
     - 调用build_train_param()
     - 基于加载的sdlora_grad计算train_mask和zero_mask (仅计算一次)
  4. 后续前向传播:
     - param_new = param + alpha * masked_scatter(..., train_mask, sdlora_adapter)
     - 只有mask内的位置参与训练
```

---

## 三、SD-LoRA vs Naive LoRA 对比

| 维度 | Naive LoRA | SD-LoRA |
|------|------------|---------|
| **参数化方式** | `W_new = W + BA` (全矩阵低秩分解) | `W_new = W + mask ⊙ Δ` (结构化稀疏) |
| **训练阶段** | 单阶段 | 两阶段 (warmup → fine-tuning) |
| **参数选择** | 手动指定秩r | 基于梯度自动选择重要通道/状态 |
| **稀疏模式** | 低秩稀疏 (秩约束) | 结构化稀疏 (通道+状态掩码) |
| **训练参数量** | `rank × (d_in + d_out)` | `num_train[state] × num_train[channel]` |
| **推理效率** | 需要额外的矩阵乘法 (`BA`) | 直接修改原参数 (mask可合并) |
| **适用场景** | 通用微调 | Mamba等SSM模型 (利用状态空间结构) |

### SD-LoRA独有特性

1. **Two-stage训练**:
   - Warmup阶段: 全参数梯度累积,识别重要通道和状态
   - Fine-tuning阶段: 只更新选中位置,其他位置冻结或置零

2. **结构化剪枝**:
   - `num_zero`: 通过掩码置+∞实现参数置零 (softmax后变0)
   - `num_freeze`: 保留预训练权重,不参与梯度更新
   - `num_train`: 基于梯度幅值动态选择

3. **Per-channel状态选择** (`SelectMode.CHANNELS_PER_STATE_CHANNELS`):
   - 先选择重要通道 (CHANNEL维度)
   - 再在每个通道内选择重要状态 (STATE维度)
   - 相比全局选择,更符合SSM的物理意义

4. **投影层特殊处理**:
   - `in_proj_x/z`, `out_proj` 使用LoRA (rank=8)
   - `A_log`, `x_proj_B/C` 使用SD-LoRA掩码

---

## 四、完整执行流程追踪

### 示例配置
- **Experiment**: `cfg/exps/benchmark/glue/cola/sd_lora_example.yaml`
  ```yaml
  peft: cfg/peft/sd_lora/500it/n0.95_d0.99.json
  data: glue-tvt_cola
  model: state-spaces/mamba-130m
  num_epochs: 10
  ```
- **PEFT**: `cfg/peft/sd_lora/500it/n0.95_d0.99.json` (见1.3节)

### 执行时间线

```
t=0: 用户执行命令
  $ DEVICES=0 python run_all.py train.py --device 0 --cfg cfg/exps/benchmark/glue/cola/sd_lora_example.yaml --override

t=1: run_all.py 解析参数
  - devices = ["0"]
  - other_args_var = ["--cfg", "cfg/exps/benchmark/glue/cola/sd_lora_example.yaml"]
  - other_args_const = ["--override"]
  - 生成命令: ["python", "train.py", "--cfg", "...", "--override"]

t=2: worker_func 启动
  - 设置 CUDA_VISIBLE_DEVICES=0
  - 执行 subprocess.run(["python", "train.py", "--cfg", "...", "--override"])

t=3: train.py main() 执行
  - 加载实验配置: data="glue-tvt_cola", peft="cfg/peft/sd_lora/500it/n0.95_d0.99.json"
  - 读取PEFT配置: peft_cfg["peft_type"] = "SD_LORA"
  - 检测到 is_sdlora=True

t=4: train.py 第一阶段调用
  run_train(
      output_dir="weights/benchmark/glue/cola/sd_lora_example",
      peft="cfg/peft/sd_lora/500it/n0.95_d0.99.json",
      data="glue-tvt_cola",
      is_sdlora=True
  )

t=5: run_train() 阶段1执行
  a. 加载Mamba模型: state-spaces/mamba-130m
  b. get_mamba_peft_model():
     - model.split_layers()
     - 创建SdLoraModel,包装原模型
     - 为每个target_module创建SdLoraParameter:
       * A_log_adapter → SdLoraParameter (state=16, channel=2560)
       * x_proj_B → SdLoraParameter
       * x_proj_C → SdLoraParameter
       * out_proj → LoraLinear (r=8)
     - 初始化sdlora_mode="warmup", it_counter=0
  c. 创建MambaTrainer
  d. trainer.train():
     - 开始训练循环
     - 每次前向传播:
       * SdLoraParameter.forward():
         - param_new = param + alpha * sdlora_grad (全参数微调)
         - it_counter += 1
       * 当 it_counter=500 时:
         - set_sdlora_mode("train")
         - should_training_stop返回True
     - Trainer.compute_loss()检测到should_training_stop:
       * model.save_config("weights/.../")
         → 保存 mixer_layers_0_mixer_A_log_adapter.pkl 等文件
       * control.should_training_stop=True
     - 训练循环退出

t=6: run_train() 阶段1结束
  - checkpoint保存在 weights/benchmark/glue/cola/sd_lora_example/
  - sdlora_grad保存在 *.pkl文件中

t=7: train.py 第二阶段调用
  if Path(output_dir).exists():
      shutil.rmtree(output_dir)  # 删除checkpoint (保留.pkl?)

  run_train(
      output_dir="weights/benchmark/glue/cola/sd_lora_example",
      peft="cfg/peft/sd_lora/500it/n0.95_d0.99.json",
      data="glue-tvt_cola",
      is_sdlora=True,
      overwrite=True
  )

t=8: run_train() 阶段2执行
  a. 再次加载Mamba模型和PEFT
  b. MambaTrainer初始化:
     - if hasattr(model, "load_config"):
       model.load_config("weights/.../")
       → 从*.pkl加载sdlora_grad
       → set_sdlora_mode("train")
  c. trainer.train():
     - 首次前向传播:
       * SdLoraParameter.forward():
         - sdlora_mode="train"
         - 调用 build_train_param():
           + 计算 train_mask = get_mask("train")
             (基于加载的sdlora_grad选择重要位置)
           + 计算 zero_mask = get_mask("zero") (对A_log)
           + param_new = masked_fill(param, zero_mask, 10)
           + bias = masked_scatter(zeros_like(param), train_mask, sdlora_adapter)
           + return param_new + alpha * bias
     - 后续迭代只更新sdlora_adapter参数 (mask固定)
     - 训练完整的num_epochs=10轮

t=9: 训练完成
  - 最终checkpoint保存在 weights/benchmark/glue/cola/sd_lora_example/
  - WandB日志记录在项目 "mamba-peft"
```

---

## 五、关键代码文件汇总

### 核心文件依赖图
```
run_all.py
    └── train.py
        ├── mamba_ssm_peft/__init__.py
        │   ├── get_mamba_peft_model()
        │   ├── load_mamba()
        │   └── load_tokenizer()
        ├── mamba_ssm_peft/peft/sd_lora.py
        │   ├── SdLoraConfig (配置类)
        │   ├── SdLoraModel (PEFT包装器)
        │   └── SdLoraParameter (核心实现)
        │       ├── forward() - 前向传播逻辑
        │       ├── get_mask() - 掩码选择
        │       ├── build_train_param() - 构建稀疏参数
        │       ├── save_config() - 保存warmup梯度
        │       └── load_config() - 加载warmup梯度
        ├── trainer/mamba_trainer.py
        │   ├── MambaTrainer
        │   │   ├── compute_loss() - 检测should_training_stop
        │   │   └── __init__() - 调用load_config()
        │   └── MambaTrainingArguments
        └── dataset/
            └── load_dataset() - 数据加载
```

### 涉及的所有文件 (按调用顺序)
1. **run_all.py** (107行): 多设备并行调度
2. **train.py** (237行): 单任务入口 + 两阶段调用
3. **mamba_ssm_peft/__init__.py** (153行): 模型加载和PEFT包装
4. **mamba_ssm_peft/peft/__init__.py** (43行): PEFT注册机制
5. **mamba_ssm_peft/peft/sd_lora.py** (438行): **SD-LoRA核心实现**
6. **trainer/mamba_trainer.py** (228行): 自定义Trainer + 状态转换检测
7. **dataset/*.py**: 数据集加载 (GLUE, Spider等)
8. **mamba_ssm_peft/utils/decoder.py**: 生成解码器 (仅eval_gen时使用)

---

## 六、与普通LoRA的代码差异

### LoRA实现 (假设使用PEFT库的标准LoRA)
```python
# peft/tuners/lora/layer.py (伪代码)
class LoraLinear(nn.Module):
    def __init__(self, base_layer, r, lora_alpha):
        self.base_layer = base_layer
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = lora_alpha / r

    def forward(self, x):
        result = self.base_layer(x)  # W·x
        lora_result = (x @ self.lora_A.T) @ self.lora_B.T  # x·A^T·B^T
        return result + lora_result * self.scaling
```

### SD-LoRA实现
```python
# mamba_ssm_peft/peft/sd_lora.py (实际代码)
class SdLoraParameter(nn.Module):
    def __init__(self, base_layer, num_warmup_it, num_freeze, ...):
        self.base_layer = base_layer
        self.sdlora_grad = nn.Parameter(torch.zeros(full_shape))  # warmup梯度
        self.sdlora_adapter = nn.Parameter(torch.zeros(num_train))  # 稀疏adapter
        self.it_counter = 0
        self.train_mask = None  # 延迟初始化

    def forward(self, x):
        # 阶段1: warmup
        if self.sdlora_mode == "warmup":
            param_new = param + alpha * self.sdlora_grad
            self.it_counter += 1
            if self.it_counter > self.num_warmup_it:
                self.set_sdlora_mode("train")

        # 阶段2: fine-tuning
        elif self.sdlora_mode == "train":
            if self.train_mask is None:
                self.train_mask = self.get_mask("train")  # 基于sdlora_grad选择
            param_new = param + alpha * masked_scatter(zeros, self.train_mask, self.sdlora_adapter)

        return F.linear(x, param_new) if self.is_layer else param_new
```

### 关键差异总结
| 特性 | LoRA | SD-LoRA |
|------|------|---------|
| 参数初始化 | `lora_A ~ N(0,1)`, `lora_B = 0` | `sdlora_grad = 0`, `sdlora_adapter = 0` |
| 训练方式 | 直接优化A和B | warmup累积梯度 → 选择mask → 优化adapter |
| 前向计算 | `W + BA` | warmup: `W + grad`, fine-tune: `W + mask⊙adapter` |
| 参数量 | `r(d_in+d_out)` | `num_train[0] × num_train[1]` (数据驱动) |
| 剪枝策略 | 固定秩约束 | 动态选择 (基于梯度) |

---

## 七、中文总结

### SD-LoRA的核心创新
1. **两阶段训练范式**:
   - 第一阶段 (warmup): 全参数前向,累积梯度幅值,不做参数更新 (或微小更新)
   - 第二阶段 (fine-tuning): 基于梯度幅值选择重要通道和状态,只训练选中位置

2. **结构化稀疏 vs 低秩稀疏**:
   - LoRA: `W ≈ W_0 + BA` (秩r约束,全矩阵近似)
   - SD-LoRA: `W[i,j] = W_0[i,j] + Δ[i,j]` if `mask[i,j]=True` (位置级稀疏)

3. **数据驱动的参数选择**:
   - 不需要手动调节秩r
   - 通过 `num_freeze` 比例控制稀疏度
   - 梯度幅值自动识别重要通道/状态

4. **针对SSM的优化**:
   - 利用Mamba的状态空间结构 (state × channel)
   - 每个通道内独立选择重要状态 (per-channel selection)
   - `A_log` 参数的特殊处理 (状态维度稀疏 + 通道维度稀疏)

### 适用场景
- **推荐使用**: Mamba, S4等状态空间模型的微调
- **优势**:
  - 更少的可训练参数 (0.05% vs 0.1%+ for LoRA)
  - 更好的推理效率 (mask可以在保存checkpoint时合并到原参数)
  - 保留预训练知识 (大部分参数冻结)
- **劣势**:
  - 需要两阶段训练 (时间成本 ≈ 1.5×单阶段)
  - warmup阶段的超参数敏感 (num_warmup_it需要调节)

### 使用建议
1. **首次使用**: 从 `num_freeze={"state": 0.95, "channel": 0.99}` 开始
2. **调节稀疏度**: 增大 `num_freeze` → 更少参数 → 可能欠拟合
3. **调节warmup**: `num_warmup_it` 建议为总训练步数的 5-10%
4. **监控指标**: warmup结束后的eval metric应接近全参数微调的初始metric

---

## 附录: 完整的配置示例

### A. 最小可运行示例
```yaml
# cfg/exps/test/sd_lora_minimal.yaml
batch_size: 4
data: glue-tvt_cola
learning_rate: 1e-4
model: state-spaces/mamba-130m
num_epochs: 3
peft: cfg/peft/sd_lora/100it/minimal.json
prec: bf16
```

```json
// cfg/peft/sd_lora/100it/minimal.json
{
    "peft_type": "SD_LORA",
    "num_zero": {"state": 0, "channel": 0},
    "num_freeze": {"state": 0.9, "channel": 0.95},
    "target_modules": ["A_log"],
    "proj_lora_r": null,
    "num_warmup_it": 99
}
```

### B. 完整配置 (论文复现)
```json
// cfg/peft/sd_lora/500it/n0.95_d0.99.json
{
    "peft_type": "SD_LORA",
    "num_zero": {"state": 0, "channel": 0},
    "num_freeze": {"state": 0.95, "channel": 0.99},
    "target_modules": ["A_log", "x_proj_B", "x_proj_C", "out_proj"],
    "proj_lora_r": 8,
    "num_warmup_it": 499,
    "select_mode": "CHANNELS_PER_STATE_CHANNELS",
    "proj_select_mode": "CHANNELS_ALL_STATES",
    "reg_scale": 0.0,
    "sdlora_alpha": null
}
```

