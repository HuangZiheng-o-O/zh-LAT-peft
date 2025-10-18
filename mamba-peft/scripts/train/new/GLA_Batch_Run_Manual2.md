<!-- Update timestamp: 2025-10-16 21:44:30 -->

# GLA Batch Run Manual — Per‑GPU Concurrency Update

## 🔄 What changed (patch recap)
- **Per‑GPU concurrency plan (`GPU_PLAN`)** added to `gla_round_new.sh`.
- **Dynamic round slicing** now uses **total parallel slots `N_SLOTS`** (not `NUM_GPUS`).
- **GPU assignment** is based on **flattened concurrency slots** (`GPU_SLOTS`) not “one job per GPU”.
- **7‑GPU hard check removed**. You can pass a custom subset via `GPU_IDS`; GPUs with **0-concurrency** are supported.
- Wrappers **`gla_tmux_nohup.sh`** and **`gla_batch_tmux.sh`** accept **`--gpus`** and **`--gpu-plan`** and forward them via env.
- Launch logs now **echo GPUs / PLAN / SLOTS** for visibility.

---

## 🚀 Quick start (new flags)

### Single job wrapper
```bash
# Minimal
bash .../gla_tmux_nohup.sh --suite E2 --round all --data glue-tvt_mrpc

# With explicit GPU subset and per‑GPU concurrency
bash .../gla_tmux_nohup.sh \
  --suite E2 --round all --data glue-tvt_mrpc \
  --gpus "0 1 2 3 5 6" \
  --gpu-plan "3,3,3,3,0,3,3"
```

### Batch wrapper (sequential steps within one tmux session)
```bash
# Two back-to-back jobs
./gla_batch_tmux.sh --suite E2 --round all --pairs "127:AAA,87:BBB"

# Per‑GPU settings applied to every job in the batch
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
./gla_batch_tmux.sh \
  --suite E2 --round all \
  --pairs "87:glue-tvt_mrpc" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "3,3,3,3,0,3,3"
  
conda activate mzsz 
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
  ./gla_batch_tmux.sh \
  --suite E2 --round all \
  --pairs "87:glue-tvt_cola 87:glue-tvt_mrpc 87:glue-tvt_mnli 87:glue-tvt_qnli 87:glue-tvt_qqp 87:glue-tvt_rte 87:glue-tvt_sst2" \
  --gpus "0 1 2 3 4 5 6" \
  --gpu-plan "3,3,3,3,3,1,1"
```

> Both wrappers export `GPU_IDS` and `GPU_PLAN` to the core launcher `gla_round_new.sh`.

---

## 🧠 Semantics: `GPU_IDS` vs `GPU_PLAN`

- **`GPU_IDS`** defines **which physical GPUs** are under this script’s control and **their order**.  
  If omitted, the script auto-detects all visible devices (or uses `CUDA_VISIBLE_DEVICES`).
- **`GPU_PLAN`** defines **how many concurrent slots each listed GPU gets**.

### Length rules
- If `GPU_PLAN` is **unset** → default **1 slot per detected GPU**.
- If `GPU_PLAN` has a **single integer** → it is **broadcast** to every detected GPU.
- If `GPU_PLAN` has **multiple integers** → its **length must equal** `len(GPU_IDS)` (or detected GPU count).

### Zero-concurrency GPUs
- A `0` in `GPU_PLAN` means **keep the GPU visible** but **do not schedule jobs to it** (useful to “reserve” a card for others).

> **Tip:** If you want a GPU to be completely invisible to this run, **remove it from `GPU_IDS`** rather than giving it `0` slots.

---

## 🧮 Slot expansion & round slicing (the new core)

Given:
```bash
GPU_IDS="0 1 2 3 4 5 6"
GPU_PLAN="3,3,3,3,0,3,3"
```
The launcher builds a flattened **slot array**:
```
GPU_SLOTS = 0 0 0  1 1 1  2 2 2  3 3 3  5 5 5  6 6 6   # (4 has 0 slots → absent)
N_SLOTS   = len(GPU_SLOTS) = 18
```

- Jobs in a round are **sliced by `N_SLOTS`**:  
  `N_ROUNDS = ceil(TOTAL_CFGS / N_SLOTS)`
- **Assignment:** job `i` uses `GPU = GPU_SLOTS[i % N_SLOTS]`.

This makes scheduling **stable** and **proportional to per‑GPU concurrency**.

---

## 🧩 “Equivalent” configurations (think harder)

**A.**
```bash
GPU_IDS="0 1 2 3 4 5 6"
GPU_PLAN="3,3,3,3,0,3,3"
```
**B.**
```bash
GPU_IDS="0 1 2 3 4 5 6"
GPU_PLAN="3,3,3,3,3,3"
```

- **Scheduling results are identical** (same `GPU_SLOTS`, same `N_SLOTS=18`, 4 never receives a job).  
- **But semantics differ:**
  - **A = 逻辑禁用**（4 可见但“0 并发”）。其他程序可能仍用到 4；NCCL/拓扑探测会看到它。
  - **B = 物理排除**（4 不可见于本脚本）。更干净，常用于避免误用/探测开销。

选择建议：
- 需要**彻底不碰某 GPU** → **B**（移出 `GPU_IDS`）。
- 需要**给他人/别的进程保留**某 GPU → **A**（`GPU_PLAN` 置 `0`）。

---

## 🧾 New logs (for sanity check)

每轮开头会打印：
```
=== Starting Round r (...; NUM_GPUS=K; N_SLOTS=S) ===
GPUs    = 0 1 2 3 5 6
PLAN    = 3 3 3 3 0 3 3  (GPU->slots)
SLOTS   = 0 0 0 1 1 1 2 2 2 3 3 3 5 5 5 6 6 6  (flattened)
```
请在启动后**确认这三行是否符合预期**。

---

## 🧰 CLI reference (wrappers)

### `gla_tmux_nohup.sh`
```
--suite <E*>           Suite 名称（传给核心脚本）
--round <N|all>        轮次编号或 all
--seed <int>           替换 FORCE_SEED（通过临时副本）
--data <name>          注入 DATA=... 环境变量
--name <str>           tmux 会话名（可选）
--logdir <dir>         日志目录（默认 ./logs 或示例中的自定义默认）
--gpus "<ids>"         设备列表（空格/逗号均可）
--gpu-plan "<ints>"    每设备并发（单值广播或与 --gpus 等长）
```

### `gla_batch_tmux.sh`
- `--pairs "SEED:DATA ..."` 多个作业**顺序**执行于同一 tmux 会话；其他 flags 同上。

---

## 🧯 Troubleshooting

- **ERROR: GPU_PLAN length ...**  
  → `--gpu-plan` 的整数个数需等于 `--gpus` 数量（或只给一个值用于广播）。

- **ERROR: Effective parallel slots is zero**  
  → 你的 `GPU_PLAN` 可能全是 0；至少给一张卡正数并发。

- **看起来“没有程序”，显存却被占**  
  - 检查谁打开了该 GPU 设备：  
    ```bash
    sudo fuser -v /dev/nvidiaX
    sudo lsof /dev/nvidiaX
    ```
  - Xorg/容器/其他用户/残留进程都可能占用显存。

- **与 NCCL/通信拓扑相关的偶发问题**  
  - **优先使用方式 B**（把不用的 GPU 从 `GPU_IDS` 移除），比 0 并发更干净。

- **tmux 常用**  
  - 列表：`tmux ls`  
  - 连接：`tmux attach -t <name>`  
  - 退出会话：`exit` 或 `Ctrl-b :kill-session`  
  - 杀全部：`tmux kill-server`（慎用）

---

## ✅ Recommended patterns

- 单卡多并发（例如显存很大）：  
  ```bash
  --gpus "0" --gpu-plan "4"
  ```

- 异构并发：  
  ```bash
  --gpus "0 1 2 3" --gpu-plan "4,2,2,1"
  ```

- 保留一张卡给其他人用：  
  ```bash
  --gpus "0 1 2 3 4 5 6" --gpu-plan "3,3,3,3,0,3,3"
  # 或者：直接移除 4
  --gpus "0 1 2 3 5 6"   --gpu-plan "3,3,3,3,3,3"
  ```

---

## 📌 Implementation notes (for maintainers)

- `GPU_IDS`/`CUDA_VISIBLE_DEVICES` → `DETECTED_GPUS`  
- `GPU_PLAN` → normalize separators → array → broadcast/validate  
- Build `GPU_SLOTS` by repeating each GPU ID by its concurrency  
- `N_SLOTS = len(GPU_SLOTS)`  
- Round slicing & modulo assignment use `N_SLOTS`  
- Wrappers export `GPU_IDS` / `GPU_PLAN`, and show them via echo

---

*End of update.*


---

# 🧭 GLA 实验批量运行手册

## 📁 文件结构示例

假设你的工程路径如下：
```
/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/
│
├── gla_round_new.sh           ← 主训练脚本（已修改为支持 DATA 环境变量）
├── gla_tmux_nohup.sh          ← 单次封装（tmux + nohup + 日志）
└── gla_batch_tmux.sh          ← 批量自动运行封装
```

日志默认保存在：
```
/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/logs/
```

---

## 🚀 一、运行单次实验（tmux + nohup 自动日志）

```bash
./gla_tmux_nohup.sh --suite E2 --round all --seed 127 --data AAA
```

这会：  
- 自动创建一个 tmux 会话（名字自动生成，如 `gla_E2_all_s127_AAA_1016_2350`）  
- 日志输出到：  
  `/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/logs/gla_E2_all_s127_AAA_1016_2350.log`

可以通过以下方式查看：
```bash
tmux attach -t gla_E2_all_s127_AAA_1016_2350    # 进入 tmux
tail -f logs/gla_E2_all_s127_AAA_1016_2350.log  # 直接看日志
tmux ls                                          # 查看会话列表
tmux kill-session -t gla_E2_all_s127_AAA_1016_2350  # 结束会话
```

---

## ⚙️ 二、批量自动实验运行

连续运行多个实验（如不同 seed / data），自动顺序执行：

```bash
./gla_batch_tmux.sh --suite E2 --round all --pairs "127:AAA,87:BBB"
```

也可以用空格：
```bash
./gla_batch_tmux.sh --suite E2 --round all --pairs "127:AAA 87:BBB"
```

### ✅ 它会执行的内容

1. 在一个 tmux 会话中自动顺序执行：
   - 第 1 个实验：seed=127, data=AAA  
   - 第 2 个实验：seed=87, data=BBB
2. 每个任务都有独立日志：  
   `/logs/step1_s127_AAA_*.log`  
   `/logs/step2_s87_BBB_*.log`
3. 还有一个**总日志**：  
   `/logs/batch_E2_all_*.log`
4. 原始 `gla_round_new.sh` 不会被修改：脚本会在 `/tmp/` 创建临时副本并自动替换 `FORCE_SEED=`。

### 📂 示例输出结构

```
logs/
├── batch_E2_all_1016_2350.log          # 总日志
├── step1_s127_AAA_1016_2350.log        # 第一个实验日志
└── step2_s87_BBB_1016_2353.log         # 第二个实验日志
```

---

## 🧩 三、可选参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--suite` | 实验系列 (E1/E2/...) | `--suite E2` |
| `--round` | 实验轮次 (数字或 all) | `--round all` |
| `--seed`  | 随机种子 | `--seed 127` |
| `--data`  | 数据集代号 | `--data AAA` |
| `--pairs` | 多个 seed:data 组合 | `"127:AAA,87:BBB,42:CCC"` |
| `--name`  | 指定 tmux 会话名 | `--name exp_AAA` |
| `--logdir` | 自定义日志目录 | `--logdir /home/user/mzs_h/log` |

---

## 🧠 四、推荐工作流

### 1️⃣ 启动批量任务
```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new
./gla_batch_tmux.sh --suite E2 --round all --pairs "127:AAA,87:BBB"
```

### 2️⃣ 查看实时进度
```bash
tmux attach -t batch_E2_all_*
```

### 3️⃣ 查看日志
```bash
cd logs
tail -f batch_E2_all_*.log
tail -f step1_s127_AAA_*.log
tail -f step2_s87_BBB_*.log
```

### 4️⃣ 中断任务
```bash
tmux kill-session -t batch_E2_all_*
```

---

## 🧩 五、路径与环境变量

- 脚本中的路径均为相对路径（以 `gla_batch_tmux.sh` 所在目录为根）。  
- 你可以在外部设置：
  ```bash
  export DATA=AAA
  export SEED=127
  ```
  或直接通过参数指定。

---

## 🧹 六、清理临时文件

每个任务会在 `/tmp/` 创建临时副本（`/tmp/gla_round_XXXXXX.sh`），任务结束后自动删除。  
如果系统异常终止，可以手动清理：
```bash
rm -f /tmp/gla_round_*.sh
```

---

## 📘 七、常见问题

### ❓日志在哪里？
默认在：
```
/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/logs/
```

### ❓我可以同时跑多个 batch 吗？
可以，每个 batch 会自动起独立 tmux 会话。建议不同的 `--name`。

### ❓怎么继续看上次的日志？
```bash
tmux attach -t <会话名>
tail -f logs/<logfile>.log
```

---

作者：**ChatGPT 自动生成**  
更新时间：2025-10-16
