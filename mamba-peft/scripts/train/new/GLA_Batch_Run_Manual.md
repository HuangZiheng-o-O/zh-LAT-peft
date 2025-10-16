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
