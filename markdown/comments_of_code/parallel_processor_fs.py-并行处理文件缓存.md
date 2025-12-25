# 并行处理文件缓存

**User:** ABC ABC (northeasternuniversitychina@gmail.com)  
**Created:** 12/25/2025 0:27:35  
**Updated:** 12/25/2025 0:28:26  
**Exported:** 12/25/2025 0:33:32  
**Link:** [https://chatgpt.com/c/694ccb3f-61cc-8328-8ac2-b5d3837708bb](https://chatgpt.com/c/694ccb3f-61cc-8328-8ac2-b5d3837708bb)  



  > ###  数据集系统：`dataset/__init__.py`
  > 功能：统一数据集加载入口
  >
  > 使用的具体数据集文件：
  > - dataset/base.py - 基础数据集类（DatasetBase, NluDatasetBase, NlgDatasetBase）
  > - ...
  > - utils/parallel_processor_fs.py - 并行数据预处理（通过 `base.py` 导入）

  总结来说：

  mamba-peft/utils/parallel_processor_fs.py 被 mamba-peft/dataset/base.py 文件导入和使用。

  它的核心功能是在数据预处理阶段，例如当需要对成千上万个数据样本进行 tokenize 或者其他转换时，利用多进程来并行处理，从而极大地加快数据加载和准备的速度。

  所以，它在调用链中的位置是：
  train_gla_only.py → load_dataset (in dataset/__init__.py) → DatasetBase 或其子类 (in dataset/base.py) → ParallelProcessorFS (in utils/parallel_processor_fs.py)



## Response:
下面是在你给出的代码基础上，补充了“非常详细”的中文注释版本（尽量覆盖：设计意图、并发细节、文件一致性/原子写、进度条策略、潜在陷阱与可改进点）。代码逻辑未改动，仅增加注释与少量不影响语义的说明性注释。

```python
from multiprocessing import Process, Value
from pathlib import Path
import pickle
from tqdm import tqdm
import os
import time

class ParallelProcessorFS:
    """
    一个“基于文件系统落盘缓存”的多进程并行处理器。

    典型用途：
      - 你有一个函数 func(idx) 可以处理第 idx 个样本/任务，并返回结果
      - 总任务数为 size（idx 范围 [0, size)）
      - 希望用 n 个进程并行跑
      - 每个进程将自己的结果写到独立的 part 文件（pickle），最终再聚合为一个总输出文件

    关键特点：
      1) 任务分配采用共享计数器 counter（multiprocessing.Value），所有进程抢占式获取下一个 idx
      2) 每个 worker 生成一个字典 out: {idx: result}，最终落盘到 worker_files[worker_idx]
      3) 聚合阶段把多个 part 文件中的结果合并到 output_all（长度为 size 的列表）
      4) 写文件采用“先写临时文件，再 os.replace 原子替换”的方式，避免进程中途崩溃导致半写文件/损坏文件
    """

    def __init__(self, func, size, n, output_file) -> None:
        """
        参数说明：
          func: 处理函数，签名建议为 func(idx) -> 任意可 pickle 的对象
          size: 总任务数量（idx 从 0 到 size-1）
          n: 并行进程数
          output_file: 最终聚合输出文件路径（pickle 文件）

        文件组织：
          output_file:         最终聚合输出，例如 results.pkl
          cache_path/parts/:   中间 part 文件目录
          worker_files:        n 个 part 文件，例如 results_part_000.pkl ... results_part_{n-1}.pkl
        """
        self.func = func
        self.size = size
        self.n = n

        # 将 output_file 转为 Path，便于跨平台路径操作与拼接
        self.output_file = Path(output_file)

        # part 文件统一放在 output_file 同级目录下的 parts/ 目录
        self.cache_path = self.output_file.parent / "parts"

        # 注意：这里用的是 output_file.stem（不带后缀的文件名）作为 part 前缀
        # worker_files 是长度为 n 的列表，每个进程写其中一个对应文件
        self.worker_files = [
            self.cache_path / f"{output_file.stem}_part_{i:03d}.pkl"
            for i in range(n)
        ]

        # 确保输出目录存在
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # 确保 parts 目录存在
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def _worker(self, worker_idx, counter):
        """
        子进程执行函数（worker）。

        worker 的核心流程：
          - 维护一个 out 字典保存本进程处理过的 idx -> result
          - 通过共享 counter 原子地领取任务 idx
          - 调用 self.func(idx) 得到结果写入 out
          - 出现异常则 out[idx] = None，并打印栈信息
          - 结束后将 out 以 pickle 写到对应 part 文件（原子替换写入）

        参数：
          worker_idx: 当前 worker 的编号（0..n-1），用于选择写入哪个 part 文件
          counter: multiprocessing.Value("i", 0)，共享整型计数器，表示下一个待处理 idx
        """
        # 每个进程只维护自己的结果映射，避免跨进程共享复杂对象
        out = {}

        # 进度条策略：
        #   - tqdm 多进程同时显示会导致输出错乱，所以只让 worker 0 显示进度条
        #   - 其它 worker 不显示 pbar
        pbar = tqdm(total=self.size, desc="Parallel processing", position=0) if worker_idx == 0 else None

        # idx_last 用于计算 pbar 每次更新多少（当前 idx - 上一次 idx）
        # 这里的写法实际上容易出现“更新不准确”（因为 idx 并不一定连续被 worker0 领取）
        # 但总体上能体现增长趋势。
        idx_last = 0

        # processed_count：当前 worker 实际成功/失败处理的样本数量计数（不区分成功与失败都 +1）
        processed_count = 0

        # start_time：用于统计 worker 的吞吐率
        start_time = time.time()

        # first_call_logged：用于记录首次调用耗时（每个 worker 各自记录一次）
        first_call_logged = False

        while True:
            # 领取任务 idx 必须加锁，否则多个进程会读到相同的 counter.value 导致重复处理
            # counter.get_lock() 提供一个跨进程同步锁
            with counter.get_lock():
                idx = counter.value

                # 如果 idx 已经达到 size，说明所有任务都领取完了，worker 退出
                if idx >= self.size:
                    break

                # 将共享计数器加 1，把“下一个任务”指针推进
                counter.value += 1

            try:
                # 计时：统计单个样本的处理耗时
                t0 = time.time()

                # 执行用户函数：核心计算/IO 发生在这里
                # 注意：func 必须是可在子进程中调用的函数（可 pickle / 可导入），
                # 在 Windows 上尤其需要注意 if __name__ == "__main__" 保护
                out[idx] = self.func(idx)

                elapsed = time.time() - t0
                processed_count += 1

                # 打印“首次样本处理耗时”：便于判断是否存在初始化开销（如模型加载、首次编译等）
                if not first_call_logged:
                    print(f"[Worker {worker_idx}] First sample processed in {elapsed:.3f}s", flush=True)
                    first_call_logged = True

                # 仅 worker0 每处理 5000 个样本打印一次整体吞吐率，减少日志噪音
                elif worker_idx == 0 and processed_count % 5000 == 0:
                    total_elapsed = time.time() - start_time
                    rate = processed_count / total_elapsed if total_elapsed > 0 else 0
                    print(f"[Worker 0] Processed {processed_count} samples, rate={rate:.1f} samples/s", flush=True)

            except Exception as e:
                # 如果 func(idx) 抛异常：
                #   - 打印异常类型与信息
                #   - 打印 traceback 方便定位具体错误发生位置
                #   - 将 out[idx] 设置为 None，表示该 idx 失败
                #     后续聚合时会警告并过滤 None
                print(f"[Worker {worker_idx}] Error processing idx={idx}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                out[idx] = None

            # 进度条更新（仅 worker0）
            if pbar is not None:
                # 这里使用 idx - idx_last 来更新，但由于 idx 的领取是“抢占式”，
                # worker0 的 idx 可能跳跃（例如从 10 跳到 50），会导致 pbar 看起来快进。
                # 如果你想要严格准确的进度，建议用一个共享的“已完成计数器”，每完成一个就 +1。
                pbar.update(idx - idx_last)
                idx_last = idx

        # worker 即将结束：关闭进度条，避免 tqdm 残留
        if pbar is not None:
            pbar.close()

        # 输出本 worker 的统计信息
        total_elapsed = time.time() - start_time
        rate = processed_count / total_elapsed if total_elapsed > 0 else 0
        print(
            f"[Worker {worker_idx}] Done: {processed_count} samples in {total_elapsed:.1f}s ({rate:.1f} samples/s)",
            flush=True
        )

        # 将本 worker 的 out 写到对应 part 文件
        final_path = self.worker_files[worker_idx]

        # 原子写：先写 tmp 文件，再替换到 final_path
        # 目的：
        #   - 避免写到一半进程崩溃导致 final_path 是损坏/半截的 pickle
        #   - os.replace 在大多数平台上是原子操作（同一文件系统内）
        tmp_path = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")

        with open(tmp_path, "wb") as f:
            pickle.dump(out, f)

        # 原子替换：如果 final_path 已存在，会被直接替换
        os.replace(tmp_path, final_path)

        print(f"Wrote {final_path}")

    def aggregate_result(self):
        """
        将 n 个 part 文件聚合为一个结果列表 output_all，并写入 self.output_file。

        聚合逻辑：
          - output_all 初始化为长度 size 的列表，默认 None
          - 逐个读取 worker_files（每个是 dict: idx -> value）
          - 对应位置填充 output_all[idx] = value
          - 统计 None 数量并警告
          - 过滤掉 None 得到最终 output_all
          - 将最终 output_all 写到 output_file（原子写）
        """
        # 先用 None 占位，方便按 idx 直接定位写入
        output_all = [None] * self.size

        # 依次读每个 worker 的 part 文件
        for worker_file in tqdm(self.worker_files, desc="Aggregating"):
            with open(worker_file, "rb") as f:
                out = pickle.load(f)

            # out 是 dict: {idx: value}
            # 注意：如果某个 idx 被重复处理（理论上不应该），后读到的会覆盖先读到的
            for k, v in out.items():
                output_all[k] = v

        # 统计失败/缺失结果数量（None）
        none_count = sum(1 for o in output_all if o is None)
        if none_count > 0:
            print(f"Warning: {none_count}/{self.size} samples returned None (will be filtered out)")

        # 过滤掉 None，得到最终有效结果列表
        # 语义注意：过滤会改变结果长度，不再与 idx 一一对应
        output_all = [o for o in output_all if o is not None]

        # 如果全部都是 None，给出显式错误提示
        if len(output_all) == 0:
            print(f"ERROR: All {self.size} samples were filtered out (all returned None)")
            print(f"Check worker logs above for errors during processing")

        # 最终输出文件也采用原子写，避免聚合写到一半程序中止导致 output_file 损坏
        tmp_path = self.output_file.with_suffix(self.output_file.suffix + f".tmp.{os.getpid()}")
        with open(tmp_path, "wb") as f:
            pickle.dump(output_all, f)
        os.replace(tmp_path, self.output_file)

        return output_all

    def run(self):
        """
        运行入口：
          1) 初始化共享计数器 counter=0
          2) 创建 n 个进程，每个进程执行 _worker(i, counter)
          3) start 所有进程
          4) join 等待所有进程结束
          5) 聚合 part 文件并返回最终结果
        """
        # 共享整型计数器，"i" 表示 C int
        # counter.value 表示下一个待领取的 idx
        counter = Value("i", 0)

        # 创建 n 个进程
        procs = [Process(target=self._worker, args=(i, counter)) for i in range(self.n)]

        # 启动进程
        for proc in procs:
            proc.start()

        # 等待所有进程结束
        for proc in procs:
            proc.join()

        print("Aggregating...")
        return self.aggregate_result()
```

如果你希望注释进一步“更偏工程化”，我可以在不改功能的前提下，补充一段“常见问题与改进建议”注释块（例如：Windows spawn 的注意事项、tqdm 更新方式的准确实现、异常重试策略、断点续跑/跳过已存在 part 文件、内存占用从 dict 改为分批落盘等）。

这段代码从一个“基础原型”进化为了一个**“生产级”健壮版本**。

核心的变化在于：**代码不再假设一切都会顺利运行**。它增加了防御性编程、详细的监控和防止数据损坏的机制。

具体来说，主要改变了以下 **4 个方面**：

### 1. 文件写入安全性：增加了“原子写入”机制

这是最关键的改动，防止文件损坏。

- **旧代码：** 直接打开最终文件写入 `pickle.dump(out, f)`。
  - *风险：* 如果在写入过程中程序崩溃、断电或被强制停止（Ctrl+C），磁盘上会留下只有一半数据的损坏文件，下次读取时会报错且无法恢复。
- **新代码：** 先写入临时文件 `.tmp`，写完后再重命名 `os.replace`。
  - *优势：* `os.replace` 在大多数系统上是原子操作。这意味着文件要么完全写好存在，要么根本不存在，绝不会出现“半个文件”的情况。

Python

```
# 新增逻辑：
tmp_path = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
with open(tmp_path, "wb") as f:
    pickle.dump(out, f)
os.replace(tmp_path, final_path) # 瞬间替换，确保安全
```

### 2. 异常处理：单个任务失败不会炸毁整个进程

- **旧代码：** 没有 `try...except`。
  - *风险：* 如果处理第 5000 个样本时 `self.func(idx)` 报错（比如除以零），整个 Worker 进程会直接挂掉，该 Worker 之前处理的所有结果（内存中）都会丢失，甚至导致主程序卡死。
- **新代码：** 包裹了 `try...except`。
  - *优势：* 如果某个样本出错，程序会捕获异常，打印错误堆栈，将该结果标记为 `None`，然后**继续处理下一个任务**。这保证了程序的鲁棒性。

Python

```
try:
    out[idx] = self.func(idx)
except Exception as e:
    # 打印错误但不退出
    print(f"[Worker {worker_idx}] Error... {e}")
    out[idx] = None 
```

### 3. 可观测性：增加了详细的进度监控和耗时统计

- **旧代码：** 只有一个 `tqdm` 进度条。
  - *局限：* 你不知道具体的处理速度（每秒处理多少个），也不知道是不是卡住了。
- **新代码：**
  - **首个样本计时：** 让你立即知道单个任务大概需要多久（`First sample processed in ...`）。
  - **速率打印：** Worker 0 会定期打印 `rate=xx samples/s`，方便预估剩余时间。
  - **最终统计：** Worker 结束时会汇总报告总耗时。

### 4. 数据完整性检查：聚合时的校验

- **旧代码：** 简单地过滤掉 `None`，不报错。
- **新代码：**
  - **警告统计：** 如果有样本返回了 `None`（说明处理出错了），会显式打印 `Warning: X/Y samples returned None`。
  - **全军覆没检查：** 如果所有样本都被过滤掉了，会打印 `ERROR` 提示用户检查日志。

------

### 总结

| **特性**     | **旧版本**           | **新版本**                          |
| ------------ | -------------------- | ----------------------------------- |
| **文件写入** | 直接写入，断电会坏档 | **原子写入** (tmp -> rename)，安全  |
| **遇到Bug**  | 进程崩溃，数据丢失   | **捕获异常**，记录为 None，继续运行 |
| **监控**     | 仅有进度条           | **首帧耗时、处理速率、详细日志**    |
| **调试**     | 出错不知道哪错了     | **打印详细 Traceback**              |
| **结果聚合** | 默默过滤             | **统计失败数量并发出警告**          |

**一句话总结：** 这个改动把一个“能跑就行”的脚本，变成了一个可以**放心跑在服务器上处理大规模数据**的工具。



