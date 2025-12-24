from multiprocessing import Process, Value
from pathlib import Path
import pickle
from tqdm import tqdm
import os
import time


class ParallelProcessorFS:
    """
    一个基于文件系统（FS）的多进程并行预处理器。

    设计目标：
    1) 将 size 个样本（按 idx=0..size-1）分配给 n 个 worker 进程并行处理；
    2) 每个 worker 把自己的 idx->result 字典写到一个独立的 part 文件（parts/xxx_part_000.pkl）；
    3) 主进程 join 所有 worker 后，再把所有 part 文件读回来合并，形成最终缓存 output_file。

    关键工程点：
    - 使用 multiprocessing.Value 作为共享计数器，保证每个 idx 只被处理一次；
    - 对 worker part 文件和最终 output_file 都使用“临时文件 + os.replace”实现原子写，
      防止半写入/崩溃导致缓存损坏；
    - 对单条样本处理异常进行捕获并记录，失败样本用 None 标记；
    - 聚合阶段会过滤掉 None，避免后续训练/加载报错（但会引入长度变化的副作用，见下文注释）。
    """

    def __init__(self, func, size, n, output_file) -> None:
        """
        Args:
            func: 预处理函数，签名通常为 func(idx) -> 任意可pickle对象（dict/tuple等）。
                  这里假设 func 是确定性的（给定 idx 与环境配置固定时输出一致），否则会影响复现性。
            size: 总样本数，idx 空间为 [0, size)。
            n: worker 数量（并行进程数）。
            output_file: 最终聚合缓存文件路径（pickle），例如 data/xxx/cache_train.pkl。
        """
        self.func = func
        self.size = size
        self.n = n
        self.output_file = Path(output_file)

        # parts 目录：用于存放每个 worker 的分片输出，便于崩溃恢复/调试
        # 注意：如果同一路径被多个“并行任务”同时运行，会发生互相覆盖，导致不一致/损坏。
        self.cache_path = self.output_file.parent / "parts"

        # 每个 worker 对应一个唯一 part 文件（按 worker_idx 命名）
        # 文件名使用 output_file.stem 作为前缀，避免不同数据集/不同split混淆（前提是 output_file 命名本身足够唯一）
        self.worker_files = [
            self.cache_path / f"{output_file.stem}_part_{i:03d}.pkl"
            for i in range(n)
        ]

        # 确保输出目录存在
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def _worker(self, worker_idx, counter):
        """
        单个 worker 的执行函数。

        核心并发机制：
        - counter 是 multiprocessing.Value("i", 0)，所有进程共享；
        - 每次取 idx 都在 counter.get_lock() 下进行，确保 idx 分配互斥；
        - 取到 idx 后立刻 counter.value += 1，保证每个 idx 只被某一个 worker 处理。

        输出：
        - out: dict[int, Any]，键是样本 idx，值是 func(idx) 的输出；失败则为 None。
        - 最终把 out 写到本 worker 的 part 文件中（原子替换）。
        """
        out = {}

        # 仅 worker 0 展示 tqdm 进度条（避免多进程同时输出导致终端乱）
        # 注意：这里的 pbar 总量是 self.size，但 worker 0 只处理其中一部分 idx，
        #       所以这个进度条本质是“全局进度的近似展示”，不是严格的每个 idx 1 次更新。
        pbar = tqdm(total=self.size, desc="Parallel processing", position=0) if worker_idx == 0 else None
        idx_last = 0  # 用于计算 pbar.update 的增量
        processed_count = 0
        start_time = time.time()

        # Debug：记录 worker 首次调用 func 的耗时（通常可用于判断 tokenizer/model 初始化等开销）
        first_call_logged = False

        while True:
            # 关键：通过共享 counter 分配任务 idx
            with counter.get_lock():
                idx = counter.value

                # 当 idx 达到 size，所有样本都已分配完毕，worker 退出循环
                if idx >= self.size:
                    break

                # 先递增再释放锁：确保其他 worker 不会拿到同一个 idx
                counter.value += 1

            # 处理样本：任何异常都捕获，避免 worker 直接崩溃导致整体任务卡死或缺 part 文件
            try:
                t0 = time.time()
                out[idx] = self.func(idx)  # 这里是最主要的业务逻辑入口
                elapsed = time.time() - t0
                processed_count += 1

                # 打印首样本耗时：常用于排查“第一条特别慢”或“初始化开销”
                if not first_call_logged:
                    print(f"[Worker {worker_idx}] First sample processed in {elapsed:.3f}s", flush=True)
                    first_call_logged = True

                # 定期打印吞吐（仅 worker 0）：用于观察整体处理速度趋势/是否卡顿
                elif worker_idx == 0 and processed_count % 5000 == 0:
                    total_elapsed = time.time() - start_time
                    rate = processed_count / total_elapsed if total_elapsed > 0 else 0
                    print(f"[Worker 0] Processed {processed_count} samples, rate={rate:.1f} samples/s", flush=True)

            except Exception as e:
                # 失败样本记为 None，聚合时会被过滤掉
                # 工程含义：
                # - 优点：单样本异常不影响整体预处理完成；
                # - 风险：过滤 None 会改变最终数据长度，且 idx 顺序/对齐关系丢失（见 aggregate_result 注释）。
                print(f"[Worker {worker_idx}] Error processing idx={idx}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                out[idx] = None

            # 进度条更新逻辑（仅 worker 0）
            # 注意：idx 分配是全局递增的，但 worker 0 并不会处理所有 idx。
            # 这里用 idx-idx_last 作为增量，能让进度条大体跟随“全局 idx 分配”的推进。
            if pbar is not None:
                pbar.update(idx - idx_last)
                idx_last = idx

        if pbar is not None:
            pbar.close()

        # worker 完成统计输出
        total_elapsed = time.time() - start_time
        rate = processed_count / total_elapsed if total_elapsed > 0 else 0
        print(f"[Worker {worker_idx}] Done: {processed_count} samples in {total_elapsed:.1f}s ({rate:.1f} samples/s)", flush=True)

        # === 原子写 part 文件 ===
        # 关键点：
        # 1) 先写 tmp（包含 pid，避免多个进程冲突）；
        # 2) 写完后 os.replace(tmp, final)，在同一文件系统中是原子替换；
        # 3) 读者要么看到旧文件，要么看到新文件，不会看到半写入。
        final_path = self.worker_files[worker_idx]
        tmp_path = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
        with open(tmp_path, "wb") as f:
            pickle.dump(out, f)
        os.replace(tmp_path, final_path)

        print(f"Wrote {final_path}")

    def aggregate_result(self):
        """
        聚合所有 worker 的 part 文件，生成最终缓存 self.output_file。

        实现策略：
        - 先创建 output_all=[None]*size，用 idx 作为位置把结果放回去；
          这样可以检测哪些 idx 缺失/失败（仍为 None）。
        - 再统计 None 个数并打印 warning；
        - 再过滤掉 None，得到最终 output_all（注意：长度会变短，且 idx 对齐语义丢失）。

        重要工程注意事项（强烈建议维护者知晓）：
        1) 你这里“过滤 None”会改变数据长度：
           - 若下游依赖“第 k 条对应原始 idx=k”，则会被破坏；
           - 若下游只需要“一个可训练的样本列表”，则通常没问题。
           因此你在上层最好有“完整性校验/失败比例阈值”，否则 silently drop 会掩盖问题。
        2) 当前实现没有显式检查 worker_file 是否存在：
           - 若某个 worker 崩溃且没有写出 part 文件，这里 open 会直接抛异常。
           - 你们第 4 周的“间歇性损坏”中，这类情况是高风险点之一。
        """
        output_all = [None] * self.size

        # 逐个读取 worker 的 part 文件并合并
        for worker_file in tqdm(self.worker_files, desc="Aggregating"):
            with open(worker_file, "rb") as f:
                out = pickle.load(f)  # out 是 dict: idx -> result/None

            # 将每个 idx 放回 output_all 的对应位置
            for k, v in out.items():
                output_all[k] = v

        # 统计失败样本（None）数量
        none_count = sum(1 for o in output_all if o is None)
        if none_count > 0:
            print(f"Warning: {none_count}/{self.size} samples returned None (will be filtered out)")

        # 过滤 None：得到最终样本列表
        # 副作用：丢失原始 idx 对齐关系，且最终长度 < size
        output_all = [o for o in output_all if o is not None]

        # 如果全都失败，给出明显错误提示（但仍会继续写空/很小文件；如需强制失败可 raise）
        if len(output_all) == 0:
            print(f"ERROR: All {self.size} samples were filtered out (all returned None)")
            print(f"Check worker logs above for errors during processing")

        # === 原子写最终聚合缓存 ===
        # 目的：避免写到一半被中断导致 cache 损坏（pickle EOF、UnpicklingError 等）。
        tmp_path = self.output_file.with_suffix(self.output_file.suffix + f".tmp.{os.getpid()}")
        with open(tmp_path, "wb") as f:
            pickle.dump(output_all, f)
        os.replace(tmp_path, self.output_file)

        return output_all

    def run(self):
        """
        启动并行处理并聚合结果。

        流程：
        1) 初始化共享 counter=0；
        2) 启动 n 个 Process，每个执行 _worker(i, counter)；
        3) join 等待所有 worker 完成；
        4) 聚合 part 文件，写最终缓存并返回。

        工程风险点（建议后续增强）：
        - 当前 join 后不检查 proc.exitcode：
          如果某个 worker 异常退出（exitcode != 0），仍会进入 aggregate_result，
          此时可能缺 part 文件或 part 文件不完整，导致聚合阶段异常。
        - parts 目录未清理：
          如果同名 output_file 多次运行，旧的 part 文件可能残留；
          你当前实现会覆盖同名 part，但如果 worker 数 n 改变，可能出现“多余旧 part”的迷惑性状态。
        """
        counter = Value("i", 0)

        # 创建 worker 进程
        procs = [Process(target=self._worker, args=(i, counter)) for i in range(self.n)]

        # 启动所有 worker
        for proc in procs:
            proc.start()

        # 等待所有 worker 结束
        for proc in procs:
            proc.join()

        print("Aggregating...")
        return self.aggregate_result()