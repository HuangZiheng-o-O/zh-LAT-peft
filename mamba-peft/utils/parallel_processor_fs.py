

from multiprocessing import Process, Value
from pathlib import Path
import pickle
from tqdm import tqdm
import os


class ParallelProcessorFS:
    def __init__(self, func, size, n, output_file) -> None:
        self.func = func
        self.size = size
        self.n = n
        self.output_file = Path(output_file)
        self.cache_path = self.output_file.parent / "parts"
        self.worker_files = [self.cache_path / f"{output_file.stem}_part_{i:03d}.pkl" for i in range(n)]

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def _worker(self, worker_idx, counter):
        out = {}

        pbar = tqdm(total=self.size, desc="Parallel processing") if worker_idx == 0 else None
        idx_last = 0

        while True:
            with counter.get_lock():
                idx = counter.value

                if idx >= self.size:
                    break

                counter.value += 1

            try:
                out[idx] = self.func(idx)
            except Exception as e:
                print(f"[Worker {worker_idx}] Error processing idx={idx}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                out[idx] = None

            if pbar is not None:
                pbar.update(idx - idx_last)
                idx_last = idx

        if pbar is not None:
            pbar.close()
        # Atomic write: write to tmp then replace
        final_path = self.worker_files[worker_idx]
        tmp_path = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
        with open(tmp_path, "wb") as f:
            pickle.dump(out, f)
        os.replace(tmp_path, final_path)
        print(f"Wrote {final_path}")

    def aggregate_result(self):
        output_all = [None] * self.size

        for worker_file in tqdm(self.worker_files, desc="Aggregating"):
            with open(worker_file, "rb") as f:
                out = pickle.load(f)
            for k, v in out.items():
                output_all[k] = v

        # Count None values before filtering
        none_count = sum(1 for o in output_all if o is None)
        if none_count > 0:
            print(f"Warning: {none_count}/{self.size} samples returned None (will be filtered out)")
        
        output_all = [o for o in output_all if o is not None]
        
        if len(output_all) == 0:
            print(f"ERROR: All {self.size} samples were filtered out (all returned None)")
            print(f"Check worker logs above for errors during processing")

        with open(self.output_file, "wb") as f:
            pickle.dump(output_all, f)

        return output_all
            

    def run(self):
        counter = Value("i", 0)

        procs = [Process(target=self._worker, args=(i, counter)) for i in range(self.n)]

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

        print("Aggregating...")
        return self.aggregate_result()
    