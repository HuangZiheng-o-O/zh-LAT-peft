from abc import abstractmethod, ABC
import random
import transformers
import torch
from pathlib import Path
from tqdm import tqdm
import pickle
import os
import time
import datetime

def _debug_print(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DEBUG][{ts}] [base.py] {msg}", flush=True)

from utils.parallel_processor_fs import ParallelProcessorFS


class DatasetBase(ABC):
    shuffle_seeds = [
        123,
        321,
        532,
        523,
    ]
    
    def __init__(self, tokenizer: transformers.AutoTokenizer, path: str, split="train", prompt_prefix=None,
                 use_cache=True, num_parallel_workers=16, subset_size=None, mode="lm", max_seqlen=None):
        _debug_print(f"DatasetBase.__init__ START: path={path}, split={split}, use_cache={use_cache}, num_parallel_workers={num_parallel_workers}")
        super().__init__()

        self.path = path
        self.split = split

        self.sep = "###"
        self.eot = "<|endoftext|>"
        self.tokenizer = tokenizer  
        self.ignore_index = -100
        self.data = None
        self.prompt_prefix = prompt_prefix
        self.prompt_prefix_ids = None
        self.mode = mode
        self.max_seqlen = max_seqlen
        if use_cache:
            cache_file_stem = self.get_cache_name()
            _debug_print(f"  cache_file_stem = {cache_file_stem}")

            if subset_size is not None:
                cache_file_stem += f"_{subset_size}"

            cache_file = Path("data") / path.replace("/", "_") / f"{cache_file_stem}.pkl"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            lock_file = cache_file.with_suffix(cache_file.suffix + ".lock")
            _debug_print(f"  cache_file = {cache_file}")
            _debug_print(f"  lock_file = {lock_file}")
            _debug_print(f"  cache_file.exists() = {cache_file.exists()}, lock_file.exists() = {lock_file.exists()}")

            # Fast-path: cache already exists and no writer lock → just load
            if cache_file.exists() and not lock_file.exists():
                _debug_print(f"  FAST-PATH: Loading from cache...")
                with open(cache_file, "rb") as f:
                    self.data = pickle.load(f)
                _debug_print(f"  FAST-PATH: Loaded {len(self.data)} samples from cache")
            else:
                _debug_print(f"  SLOW-PATH: Cache miss or locked, need to build data...")
                # Cooperative lock to prevent multiple processes writing the same cache concurrently
                got_lock = False
                try:
                    fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    with os.fdopen(fd, "w") as lf:
                        lf.write(f"pid={os.getpid()} time={time.time()}\n")
                    got_lock = True
                    _debug_print(f"  Acquired lock (pid={os.getpid()})")
                except FileExistsError:
                    got_lock = False
                    _debug_print(f"  Lock already held by another process")

                if got_lock:
                    try:
                        if num_parallel_workers > 0:
                            assert subset_size is None
                            _debug_print(f"  Calling len(self) to get dataset size...")
                            dataset_len = len(self)
                            _debug_print(f"  len(self) = {dataset_len}")
                            data_ind = list(range(dataset_len))
                            _debug_print(f"  Starting ParallelProcessorFS with {num_parallel_workers} workers...")
                            # ParallelProcessorFS is responsible for atomic writes to cache_file
                            self.data = ParallelProcessorFS(self.preproc, len(data_ind), num_parallel_workers, cache_file).run()
                            _debug_print(f"  ParallelProcessorFS completed, got {len(self.data)} samples")
                        else:
                            _debug_print(f"  Sequential processing (num_parallel_workers=0)...")
                            data_ind = list(range(len(self)))
                            if subset_size is not None:
                                random.Random(0).shuffle(data_ind)
                                data_ind = data_ind[:subset_size]
                            self.data = [self.preproc(idx) for idx in tqdm(data_ind)]
                            self.data = [d for d in self.data if d is not None]
                            # Atomic write final cache
                            tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
                            with open(tmp_file, "wb") as f:
                                pickle.dump(self.data, f)
                            os.replace(tmp_file, cache_file)
                            _debug_print(f"  Sequential processing done, got {len(self.data)} samples")
                    finally:
                        # Release lock
                        try:
                            if lock_file.exists():
                                os.remove(lock_file)
                                _debug_print(f"  Released lock")
                        except Exception:
                            pass
                else:
                    # Waiter: spin until cache file materializes (another process is writing it)
                    # But detect stale locks: if lock is older than 10 minutes, assume it's stale and remove it
                    _debug_print(f"  WAITER: Spinning until cache file appears (lock held by another process)...")
                    spin_count = 0
                    max_lock_age_seconds = 600  # 10 minutes
                    while lock_file.exists() or not cache_file.exists():
                        spin_count += 1
                        # Check for stale lock
                        if lock_file.exists():
                            try:
                                lock_age = time.time() - os.path.getmtime(str(lock_file))
                                if lock_age > max_lock_age_seconds:
                                    _debug_print(f"  STALE LOCK DETECTED: lock is {lock_age:.0f}s old (>{max_lock_age_seconds}s), removing...")
                                    os.remove(lock_file)
                                    _debug_print(f"  Stale lock removed, will try to acquire lock on next iteration")
                                    # Don't break here; let the loop re-check and potentially acquire lock
                            except Exception as e:
                                _debug_print(f"  Error checking lock age: {e}")
                        if spin_count % 5 == 0:
                            lock_age_str = ""
                            if lock_file.exists():
                                try:
                                    lock_age_str = f", lock_age={time.time() - os.path.getmtime(str(lock_file)):.0f}s"
                                except Exception:
                                    pass
                            _debug_print(f"    Still waiting... (spin_count={spin_count}, lock_exists={lock_file.exists()}, cache_exists={cache_file.exists()}{lock_age_str})")
                        time.sleep(2)
                    _debug_print(f"  WAITER: Cache file ready, loading...")
                    with open(cache_file, "rb") as f:
                        self.data = pickle.load(f)
                    _debug_print(f"  WAITER: Loaded {len(self.data)} samples")
        else:
            _debug_print(f"  NO-CACHE mode: Building data in-memory...")
            # Build data in-memory without writing cache
            data_ind = list(range(len(self)))
            if subset_size is not None:
                random.Random(0).shuffle(data_ind)
                data_ind = data_ind[:subset_size]
            self.data = [self.preproc(idx) for idx in tqdm(data_ind)]
            self.data = [d for d in self.data if d is not None]
            _debug_print(f"  NO-CACHE mode: Built {len(self.data)} samples")
        
        _debug_print(f"DatasetBase.__init__ DONE")

    def _ensure_materialized(self):
        """Lazily load/build self.data if it is None (e.g., after unpickling in a different context)."""
        if self.data is not None:
            return
        cache_file_stem = self.get_cache_name()
        cache_file = Path("data") / self.path.replace("/", "_") / f"{cache_file_stem}.pkl"
        try:
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    self.data = pickle.load(f)
            else:
                # Fallback: build in-memory without cache
                data_ind = list(range(len(self)))
                self.data = [self.preproc(idx) for idx in data_ind]
                self.data = [d for d in self.data if d is not None]
        except Exception:
            # As last resort, mark empty to avoid NoneType errors; caller will handle emptiness.
            self.data = []

    def get_cache_name(self):
        base = f"cache_{self.path.replace('/', ' ')}_{self.split}"
        # Optional namespacing to avoid cross-job collisions (set in env)
        tag = os.environ.get("DATA_CACHE_TAG") or os.environ.get("CACHE_TAG")
        if tag:
            # sanitize tag
            safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(tag))
            if safe:
                base += f"_{safe}"
        return base

    def encode(self, seq):
        return torch.LongTensor(self.tokenizer.encode(seq))

    def preproc(self, idx):
        input, label = self.get_input_label(idx)
        
        # Handle case where get_input_label returns (None, None) for invalid samples
        if input is None or label is None:
            return None
        
        input_prepoc, label_preproc = self.preproc_input_label(input, label)
        input_ids, label_ids = self.encode(input_prepoc), self.encode(label_preproc)

        if self.max_seqlen is not None and (input_ids.shape[0] + label_ids.shape[0]) > self.max_seqlen:
            return None

        return input_ids, label_ids
    
    def get_ids(self, idx):
        # Guard against None in edge cases
        if self.data is None:
            self._ensure_materialized()
        return self.data[idx]

    def __getitem__(self, idx):
        input_ids, label_ids = self.get_ids(idx)

        if self.prompt_prefix is not None:
            if self.prompt_prefix_ids is None:
                self.prompt_prefix_ids = self.encode(self.prompt_prefix)

            input_ids = torch.cat([self.prompt_prefix_ids, input_ids])

        if self.mode == "lm":
            ids = torch.cat([input_ids, label_ids])
            label_len = label_ids.shape[0]

            input_ids = ids[:-1]
            label_ids = torch.nn.functional.pad(ids[-label_len:], (input_ids.shape[0] - label_len, 0), value=self.ignore_index)
        elif self.mode == "gen":
            pass
        else:
            raise Exception(self.mode)
        
        return dict(input_ids=input_ids, label_ids=label_ids)

    @abstractmethod
    def get_input_label(self, idx):
        pass

    @abstractmethod
    def preproc_input_label(self, input, label):
        pass

    @abstractmethod
    def compute_metrics(self, eval_preds):
        pass


class NluDatasetBase(DatasetBase):
    def label_int_to_str(self, label):
        assert 0 <= label <= 9
        return str(label)
    
    def label_str_to_int(self, label):
        return int(label)
    
    def preproc_input_label(self, input, label):
        if isinstance(label, int):
            label = self.label_int_to_str(label)
        sep = getattr(self.tokenizer, "sep_token", None) or getattr(self.tokenizer, "eos_token", None) or ""
        return input + sep, label
    
    # workaround for old cache file, which store input and label concatenated
    def get_ids(self, idx):
        if self.data is None:
            self._ensure_materialized()
        sample = self.data[idx]
    
        if not isinstance(sample, (tuple, list)):
            input_ids, label_ids = sample[:-1], sample[-1:]
        else:
            input_ids, label_ids = sample

        return input_ids, label_ids


class NlgDatasetBase(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preproc_input_label(self, input, label):
        sep = getattr(self.tokenizer, "sep_token", None) or getattr(self.tokenizer, "eos_token", None) or ""
        eos = getattr(self.tokenizer, "eos_token", None) or ""
        return input + sep, label + eos
    