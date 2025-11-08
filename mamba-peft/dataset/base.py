from abc import abstractmethod, ABC
import random
import transformers
import torch
from pathlib import Path
from tqdm import tqdm
import pickle
import os
import time

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

            if subset_size is not None:
                cache_file_stem += f"_{subset_size}"

            cache_file = Path("data") / path.replace("/", "_") / f"{cache_file_stem}.pkl"
            if not cache_file.exists():
                # Cooperative lock to prevent multiple processes writing the same cache concurrently
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                lock_file = cache_file.with_suffix(cache_file.suffix + ".lock")

                got_lock = False
                try:
                    # O_EXCL to ensure single writer
                    fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    with os.fdopen(fd, "w") as lf:
                        lf.write(f"pid={os.getpid()} time={time.time()}\n")
                    got_lock = True
                except FileExistsError:
                    got_lock = False

                if got_lock:
                    try:
                        if num_parallel_workers > 0:
                            assert subset_size is None
                            data_ind = list(range(len(self)))
                            self.data = ParallelProcessorFS(self.preproc, len(data_ind), num_parallel_workers, cache_file).run()
                        else:
                            data_ind = list(range(len(self)))
                            
                            if subset_size is not None:
                                random.Random(0).shuffle(data_ind)
                                data_ind = data_ind[:subset_size]

                            self.data = [self.preproc(idx) for idx in tqdm(data_ind)]
                            self.data = [d for d in self.data if d is not None]

                            if use_cache:
                                with open(cache_file, "wb") as f:
                                    pickle.dump(self.data, f)
                    finally:
                        # Release lock
                        try:
                            if lock_file.exists():
                                os.remove(lock_file)
                        except Exception:
                            pass
                else:
                    # Waiter: spin until cache file materializes (another process is writing it)
                    while not cache_file.exists():
                        time.sleep(2)
                    with open(cache_file, "rb") as f:
                        self.data = pickle.load(f)
            else:
                with open(cache_file, "rb") as f:
                    self.data = pickle.load(f)

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
        input_prepoc, label_preproc = self.preproc_input_label(input, label)
        input_ids, label_ids = self.encode(input_prepoc), self.encode(label_preproc)

        if self.max_seqlen is not None and (input_ids.shape[0] + label_ids.shape[0]) > self.max_seqlen:
            return None

        return input_ids, label_ids
    
    def get_ids(self, idx):
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
    