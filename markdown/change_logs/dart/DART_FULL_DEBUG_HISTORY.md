## DART Full Debug History (continuous log)\n
\n
This document records high-signal incidents, root causes, and fixes across the DART dataset pipeline (data loading, caching, training, and evaluation), along with a comprehensive index of relevant Python and bash files that are useful for analysis and reproduction.\n
\n

## DART 数据集全量排障与修复纪实（从首次报错到全面解决）

本文件完整记录了过去一周围绕 DART（GEM/dart）训练流水线的所有问题、分析与修复。包含：错误现象、根因、代码改动、验证方式、脚本与命令、经验教训与复用清单。目标是让后续读者可以“一次读懂、一次复现”。


### 目录
- 背景与范围
- 时间线（按发生顺序）
- 关键问题与修复（按主题归档）
- 落盘的脚本与用法
- 稳定的训练命令与变量
- 缓存管理与常见恢复手段
- 经验教训与建议
- 复用清单（Quick checklist）


## 背景与范围

- 工程：`zh-LAT-peft/mamba-peft`（GLA + LoRA/PEFT 训练框架）
- 目标：让 DART（RDF→文本）在本工程中可以稳定完成训练/验证/测试，配合自动化脚本批量运行与聚合。
- 相关组件：
  - Hugging Face `datasets` 与 `transformers`
  - 本仓库的数据模块（`mamba-peft/dataset/*`）与训练脚本（`mamba-peft/train.py`、`scripts/train/new/*.sh`）
  - 自研并行预处理器 `utils/parallel_processor_fs.py`
  - 评测与生成相关配置（`eval_gen` 自动注入、环境变量透传等）


## 时间线（按发生顺序）

> 注：时间来自你提供的日志与会话记录（UTC+8）。

### 11-04 早间（首次失败）
- 症状：`RuntimeError: Dataset scripts are no longer supported, but found dart.py`
- 根因：HF Datasets v3 起不再支持“脚本式”数据集（同名冲突）；仓库内本地 `dart.py` 与 Hub 数据集 ID 冲突。
- 处理：将 `dart.py` → `dart_data.py`，同时 `samsum.py` → `samsum_data.py`，`spider.py` → `spider_data.py` 并更新所有导入路径。

### 11-04 夜间
- 症状：`AssertionError: GEM/dart train files not found under .../data/GEM_dart`
- 根因：本地目录结构与 Hub 快照不一致；仅有 `train.json / validation.json / test.json` 扁平文件；原逻辑只找特定结构或 parquet。
- 处理：实现稳健的文件发现与回退：`_snapshot_local_root`、`_find_split_files`、`_download_candidates`，优先本地文件，其次快照，最后显式下载候选文件名。

### 11-05 上午
- 症状：`KeyError: 'source'` 或 `KeyError: 'text'`（DataFrame 处理时 explode 等步骤失败）。
- 根因：不同版本/来源的 DART JSON 存在多种 schema：`annotations`/`references`/`targets`/`text`/`outputs` 等键名与结构差异，且有 list/dict/str 多形态。
- 处理：重写 `build_lists` 正规化逻辑，保证无论输入如何，都能构建出 `source: list[str]` 与 `text: list[str]`；在 LM 模式下再将多参考展开为逐行样本。

### 11-05 中午
- 症状：`ValueError: evaluation strategy steps requires either non-zero --eval_steps or --logging_steps`
- 根因：环境变量 `HP_EVAL_STEPS/HP_SAVE_STEPS/HP_LOGGING_STEPS` 未被启动脚本正确透传。
- 处理：修复 `scripts/train/new/gla_batch_tmux.sh`，显式收集并透传上述 `HP_*` 与 `EVAL_GEN*`、`GRADIENT_CHECKPOINTING`、`LOGITS_TO_KEEP` 等变量。

### 11-06 凌晨
- 症状：`ValueError: num_samples should be a positive integer value, but got num_samples=0`
- 伴随现象：日志只看到 `val_gen` 缓存写入，`train` 未写入或为空。
- 根因：命中“空缓存”（历史失败的缓存文件被直接读取）；并行预处理未打印任何异常细节，导致误判。
- 处理：
  - 清理缓存：`data/GEM_dart/cache_GEM_dart_train*.pkl` 与 `data/GEM_dart/parts/cache_GEM_dart_train_part_*.pkl`。
  - 加固 `parallel_processor_fs.py`：
    - Worker 内 `try/except` 捕获并打印 traceback。
    - 聚合阶段统计与告警 `None` 数量，若全为 `None` 则明确报错。
  - 提供快速校验脚本（见“落盘脚本与用法”）。

### 11-08 凌晨（修复后再次推进）
- 现象：`load_df()` 成功（展开后 ~62,659 行），但训练并行阶段日志暴增 ~50MB，重复错误：
  - `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`
- 根因（两处）：
  1) pandas `to_pandas()` 将嵌套 `list` 转为 `numpy.ndarray`。`build_lists` 仅检查 `isinstance(ann, list)`，导致本应可用的 `annotations` 被跳过（早期问题点，已修）。
  2) `linearize_triples()` 中 `triples = triples or []` 在 `triples` 为 `np.ndarray` 时触发歧义报错（本次新暴露问题）。


## 关键问题与修复（按主题归档）

### 1) HF Datasets v3 禁止脚本式数据集 & 同名冲突
- 症状：`Dataset scripts are no longer supported, but found dart.py`
- 根因：本地 `dart.py` 与 Hub `GEM/dart` 同名，引发新版策略拦截。
- 修复：文件更名并更新导入：`dart_data.py` / `samsum_data.py` / `spider_data.py`；集中在 `dataset/__init__.py` 注入新类名。

### 2) 本地文件发现与离线优先
- 症状：看不到 `train/val/test`，`AssertionError: ... files not found`。
- 根因：目录结构/命名差异、parquet/jsonl/json 混用。
- 修复：
  - `_snapshot_local_root()`：优先使用 `DART_LOCAL_DIR/HP_DART_LOCAL_DIR` 或 `data/GEM_dart`；检测离线或已有文件时直接短路。
  - `_find_split_files()`：对 `train/val/test` 多 hint、多后缀优先级（parquet > jsonl > json）。
  - `_download_candidates()`：显式下载常见文件名。

### 3) 规范化 `source/text` 与 LM 模式展开
- 症状：`KeyError: 'source'/'text'` 或 `explode` 失败。
- 根因：多种 schema 混杂；字段缺失或是非列表。
- 修复：`build_lists` 统一收敛为 `list[str]`；LM 模式将多参考拆分为多行，最终 `DataFrame(tripleset, source, text)` 三列齐备。

### 4) 训练循环频率参数未透传
- 症状：`evaluation strategy steps requires either non-zero --eval_steps or --logging_steps`
- 根因：启动脚本未把 `HP_*` 传到 Python 进程。
- 修复：`gla_batch_tmux.sh` 增加采集与透传打印；`train.py` 支持 `HP_VAL_SPLIT/HP_DATA`、自动注入 `eval_gen`（若数据是生成任务或 `EVAL_GEN=1`）。

### 5) 空缓存导致 `num_samples=0`
- 症状：DataLoader 取样本时报 `num_samples=0`；日志只见 `val_gen` 写入。
- 根因：历史失败缓存被沿用；并行异常未打印（误以为数据集为空）。
- 修复：
  - 清缓存（见后文命令）。
  - `parallel_processor_fs.py`：
    - Worker 层捕获异常并打印详细 traceback。
    - 聚合阶段统计 `None` 并打印 Warning/ERROR。
  - 验证脚本确保“真实数据长度”与“缓存样本数”一致。

### 6) pandas 将 list → numpy.ndarray 引发两类 Bug
- 现象 A：`build_lists` 只识别 `list`，`annotations` 变 `ndarray` 后被跳过 → `load_df()` 为空。
  - 修复：`isinstance(ann, (list, np.ndarray))`；全面接受两种类型。
- 现象 B：`linearize_triples` 中 `triples = triples or []` 对 `ndarray` 抛 `ValueError`。
  - 修复：显式判断 `None` 或空：
    - `if triples is None or (isinstance(triples, (list, np.ndarray)) and len(triples) == 0): triples = []`

### [2025-11-08] DataLoader first batch TypeError caused by dataset cache not being loaded\n
\n
- Symptoms\n
  - Trainers and minimal probes crashed on the first batch with:\n
\n
```\n
TypeError: 'NoneType' object is not subscriptable\n
  File .../dataset/base.py, line 132, in get_ids\n
    return self.data[idx]\n
```\n
\n
  - Reproduced even with num_workers=0 and with a valid, large train set (`len(ds)=62659`).\n
\n
- Root cause\n
  - In `DatasetBase.__init__` (when `use_cache=True`): if a cache file already exists before dataset instantiation and there is no active writer, the code did not load the cache into `self.data`. Only two paths were handled:\n
    - “no cache yet” → build and write (sets `self.data`)\n
    - “another process writing” → wait, then load (sets `self.data`)\n
  - If the cache already existed, initialization skipped both paths and left `self.data=None`. `len(ds)` still worked (via `load_df()`), but `__getitem__` failed at the first batch.\n
\n
- Why it surfaced “only sometimes”\n
  - First runs after cache clear: building path sets `self.data` → OK.\n
  - Subsequent runs (cache present): init didn’t load existing cache → `self.data` stayed None until `__getitem__` attempted indexing.\n
\n
- Fix (elegant, minimal)\n
  - Add a fast-path that, when `cache_file` exists and no lock file is present, loads the cache in `__init__` immediately.\n
  - Keep cooperative locking for build cases; add atomic write (`.tmp` then `os.replace`) to prevent partial or corrupted caches.\n
  - If another process is building (lock exists), wait until the lock clears and then load the finished cache.\n
\n
- Validation\n
  - `python -u mamba-peft/tools/debug_train_start.py`: train/val both print batch0 keys and tensor shapes; no `NoneType` errors.\n
  - Full training proceeds; evaluation no longer crashes from malformed batches.\n
\n
---\n
\n
## Relevant files index (Python and Bash)\n
\n
The following files are the most relevant to DART loading, preprocessing, caching, training, generation, evaluation, and orchestration. This list is intentionally exhaustive to aid future investigations.\n
\n
### Dataset and preprocessing (Python)\n
- `mamba-peft/dataset/dart_data.py` — DART-specific dataset, split resolution, normalization of source/text, linearization, get_input_label, metrics.\n
- `mamba-peft/dataset/dart.py` — Additional DART dataset utilities (if used by older paths).\n
- `mamba-peft/dataset/base.py` — Base dataset class, cache building/loading, sample preprocessing, mode handling.\n
- `mamba-peft/dataset/collator.py` — Data collator used by DataLoader and Trainer.\n
- `mamba-peft/scripts/preproc/preproc_dart.py` — Optional standalone preprocessing script for DART.\n
- `dataset_downloader_v3.py` — Helper for dataset downloads (if used to stage local data).\n
\n
### Parallel preprocessing and utilities (Python)\n
- `mamba-peft/utils/parallel_processor_fs.py` — Parallel workers, atomic part-file writes, aggregation.\n
- `mamba-peft/utils/utils.py` — General utilities.\n
- `mamba-peft/utils/model_utils.py` — Model-related utilities used during setup/evaluation.\n
\n
### Training and orchestration (Python)\n
- `mamba-peft/train.py` — Top-level training entry.\n
- `mamba-peft/train_shared.py` — Trainer construction, argument plumbing, training/eval hooks.\n
- `mamba-peft/trainer/mamba_trainer.py` — Custom Trainer (evaluation and generation hooks, logging, guards).\n
- `mamba-peft/trainer/trainer_utils.py` — Trainer utility functions used by the training loop.\n
- `prefetch_mamba.py` — Optional prefetch helpers.\n
\n
### Generation/decoding (Python)\n
- `mamba-peft/mamba_ssm_peft/utils/decoder.py` — Beam search decoding; earlier scoping fix for BeamSearchScorer.\n
- `mamba-peft/mamba_ssm_peft/utils/generation.py` — Generation utilities used during evaluation.\n
- `mamba-peft/mamba_ssm_peft/utils/beam_search.py` — Beam search helpers.\n
- `mamba-peft/mamba_ssm_peft/utils/hf.py` — HF integration helpers.\n
\n
### Debug and verification scripts (Python)\n
- `mamba-peft/tools/debug_train_start.py` — Minimal probe for cache presence, `self.data` state, and first DataLoader batch.\n
- `quick_check_dart_eval.py` (recommended placement: repo root) — End-to-end quick check for train/val dataset shapes and first batches, cached and uncached.\n
\n
### Bash scripts for training and preprocessing\n
- `mamba-peft/scripts/train/new/gla_batch_tmux.sh` — Batch orchestrator; passes/exports critical env vars to runner scripts.\n
- `mamba-peft/scripts/train/new/all.sh` — Training runner.\n
- `mamba-peft/scripts/train/new/dart130m.sh` — DART-targeted training runner (example config).\n
- `mamba-peft/scripts/train/gla_rounds.sh` — Round-based orchestrations (if used).\n
- `mamba-peft/scripts/preproc/preproc_data.sh` — Data preprocessing driver.\n
- `run_gla_finetune.sh` — Repo-level convenience script for finetuning runs.\n
- `run_dart_training.sh` — Repo-level convenience script for DART runs.\n
\n
### Environment variables (high-signal for DART)\n
- Data/cache control: `DART_LOCAL_DIR`, `DATA_CACHE_TAG` (or `CACHE_TAG`).\n
- Trainer/eval cadence: `HP_EVAL_STEPS`, `HP_SAVE_STEPS`, `HP_LOGGING_STEPS`.\n
- Generation: `EVAL_GEN`, `EVAL_GEN_MAX_LENGTH`, `EVAL_GEN_MIN_LENGTH`, `EVAL_GEN_NUM_BEAMS`.\n
- Performance: `NUM_DATA_WORKERS`, `GRADIENT_CHECKPOINTING`, `LOGITS_TO_KEEP`, `PREFETCH_FACTOR`.\n
- Torch/tokenizers: `PYTORCH_CUDA_ALLOC_CONF`, `TOKENIZERS_PARALLELISM`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`.\n
\n
---\n
\n
## Notes\n
- When pre-warming caches (single-experiment phase), ensure the same `DATA_CACHE_TAG` and env are used as subsequent parallel runs, so all workers load the same cache fast-path.\n
- After changes to dataset normalization (e.g., list flattening, sep token handling), clear only the affected split caches (e.g., `val_gen`) to expedite re-tests.\n
\n
## 落盘的脚本与用法

- 清理/验证/训练一体化脚本：
  - `run_dart_training.sh`（最终训练一键脚本，含清缓存与启动）
  - `fix_dart_remote.sh`（清缓存 + 子集校验 + 命令提示）
  - `QUICK_FIX_COMMANDS.txt`（手工执行的完整命令序列）

- 调试定位脚本：
  - `debug_dart_loading.py`（逐步检查本地文件、HF 加载、`load_df`、`get_input_label`、`preproc`）
  - `quick_debug_dart.py`（极简版：头部样本、关键字段结构）
  - `trace_build_lists.py`（逐行追踪 `build_lists` 返回值，核对 schema）
  - `compare_loading_methods.py`（对比两种加载路径的 `annotations` 类型与内容一致性）
  - `test_dartdataset_full.py`（完整贯通 `DartDataset` 每个关键方法并打印中间态）
  - `verify_dart_fix.py`（最终验证：`load_df()` 行数 + 最小数据集初始化 + 取首样本）

> 这些脚本配合改进后的并行预处理日志（Worker 级 traceback + 聚合统计），让“定位→修复→验证”的闭环极快收敛。


## 稳定的训练命令与变量

```bash
# 训练前建议先清理旧缓存（特别是 parts/）：
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft
rm -f data/GEM_dart/cache_GEM_dart_train*.pkl
rm -f data/GEM_dart/parts/cache_GEM_dart_train_part_*.pkl

cd scripts/train/new
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUM_DATA_WORKERS=4 \
GRADIENT_CHECKPOINTING=true \
LOGITS_TO_KEEP=1 \
HP_EVAL_STEPS=2000 HP_SAVE_STEPS=2000 HP_LOGGING_STEPS=200 \
EVAL_GEN=1 EVAL_GEN_MAX_LENGTH=128 EVAL_GEN_MIN_LENGTH=5 EVAL_GEN_NUM_BEAMS=5 \
./gla_batch_tmux.sh --suite E10 --round all \
  --pairs "87:dart" \
  --gpus "1" \
  --gpu-plan "1"
```

- 说明：
  - `EVAL_GEN=*` 将由 `train.py` 自动注入生成参数（若数据为生成任务或 `EVAL_GEN=1`）。
  - `HP_VAL_SPLIT` 可设置为 `train|val|test` 控制评估 split。
  - 大数据集可将 `HP_*_STEPS` 调大、或调整并行/线程参数。


## 缓存管理与常见恢复手段

- 何时清缓存：
  - 看到 `num_samples=0`、或日志中并无 `Parallel processing`/`Wrote ... part_*.pkl`/`Aggregating`。
  - 训练集只生成 `val_gen` 缓存时（典型“命中空缓存”信号）。
  - 修改了预处理逻辑、schema 适配、或 `eval_gen`/模式切换后。

- 清理指令（训练前）：
```bash
rm -f data/GEM_dart/cache_GEM_dart_train*.pkl
rm -f data/GEM_dart/parts/cache_GEM_dart_train_part_*.pkl
```

- 快速验真：
```bash
python - <<'PY'
import os, sys
sys.path.insert(0, "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")
os.environ["DART_LOCAL_DIR"] = "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart"
from transformers import AutoTokenizer
from dataset.dart_data import DartDataset
tok = AutoTokenizer.from_pretrained("/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B", trust_remote_code=True)
ds = DartDataset(tok, split="train", use_cache=False)
print("train samples (no cache):", len(ds))
PY
```


## 经验教训与建议

1) 生态变化要“避坑”：HF Datasets v3 禁用脚本式数据集与 `trust_remote_code` 的策略变化，需要尽快切到“文件级加载 + 本地优先 + 离线可用”。  
2) schema 必须“自适应”：统一入口 `build_lists` 收敛多种字段/结构，LM 展开逻辑与生成模式隔离。  
3) pandas/ndarray 兼容性：凡是从 HF Dataset `to_pandas()` 的字段，必须考虑 `list` 与 `np.ndarray` 两态，避免 `or []`、`isinstance(..., list)` 等陷阱。  
4) 并行预处理要“可解释”：Worker try/except + 聚合统计是关键。静默失败会极大增加排障难度。  
5) 变量透传“一次到位”：启动脚本务必打印并透传所有 `HP_*`/`EVAL_GEN*`，避免“命令行参数生效假象”。  
6) 缓存不可迷信：看到“异常现象”第一时间清缓存；配合“no cache”快速验真脚本。  


## 复用清单（Quick checklist）

- [ ] 训练前清理 `data/GEM_dart/cache_*` 与 `data/GEM_dart/parts/*`（尤其在改动逻辑后）  
- [ ] `DART_LOCAL_DIR` 指向可用的本地数据目录（含 `train/validation/test.json`）  
- [ ] `build_lists` 支持 `list` 与 `np.ndarray`；`linearize_triples` 不使用布尔短路判断数组  
- [ ] 并行预处理日志出现：`Parallel processing`、`Wrote ... part_*.pkl`、`Aggregating ...`  
- [ ] 启动脚本打印并透传 `HP_*`、`EVAL_GEN*`、`GRADIENT_CHECKPOINTING`、`LOGITS_TO_KEEP`  
- [ ] `train.py` 自动注入 `eval_gen`（或通过 `EVAL_GEN=1` 强制）  
- [ ] 失败时优先跑最小子集/无缓存验证脚本，定位问题点  


---

如需对 SAMSum / Spider 复用本路线：按本文件的“文件级加载 + schema 正规化 + ndarray 兼容 + 并行可解释 + 缓存策略 + 启动透传”六步走，即可快速稳定落地。祝训练顺利！


