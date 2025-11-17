
# Bugfix Log: 修复 Spider 评测 `'select' not found` 崩溃 & 索引错位隐患

## 一、问题背景

在 Spider Text-to-SQL 任务上训练/评估时，`trainer.evaluate()` 阶段周期性触发 SQL metric 计算。  
在这个阶段，程序在 Spider 官方的 SQL 解析脚本中崩溃，日志类似：

```text
Computing SQL metrics...:   0%|          | 0/1375 [00:00<?, ?it/s]
...
File "metrics/spider/process_sql.py", line 345, in parse_select
    assert toks[idx] == 'select', "'select' not found"
AssertionError: 'select' not found
````

堆栈可追踪到：

* `SpiderDataset.compute_metrics(...)` 调用 `SpiderMetric.compute(...)`
* `SpiderMetric.compute(...)` 再调用 Spider 官方 `evaluate(...)`
* `evaluate(...)` 内部将字符串 SQL 解析为 AST，要求 SQL 以 `select` 开头

---

## 二、根因分析

### 1. 使用了“解码后的 label 文本”当作 reference SQL

原始逻辑中，`SpiderDataset.compute_metrics` 构造 metrics 输入时：

```python
db_ids = [self.get_db_id(i) for i in range(len(self))]
assert len(db_ids) == len(eval_preds.preds)
assert len(db_ids) == len(eval_preds.labels)

predictions = list(zip(eval_preds.preds, db_ids))
references = list(zip(eval_preds.labels, db_ids))
```

这里的 `eval_preds.labels` 是 **从 token IDs decode 回来的文本**，而不是数据集中原始的 `query` 字段。
在训练/生成过程中，label 通常会套在 prompt 模板后面，可能包含：

* instruction 前缀（如 `[INST] ... [/INST]`）
* 特殊 token（如 `<s>`、`</s>`）
* 其他非 SQL 的上下文文本

当这些“前缀垃圾”出现在字符串开头时，Spider 的 parser 看到的第一个 token 就**不是** `select`，于是触发：

```python
assert toks[idx] == 'select', "'select' not found"
```

👉 **直接导致评估阶段崩溃**。

---

### 2. 潜在隐患：HF Dataset 索引与 `self.data` 索引错位

`SpiderDataset` 继承自 `NlgDatasetBase`，数据构建路径大致为：

* 原始 HF Dataset：`self.hf_dataset[0]`
* 通过 `preproc` 构造实际用于训练/评估的 `self.data`，结构类似：

  ```python
  self.data[i] = (inputs_labels, meta)
  ```

原来的 `preproc`：

```python
def preproc(self, idx):
    inputs_labels = super().preproc(idx)
    if inputs_labels is None:
        return None
    return inputs_labels, {"db_id": self.hf_dataset[0]["db_id"][idx]}
```

这是一个关键点：

* `super().preproc(idx)` 可能返回 `None` → 该样本被过滤（太长等）
* 只要返回 `None`，该样本就不会写入 `self.data`

然而旧的 metric 代码里：

```python
db_ids   = [self.get_db_id(i) for i in range(len(self))]  # 基于 self.data
labels   = eval_preds.labels                               # Trainer 基于 self.data 遍历出的预测/label
# 但如果从 hf_dataset[0] 再去按 i 访问 query，就会假设 "i == 原始 idx"，这是不可靠的
```

一旦有样本在 `preproc` 阶段被丢弃，**原始 HF Dataset 的索引就不再与 `self.data` 一一对应**。
这会导致：

* prediction / reference / db_id 之间出现静默错位；
* 即使不崩溃，评估指标也会严重失真。

---

## 三、修复思路概览

目标有三：

1. **彻底不再依赖 decoded label 文本作为 reference SQL**
2. **从源头保证 prediction / reference / db_id 三者索引完美对齐**
3. **兼容已有数据 cache，但避免“静默错位”——旧 cache 一律显式报错提示清理**

为此，我们对 `SpiderDataset` 做了两类修改：

1. 在 `preproc` 阶段，将 ground-truth SQL 写入 per-sample metadata：

   * `meta = {"db_id": ..., "query": canonical_sql}`
2. 在 `compute_metrics` 中，仅使用 `self.data` 中的 `db_id` 与 `query`：

   * 不再依赖 `hf_dataset[0][i]` / `eval_preds.labels`

---

## 四、具体代码改动

### 1. `get_input_label`：更清晰地使用 HF Dataset

**原始实现：**

```python
def get_input_label(self, idx):
    self.get_hf_dataset()

    question = self.hf_dataset[0]["question"][idx]
    db_id = self.hf_dataset[0]["db_id"][idx]
    query = self.hf_dataset[0]["query"][idx]

    table = self.hf_dataset[1][db_id]

    input = f"Question: {question}\nSchema: {table}\n"
    label = query.lower().strip()
        
    return input, label
```

**新实现：**

```python
def get_input_label(self, idx):
    hf_ds, _ = self.get_hf_dataset()

    question = hf_ds["question"][idx]
    db_id = hf_ds["db_id"][idx]
    query = hf_ds["query"][idx]

    table = self.hf_dataset[1][db_id]

    input = f"Question: {question}\nSchema: {table}\n"
    label = query.lower().strip()
        
    return input, label
```

**变化点：**

* 使用解构 `hf_ds, _ = self.get_hf_dataset()`，使得 `hf_ds` 语义更清晰（HF Dataset 本体），避免到处写 `self.hf_dataset[0]`。
* 行为上不变：`label` 仍然是 `query.lower().strip()`。

> 这一改动主要是可读性提升，为后续逻辑统一打基础。

---

### 2. `preproc`：将 canonical SQL 写入 metadata（关键改动）

**原始实现：**

```python
def preproc(self, idx):
    inputs_labels = super().preproc(idx)

    if inputs_labels is None:
        return None

    return inputs_labels, {"db_id": self.hf_dataset[0]["db_id"][idx]}
```

**新实现：**

```python
def preproc(self, idx):
    """
    Build one training/eval example.
    We:
    - Use the base class to create (input_ids, label_ids)
    - Attach metadata with db_id and canonical SQL query (lower+strip) so that
      generation metrics can safely use the ground-truth SQL without relying
      on decoded labels or fragile index assumptions.
    """
    inputs_labels = super().preproc(idx)

    if inputs_labels is None:
        return None

    hf_ds, _ = self.get_hf_dataset()
    meta = {
        "db_id": hf_ds["db_id"][idx],
        "query": str(hf_ds["query"][idx]).lower().strip(),
    }
    return inputs_labels, meta
```

**核心变化：**

* `meta` 中新增了 `"query"` 字段（canonical SQL：`lower().strip()`）。
* 由于 `preproc` 只在样本保留时才返回 `(inputs_labels, meta)`，
  👉 **`self.data[i]` 中的 `meta` 与 Trainer 实际使用的样本完全同步**。

这为后续 metrics 使用 `self.data[i][1]["query"]` 作为 reference SQL 奠定了基础。

---

### 3. `compute_metrics`：仅从 `self.data` 读 metadata，彻底丢弃 decoded labels

**原始实现：**

```python
def compute_metrics(self, eval_preds, eval_mask=None):
    if self.mode == "gen":
        metric = SpiderMetric()

        db_ids = [self.get_db_id(i) for i in range(len(self))]
        assert len(db_ids) == len(eval_preds.preds)
        assert len(db_ids) == len(eval_preds.labels)

        predictions = list(zip(eval_preds.preds, db_ids))
        references = list(zip(eval_preds.labels, db_ids))

        if eval_mask is not None:
            predictions = [predictions[i] for i in eval_mask]
            references = [references[i] for i in eval_mask]

        metrics = metric.compute(predictions, references)

        # important metric first
        return {
            "all/exec": None,
            **metrics
        }
    else:
        return {}
```

**新实现：**

```python
def compute_metrics(self, eval_preds, eval_mask=None):
    if self.mode == "gen":
        # Ensure dataset and HF view are initialized (handles lazy reload cases)
        if self.data is None:
            # Import from base without circular import
            from .base import DatasetBase  # type: ignore
            DatasetBase._ensure_materialized(self)  # type: ignore[attr-defined]

        metric = SpiderMetric()

        size = len(self)
        # Guard against legacy caches that don't carry 'query' in metadata
        sample_meta = self.data[0][1] if (self.data and len(self.data) > 0) else {}
        if "query" not in sample_meta:
            raise RuntimeError(
                "SpiderDataset.compute_metrics expected per-sample metadata with 'query', "
                "but current cache is missing it. Please clear the Spider cache directory "
                "(e.g., data/xlangai_spider_*/cache_*.pkl) or set DATA_CACHE_TAG to a new "
                "value and rerun so the dataset can be rebuilt."
            )

        # For each in-memory example, use the attached db_id + canonical SQL query.
        db_ids = [self.data[i][1]["db_id"] for i in range(size)]
        gt_queries = [self.data[i][1]["query"] for i in range(size)]

        assert len(db_ids) == len(eval_preds.preds) == len(gt_queries)

        predictions = list(zip(eval_preds.preds, db_ids))
        references = list(zip(gt_queries, db_ids))

        if eval_mask is not None:
            predictions = [predictions[i] for i in eval_mask]
            references = [references[i] for i in eval_mask]

        metrics = metric.compute(predictions, references)

        # important metric first
        return {
            "all/exec": None,
            **metrics
        }
    else:
        return {}
```

**关键点说明：**

1. **Lazy materialization 防御：**

   ```python
   if self.data is None:
       from .base import DatasetBase
       DatasetBase._ensure_materialized(self)
   ```

   * 确保在 `compute_metrics` 时，`self.data` 已经构建完成。
   * 适配某些 lazy 场景：评估可能发生在首次 materialize 之前。

2. **旧 cache 显式报错（防止静默错位）：**

   ```python
   sample_meta = self.data[0][1] if (self.data and len(self.data) > 0) else {}
   if "query" not in sample_meta:
       raise RuntimeError(
           "SpiderDataset.compute_metrics expected per-sample metadata with 'query', "
           "but current cache is missing it. Please clear the Spider cache directory "
           "(e.g., data/xlangai_spider_*/cache_*.pkl) or set DATA_CACHE_TAG to a new "
           "value and rerun so the dataset can be rebuilt."
       )
   ```

   * 旧版本 cache 中的 metadata 只有 `db_id`，没有 `query`。
   * 为避免“悄悄用错 reference”，统一在第一次 eval 时抛出明确错误，提示清理/刷新 cache。

3. **对齐保证：三个向量都来自 `self.data` 的同一索引空间**

   ```python
   db_ids     = [self.data[i][1]["db_id"]  for i in range(size)]
   gt_queries = [self.data[i][1]["query"]  for i in range(size)]

   assert len(db_ids) == len(eval_preds.preds) == len(gt_queries)
   ```

   * `eval_preds.preds` 是 Trainer 顺序遍历 `self.data` 的输出；
   * `db_ids` / `gt_queries` 也是按 `self.data[i]` 构造；
   * 三者长度一致 → **索引严格对齐**。

4. **彻底丢弃 decoded label 作为 reference：**

   ```python
   predictions = list(zip(eval_preds.preds, db_ids))
   references  = list(zip(gt_queries, db_ids))
   ```

   * `eval_preds.labels` 不再参与 reference 构造；
   * reference 永远来自 canonical ground-truth SQL（`meta["query"]`）。

---

## 五、向后兼容与缓存策略

* **旧 cache 场景：**

  * 旧版本构建的 dataset cache 不包含 `meta["query"]`，在 `compute_metrics` 中会被检测到。
  * 行为：立即抛出 `RuntimeError`，提示清理 `data/xlangai_spider_*/cache_*.pkl` 或 bump `DATA_CACHE_TAG`。
  * 设计目的：避免“旧 cache + 新代码”导致指标悄悄错位。

* **新 cache 场景：**

  * 只要重新构建 SpiderDataset（或使用新的 `DATA_CACHE_TAG`），`meta` 中就会包括 `query` 字段。
  * 此时 `compute_metrics` 正常运行，Spider parser 看到的是干净 SQL，不再触发 `'select' not found'`。

---

## 六、验证步骤（秒级 sanity check）

在配置好 Spider 环境（`SPIDER_LOCAL_DIR`、`NLTK_DATA` 等）后，无需启动完整训练，即可快速验证整条 metric 链路：

```bash
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft

python - <<'PY'
import os, sys, random
sys.path.insert(0, "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft")

from transformers import AutoTokenizer
from dataset.spider_data import SpiderDataset
from metrics.spider.spider import SpiderMetric
from trainer.trainer_utils import MambaEvalPrediction

# 1) 构造 SpiderDataset（用和训练一致的本地 JSON）
tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
ds  = SpiderDataset(tok, split="train", has_test_split=True, use_cache=True)

print("len(ds) =", len(ds))
print("sample meta[0] =", ds.data[0][1])

# 2) 抽 N 条样本，构造“伪 eval_preds”
N = min(5, len(ds))
idxs = list(range(N))

# 模拟 evaluate_generation 的输出：preds == labels（“完美预测”）
input_ids = [ds.data[i][0][0] for i in idxs]
label_ids = [ds.data[i][0][1] for i in idxs]

eval_pred = MambaEvalPrediction(
    tokenizer=tok,
    input_ids=input_ids,
    pred_ids=label_ids,
    label_ids=label_ids,
    save_file=None,
    remove_eos=True,
)

metric = SpiderMetric()
db_ids = [ds.data[i][1]["db_id"] for i in idxs]
predictions = list(zip(eval_pred.preds, db_ids))
references  = list(zip([ds.data[i][1]["query"] for i in idxs], db_ids))

print("Sanity: first pred/ref pair:")
print("  pred =", predictions[0][0])
print("  ref  =", references[0][0])

out = metric.compute(predictions, references)
keys = [k for k in out.keys() if k.endswith("/exact")]
print("metric keys (sample):", keys[:5])
for k in sorted(keys)[:5]:
    print(k, "->", out[k])
PY
```

**预期：**

* 若 cache 为旧版本（无 `query`），会直接抛出 `RuntimeError`，提示清理/重建。
* 若 cache 为新版本：

  * `len(ds)` 为正；
  * `sample meta[0]` 中包含 `{"db_id": ..., "query": "select ..."}`
  * `pred` 与 `ref` 文本非常接近（因为我们模拟了“完美预测”）；
  * `metric` 输出中的 `*/exact` 指标接近 1。

一旦这个 sanity check 通过，就可以较为放心地启动完整 Spider 训练/评估流程，不再担心：

* `'select' not found'` 崩溃；
* 或 prediction/reference 错位导致的诡异 metric。

---

## 七、影响面与结论

* **影响代码范围：**

  * 仅限 `mamba-peft/dataset/spider_data.py` 中 `SpiderDataset` 的数据构建与 metrics 计算逻辑。

* **模型行为影响：**

  * 训练阶段（loss 计算）不变；
  * 生成预测（predictions）不变；
  * **仅 evaluation 阶段的 reference 获取逻辑改变**：

    * 从 decoded labels → 改为 HF Dataset 中的 canonical ground-truth SQL (`query.lower().strip()`)

* **修复收益：**

  1. 彻底消除 Spider parser `'select' not found'` 崩溃。
  2. 避免因样本过滤造成的索引错位，保证 metrics 对齐正确。
  3. 对旧 cache 采用“显式 fail fast”，防止静默错误。

整体来看，这次修复将 Spider 评测链路从“脆弱 + 易错位”升级为“索引严格对齐 + parser 输入稳定”，可以安全支撑后续大规模训练与对比实验。

  
### 最小必要清理

在你已经：

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft
rm -f data/xlangai_spider/cache_xlangai_spider-tvt_*_seqlen*.pkl
```

的基础上，为了 100% 确保用到的是**带 `query` meta 的新 cache**，建议再做两件事（二选一）：

- **方案 A：一次性彻底删 Spider 的所有旧 cache**
  ```bash
  rm -f data/xlangai_spider/cache_*.pkl
  rm -f data/xlangai_spider/cache_*.pkl.tmp 2>/dev/null || true
  rm -f data/xlangai_spider/cache_*.pkl.lock 2>/dev/null || true
  ```
  这样无论是 `train` 还是 `val`，都会在下次运行时用新代码重建。
 